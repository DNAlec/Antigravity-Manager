//! Shared count-resource discovery. Generation routes are never rewritten here.
//!
//! A successful HTTP response alone is insufficient: some Cloud Code resources
//! count encoded image bytes. Probe compression invariance before sending image
//! history, and keep capability results scoped to the account and endpoint.

use base64::Engine;
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::ImageEncoder;
use parking_lot::Mutex;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};
use tokio::sync::{Mutex as AsyncMutex, Semaphore};

use super::InboundThinkingPipeline;

const MAX_CANDIDATES: usize = 3;
const MAX_CACHE_ENTRIES: usize = 1024;
const CAPABILITY_TTL: Duration = Duration::from_secs(600);
const UNSUPPORTED_TTL: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CountError {
    InvalidRequest,
    Transport,
    Http(u16),
    InvalidResponse,
    NoReliableResource,
    Timeout,
}

impl CountError {
    pub fn message(self) -> &'static str {
        match self {
            Self::InvalidRequest => "Invalid countTokens request",
            Self::Transport => "Token counting upstream is unavailable",
            Self::Http(_) => "Token counting request was rejected by upstream",
            Self::InvalidResponse => "Upstream returned an invalid token count",
            Self::NoReliableResource => {
                "No image counting resource passed compression validation for this model"
            }
            Self::Timeout => "Token counting exceeded the request deadline",
        }
    }

    pub fn status(self) -> u16 {
        match self {
            Self::InvalidRequest => 400,
            Self::Http(code) if (400..600).contains(&code) => code,
            Self::Timeout => 504,
            _ => 502,
        }
    }

    fn try_next_endpoint(self) -> bool {
        matches!(
            self,
            Self::Transport | Self::Timeout | Self::Http(408 | 404 | 500..=599)
        )
    }
}

pub struct CountResult {
    pub tokens: u64,
    pub model: String,
    pub image_checked: bool,
}

pub(crate) trait CountTransport: Sync {
    fn send(
        &self,
        endpoint: &str,
        body: Value,
    ) -> impl Future<Output = Result<Value, CountError>> + Send;
}

/// Exact route first; known effort syntax only supplies *candidates*, never an
/// unconditional model mapping. Unknown names keep their original spelling.
pub fn model_candidates(target: &str) -> Vec<String> {
    let mut current = target.trim().trim_start_matches("models/").to_string();
    let mut candidates = vec![current.clone()];
    while candidates.len() < MAX_CANDIDATES {
        let next = crate::proxy::common::model_mapping::strip_tier_suffix(&current);
        if next.is_empty() || next == current {
            break;
        }
        candidates.push(next.clone());
        current = next;
    }
    candidates
}

pub fn parse_count(response: &Value) -> Result<u64, CountError> {
    response
        .get("response")
        .and_then(|r| r.get("totalTokens"))
        .or_else(|| response.get("totalTokens"))
        .and_then(Value::as_u64)
        .ok_or(CountError::InvalidResponse)
}

fn has_image(value: &Value) -> bool {
    match value {
        Value::Object(obj) => {
            for key in ["inlineData", "inline_data", "fileData", "file_data"] {
                if let Some(media) = obj.get(key) {
                    let mime = media.get("mimeType").or_else(|| media.get("mime_type"));
                    if mime
                        .and_then(Value::as_str)
                        .map_or(true, |s| s.to_ascii_lowercase().starts_with("image/"))
                    {
                        return true;
                    }
                }
            }
            obj.values().any(has_image)
        }
        Value::Array(values) => values.iter().any(has_image),
        _ => false,
    }
}

/// The two PNGs encode identical pixels. No client content enters a probe.
fn probe_images() -> Result<&'static [String; 2], CountError> {
    static IMAGES: LazyLock<Result<[String; 2], CountError>> = LazyLock::new(|| {
        let pixels = [68, 136, 204].repeat(768 * 768);
        let mut encoded = Vec::new();
        for compression in [CompressionType::Fast, CompressionType::Best] {
            let mut png = Vec::new();
            PngEncoder::new_with_quality(&mut png, compression, FilterType::NoFilter)
                .write_image(&pixels, 768, 768, image::ExtendedColorType::Rgb8)
                .map_err(|_| CountError::InvalidResponse)?;
            encoded.push(base64::engine::general_purpose::STANDARD.encode(png));
        }
        if encoded[0].len() == encoded[1].len() {
            return Err(CountError::InvalidResponse);
        }
        Ok([encoded.remove(0), encoded.remove(0)])
    });
    IMAGES.as_ref().map_err(|e| *e)
}

#[derive(Clone, Copy)]
struct Capability {
    supported: bool,
    expires: Instant,
}

type CacheKey = (String, String, String);
type CacheSlot = Arc<AsyncMutex<Option<Capability>>>;

pub struct CountCapabilities {
    entries: Mutex<HashMap<CacheKey, CacheSlot>>,
    probes: Semaphore,
}

impl Default for CountCapabilities {
    fn default() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            probes: Semaphore::new(2),
        }
    }
}

impl CountCapabilities {
    pub fn clear(&self) {
        self.entries.lock().clear();
    }

    fn slot(&self, key: CacheKey) -> CacheSlot {
        let mut entries = self.entries.lock();
        if let Some(slot) = entries.get(&key) {
            return Arc::clone(slot);
        }
        if entries.len() >= MAX_CACHE_ENTRIES {
            entries.retain(|_, slot| {
                Arc::strong_count(slot) > 1
                    || slot.try_lock().map_or(true, |entry| {
                        entry.is_some_and(|v| v.expires > Instant::now())
                    })
            });
        }
        let slot = Arc::new(AsyncMutex::new(None));
        // A full cache must not grow without bound or evict in-flight probes.
        if entries.len() < MAX_CACHE_ENTRIES {
            entries.insert(key, Arc::clone(&slot));
        }
        slot
    }

    async fn supports_images(
        &self,
        transport: &impl CountTransport,
        endpoint: &str,
        account: &str,
        model: &str,
    ) -> Result<bool, CountError> {
        let slot = self.slot((account.into(), endpoint.into(), model.into()));
        // The caller's deadline also bounds waiting for another probe.
        let mut cached = slot.lock().await;
        if let Some(value) = *cached {
            if value.expires > Instant::now() {
                return Ok(value.supported);
            }
        }
        let _permit = self
            .probes
            .acquire()
            .await
            .map_err(|_| CountError::Transport)?;
        let mut counts = Vec::new();
        for image in probe_images()? {
            let request = json!({"contents": [{"role": "user", "parts": [{
                "inlineData": {"mimeType": "image/png", "data": image}
            }]}]});
            let envelope = InboundThinkingPipeline::build_count_tokens_envelope(&request, model);
            // A slow capability probe must leave time for the next endpoint.
            let response =
                tokio::time::timeout(Duration::from_secs(15), transport.send(endpoint, envelope))
                    .await
                    .map_err(|_| CountError::Timeout)?;
            match response {
                Ok(response) => match parse_count(&response) {
                    Ok(count) => counts.push(count),
                    Err(_) => break,
                },
                // The probe has a known valid shape. These statuses indicate
                // that this resource cannot count it; auth/rate limits do not.
                Err(CountError::Http(400 | 404 | 422)) => break,
                Err(error) => return Err(error),
            }
        }
        let supported = counts.len() == 2 && counts[0] > 0 && counts[0] == counts[1];
        *cached = Some(Capability {
            supported,
            expires: Instant::now()
                + if supported {
                    CAPABILITY_TTL
                } else {
                    UNSUPPORTED_TTL
                },
        });
        Ok(supported)
    }

    pub(crate) async fn count(
        &self,
        transport: &impl CountTransport,
        endpoints: &[&str],
        account: &str,
        target: &str,
        request: &Value,
    ) -> Result<CountResult, CountError> {
        let request = request.get("request").unwrap_or(request);
        if !request.is_object() || target.trim().trim_start_matches("models/").is_empty() {
            return Err(CountError::InvalidRequest);
        }
        let image_checked = has_image(request);
        let candidates = model_candidates(target);
        let mut last_error = CountError::NoReliableResource;
        'endpoints: for endpoint in endpoints {
            for model in &candidates {
                if image_checked {
                    match self
                        .supports_images(transport, endpoint, account, model)
                        .await
                    {
                        Ok(true) => {}
                        Ok(false) => continue,
                        Err(error) if error.try_next_endpoint() => {
                            last_error = error;
                            continue 'endpoints;
                        }
                        Err(error) => return Err(error),
                    }
                }
                let envelope = InboundThinkingPipeline::build_count_tokens_envelope(request, model);
                match transport.send(endpoint, envelope).await {
                    Ok(response) => {
                        let tokens = parse_count(&response)?;
                        if image_checked && tokens == 0 {
                            return Err(CountError::InvalidResponse);
                        }
                        return Ok(CountResult {
                            tokens,
                            model: model.clone(),
                            image_checked,
                        });
                    }
                    Err(error) if error.try_next_endpoint() => {
                        last_error = error;
                        continue 'endpoints;
                    }
                    Err(error) => return Err(error),
                }
            }
        }
        Err(last_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    enum Counter {
        Pixels,
        Bytes,
        Error(CountError),
        Missing,
        Stall,
    }

    struct FakeTransport {
        counters: HashMap<(&'static str, &'static str), Counter>,
        calls: Mutex<Vec<(String, Value)>>,
    }

    impl FakeTransport {
        fn new(counters: &[((&'static str, &'static str), Counter)]) -> Self {
            Self {
                counters: counters.iter().copied().collect(),
                calls: Mutex::new(Vec::new()),
            }
        }
    }

    impl CountTransport for FakeTransport {
        async fn send(&self, endpoint: &str, body: Value) -> Result<Value, CountError> {
            self.calls.lock().push((endpoint.into(), body.clone()));
            tokio::task::yield_now().await;
            let model = body["request"]["model"]
                .as_str()
                .unwrap()
                .strip_prefix("models/")
                .unwrap();
            match self
                .counters
                .get(&(endpoint, model))
                .copied()
                .unwrap_or(Counter::Error(CountError::Http(404)))
            {
                Counter::Pixels => Ok(json!({"totalTokens": 258})),
                Counter::Bytes => {
                    let data = body["request"]["contents"][0]["parts"][0]["inlineData"]["data"]
                        .as_str()
                        .unwrap();
                    Ok(
                        json!({"totalTokens": base64::engine::general_purpose::STANDARD.decode(data).unwrap().len()}),
                    )
                }
                Counter::Error(error) => Err(error),
                Counter::Missing => Ok(json!({})),
                Counter::Stall => std::future::pending().await,
            }
        }
    }

    fn image_request() -> Value {
        json!({"contents": [{"role": "user", "parts": [{"inlineData": {
            "mimeType": "image/png", "data": probe_images().unwrap()[0]
        }}]}], "systemInstruction": {"parts": [{"text": "client context"}]}})
    }

    #[test]
    fn candidates_cover_future_families_without_name_tables() {
        for model in ["future-99-nebula", "claude-future", "gpt-future", "custom"] {
            assert_eq!(
                model_candidates(&format!(" models/{model}-tiered-xhigh ")),
                vec![
                    format!("{model}-tiered-xhigh"),
                    format!("{model}-tiered"),
                    model.into()
                ]
            );
        }
        assert_eq!(
            model_candidates("future-high-preview"),
            vec!["future-high-preview"]
        );
    }

    #[test]
    fn probes_have_identical_pixels_and_different_byte_sizes() {
        let images = probe_images().unwrap().each_ref().map(|data| {
            base64::engine::general_purpose::STANDARD
                .decode(data)
                .unwrap()
        });
        assert_ne!(images[0].len(), images[1].len());
        let decoded = images.map(|bytes| image::load_from_memory(&bytes).unwrap().to_rgb8());
        assert_eq!(decoded[0], decoded[1]);
        assert_eq!(decoded[0].dimensions(), (768, 768));
    }

    #[test]
    fn rejects_missing_negative_and_fractional_counts() {
        for value in [
            json!({}),
            json!({"totalTokens": -1}),
            json!({"totalTokens": 1.5}),
            json!({"totalTokens": "2"}),
        ] {
            assert_eq!(parse_count(&value), Err(CountError::InvalidResponse));
        }
        assert_eq!(
            parse_count(&json!({"response": {"totalTokens": 42}})),
            Ok(42)
        );
        assert_eq!(parse_count(&json!({"totalTokens": 0})), Ok(0));
    }

    #[test]
    fn recognizes_snake_case_file_data_and_nested_tool_images() {
        for part in [
            json!({"inline_data": {"mime_type": "image/jpeg", "data": "x"}}),
            json!({"fileData": {"fileUri": "gs://image", "mimeType": "image/png"}}),
            json!({"functionResponse": {"parts": [{"file_data": {"file_uri": "gs://image"}}]}}),
        ] {
            assert!(has_image(
                &json!({"request": {"contents": [{"parts": [part]}]}})
            ));
        }
        assert!(!has_image(
            &json!({"inlineData": {"mimeType": "audio/wav", "data": "x"}})
        ));
    }

    #[tokio::test]
    async fn unverified_tier_falls_back_only_after_base_passes_probes() {
        let transport = FakeTransport::new(&[
            (("daily", "future-high"), Counter::Bytes),
            (("daily", "future"), Counter::Pixels),
        ]);
        let cache = CountCapabilities::default();
        let request = image_request();
        let result = cache
            .count(&transport, &["daily"], "account", "future-high", &request)
            .await
            .unwrap();
        assert_eq!(result.model, "future");
        assert_eq!(result.tokens, 258);
        assert!(result.image_checked);
        let calls = transport.calls.lock();
        assert_eq!(calls.len(), 5);
        assert_eq!(
            calls.last().unwrap().1["request"]["contents"],
            request["contents"]
        );
        assert_eq!(
            calls.last().unwrap().1["request"]["systemInstruction"],
            request["systemInstruction"]
        );
    }

    #[tokio::test]
    async fn supported_original_resource_is_never_stripped() {
        let transport = FakeTransport::new(&[(("daily", "future-high"), Counter::Pixels)]);
        let result = CountCapabilities::default()
            .count(
                &transport,
                &["daily"],
                "account",
                "future-high",
                &image_request(),
            )
            .await
            .unwrap();
        assert_eq!(result.model, "future-high");
        assert_eq!(transport.calls.lock().len(), 3);
    }

    #[tokio::test]
    async fn unreliable_resources_return_error_without_sending_client_history() {
        let transport = FakeTransport::new(&[
            (("daily", "future-high"), Counter::Bytes),
            (("daily", "future"), Counter::Bytes),
        ]);
        let cache = CountCapabilities::default();
        let error = cache
            .count(
                &transport,
                &["daily"],
                "account",
                "future-high",
                &image_request(),
            )
            .await
            .err();
        assert_eq!(error, Some(CountError::NoReliableResource));
        assert!(transport
            .calls
            .lock()
            .iter()
            .all(|(_, body)| body["request"].get("systemInstruction").is_none()));
        let before = transport.calls.lock().len();
        assert!(cache
            .count(
                &transport,
                &["daily"],
                "account",
                "future-high",
                &image_request()
            )
            .await
            .is_err());
        assert_eq!(transport.calls.lock().len(), before);
    }

    #[tokio::test]
    async fn concurrent_requests_share_probes_but_not_client_counts() {
        let transport = FakeTransport::new(&[(("daily", "future"), Counter::Pixels)]);
        let cache = CountCapabilities::default();
        let request = image_request();
        let (a, b) = tokio::join!(
            cache.count(&transport, &["daily"], "a", "future", &request),
            cache.count(&transport, &["daily"], "a", "future", &request)
        );
        assert!(a.is_ok() && b.is_ok());
        assert_eq!(transport.calls.lock().len(), 4);
        cache
            .count(&transport, &["daily"], "b", "future", &request)
            .await
            .unwrap();
        assert_eq!(transport.calls.lock().len(), 7);
    }

    #[tokio::test]
    async fn endpoint_fallback_requires_its_own_validation() {
        let transport = FakeTransport::new(&[
            (("daily", "future"), Counter::Bytes),
            (("sandbox", "future"), Counter::Pixels),
        ]);
        let result = CountCapabilities::default()
            .count(
                &transport,
                &["daily", "sandbox"],
                "a",
                "future",
                &image_request(),
            )
            .await
            .unwrap();
        assert_eq!(result.tokens, 258);
        assert_eq!(
            transport
                .calls
                .lock()
                .iter()
                .map(|(endpoint, _)| endpoint.as_str())
                .collect::<Vec<_>>(),
            ["daily", "daily", "sandbox", "sandbox", "sandbox"]
        );
    }

    #[tokio::test]
    async fn rate_limits_are_not_cached_as_unsupported_models() {
        let transport =
            FakeTransport::new(&[(("daily", "future"), Counter::Error(CountError::Http(429)))]);
        let cache = CountCapabilities::default();
        for _ in 0..2 {
            assert_eq!(
                cache
                    .count(&transport, &["daily"], "a", "future", &image_request())
                    .await
                    .err(),
                Some(CountError::Http(429))
            );
        }
        assert_eq!(transport.calls.lock().len(), 2);
    }

    #[tokio::test]
    async fn expired_capability_is_revalidated() {
        let cache = CountCapabilities::default();
        let slot = cache.slot(("a".into(), "daily".into(), "future".into()));
        *slot.lock().await = Some(Capability {
            supported: false,
            expires: Instant::now(),
        });
        let transport = FakeTransport::new(&[(("daily", "future"), Counter::Pixels)]);
        assert!(cache
            .count(&transport, &["daily"], "a", "future", &image_request())
            .await
            .is_ok());
        assert_eq!(transport.calls.lock().len(), 3);
    }

    #[tokio::test]
    async fn text_keeps_original_route_without_image_probes() {
        let transport = FakeTransport::new(&[(("daily", "future-high"), Counter::Pixels)]);
        let request = json!({"request": {"contents": [{"parts": [{"text": "Hello world"}]}]}});
        let result = CountCapabilities::default()
            .count(&transport, &["daily"], "a", "future-high", &request)
            .await
            .unwrap();
        assert!(!result.image_checked);
        assert_eq!(transport.calls.lock().len(), 1);
        assert_eq!(
            transport.calls.lock()[0].1["request"]["contents"],
            request["request"]["contents"]
        );
    }

    #[tokio::test]
    async fn missing_count_is_an_error_instead_of_successful_zero() {
        let transport = FakeTransport::new(&[(("daily", "future"), Counter::Missing)]);
        assert_eq!(
            CountCapabilities::default()
                .count(
                    &transport,
                    &["daily"],
                    "a",
                    "future",
                    &json!({"contents": []})
                )
                .await
                .err(),
            Some(CountError::InvalidResponse)
        );
    }

    #[tokio::test(start_paused = true)]
    async fn stalled_probe_times_out_and_falls_back_to_another_endpoint() {
        let transport = FakeTransport::new(&[
            (("daily", "future"), Counter::Stall),
            (("sandbox", "future"), Counter::Pixels),
        ]);
        let result = CountCapabilities::default()
            .count(
                &transport,
                &["daily", "sandbox"],
                "a",
                "future",
                &image_request(),
            )
            .await
            .unwrap();
        assert_eq!(result.tokens, 258);
        assert_eq!(transport.calls.lock().len(), 4);
    }
}
