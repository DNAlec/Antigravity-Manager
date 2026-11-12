//! Observes upstream token counters without changing protocol output or billing.
//!
//! Missing counters remain missing. Repeated SSE usage is cumulative: only fields
//! actually present in a later observation replace earlier values, including zero.

use bytes::Bytes;
use futures::{Stream, StreamExt};
use serde::Serialize;
use serde_json::{json, Map, Value};
use std::pin::Pin;
use std::sync::{Arc, Mutex};

const MAX_ATTEMPTS: usize = 32;
const MAX_EVENT_BYTES: usize = 256 * 1024;
const COUNTERS: &[&str] = &[
    "promptTokenCount",
    "candidatesTokenCount",
    "cachedContentTokenCount",
    "totalTokenCount",
    "thoughtsTokenCount",
    "total_input_tokens",
    "total_output_tokens",
    "total_cached_tokens",
    "total_thought_tokens",
    "total_tool_use_tokens",
    "cached_content_token_count",
    "candidates_token_count",
];

#[derive(Default)]
pub(crate) struct UsageDiagnostics {
    attempts: Vec<Arc<Mutex<UsageRecord>>>,
    omitted_attempts: usize,
}

#[derive(Serialize)]
struct UsageRecord {
    attempt: usize,
    model: String,
    account_id: String,
    source: String,
    usage_present: bool,
    raw_usage: Map<String, Value>,
    observations: usize,
    malformed_usage_events: usize,
    truncated_events: usize,
    invalid_counter_fields: Vec<String>,
    transport_end: &'static str,
}

impl UsageDiagnostics {
    pub(crate) fn begin(
        &mut self,
        model: &str,
        account_id: &str,
        source: &str,
    ) -> Option<UsageAttempt> {
        if self.attempts.len() >= MAX_ATTEMPTS {
            self.omitted_attempts = self.omitted_attempts.saturating_add(1);
            return None;
        }
        let bounded = |s: &str| s.chars().take(128).collect();
        let record = Arc::new(Mutex::new(UsageRecord {
            attempt: self.attempts.len() + 1,
            model: bounded(model),
            account_id: bounded(account_id),
            source: bounded(source),
            usage_present: false,
            raw_usage: Map::new(),
            observations: 0,
            malformed_usage_events: 0,
            truncated_events: 0,
            invalid_counter_fields: Vec::new(),
            transport_end: "pending",
        }));
        self.attempts.push(record.clone());
        Some(UsageAttempt { record })
    }

    pub(crate) fn snapshot(&self) -> Option<Value> {
        if self.attempts.is_empty() && self.omitted_attempts == 0 {
            return None;
        }
        let attempts: Vec<Value> = self
            .attempts
            .iter()
            .filter_map(|record| {
                record
                    .lock()
                    .ok()
                    .and_then(|r| serde_json::to_value(&*r).ok())
            })
            .collect();
        Some(
            json!({"source": "upstream_raw", "attempts": attempts, "omitted_attempts": self.omitted_attempts}),
        )
    }
}

/// Owns one upstream response observation. Dropping it never claims a complete
/// response; this includes downstream cancellation and protocol early termination.
pub struct UsageAttempt {
    record: Arc<Mutex<UsageRecord>>,
}

impl UsageAttempt {
    /// Capture the holder while the request task-local is still in scope.
    pub fn current(model: &str, account_id: &str, source: &str) -> Option<Self> {
        crate::proxy::monitor::UpstreamRequestBodyHolder::current()?
            .begin_usage_attempt(model, account_id, source)
    }

    fn observe_json(&self, value: &Value) {
        let inner = value.get("response").unwrap_or(value);
        let Some(usage) = inner
            .get("usageMetadata")
            .or_else(|| value.get("usageMetadata"))
        else {
            return;
        };
        if let Ok(mut record) = self.record.lock() {
            record.usage_present = true;
            record.observations = record.observations.saturating_add(1);
            if !usage.is_object() {
                record.malformed_usage_events = record.malformed_usage_events.saturating_add(1);
                return;
            }
            for &field in COUNTERS {
                if let Some(value) = usage.get(field) {
                    if let Some(count) = value.as_u64() {
                        record.raw_usage.insert(field.to_string(), json!(count));
                    } else if !record.invalid_counter_fields.iter().any(|f| f == field) {
                        record.invalid_counter_fields.push(field.to_string());
                    }
                }
            }
        }
    }

    fn finish(&self, outcome: &'static str) {
        if let Ok(mut record) = self.record.lock() {
            if record.transport_end == "pending" {
                record.transport_end = outcome;
            }
        }
    }

    fn malformed(&self) {
        if let Ok(mut record) = self.record.lock() {
            record.malformed_usage_events = record.malformed_usage_events.saturating_add(1);
        }
    }

    fn truncated(&self) {
        if let Ok(mut record) = self.record.lock() {
            record.truncated_events = record.truncated_events.saturating_add(1);
        }
    }

    /// Call after a fully read upstream JSON body, before any mapper mutates it.
    pub fn finish_json(self, raw: &Value) {
        self.observe_json(raw);
        self.finish("eof");
    }

    /// A read/parse failure is distinct from a response with no usage metadata.
    pub fn finish_error(self) {
        self.finish("error");
    }
}

impl Drop for UsageAttempt {
    fn drop(&mut self) {
        self.finish("dropped");
    }
}

#[derive(Default)]
struct SseObserver {
    line: Vec<u8>,
    data: Vec<u8>,
    discard_line: bool,
    discard_event: bool,
}

impl SseObserver {
    fn feed(&mut self, bytes: &[u8], attempt: &UsageAttempt) {
        for &byte in bytes {
            if byte == b'\n' {
                self.finish_line(attempt);
            } else if !self.discard_line {
                if self.line.len() < MAX_EVENT_BYTES {
                    self.line.push(byte);
                } else {
                    self.line.clear();
                    self.discard_line = true;
                    self.discard_event = true;
                }
            }
        }
    }

    fn finish_line(&mut self, attempt: &UsageAttempt) {
        if self.discard_line {
            self.discard_line = false;
            return;
        }
        if self.line.last() == Some(&b'\r') {
            self.line.pop();
        }
        if self.line.is_empty() {
            self.finish_event(attempt);
        } else if let Some(data) = self.line.strip_prefix(b"data:") {
            let data = data.strip_prefix(b" ").unwrap_or(data);
            if !self.discard_event {
                if self.data.len().saturating_add(data.len()).saturating_add(1) <= MAX_EVENT_BYTES {
                    if !self.data.is_empty() {
                        self.data.push(b'\n');
                    }
                    self.data.extend_from_slice(data);
                } else {
                    self.data.clear();
                    self.discard_event = true;
                }
            }
        }
        self.line.clear();
    }

    fn finish_event(&mut self, attempt: &UsageAttempt) {
        if self.discard_event {
            attempt.truncated();
        } else if self
            .data
            .windows(b"usageMetadata".len())
            .any(|part| part == b"usageMetadata")
            || self.data.windows(2).any(|part| part == b"\\u")
        {
            // Most frames contain content only. Avoid building a second JSON tree
            // for those; escaped keys still take the parser path. Malformed counts
            // describe usage candidates, not protocol validation of every event.
            match serde_json::from_slice::<Value>(&self.data) {
                Ok(value) => attempt.observe_json(&value),
                Err(_) => attempt.malformed(),
            }
        }
        self.data.clear();
        self.discard_event = false;
    }

    fn finish(&mut self, attempt: &UsageAttempt) {
        if !self.line.is_empty() || self.discard_line {
            self.finish_line(attempt);
        }
        self.finish_event(attempt);
    }
}

/// Observe bytes before debug, auto-heal, or protocol conversion. The attempt is
/// captured before polling, so task-local scope loss cannot lose diagnostics.
/// No extra reads or retries are introduced; bytes and errors are forwarded as-is.
pub fn observe_stream<S, E>(
    stream: S,
    attempt: Option<UsageAttempt>,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, E>> + Send>>
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    E: Send + 'static,
{
    let Some(attempt) = attempt else {
        return Box::pin(stream);
    };
    Box::pin(async_stream::stream! {
        let mut stream = Box::pin(stream);
        let mut observer = SseObserver::default();
        while let Some(item) = stream.next().await {
            match &item {
                Ok(bytes) => observer.feed(bytes, &attempt),
                Err(_) => attempt.finish("error"),
            }
            yield item;
        }
        observer.finish(&attempt);
        attempt.finish("eof");
    })
}

/// Read exactly as the original response.json() path, recording raw counters
/// before protocol conversion and preserving the original success/error result.
pub async fn read_json_response(
    response: rquest::Response,
    attempt: Option<UsageAttempt>,
) -> Result<Value, rquest::Error> {
    let result = response.json::<Value>().await;
    if let Some(attempt) = attempt {
        match &result {
            Ok(raw) => attempt.finish_json(raw),
            Err(_) => attempt.finish_error(),
        }
    }
    result
}

/// Attach to the local log only. The wire response and usage accounting are not
/// modified. Non-JSON logs remain untouched; their bounded counters use tracing.
pub fn attach_to_log(
    log: &mut crate::proxy::monitor::ProxyRequestLog,
    holder: &crate::proxy::monitor::UpstreamRequestBodyHolder,
) {
    let Some(diagnostics) = holder.usage_snapshot() else {
        return;
    };
    if let Some(mut body) = log
        .response_body
        .as_deref()
        .and_then(|s| serde_json::from_str::<Value>(s).ok())
        .filter(Value::is_object)
    {
        body["_usage_diagnostics"] = diagnostics;
        if let Ok(encoded) = serde_json::to_string(&body) {
            log.response_body = Some(encoded);
        }
    } else {
        tracing::info!(request_id = %log.id, upstream_usage = %diagnostics, "Raw upstream usage diagnostics");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proxy::monitor::{UpstreamRequestBodyHolder, CURRENT_UPSTREAM_CAPTURE};
    use futures::stream;

    #[test]
    fn json_preserves_missing_zero_and_attempt_isolation() {
        let holder = UpstreamRequestBodyHolder::new();
        holder.begin_usage_attempt("m", "a", "json").unwrap().finish_json(&json!({"response":{"usageMetadata":{"promptTokenCount":99,"cachedContentTokenCount":0,"secret":"never capture"}}}));
        holder
            .begin_usage_attempt("m", "b", "json")
            .unwrap()
            .finish_json(&json!({"usageMetadata":{"promptTokenCount":50}}));
        let snapshot = holder.usage_snapshot().unwrap();
        assert_eq!(
            snapshot["attempts"][0]["raw_usage"],
            json!({"promptTokenCount":99,"cachedContentTokenCount":0})
        );
        assert!(snapshot["attempts"][1]["raw_usage"]
            .get("cachedContentTokenCount")
            .is_none());
        assert_eq!(snapshot["attempts"][1]["transport_end"], "eof");
    }

    #[tokio::test]
    async fn stream_survives_scope_exit_and_forwards_exact_chunks() {
        let holder = UpstreamRequestBodyHolder::new();
        let chunks = vec![Bytes::from_static(b"data: {\"usageMeta"), Bytes::from_static(b"data\":{\"promptTokenCount\":10,\"cachedContentTokenCount\":8}}\r\n\r\ndata: {\"usageMetadata\":{\"cachedContentTokenCount\":0}}")];
        let input = chunks.clone();
        let observed = CURRENT_UPSTREAM_CAPTURE
            .scope(holder.clone(), async {
                observe_stream(
                    stream::iter(input.into_iter().map(Ok::<_, &'static str>)),
                    UsageAttempt::current("m", "a", "stream"),
                )
            })
            .await;
        let output: Vec<_> = observed.collect().await;
        assert_eq!(output, chunks.into_iter().map(Ok).collect::<Vec<_>>());
        let snapshot = holder.usage_snapshot().unwrap();
        assert_eq!(
            snapshot["attempts"][0]["raw_usage"],
            json!({"promptTokenCount":10,"cachedContentTokenCount":0})
        );
        assert_eq!(snapshot["attempts"][0]["observations"], 2);
    }

    #[tokio::test]
    async fn stream_error_and_unpolled_drop_are_not_complete() {
        let holder = UpstreamRequestBodyHolder::new();
        let output: Vec<_> = observe_stream(
            stream::iter(vec![Err::<Bytes, _>("same error")]),
            holder.begin_usage_attempt("m", "a", "stream"),
        )
        .collect()
        .await;
        assert_eq!(output, vec![Err("same error")]);
        drop(observe_stream(
            stream::pending::<Result<Bytes, &str>>(),
            holder.begin_usage_attempt("m", "a", "stream"),
        ));
        let snapshot = holder.usage_snapshot().unwrap();
        assert_eq!(snapshot["attempts"][0]["transport_end"], "error");
        assert_eq!(snapshot["attempts"][1]["transport_end"], "dropped");
    }

    #[test]
    fn oversized_event_recovers_and_attempt_count_is_bounded() {
        let holder = UpstreamRequestBodyHolder::new();
        let attempt = holder.begin_usage_attempt("m", "a", "stream").unwrap();
        let mut observer = SseObserver::default();
        observer.feed(&vec![b'x'; MAX_EVENT_BYTES + 1], &attempt);
        observer.feed(
            b"\n\ndata: {\"usageMetadata\":{\"cachedContentTokenCount\":0}}\n\n",
            &attempt,
        );
        drop(attempt);
        for _ in 0..MAX_ATTEMPTS {
            drop(holder.begin_usage_attempt("m", "a", "json"));
        }
        let snapshot = holder.usage_snapshot().unwrap();
        assert_eq!(snapshot["attempts"].as_array().unwrap().len(), MAX_ATTEMPTS);
        assert_eq!(snapshot["omitted_attempts"], 1);
        assert_eq!(snapshot["attempts"][0]["truncated_events"], 1);
        assert_eq!(
            snapshot["attempts"][0]["raw_usage"]["cachedContentTokenCount"],
            0
        );
    }

    #[test]
    fn malformed_usage_is_distinct_from_missing_usage() {
        let holder = UpstreamRequestBodyHolder::new();
        holder
            .begin_usage_attempt("m", "a", "json")
            .unwrap()
            .finish_json(&json!({"usageMetadata": {"cachedContentTokenCount": "bad"}}));
        holder
            .begin_usage_attempt("m", "a", "json")
            .unwrap()
            .finish_json(&json!({"candidates": []}));
        let snapshot = holder.usage_snapshot().unwrap();
        assert_eq!(
            snapshot["attempts"][0]["invalid_counter_fields"],
            json!(["cachedContentTokenCount"])
        );
        assert_eq!(snapshot["attempts"][1]["usage_present"], false);
    }

    #[test]
    fn log_attachment_preserves_scaled_client_usage_and_simple_storage() {
        let holder = UpstreamRequestBodyHolder::new();
        holder
            .begin_usage_attempt("m", "a", "json")
            .unwrap()
            .finish_json(&json!({
                "usageMetadata": {"promptTokenCount": 100000, "cachedContentTokenCount": 80000}
            }));
        let client_usage = crate::proxy::pipeline::CanonicalUsage::new(100000, 80000, 5, 0)
            .to_claude_usage(true, 1_000_000);
        let mut log = crate::proxy::monitor::prompt_log_tests::sample_log("usage", 0);
        log.input_tokens = Some(123);
        log.cached_tokens = Some(45);
        log.response_body = Some(json!({"usage": client_usage.clone()}).to_string());
        attach_to_log(&mut log, &holder);
        let stored: Value = serde_json::from_str(log.response_body.as_deref().unwrap()).unwrap();
        assert_eq!(stored["usage"], client_usage);
        assert_eq!((log.input_tokens, log.cached_tokens), (Some(123), Some(45)));
        assert_eq!(
            stored["_usage_diagnostics"]["attempts"][0]["raw_usage"]["promptTokenCount"],
            100000
        );
        let simple = crate::proxy::payload_audit::simplify_payload_json(&stored);
        assert_eq!(simple["_usage_diagnostics"], stored["_usage_diagnostics"]);
    }

    #[test]
    fn multiline_events_keep_numeric_fields_only_and_holders_are_isolated() {
        let holder = UpstreamRequestBodyHolder::new();
        let other_holder = UpstreamRequestBodyHolder::new();
        let attempt = holder.begin_usage_attempt("m", "a", "stream").unwrap();
        let mut observer = SseObserver::default();
        observer.feed(b": ping\r\ndata: {\r\ndata: \"usageMetadata\": {\"promptTokenCount\":3,\"unknown\":\"secret\"}}\r\n\r\n", &attempt);
        observer.finish(&attempt);
        assert_eq!(
            holder.usage_snapshot().unwrap()["attempts"][0]["raw_usage"],
            json!({"promptTokenCount":3})
        );
        assert!(other_holder.usage_snapshot().is_none());
    }
}
