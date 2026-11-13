//! Preserve structured upstream SSE errors before protocol adapters flatten them.
//! The observer never changes response bytes, retries, or scans generated text for
//! error keywords. Each upstream response must use a fresh observer.

use bytes::Bytes;
use futures::{Stream, StreamExt};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

const MAX_EVENT_BYTES: usize = 256 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpstreamStreamError {
    /// HTTP-equivalent status supplied by the structured upstream error envelope.
    pub status: u16,
    /// Preserve details such as google.rpc.QuotaFailure for the shared HTTP policy.
    pub body: String,
    /// Final cumulative counters may accompany the upstream error envelope.
    pub usage: Option<Value>,
    /// Replaying after text, thinking, tool calls or media would duplicate output.
    /// Also true if an oversized frame made it unsafe to prove no output occurred.
    pub after_output: bool,
}

impl UpstreamStreamError {
    /// A response may be replayed only before both output and measured work.
    pub fn replay_safe(&self) -> bool {
        !self.after_output && !has_measured_usage(self.usage.as_ref())
    }

    /// The protocol collector has complete events; the bounded observer can miss
    /// a large frame's newer counters. Use it only as a fallback for missing data.
    pub fn usage_with_response(&self, response: Option<&Value>) -> Option<Value> {
        let mut usage = self.usage.clone();
        if let Some(update) = response.and_then(|value| value.get("usageMetadata")) {
            super::usage::merge_cumulative_usage(&mut usage, update);
        }
        usage
    }
}

#[derive(Default)]
struct Observation {
    failure: Option<UpstreamStreamError>,
    has_output: bool,
    usage: Option<Value>,
}

#[derive(Clone, Default)]
pub struct UpstreamStreamErrorObserver {
    state: Arc<Mutex<Observation>>,
}

impl UpstreamStreamErrorObserver {
    pub fn failure(&self) -> Option<UpstreamStreamError> {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.failure.clone().map(|mut failure| {
            failure.after_output |= state.has_output;
            failure.usage = state.usage.clone();
            failure
        })
    }

    /// Transport/EOF failures have no structured envelope. Preserve measured
    /// work before adapters flatten them into a generic collection error.
    pub fn failure_or_incomplete(&self, incomplete: Option<&str>) -> Option<UpstreamStreamError> {
        if let Some(failure) = self.failure() {
            return Some(failure);
        }
        let message = incomplete?;
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if !state.has_output && !has_measured_usage(state.usage.as_ref()) {
            return None;
        }
        Some(UpstreamStreamError {
            status: 502,
            body: json!({"error":{"code":502,"message":message}}).to_string(),
            usage: state.usage.clone(),
            after_output: state.has_output,
        })
    }

    pub fn take(&self) -> Option<UpstreamStreamError> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.failure.take().map(|mut failure| {
            failure.after_output |= state.has_output;
            failure.usage = state.usage.clone();
            failure
        })
    }

    pub fn has_measured_usage(&self) -> bool {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        has_measured_usage(state.usage.as_ref())
    }

    pub fn has_output(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .has_output
    }

    fn mark_output_unknown(&self) {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .has_output = true;
    }

    fn observe(&self, value: &Value) {
        let inner = value
            .get("response")
            .filter(|v| v.is_object())
            .unwrap_or(value);
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.has_output |= has_substantive_output(inner);
        if let Some(usage) = inner.get("usageMetadata") {
            super::usage::merge_cumulative_usage(&mut state.usage, usage);
        }
        if state.failure.is_some() {
            return;
        }
        if let Some(mut failure) = upstream_error_from_event(value) {
            failure.after_output = state.has_output;
            state.failure = Some(failure);
        }
    }
}

/// Provider refusals are completed responses, not empty-thinking dropouts.
pub fn is_refusal_finish_reason(reason: &str) -> bool {
    matches!(
        reason,
        "SAFETY"
            | "RECITATION"
            | "BLOCKLIST"
            | "PROHIBITED_CONTENT"
            | "SPII"
            | "IMAGE_SAFETY"
            | "IMAGE_PROHIBITED_CONTENT"
            | "MODEL_ARMOR"
    )
}

/// A blocked prompt terminates the entire response; a refused candidate may
/// coexist with other candidates that are still producing output.
pub fn response_has_prompt_refusal(value: &Value) -> bool {
    let raw = value
        .get("response")
        .filter(|value| value.is_object())
        .unwrap_or(value);
    raw.pointer("/promptFeedback/blockReason")
        .and_then(Value::as_str)
        .is_some_and(|reason| !reason.is_empty() && reason != "BLOCK_REASON_UNSPECIFIED")
}

pub fn response_is_refusal(value: &Value) -> bool {
    let raw = value
        .get("response")
        .filter(|value| value.is_object())
        .unwrap_or(value);
    response_has_prompt_refusal(raw)
        || raw
            .get("candidates")
            .and_then(Value::as_array)
            .is_some_and(|candidates| {
                candidates.iter().any(|candidate| {
                    candidate
                        .get("finishReason")
                        .or_else(|| candidate.get("finish_reason"))
                        .and_then(Value::as_str)
                        .is_some_and(is_refusal_finish_reason)
                })
            })
}

/// Recognize only a structured error envelope, never generated text.
/// Shared by the upstream observer, lifecycle, collectors and monitor.
pub fn error_envelope(value: &Value) -> Option<&Value> {
    value
        .get("response")
        .and_then(|inner| inner.get("error"))
        .filter(|error| error.is_object())
        .or_else(|| value.get("error").filter(|error| error.is_object()))
        .or_else(|| {
            (value.get("type").and_then(Value::as_str) == Some("error")
                && value.get("message").is_some())
            .then_some(value)
        })
}

pub fn upstream_error_from_event(value: &Value) -> Option<UpstreamStreamError> {
    let error = error_envelope(value)?;
    Some(UpstreamStreamError {
        status: error_status(error).unwrap_or(502),
        body: json!({"error": error}).to_string(),
        usage: value
            .get("response")
            .and_then(|inner| inner.get("usageMetadata"))
            .or_else(|| value.get("usageMetadata"))
            .filter(|usage| usage.is_object())
            .cloned(),
        after_output: false,
    })
}

/// Measured work is evidence against replay even when no visible output arrived.
/// Use allocatable counters only: an isolated aggregate total cannot establish
/// input/output/cache costs. Explicit zero, null and malformed fields are absent work.
pub fn has_measured_usage(usage: Option<&Value>) -> bool {
    let Some(usage) = usage else {
        return false;
    };
    let mut observed = super::usage::ObservedUsage::default();
    observed.merge_usage(usage);
    [
        observed.input_tokens(),
        observed.cached_tokens(),
        observed.output_tokens(),
    ]
    .into_iter()
    .flatten()
    .any(|value| value > 0)
        || usage
            .get("cache_creation_input_tokens")
            .and_then(Value::as_u64)
            .is_some_and(|value| value > 0)
}

/// Attach authoritative usage to an error response without changing its error
/// status or replay policy. This is the upstream Gemini envelope shared by adapters.
pub fn attach_usage_metadata(response: &mut Value, usage: Option<&Value>) {
    if let (Some(response), Some(usage)) = (response.as_object_mut(), usage) {
        response.insert("usageMetadata".into(), usage.clone());
    }
}

/// Error responses retain both raw upstream counters and the caller's usage
/// dialect. Keep sparse fields sparse and never scale billing measurements.
pub fn attach_error_usage(response: &mut Value, protocol: &str, usage: Option<&Value>) {
    attach_usage_metadata(response, usage);
    let (Some(response), Some(usage)) = (response.as_object_mut(), usage) else {
        return;
    };
    if protocol == "gemini" {
        return;
    }
    let mut observed = super::usage::ObservedUsage::default();
    observed.merge_usage(usage);
    let canonical = super::usage::CanonicalUsage::from_gemini(usage);
    let mut rendered = serde_json::Map::new();
    let claude = protocol == "claude" || protocol == "messages";
    let responses = protocol == "responses";
    if let Some(input) = observed.input_tokens() {
        rendered.insert(
            if claude || responses {
                "input_tokens"
            } else {
                "prompt_tokens"
            }
            .into(),
            json!(if claude {
                canonical.uncached_input_tokens
            } else {
                input
            }),
        );
    }
    if let Some(output) = observed.output_tokens() {
        rendered.insert(
            if claude || responses {
                "output_tokens"
            } else {
                "completion_tokens"
            }
            .into(),
            json!(output),
        );
    }
    if let Some(cached) = observed.cached_tokens() {
        if claude {
            rendered.insert("cache_read_input_tokens".into(), json!(cached));
        } else {
            rendered.insert(
                if responses {
                    "input_tokens_details"
                } else {
                    "prompt_tokens_details"
                }
                .into(),
                json!({"cached_tokens":cached}),
            );
        }
    }
    if claude {
        if let Some(created) = usage
            .get("cache_creation_input_tokens")
            .and_then(Value::as_u64)
        {
            rendered.insert("cache_creation_input_tokens".into(), json!(created));
        }
    } else {
        if let Some(total) = observed.total_tokens() {
            rendered.insert("total_tokens".into(), json!(total));
        }
        if let Some(reasoning) = observed.reasoning_tokens() {
            rendered.insert(
                if responses {
                    "output_tokens_details"
                } else {
                    "completion_tokens_details"
                }
                .into(),
                json!({"reasoning_tokens":reasoning}),
            );
        }
    }
    response.insert("usage".into(), Value::Object(rendered));
}

/// The lifecycle assembles complete protocol events. Only the diagnostic
/// observer is bounded: its capture limit must never truncate protocol data.
pub(crate) struct UpstreamSseErrorDetector {
    parser: SseErrorParser,
    observer: UpstreamStreamErrorObserver,
}
impl Default for UpstreamSseErrorDetector {
    fn default() -> Self {
        Self {
            parser: SseErrorParser {
                capture_limit: None,
                ..Default::default()
            },
            observer: UpstreamStreamErrorObserver::default(),
        }
    }
}
impl UpstreamSseErrorDetector {
    pub(crate) fn process_line(
        &mut self,
        line: &str,
    ) -> Result<Option<String>, UpstreamStreamError> {
        let pending = if line.is_empty()
            && self.parser.data.starts_with(b"{")
            && serde_json::from_slice::<Value>(&self.parser.data).is_err()
        {
            Some(format!(
                "data: {}",
                String::from_utf8_lossy(&self.parser.data)
            ))
        } else {
            None
        };
        self.parser.feed(line.as_bytes(), &self.observer);
        self.parser.feed(b"\n", &self.observer);
        if let Some(error) = self.observer.take() {
            return Err(error);
        }
        if line.starts_with("data:") {
            let data = &self.parser.data;
            if data == b"[DONE]" || serde_json::from_slice::<Value>(data).is_ok() {
                return Ok(Some(format!("data: {}", String::from_utf8_lossy(data))));
            }
            // Hold JSON fragments until the event is complete. Non-JSON lines
            // retain the adapter's existing handling (including parse errors).
            if data.starts_with(b"{") || data.starts_with(b"[") {
                return Ok(None);
            }
            return Ok(Some(line.to_owned()));
        }
        Ok(pending.or_else(|| (!line.is_empty()).then(|| line.to_owned())))
    }
}

fn error_status(error: &Value) -> Option<u16> {
    if let Some(code) = error.get("code").and_then(Value::as_u64) {
        if (400..=599).contains(&code) {
            return Some(code as u16);
        }
    }
    match error.get("status").and_then(Value::as_str)? {
        "INVALID_ARGUMENT" | "FAILED_PRECONDITION" | "OUT_OF_RANGE" => Some(400),
        "UNAUTHENTICATED" => Some(401),
        "PERMISSION_DENIED" => Some(403),
        "NOT_FOUND" => Some(404),
        "ALREADY_EXISTS" | "ABORTED" => Some(409),
        "RESOURCE_EXHAUSTED" => Some(429),
        "CANCELLED" => Some(499),
        "INTERNAL" | "UNKNOWN" | "DATA_LOSS" => Some(500),
        "UNIMPLEMENTED" => Some(501),
        "UNAVAILABLE" => Some(503),
        "DEADLINE_EXCEEDED" => Some(504),
        _ => None,
    }
}

fn has_substantive_output(value: &Value) -> bool {
    value
        .get("candidates")
        .and_then(Value::as_array)
        .is_some_and(|candidates| {
            candidates.iter().any(|candidate| {
                candidate
                    .get("content")
                    .and_then(|content| content.get("parts"))
                    .and_then(Value::as_array)
                    .is_some_and(|parts| {
                        parts.iter().any(|part| {
                            part.get("text")
                                .and_then(Value::as_str)
                                .is_some_and(|text| !text.is_empty())
                                || part
                                    .get("functionCall")
                                    .or_else(|| part.get("function_call"))
                                    .is_some_and(Value::is_object)
                                || part
                                    .get("inlineData")
                                    .or_else(|| part.get("inline_data"))
                                    .and_then(|data| data.get("data"))
                                    .and_then(Value::as_str)
                                    .is_some_and(|data| !data.is_empty())
                                || part
                                    .get("fileData")
                                    .or_else(|| part.get("file_data"))
                                    .and_then(|data| {
                                        data.get("fileUri").or_else(|| data.get("file_uri"))
                                    })
                                    .and_then(Value::as_str)
                                    .is_some_and(|uri| !uri.is_empty())
                        })
                    })
            })
        })
}

struct SseErrorParser {
    capture_limit: Option<usize>,
    line: Vec<u8>,
    data: Vec<u8>,
    discard_line: bool,
    discard_event: bool,
    done: bool,
}

impl Default for SseErrorParser {
    fn default() -> Self {
        Self {
            capture_limit: Some(MAX_EVENT_BYTES),
            line: Vec::new(),
            data: Vec::new(),
            discard_line: false,
            discard_event: false,
            done: false,
        }
    }
}

impl SseErrorParser {
    fn feed(&mut self, bytes: &[u8], observer: &UpstreamStreamErrorObserver) {
        for &byte in bytes {
            if self.done {
                break;
            }
            if byte == b'\n' {
                self.finish_line(observer);
            } else if !self.discard_line {
                if self
                    .capture_limit
                    .is_none_or(|limit| self.line.len() < limit)
                {
                    self.line.push(byte);
                } else {
                    self.line.clear();
                    self.discard_line = true;
                    self.discard_event = true;
                    observer.mark_output_unknown();
                }
            }
        }
    }

    fn finish_line(&mut self, observer: &UpstreamStreamErrorObserver) {
        if self.discard_line {
            self.discard_line = false;
            return;
        }
        if self.line.last() == Some(&b'\r') {
            self.line.pop();
        }
        if self.line.is_empty() {
            self.finish_event(observer);
        } else if let Some(data) = self.line.strip_prefix(b"data:") {
            let data = data.strip_prefix(b" ").unwrap_or(data);
            if !self.discard_event {
                if self.capture_limit.is_none_or(|limit| {
                    self.data.len().saturating_add(data.len()).saturating_add(1) <= limit
                }) {
                    if !self.data.is_empty() {
                        self.data.push(b'\n');
                    }
                    self.data.extend_from_slice(data);
                    // Adapters may stop after this data line before reading the
                    // blank delimiter. Observe complete JSON before yielding it.
                    if let Ok(value) = serde_json::from_slice::<Value>(&self.data) {
                        observer.observe(&value);
                    }
                } else {
                    self.data.clear();
                    self.discard_event = true;
                    observer.mark_output_unknown();
                }
            }
        }
        self.line.clear();
    }

    fn finish_event(&mut self, observer: &UpstreamStreamErrorObserver) {
        if !self.discard_event {
            if self.data == b"[DONE]" {
                self.done = true;
            } else if let Ok(value) = serde_json::from_slice::<Value>(&self.data) {
                observer.observe(&value);
            }
        }
        self.data.clear();
        self.discard_event = false;
    }

    fn finish(&mut self, observer: &UpstreamStreamErrorObserver) {
        if !self.done {
            if !self.line.is_empty() || self.discard_line {
                self.finish_line(observer);
            }
            self.finish_event(observer);
        }
    }
}

/// Wrap the raw upstream SSE after any same-account auto-heal wrapper and before
/// protocol conversion. Results, transport errors and chunk boundaries pass through.
pub fn observe_stream_errors<S, E>(
    stream: S,
    observer: UpstreamStreamErrorObserver,
) -> impl Stream<Item = Result<Bytes, E>>
where
    S: Stream<Item = Result<Bytes, E>>,
{
    observe_response_errors(stream, observer, false)
}

/// Aggregated responses need complete counters even when one tool/media event
/// exceeds the diagnostic cap. Streaming diagnostics stay bounded; collection
/// already requires complete events in the lifecycle and protocol collector.
pub fn observe_response_errors<S, E>(
    stream: S,
    observer: UpstreamStreamErrorObserver,
    collecting: bool,
) -> impl Stream<Item = Result<Bytes, E>>
where
    S: Stream<Item = Result<Bytes, E>>,
{
    async_stream::stream! {
        futures::pin_mut!(stream);
        let mut parser = SseErrorParser::default();
        if collecting { parser.capture_limit = None; }
        while let Some(item) = stream.next().await {
            match &item {
                Ok(bytes) => parser.feed(bytes, &observer),
                Err(_) => parser.finish(&observer),
            }
            yield item;
        }
        parser.finish(&observer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn collected_large_events_keep_authoritative_failure_usage() {
        for collecting in [true, false] {
            for structured in [true, false] {
                let observer = UpstreamStreamErrorObserver::default();
                let large = json!({"candidates":[{"content":{"parts":[{"functionCall":{"name":"tool","args":{"text":"x".repeat(MAX_EVENT_BYTES + 1)}}}]}}],"usageMetadata":{"promptTokenCount":4096,"cachedContentTokenCount":3072}});
                let mut frames = vec![Ok::<_, String>(Bytes::from(format!("data: {large}\n\n")))];
                if structured {
                    frames.push(Ok(Bytes::from_static(b"data: {\"error\":{\"code\":429},\"usageMetadata\":{\"candidatesTokenCount\":0}}\n\n")));
                } else {
                    frames.push(Err("connection reset".into()));
                }
                let _: Vec<_> = observe_response_errors(
                    futures::stream::iter(frames),
                    observer.clone(),
                    collecting,
                )
                .collect()
                .await;
                let failure = observer
                    .failure_or_incomplete(Some("connection reset"))
                    .unwrap();
                assert!(!failure.replay_safe());
                if collecting {
                    assert_eq!(failure.usage.as_ref().unwrap()["promptTokenCount"], 4096);
                    assert_eq!(
                        failure.usage.as_ref().unwrap()["cachedContentTokenCount"],
                        3072
                    );
                    let mut rendered = json!({"error":{"code":429}});
                    attach_error_usage(&mut rendered, "claude", failure.usage.as_ref());
                    assert_eq!(rendered["usage"]["input_tokens"], 1024);
                    assert_eq!(rendered["usage"]["cache_read_input_tokens"], 3072);
                } else {
                    assert!(
                        failure.after_output,
                        "bounded diagnostic uncertainty forbids replay"
                    );
                    assert!(
                        failure
                            .usage
                            .as_ref()
                            .and_then(|u| u.get("promptTokenCount"))
                            .is_none(),
                        "streaming diagnostics retain their capture cap"
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn measured_transport_and_eof_failures_are_not_replayed() {
        for transport in [true, false] {
            for measured in [true, false] {
                let observer = UpstreamStreamErrorObserver::default();
                let count = if measured { 4096 } else { 0 };
                let mut frames = vec![Ok::<_, String>(Bytes::from(format!(
                    "data: {{\"usageMetadata\":{{\"promptTokenCount\":{count}}}}}\n\n"
                )))];
                if transport {
                    frames.push(Err("connection reset".into()));
                }
                let _: Vec<_> =
                    observe_stream_errors(futures::stream::iter(frames), observer.clone())
                        .collect()
                        .await;
                assert!(observer.failure().is_none());
                assert!(
                    observer.failure_or_incomplete(None).is_none(),
                    "successful no-output responses are not synthesized errors"
                );
                let failure = observer.failure_or_incomplete(Some(if transport {
                    "connection reset"
                } else {
                    "empty response stream"
                }));
                assert_eq!(failure.is_some(), measured);
                if let Some(failure) = failure {
                    assert!(!failure.replay_safe());
                    assert_eq!(failure.usage.as_ref().unwrap()["promptTokenCount"], 4096);
                }
            }
        }
    }

    #[test]
    fn measured_failure_disables_replay_and_error_usage_keeps_protocol_units() {
        for status in [400, 429, 503] {
            let failure = upstream_error_from_event(&json!({
                "error":{"code":status},
                "usageMetadata":{"promptTokenCount":100,"cachedContentTokenCount":60,"candidatesTokenCount":3,"thoughtsTokenCount":2}
            })).unwrap();
            assert!(!failure.replay_safe());
            for protocol in ["claude", "openai", "responses", "gemini"] {
                let mut response = json!({"error":{"code":status}});
                attach_error_usage(&mut response, protocol, failure.usage.as_ref());
                assert_eq!(response["usageMetadata"]["promptTokenCount"], 100);
                match protocol {
                    "claude" => {
                        assert_eq!(response["usage"]["input_tokens"], 40);
                        assert_eq!(response["usage"]["cache_read_input_tokens"], 60);
                        assert_eq!(response["usage"]["output_tokens"], 5);
                    }
                    "responses" => {
                        assert_eq!(response["usage"]["input_tokens"], 100);
                        assert_eq!(
                            response["usage"]["input_tokens_details"]["cached_tokens"],
                            60
                        );
                        assert_eq!(response["usage"]["output_tokens"], 5);
                    }
                    "openai" => {
                        assert_eq!(response["usage"]["prompt_tokens"], 100);
                        assert_eq!(
                            response["usage"]["prompt_tokens_details"]["cached_tokens"],
                            60
                        );
                        assert_eq!(response["usage"]["completion_tokens"], 5);
                    }
                    _ => assert!(response.get("usage").is_none()),
                }
            }
        }
        for usage in [
            json!({}),
            json!({"totalTokenCount":50}),
            json!({"promptTokenCount":0,"cachedContentTokenCount":0,"candidatesTokenCount":0}),
            json!({"output_tokens":0,"reasoning_tokens":5}),
        ] {
            let mut failure =
                upstream_error_from_event(&json!({"error":{"code":400},"usageMetadata":usage}))
                    .unwrap();
            assert!(failure.replay_safe());
            failure.after_output = true;
            assert!(!failure.replay_safe());
        }
        let mut sparse = json!({"error":{"code":503}});
        attach_error_usage(&mut sparse, "claude", Some(&json!({"promptTokenCount":9})));
        assert_eq!(sparse["usage"]["input_tokens"], 9);
        assert!(sparse["usage"].get("output_tokens").is_none());
        assert!(sparse["usage"].get("cache_read_input_tokens").is_none());
    }

    #[test]
    fn output_after_error_upgrades_snapshot_and_take_for_any_chunking() {
        let error = b"data: {\"error\":{\"code\":429}}\n\n";
        let content =
            b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"later output\"}]}}]}\n\n";
        for separate_chunks in [false, true] {
            let observer = UpstreamStreamErrorObserver::default();
            let mut parser = SseErrorParser::default();
            if separate_chunks {
                parser.feed(error, &observer);
                assert!(!observer.failure().unwrap().after_output);
                parser.feed(content, &observer);
            } else {
                parser.feed(&[error.as_slice(), content.as_slice()].concat(), &observer);
            }
            assert!(observer.failure().unwrap().after_output);
            assert!(observer.take().unwrap().after_output);
            assert!(observer.failure().is_none());
        }
        let observer = UpstreamStreamErrorObserver::default();
        let mut parser = SseErrorParser::default();
        parser.feed(error, &observer);
        parser.feed(&vec![b'x'; MAX_EVENT_BYTES + 32], &observer);
        assert!(
            observer.failure().unwrap().after_output,
            "uninspectable later output also blocks replay"
        );
        assert!(observer.take().unwrap().after_output);
    }

    // Exercise the actual stream adapters and collectors: their presentation or
    // terminal behavior must not erase the HTTP policy input held by the observer.
    #[tokio::test]
    async fn structured_error_survives_claude_and_openai_mapper_collection() {
        for protocol in ["claude", "openai"] {
            for with_text in [false, true] {
                for status in [401, 403, 429] {
                    let error = json!({"code":status,"message":"upstream rejected request",
                        "details":[{"@type":"type.googleapis.com/google.rpc.QuotaFailure","violations":[{"subject":"daily quota"}]}]});
                    let mut frames = Vec::new();
                    if with_text {
                        frames.push(json!({"response":{"candidates":[{"content":{"parts":[{"text":"partial answer"}]}}]}}));
                    }
                    frames.push(json!({"response":{"error":error}}));
                    let chunks: Vec<_> = frames
                        .into_iter()
                        .flat_map(|frame| {
                            format!("data: {frame}\n\n")
                                .into_bytes()
                                .chunks(7)
                                .map(|chunk| Ok::<_, String>(Bytes::copy_from_slice(chunk)))
                                .collect::<Vec<_>>()
                        })
                        .collect();
                    let observer = UpstreamStreamErrorObserver::default();
                    let observed =
                        observe_stream_errors(futures::stream::iter(chunks), observer.clone());
                    if protocol == "claude" {
                        let converted = crate::proxy::mappers::claude::create_claude_sse_stream(
                            Box::pin(observed),
                            "stream-error-test".into(),
                            "test@example.test".into(),
                            None,
                            false,
                            1_000_000,
                            None,
                            1,
                            None,
                            vec![],
                        );
                        let converted = converted.map(|chunk| chunk.map_err(std::io::Error::other));
                        let _ = crate::proxy::mappers::claude::collect_stream_to_json(Box::pin(
                            converted,
                        ))
                        .await;
                    } else {
                        let converted =
                            crate::proxy::mappers::openai::streaming::create_openai_sse_stream(
                                Box::pin(observed),
                                "gemini-test".into(),
                                "stream-error-test".into(),
                                1,
                                None,
                                true,
                            );
                        let _ = crate::proxy::mappers::openai::collector::collect_stream_to_json(
                            converted,
                        )
                        .await;
                    }
                    let failure = observer.failure().unwrap_or_else(|| {
                        panic!("{protocol}: status {status} missing after collection")
                    });
                    assert_eq!(failure.status, status);
                    assert_eq!(
                        failure.after_output, with_text,
                        "{protocol}: consumed output determines replay eligibility"
                    );
                    assert_eq!(
                        serde_json::from_str::<Value>(&failure.body).unwrap(),
                        json!({"error":error})
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn structured_quota_error_survives_fragmentation_and_wrapping() {
        let error = json!({"code":429,"status":"RESOURCE_EXHAUSTED","message":"quota exhausted",
            "details":[{"@type":"type.googleapis.com/google.rpc.QuotaFailure","violations":[{"subject":"daily quota"}]}]});
        let bytes = format!(
            "event: error\r\ndata: {}\r\n\r\n",
            json!({"response":{"error":error}})
        )
        .into_bytes();
        let input: Vec<_> = bytes
            .chunks(3)
            .map(|part| Ok::<_, String>(Bytes::copy_from_slice(part)))
            .collect();
        let observer = UpstreamStreamErrorObserver::default();
        let output: Vec<_> =
            observe_stream_errors(futures::stream::iter(input.clone()), observer.clone())
                .collect()
                .await;
        assert_eq!(output, input);
        let failure = observer.failure().unwrap();
        assert_eq!(failure.status, 429);
        assert!(!failure.after_output);
        assert_eq!(
            serde_json::from_str::<Value>(&failure.body).unwrap(),
            json!({"error":error})
        );
        assert!(observer.take().is_some());
        assert!(observer.failure().is_none());
    }

    #[test]
    fn authentication_status_multiline_and_unterminated_eof_are_preserved() {
        let observer = UpstreamStreamErrorObserver::default();
        let mut parser = SseErrorParser::default();
        parser.feed(b"data: {\n\
            data: \"response\":{\"error\":{\"status\":\"UNAUTHENTICATED\",\"message\":\"expired\"}}}", &observer);
        assert!(observer.failure().is_none());
        parser.finish(&observer);
        assert_eq!(observer.failure().unwrap().status, 401);
    }

    #[test]
    fn capture_complete_data_line_before_downstream_stops_reading() {
        let observer = UpstreamStreamErrorObserver::default();
        let mut parser = SseErrorParser::default();
        parser.feed(
            b"data: {\"error\":{\"code\":403,\"message\":\"denied\"}}\n",
            &observer,
        );
        assert_eq!(observer.failure().unwrap().status, 403);
    }

    #[test]
    fn generated_error_text_is_not_an_upstream_error_and_output_blocks_replay() {
        for part in [
            json!({"text":"{\"error\":{\"code\":429,\"message\":\"quota\"}}"}),
            json!({"text":"thinking","thought":true}),
            json!({"functionCall":{"name":"tool","args":{"error":{"code":403}}}}),
            json!({"inlineData":{"mimeType":"image/png","data":"aGVsbG8="}}),
            json!({"fileData":{"fileUri":"gs://output/result"}}),
        ] {
            let observer = UpstreamStreamErrorObserver::default();
            let mut parser = SseErrorParser::default();
            let content = json!({"response":{"candidates":[{"content":{"parts":[part]}}]}});
            parser.feed(format!("data: {content}\n\n").as_bytes(), &observer);
            assert!(observer.failure().is_none());
            assert!(observer.has_output());
            parser.feed(b"data: {\"error\":{\"code\":429}}\n\n", &observer);
            assert!(observer.failure().unwrap().after_output);
        }
    }

    #[test]
    fn malformed_and_post_done_envelopes_do_not_classify_unknown_errors_use_502() {
        for (payload, expected_status) in [
            ("{\"error\":\"unauthorized quota exhausted\"}", None),
            (
                "{\"error\":{\"message\":\"unauthorized quota exhausted\"}}",
                Some(502),
            ),
            ("{\"error\":{\"code\":200}}", Some(502)),
            ("{\"error\":{\"status\":\"not a status\"}}", Some(502)),
            ("{bad json}", None),
        ] {
            let observer = UpstreamStreamErrorObserver::default();
            let mut parser = SseErrorParser::default();
            parser.feed(format!("data: {payload}\n\n").as_bytes(), &observer);
            // Error words never infer authentication/quota status. An explicit
            // object still terminates as a generic bad gateway failure.
            assert_eq!(
                observer.failure().map(|failure| failure.status),
                expected_status,
                "{payload}"
            );
        }
        let observer = UpstreamStreamErrorObserver::default();
        let mut parser = SseErrorParser::default();
        parser.feed(
            b"data: [DONE]\n\ndata: {\"error\":{\"code\":429}}\n\n",
            &observer,
        );
        assert!(observer.failure().is_none());
    }

    #[test]
    fn oversized_event_is_bounded_and_parser_recovers_for_next_error() {
        let observer = UpstreamStreamErrorObserver::default();
        let mut parser = SseErrorParser::default();
        parser.feed(b"data: ", &observer);
        parser.feed(&vec![b'x'; MAX_EVENT_BYTES + 32], &observer);
        assert!(parser.line.len() <= MAX_EVENT_BYTES);
        assert!(parser.data.len() <= MAX_EVENT_BYTES);
        parser.feed(b"\n\ndata: {\"error\":{\"code\":503}}\n\n", &observer);
        assert_eq!(observer.failure().unwrap().status, 503);
        assert!(observer.failure().unwrap().after_output);
    }

    #[tokio::test]
    async fn transport_errors_are_unchanged_and_do_not_invent_http_status() {
        let observer = UpstreamStreamErrorObserver::default();
        let input = vec![Err::<Bytes, _>("reset".to_string())];
        let output: Vec<_> =
            observe_stream_errors(futures::stream::iter(input.clone()), observer.clone())
                .collect()
                .await;
        assert_eq!(output, input);
        assert!(observer.failure().is_none());
    }
}

#[cfg(test)]
mod protocol_error_regressions {
    use super::*;
    use std::pin::Pin;

    #[test]
    fn observer_capture_limit_does_not_truncate_protocol_frames() {
        let mut detector = UpstreamSseErrorDetector::default();
        let frame = format!(
            "data: {}",
            json!({"candidates":[{"content":{"parts":[{"text":"x".repeat(MAX_EVENT_BYTES + 32)}]}}]})
        );
        assert_eq!(detector.process_line(&frame).unwrap(), Some(frame));
    }

    #[tokio::test]
    async fn large_multiline_protocol_event_preserves_content_and_usage() {
        let first = "甲".repeat(MAX_EVENT_BYTES / 4);
        let second = "乙".repeat(MAX_EVENT_BYTES / 4);
        let first_line = format!(
            "data: {{\"candidates\":[{{\"content\":{{\"parts\":[{},",
            json!({"text":first})
        );
        let second_line = format!(
            "data: {}]}} ,\"finishReason\":\"STOP\"}}],\"usageMetadata\":{}}}",
            json!({"text":second}),
            json!({"promptTokenCount":100,"cachedContentTokenCount":80,"candidatesTokenCount":12})
        );
        assert!(first_line.len() < MAX_EVENT_BYTES);
        assert!(second_line.len() < MAX_EVENT_BYTES);
        assert!(first_line.len() + second_line.len() > MAX_EVENT_BYTES);
        let expected: Value =
            serde_json::from_str(&format!("{}\n{}", &first_line[6..], &second_line[6..])).unwrap();
        for trailing_delimiter in [false, true] {
            let wire = format!(
                "{first_line}\n{second_line}{}",
                if trailing_delimiter { "\n\n" } else { "" }
            );
            let chunks = wire
                .as_bytes()
                .chunks(8191)
                .map(|chunk| Ok::<_, String>(Bytes::copy_from_slice(chunk)))
                .collect::<Vec<_>>();
            let observer = UpstreamStreamErrorObserver::default();
            let upstream = Box::pin(observe_stream_errors(
                futures::stream::iter(chunks),
                observer.clone(),
            ));
            let handler = crate::proxy::mappers::gemini::streaming::GeminiStreamHandler::new(
                None,
                "large-multiline-test".into(),
                "gemini-test".into(),
                1,
                "root".into(),
                false,
                None,
                String::new(),
                String::new(),
            );
            let stream = crate::proxy::pipeline::run_stream_lifecycle(
                upstream,
                handler,
                crate::proxy::pipeline::StreamLifecycleConfig::new(
                    "gemini",
                    "large_multiline_test",
                    String::new(),
                ),
            );
            let mut events = Vec::new();
            for output in stream.collect::<Vec<_>>().await {
                let output = output.expect("valid large protocol events must not become errors");
                for line in std::str::from_utf8(&output).unwrap().lines() {
                    if let Some(data) = line.strip_prefix("data: ") {
                        events.push(serde_json::from_str::<Value>(data).unwrap());
                    }
                }
            }
            assert_eq!(events, vec![expected.clone()]);
            assert!(
                observer.has_output(),
                "bounded observer must conservatively block replay"
            );
            assert!(observer.failure().is_none());
        }
    }

    #[tokio::test]
    async fn gemini_failed_collection_preserves_cumulative_usage_and_blocks_replay() {
        for one_chunk in [false, true] {
            let mut frames = vec![
                format!(
                    "data: {}\n\n",
                    json!({"response": {
                        "candidates":[{"content":{"parts":[{"text":"Partial answer"}]}}],
                        "usageMetadata":{"promptTokenCount":100,"cachedContentTokenCount":80,"candidatesTokenCount":1}
                    }})
                ),
                format!(
                    "data: {}\n\n",
                    json!({"response": {
                        "error":{"code":503,"status":"UNAVAILABLE","message":"provider overloaded"},
                        "usageMetadata":{"candidatesTokenCount":5,"cachedContentTokenCount":0}
                    }})
                ),
            ];
            if one_chunk {
                frames = vec![frames.concat()];
            }
            let chunks = frames
                .into_iter()
                .map(|frame| Ok::<_, String>(Bytes::from(frame)));
            let observer = UpstreamStreamErrorObserver::default();
            let upstream = Box::pin(observe_stream_errors(
                futures::stream::iter(chunks),
                observer.clone(),
            ));
            let handler = crate::proxy::mappers::gemini::streaming::GeminiStreamHandler::new(
                None,
                "failed-usage-test".into(),
                "gemini-test".into(),
                1,
                "root".into(),
                false,
                None,
                String::new(),
                String::new(),
            );
            let stream = crate::proxy::pipeline::run_stream_lifecycle(
                upstream,
                handler,
                crate::proxy::pipeline::StreamLifecycleConfig::new(
                    "gemini",
                    "failed_usage_test",
                    String::new(),
                ),
            );
            let collected = crate::proxy::mappers::gemini::collector::collect_stream_to_json(
                stream,
                "failed-usage-test",
            )
            .await
            .unwrap();
            let failure = observer.failure().unwrap();
            let expected = json!({"promptTokenCount":100,"cachedContentTokenCount":0,"candidatesTokenCount":5});
            assert!(failure.after_output, "consumed output cannot be replayed");
            assert_eq!(failure.usage, Some(expected.clone()));
            assert_eq!(collected["error"]["code"], 503);
            assert!(collected.get("candidates").is_none());
            assert_eq!(collected["usageMetadata"], expected);
            // The handler's final error presentation must retain these counters.
            let mut response: Value = serde_json::from_str(&failure.body).unwrap();
            attach_usage_metadata(&mut response, failure.usage.as_ref());
            assert_eq!(response["usageMetadata"], expected);
            assert_eq!(response["error"]["code"], 503);
        }
    }

    #[tokio::test]
    async fn failed_collection_overrides_stale_bounded_observer_usage() {
        let initial = json!({"usageMetadata":{"promptTokenCount":100,"cachedContentTokenCount":80,"candidatesTokenCount":1}});
        let large = json!({
            "candidates":[{"content":{"parts":[{"text":"x".repeat(MAX_EVENT_BYTES + 32)}]}}],
            "usageMetadata":{"promptTokenCount":200,"cachedContentTokenCount":160,"candidatesTokenCount":10}
        });
        let error = json!({"error":{"code":503,"status":"UNAVAILABLE","message":"overloaded"},"usageMetadata":{"candidatesTokenCount":11}});
        let chunks = [initial, large, error]
            .into_iter()
            .map(|frame| Ok::<_, String>(Bytes::from(format!("data: {frame}\n\n"))));
        let observer = UpstreamStreamErrorObserver::default();
        let upstream = Box::pin(observe_stream_errors(
            futures::stream::iter(chunks),
            observer.clone(),
        ));
        let handler = crate::proxy::mappers::gemini::streaming::GeminiStreamHandler::new(
            None,
            "large-failed-usage-test".into(),
            "gemini-test".into(),
            1,
            "root".into(),
            false,
            None,
            String::new(),
            String::new(),
        );
        let stream = crate::proxy::pipeline::run_stream_lifecycle(
            upstream,
            handler,
            crate::proxy::pipeline::StreamLifecycleConfig::new(
                "gemini",
                "large_failed_usage_test",
                String::new(),
            ),
        );
        let collected = crate::proxy::mappers::gemini::collector::collect_stream_to_json(
            stream,
            "large-failed-usage-test",
        )
        .await
        .unwrap();
        let failure = observer.failure().unwrap();
        assert!(failure.after_output);
        assert_eq!(
            failure.usage.as_ref().unwrap()["promptTokenCount"],
            100,
            "the bounded observer deliberately skipped the large frame"
        );
        let usage = failure.usage_with_response(Some(&collected));
        assert_eq!(
            usage,
            Some(
                json!({"promptTokenCount":200,"cachedContentTokenCount":160,"candidatesTokenCount":11})
            )
        );
        let mut response: Value = serde_json::from_str(&failure.body).unwrap();
        attach_usage_metadata(&mut response, usage.as_ref());
        assert_eq!(response["error"]["code"], 503);
        assert_eq!(response["usageMetadata"]["cachedContentTokenCount"], 160);
        assert!(response.get("candidates").is_none());
    }

    #[test]
    fn measured_usage_requires_known_positive_counters_not_aggregate_totals() {
        for usage in [
            json!({"promptTokenCount":1}),
            json!({"total_input_tokens":1}),
            json!({"cachedContentTokenCount":1}),
            json!({"total_cached_tokens":1}),
            json!({"candidatesTokenCount":1}),
            json!({"total_output_tokens":1}),
            json!({"thoughtsTokenCount":1}),
            json!({"total_thought_tokens":1}),
            json!({"output_tokens_details":{"reasoning_tokens":1}}),
            json!({"candidatesTokenCount":0,"thoughtsTokenCount":5}),
            json!({"input_tokens":1,"output_tokens":0,"output_tokens_details":{"reasoning_tokens":5}}),
            json!({"cache_creation_input_tokens":1}),
        ] {
            assert!(has_measured_usage(Some(&usage)), "{usage}");
        }
        for usage in [
            json!({}),
            json!({"total_tokens":100,"totalTokenCount":100}),
            json!({"output_tokens":0,"output_tokens_details":{"reasoning_tokens":5}}),
            json!({"completion_tokens":0,"thoughtsTokenCount":5}),
            json!({"promptTokenCount":0,"cachedContentTokenCount":0,"candidatesTokenCount":0,"thoughtsTokenCount":0,"cache_creation_input_tokens":0}),
            json!({"promptTokenCount":null,"cachedContentTokenCount":-1,"candidatesTokenCount":"12","thoughtsTokenCount":0.5,"cache_creation_input_tokens":null}),
        ] {
            assert!(!has_measured_usage(Some(&usage)), "{usage}");
        }
        assert!(!has_measured_usage(None));
    }

    #[tokio::test]
    async fn positive_usage_before_visible_output_survives_failure_without_replay() {
        for cache in [80, 0] {
            let frames = [
                json!({"usageMetadata":{"promptTokenCount":100,"cachedContentTokenCount":cache}}),
                json!({"error":{"code":503,"status":"UNAVAILABLE","message":"overloaded"},"usageMetadata":{"candidatesTokenCount":0}}),
            ];
            let chunks = frames
                .into_iter()
                .map(|frame| Ok::<_, String>(Bytes::from(format!("data: {frame}\n\n"))));
            let observer = UpstreamStreamErrorObserver::default();
            let upstream = Box::pin(observe_stream_errors(
                futures::stream::iter(chunks),
                observer.clone(),
            ));
            let handler = crate::proxy::mappers::gemini::streaming::GeminiStreamHandler::new(
                None,
                "prefill-failed-usage-test".into(),
                "gemini-test".into(),
                1,
                "root".into(),
                false,
                None,
                String::new(),
                String::new(),
            );
            let stream = crate::proxy::pipeline::run_stream_lifecycle(
                upstream,
                handler,
                crate::proxy::pipeline::StreamLifecycleConfig::new(
                    "gemini",
                    "prefill_failed_usage_test",
                    String::new(),
                ),
            );
            let collected = crate::proxy::mappers::gemini::collector::collect_stream_to_json(
                stream,
                "prefill-failed-usage-test",
            )
            .await
            .unwrap();
            let failure = observer.failure().unwrap();
            assert!(!failure.after_output, "the upstream has emitted no content");
            let usage = failure.usage_with_response(Some(&collected));
            assert!(
                has_measured_usage(usage.as_ref()),
                "measured prefill blocks replay"
            );
            let mut response: Value = serde_json::from_str(&failure.body).unwrap();
            attach_usage_metadata(&mut response, usage.as_ref());
            assert_eq!(response["error"]["code"], 503);
            assert_eq!(
                response["usageMetadata"],
                json!({"promptTokenCount":100,"cachedContentTokenCount":cache,"candidatesTokenCount":0})
            );
            assert!(response.get("candidates").is_none());
        }
    }

    #[test]
    fn explicit_unclassified_error_is_still_terminal() {
        let mut detector = UpstreamSseErrorDetector::default();
        assert!(detector
            .process_line("data: {\"error\":")
            .unwrap()
            .is_none());
        let error = detector
            .process_line("data: {\"message\":\"upstream failed\"}}")
            .unwrap_err();
        assert_eq!(error.status, 502);
        assert!(error.body.contains("upstream failed"));
    }

    #[tokio::test]
    async fn structured_error_after_output_is_terminal_in_all_protocols() {
        for protocol in ["claude", "chat", "responses", "gemini"] {
            for same_chunk in [false, true] {
                let content = json!({"candidates":[{"content":{"parts":[{"text":"Partial answer"}]}}],"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":1}});
                let failure = json!({"error":{"code":429,"status":"RESOURCE_EXHAUSTED","message":"upstream quota exhausted"},"usageMetadata":{"cachedContentTokenCount":80,"candidatesTokenCount":3,"thoughtsTokenCount":5}});
                let mut chunks = vec![
                    format!("data: {content}\n\n"),
                    "data: {\"error\":\n".to_string(),
                    format!(
                        "data: {},\"usageMetadata\":{}}}\n\n",
                        failure["error"], failure["usageMetadata"]
                    ),
                ];
                if same_chunk {
                    chunks = vec![chunks.concat()];
                }
                let chunks = chunks
                    .into_iter()
                    .map(|s| Ok::<_, String>(Bytes::from(s)))
                    .collect::<Vec<_>>();
                let observer = UpstreamStreamErrorObserver::default();
                let upstream = Box::pin(observe_stream_errors(
                    futures::stream::iter(chunks),
                    observer.clone(),
                ));
                let mut mapped: Pin<Box<dyn Stream<Item = Result<Bytes, String>> + Send>> =
                    match protocol {
                        "claude" => crate::proxy::mappers::claude::create_claude_sse_stream(
                            upstream,
                            "errors".into(),
                            "test@example.test".into(),
                            None,
                            false,
                            1_000_000,
                            None,
                            1,
                            None,
                            vec![],
                        ),
                        "chat" => {
                            crate::proxy::mappers::openai::streaming::create_openai_sse_stream(
                                upstream,
                                "gemini".into(),
                                "errors".into(),
                                1,
                                None,
                                true,
                            )
                        }
                        "responses" => {
                            crate::proxy::mappers::openai::streaming::create_codex_sse_stream(
                                upstream,
                                "gemini".into(),
                                "errors".into(),
                                1,
                                1,
                                "resp_errors".into(),
                                None,
                                false,
                            )
                        }
                        _ => crate::proxy::pipeline::run_stream_lifecycle(
                            upstream,
                            crate::proxy::mappers::gemini::streaming::GeminiStreamHandler::new(
                                None,
                                "errors".into(),
                                "gemini".into(),
                                1,
                                "anchor".into(),
                                false,
                                None,
                                "account".into(),
                                "gemini".into(),
                            ),
                            crate::proxy::pipeline::StreamLifecycleConfig::new(
                                "gemini",
                                "errors",
                                "test".into(),
                            ),
                        ),
                    };
                let mut rendered = Vec::new();
                while let Some(chunk) = mapped.next().await {
                    rendered.push(chunk.unwrap());
                }
                let output = rendered
                    .iter()
                    .map(|b| String::from_utf8_lossy(b))
                    .collect::<String>();
                assert!(output.contains("Partial answer"), "{protocol}: {output}");
                assert!(
                    output.contains("upstream quota exhausted"),
                    "{protocol}: {output}"
                );
                assert!(
                    !output.contains("\"stop_reason\":\"end_turn\""),
                    "{protocol}: {output}"
                );
                assert!(
                    !output.contains("\"finish_reason\":\"stop\""),
                    "{protocol}: {output}"
                );
                assert!(
                    !output.contains("response.completed"),
                    "{protocol}: {output}"
                );
                let failure = observer.failure().unwrap();
                assert_eq!(failure.status, 429);
                assert!(failure.after_output);
                let events = output
                    .lines()
                    .filter_map(|line| line.strip_prefix("data: "))
                    .filter_map(|line| serde_json::from_str::<Value>(line).ok())
                    .collect::<Vec<_>>();
                assert!(events.iter().any(|event| error_envelope(event).is_some()));
                match protocol {
                    "claude" => {
                        assert!(!output.contains("message_stop"));
                        let error = events.iter().find(|e| e["type"] == "error").unwrap();
                        assert_eq!(error["usage"]["cache_read_input_tokens"], 80);
                        assert_eq!(error["usage"]["output_tokens"], 8);
                        let stream = futures::stream::iter(
                            rendered.into_iter().map(Ok::<_, std::io::Error>),
                        );
                        assert!(
                            crate::proxy::mappers::claude::collect_stream_to_json(stream)
                                .await
                                .is_err()
                        );
                    }
                    "chat" => {
                        let error = events.iter().find(|e| e.get("error").is_some()).unwrap();
                        assert_eq!(error["usage"]["prompt_tokens_details"]["cached_tokens"], 80);
                        assert_eq!(error["usage"]["completion_tokens"], 8);
                        let stream =
                            futures::stream::iter(rendered.into_iter().map(Ok::<_, String>));
                        assert!(
                            crate::proxy::mappers::openai::collector::collect_stream_to_json(
                                stream
                            )
                            .await
                            .is_err()
                        );
                    }
                    "responses" => {
                        let failed = events
                            .iter()
                            .find(|e| e["type"] == "response.failed")
                            .unwrap();
                        assert_eq!(
                            failed["response"]["usage"]["input_tokens_details"]["cached_tokens"],
                            80
                        );
                        assert_eq!(failed["response"]["usage"]["output_tokens"], 8);
                    }
                    _ => {
                        assert_eq!(
                            events
                                .iter()
                                .filter(|e| error_envelope(e).is_some())
                                .count(),
                            1
                        );
                        let error = events.iter().find(|e| e.get("error").is_some()).unwrap();
                        assert_eq!(error["error"]["code"], 429);
                        let stream =
                            futures::stream::iter(rendered.into_iter().map(Ok::<_, String>));
                        let collected =
                            crate::proxy::mappers::gemini::collector::collect_stream_to_json(
                                stream, "errors",
                            )
                            .await
                            .unwrap();
                        assert_eq!(collected["error"]["code"], 429);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod refusal_protocol_tests {
    use super::*;
    use crate::proxy::{mappers, pipeline};
    use std::pin::Pin;
    type Input = Pin<Box<dyn Stream<Item = Result<Bytes, String>> + Send>>;
    fn frame(value: Value) -> Result<Bytes, String> {
        Ok(Bytes::from(format!("data: {value}\n\n")))
    }
    fn mapped(protocol: &str, source: Input) -> Input {
        match protocol {
            "claude" => mappers::claude::create_claude_sse_stream(
                source,
                "refusal".into(),
                "test@example.test".into(),
                None,
                false,
                1_000_000,
                None,
                1,
                None,
                vec![],
            ),
            "chat" | "chat_no_usage" => mappers::openai::streaming::create_openai_sse_stream(
                source,
                "gemini".into(),
                "refusal".into(),
                1,
                None,
                protocol == "chat",
            ),
            "responses" => mappers::openai::streaming::create_codex_sse_stream(
                source,
                "gemini".into(),
                "refusal".into(),
                1,
                1,
                "resp_refusal".into(),
                None,
                false,
            ),
            _ => pipeline::run_stream_lifecycle(
                source,
                mappers::gemini::streaming::GeminiStreamHandler::new(
                    None,
                    "refusal".into(),
                    "gemini".into(),
                    1,
                    "root".into(),
                    false,
                    None,
                    "acct".into(),
                    "gemini".into(),
                ),
                pipeline::StreamLifecycleConfig::new("gemini", "refusal", "trace".into()),
            ),
        }
    }
    #[tokio::test]
    async fn refusal_matrix_completes_without_content_and_retains_usage() {
        for reason in [
            "SAFETY",
            "RECITATION",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
            "SPII",
            "IMAGE_SAFETY",
            "IMAGE_PROHIBITED_CONTENT",
            "MODEL_ARMOR",
            "PROMPT_OTHER",
        ] {
            for thinking in [false, true] {
                for protocol in ["gemini", "claude", "chat", "chat_no_usage", "responses"] {
                    let value = if reason == "PROMPT_OTHER" {
                        json!({"promptFeedback":{"blockReason":"OTHER"},"usageMetadata":{"promptTokenCount":100}})
                    } else {
                        json!({"candidates":[{"content":{"parts":if thinking {vec![json!({"thought":true,"text":"thinking"})]}else{vec![]}},"finishReason":reason}],"usageMetadata":{"promptTokenCount":100}})
                    };
                    let source = Box::pin(futures::stream::iter(vec![
                        frame(value),
                        frame(
                            json!({"usageMetadata":{"cachedContentTokenCount":80,"thoughtsTokenCount":5}}),
                        ),
                    ]));
                    let mut stream = mapped(protocol, source);
                    let mut wire = String::new();
                    let mut events = vec![];
                    while let Some(chunk) = stream.next().await {
                        wire.push_str(&String::from_utf8_lossy(&chunk.unwrap_or_else(|err| {
                            panic!("{protocol}/{reason}/{thinking}: {err}")
                        })));
                    }
                    for line in wire.lines() {
                        if let Some(data) = line.strip_prefix("data: ") {
                            if let Ok(value) = serde_json::from_str::<Value>(data) {
                                events.push(value);
                            }
                        }
                    }
                    assert!(!wire.contains("task ready"));
                    assert!(
                        events.iter().all(|event| error_envelope(event).is_none()),
                        "{protocol}/{reason}: {wire}"
                    );
                    let usage = match protocol {
                        "gemini" => {
                            let value = mappers::gemini::collector::collect_stream_to_json(
                                futures::stream::iter(vec![Ok::<_, String>(Bytes::from(
                                    wire.clone(),
                                ))]),
                                "refusal",
                            )
                            .await
                            .unwrap();
                            if reason == "PROMPT_OTHER" {
                                assert_eq!(value["promptFeedback"]["blockReason"], "OTHER");
                                assert_eq!(value["candidates"], json!([]));
                            } else {
                                assert_eq!(value["candidates"][0]["finishReason"], reason);
                            }
                            pipeline::CanonicalUsage::from_gemini(&value)
                        }
                        "claude" => {
                            assert_eq!(
                                events
                                    .iter()
                                    .filter(|event| event["type"] == "message_stop")
                                    .count(),
                                1
                            );
                            let event = events
                                .iter()
                                .find(|event| event["type"] == "message_delta")
                                .unwrap();
                            assert_eq!(event["delta"]["stop_reason"], "refusal");
                            pipeline::CanonicalUsage::from_gemini(&event["usage"])
                        }
                        "chat" | "chat_no_usage" => {
                            assert_eq!(wire.matches("data: [DONE]").count(), 1);
                            assert_eq!(
                                events
                                    .iter()
                                    .filter(|event| event.pointer("/choices/0/finish_reason")
                                        == Some(&json!("content_filter")))
                                    .count(),
                                1
                            );
                            pipeline::CanonicalUsage::from_gemini(
                                &events
                                    .iter()
                                    .rev()
                                    .find(|event| event.get("usage").is_some())
                                    .unwrap()["usage"],
                            )
                        }
                        _ => {
                            let event = events
                                .iter()
                                .find(|event| event["type"] == "response.incomplete")
                                .unwrap();
                            assert_eq!(
                                event["response"]["incomplete_details"]["reason"],
                                "content_filter"
                            );
                            assert!(event["response"]["error"].is_null());
                            pipeline::CanonicalUsage::from_gemini(&event["response"]["usage"])
                        }
                    };
                    assert_eq!(
                        (
                            usage.total_input_tokens,
                            usage.cached_tokens,
                            usage.output_tokens
                        ),
                        (100, 80, 5),
                        "{protocol}/{reason}: {wire}"
                    );
                }
            }
        }
    }
    #[tokio::test(start_paused = true)]
    async fn candidate_refusal_does_not_cut_off_other_candidates() {
        for protocol in ["gemini", "chat"] {
            let source = async_stream::stream! {
                yield frame(json!({"candidates":[{"index":0,"finishReason":"SAFETY"},{"index":1,"content":{"parts":[{"text":"begin "}]}}],"usageMetadata":{"promptTokenCount":100}}));
                tokio::time::sleep(std::time::Duration::from_millis(400)).await;
                yield frame(json!({"candidates":[{"index":1,"content":{"parts":[{"text":"delayed other candidate"}]},"finishReason":"STOP"}],"usageMetadata":{"cachedContentTokenCount":80,"candidatesTokenCount":5}}));
            };
            let mut stream = mapped(protocol, Box::pin(source));
            let mut wire = String::new();
            while let Some(chunk) = stream.next().await {
                wire.push_str(&String::from_utf8_lossy(
                    &chunk.expect("mixed candidates complete"),
                ));
            }
            assert!(
                wire.contains("delayed other candidate"),
                "{protocol}: {wire}"
            );
            let mut usage = crate::proxy::pipeline::usage::ObservedUsage::default();
            for line in wire.lines() {
                if let Some(data) = line.strip_prefix("data: ") {
                    if let Ok(value) = serde_json::from_str::<Value>(data) {
                        usage.merge_event(&value);
                    }
                }
            }
            assert_eq!(usage.cached_tokens(), Some(80));
        }
    }
}
