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
    /// Replaying after text, thinking, tool calls or media would duplicate output.
    /// Also true if an oversized frame made it unsafe to prove no output occurred.
    pub after_output: bool,
}

#[derive(Default)]
struct Observation {
    failure: Option<UpstreamStreamError>,
    has_output: bool,
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
            failure
        })
    }

    pub fn take(&self) -> Option<UpstreamStreamError> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.failure.take().map(|mut failure| {
            failure.after_output |= state.has_output;
            failure
        })
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
        if state.failure.is_some() {
            return;
        }
        let Some(error) = inner
            .get("error")
            .or_else(|| value.get("error"))
            .filter(|v| v.is_object())
        else {
            return;
        };
        let Some(status) = error_status(error) else {
            return;
        };
        state.failure = Some(UpstreamStreamError {
            status,
            body: json!({"error": error}).to_string(),
            after_output: state.has_output,
        });
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

#[derive(Default)]
struct SseErrorParser {
    line: Vec<u8>,
    data: Vec<u8>,
    discard_line: bool,
    discard_event: bool,
    done: bool,
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
                if self.line.len() < MAX_EVENT_BYTES {
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
                if self.data.len().saturating_add(data.len()).saturating_add(1) <= MAX_EVENT_BYTES {
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
    async_stream::stream! {
        futures::pin_mut!(stream);
        let mut parser = SseErrorParser::default();
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
    fn malformed_unknown_and_post_done_envelopes_do_not_classify() {
        for payload in [
            "{\"error\":\"unauthorized quota exhausted\"}",
            "{\"error\":{\"message\":\"unauthorized quota exhausted\"}}",
            "{\"error\":{\"code\":200}}",
            "{\"error\":{\"status\":\"not a status\"}}",
            "{bad json}",
        ] {
            let observer = UpstreamStreamErrorObserver::default();
            let mut parser = SseErrorParser::default();
            parser.feed(format!("data: {payload}\n\n").as_bytes(), &observer);
            assert!(observer.failure().is_none(), "{payload}");
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
