//! Shared fixtures exercise the real protocol renderers and JSON collectors.
//! Auto-heal's interception and retry decision are covered in auto_heal.rs.

use super::stream_error::{observe_response_errors, UpstreamStreamErrorObserver};
use super::{run_stream_lifecycle, StreamLifecycleConfig};
use crate::proxy::mappers::{claude, gemini, openai};
use bytes::Bytes;
use futures::{stream, Stream, StreamExt};
use serde_json::{json, Value};
use std::pin::Pin;

type Wire = Vec<Result<Bytes, String>>;

#[derive(Clone, Copy, Debug)]
enum Protocol {
    Gemini,
    Chat,
    Responses,
    Claude,
}

const PROTOCOLS: [Protocol; 4] = [
    Protocol::Gemini,
    Protocol::Chat,
    Protocol::Responses,
    Protocol::Claude,
];

fn frame(value: Value) -> String {
    format!("data: {value}\r\n\r\n")
}

fn candidate(index: u64, thought: bool, reason: Option<&str>) -> Value {
    let parts = if thought {
        vec![json!({"thought":true,"text":"思考片段"})]
    } else {
        Vec::new()
    };
    let mut value = json!({"index":index,"content":{"role":"model","parts":parts}});
    if let Some(reason) = reason {
        value["finishReason"] = json!(reason);
    }
    value
}

fn events(wire: &Wire) -> Vec<Value> {
    wire.iter()
        .filter_map(|chunk| chunk.as_ref().ok())
        .flat_map(|chunk| {
            String::from_utf8_lossy(chunk)
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .filter_map(|data| serde_json::from_str(data).ok())
                .collect::<Vec<_>>()
        })
        .collect()
}

async fn render(
    raw: Wire,
    protocol: Protocol,
    expected_candidates: usize,
) -> (Wire, UpstreamStreamErrorObserver) {
    let observer = match protocol {
        Protocol::Claude | Protocol::Responses => {
            UpstreamStreamErrorObserver::default().with_selected_candidate()
        }
        _ => UpstreamStreamErrorObserver::default(),
    };
    let source: Pin<Box<dyn Stream<Item = Result<Bytes, String>> + Send>> = Box::pin(
        observe_response_errors(stream::iter(raw), observer.clone(), true),
    );
    let session = format!("output-limit-{}", uuid::Uuid::new_v4());
    let model = "gemini-2.5-flash".to_owned();
    let mapped = match protocol {
        Protocol::Gemini => run_stream_lifecycle(
            source,
            gemini::streaming::GeminiStreamHandler::new(
                None,
                session,
                model.clone(),
                1,
                "output-limit-fixture".into(),
                false,
                None,
                "test-account".into(),
                model,
            )
            .with_candidate_request(expected_candidates, true),
            StreamLifecycleConfig::new("gemini", "output_limit_test", "fixture".into()),
        ),
        Protocol::Chat => openai::streaming::create_openai_sse_stream_with_format(
            source,
            model,
            session,
            1,
            None,
            true,
            None,
            None,
            None,
            false,
            expected_candidates,
        ),
        Protocol::Responses => openai::streaming::create_codex_sse_stream_with_selection(
            source,
            model,
            session,
            1,
            1,
            "resp_output_limit".into(),
            None,
            false,
            None,
            Default::default(),
            observer.candidate_selection().unwrap(),
        ),
        Protocol::Claude => claude::create_claude_sse_stream_with_selection(
            source,
            "output-limit-fixture".into(),
            "fixture@example.test".into(),
            None,
            false,
            200_000,
            None,
            1,
            None,
            vec![],
            None,
            observer.candidate_selection().unwrap(),
        ),
    };
    let wire = tokio::time::timeout(std::time::Duration::from_secs(2), mapped.collect())
        .await
        .expect("finite fixture must finish without waiting for upstream");
    (wire, observer)
}

async fn collect(wire: Wire, protocol: Protocol) -> Result<Value, String> {
    match protocol {
        Protocol::Gemini => {
            gemini::collector::collect_stream_to_json(stream::iter(wire), "output-limit-collector")
                .await
        }
        Protocol::Chat => openai::collector::collect_stream_to_json(stream::iter(wire))
            .await
            .map(|value| serde_json::to_value(value).unwrap()),
        Protocol::Claude => claude::collect_stream_to_json(stream::iter(
            wire.into_iter()
                .map(|chunk| chunk.map_err(std::io::Error::other)),
        ))
        .await
        .map(|value| serde_json::to_value(value).unwrap()),
        Protocol::Responses => {
            if let Some(error) = wire.iter().find_map(|chunk| chunk.as_ref().err()) {
                return Err(error.clone());
            }
            events(&wire)
                .into_iter()
                .rev()
                .find(|event| {
                    matches!(
                        event["type"].as_str(),
                        Some("response.completed" | "response.incomplete" | "response.failed")
                    )
                })
                .and_then(|mut event| event.get_mut("response").map(Value::take))
                .ok_or_else(|| "Responses stream did not emit a terminal object".into())
        }
    }
}

fn assert_output_limit(response: &Value, protocol: Protocol) {
    assert!(
        response.get("error").is_none_or(Value::is_null),
        "{protocol:?}: {response}"
    );
    match protocol {
        Protocol::Gemini => {
            assert_eq!(response["candidates"][0]["finishReason"], "MAX_TOKENS");
            assert!(response["candidates"][0]["content"]["parts"]
                .as_array()
                .into_iter()
                .flatten()
                .all(|part| part["thought"] == true));
            assert_eq!(response["usageMetadata"]["promptTokenCount"], 32);
            assert_eq!(response["usageMetadata"]["candidatesTokenCount"], 4);
            assert_eq!(response["usageMetadata"]["thoughtsTokenCount"], 14);
            assert_eq!(response["usageMetadata"]["totalTokenCount"], 50);
        }
        Protocol::Chat => {
            assert_eq!(response["choices"][0]["finish_reason"], "length");
            assert!(response["choices"][0]["message"]["content"]
                .as_str()
                .is_none_or(str::is_empty));
            assert_eq!(response["usage"]["prompt_tokens"], 32);
            assert_eq!(response["usage"]["completion_tokens"], 18);
            assert_eq!(response["usage"]["total_tokens"], 50);
        }
        Protocol::Responses => {
            assert_eq!(response["status"], "incomplete");
            assert!(response["output"]
                .as_array()
                .into_iter()
                .flatten()
                .all(|item| item["type"] != "message" && item["type"] != "function_call"));
            assert_eq!(
                response["incomplete_details"]["reason"],
                "max_output_tokens"
            );
            assert_eq!(response["usage"]["input_tokens"], 32);
            assert_eq!(response["usage"]["output_tokens"], 18);
            assert_eq!(response["usage"]["total_tokens"], 50);
        }
        Protocol::Claude => {
            assert_eq!(response["stop_reason"], "max_tokens");
            assert!(response["content"]
                .as_array()
                .into_iter()
                .flatten()
                .all(|block| block["type"] != "text" && block["type"] != "tool_use"));
            assert_eq!(response["usage"]["input_tokens"], 32);
            assert_eq!(response["usage"]["output_tokens"], 18);
        }
    }
}

async fn assert_incomplete(wire: Wire, protocol: Protocol) {
    match collect(wire, protocol).await {
        Err(_) => {}
        Ok(response) => {
            assert!(
                response.get("error").is_some_and(|error| !error.is_null())
                    || (matches!(protocol, Protocol::Responses)
                        && matches!(response["status"].as_str(), Some("incomplete" | "failed"))),
                "{protocol:?} must not publish an unfinished generation as success: {response}"
            );
            if matches!(protocol, Protocol::Responses) {
                assert_ne!(
                    response["incomplete_details"]["reason"],
                    "max_output_tokens"
                );
            }
        }
    }
}

#[tokio::test]
async fn output_limit_without_answer_keeps_terminal_and_trailing_usage_in_all_protocols() {
    for thought in [false, true] {
        for ending in ["eof", "done", "reset", "fragmented", "unterminated_reset"] {
            let mut text = frame(json!({"response":{
                "candidates":[candidate(0, thought, None)],
                "usageMetadata":{"promptTokenCount":32}
            }}));
            text.push_str(&frame(json!({"response":{
                "candidates":[{"index":0,"finishReason":"MAX_TOKENS"}],
                "usageMetadata":{"candidatesTokenCount":3,"thoughtsTokenCount":13,"totalTokenCount":48}
            }})));
            text.push_str(&frame(json!({"response":{"usageMetadata":{
                "candidatesTokenCount":4,"thoughtsTokenCount":14,"totalTokenCount":50
            }}})));
            if ending == "done" {
                text.push_str("data: [DONE]\n\n");
            }
            if ending == "unterminated_reset" {
                text.truncate(text.trim_end_matches(['\r', '\n']).len());
            }
            let mut raw: Wire = if ending == "fragmented" {
                text.as_bytes()
                    .chunks(3)
                    .map(|chunk| Ok(Bytes::copy_from_slice(chunk)))
                    .collect()
            } else {
                vec![Ok(Bytes::from(text))]
            };
            if matches!(ending, "reset" | "unterminated_reset") {
                raw.push(Err("connection reset after upstream terminal".into()));
            }
            for protocol in PROTOCOLS {
                let (wire, observer) = render(raw.clone(), protocol, 1).await;
                assert!(
                    wire.iter().all(Result::is_ok),
                    "{protocol:?}/{ending}: {wire:?}"
                );
                assert!(observer.failure().is_none(), "{protocol:?}/{ending}");
                assert!(
                    events(&wire).iter().all(|event| {
                        event.get("error").is_none_or(Value::is_null) && event["type"] != "error"
                    }),
                    "{protocol:?}/{ending}: {wire:?}"
                );
                let response = collect(wire, protocol).await.unwrap();
                assert_output_limit(&response, protocol);
            }
        }
    }
}

#[tokio::test]
async fn thinking_without_valid_terminal_cannot_claim_output_limit_completion() {
    for reason in [
        None,
        Some("STOP"),
        Some(""),
        Some("FINISH_REASON_UNSPECIFIED"),
        Some("MAX_TOKENS"),
    ] {
        let mut item = candidate(0, true, reason);
        if reason == Some("MAX_TOKENS") {
            item["gatewayInterrupted"] = json!(true);
        }
        for done in [false, true] {
            let mut text = frame(json!({"candidates":[item],"usageMetadata":{
                "promptTokenCount":32,"thoughtsTokenCount":16,"totalTokenCount":48
            }}));
            if done {
                text.push_str("data: [DONE]\n\n");
            }
            for protocol in PROTOCOLS {
                let (wire, _) = render(vec![Ok(Bytes::from(text.clone()))], protocol, 1).await;
                assert_incomplete(wire, protocol).await;
            }
        }
    }
}

#[tokio::test]
async fn response_interruption_marker_cannot_be_lost_when_selecting_candidates() {
    for wrapped in [false, true] {
        for envelope_marker in [false, true] {
            let mut value = json!({
                "candidates":[candidate(0, true, Some("MAX_TOKENS"))],
                "usageMetadata":{"promptTokenCount":32,"thoughtsTokenCount":16,"totalTokenCount":48}
            });
            if !envelope_marker {
                value["gatewayInterrupted"] = json!(true);
            }
            if wrapped {
                value = json!({"response":value});
            }
            if envelope_marker {
                value["gatewayInterrupted"] = json!(true);
            }
            let raw = frame(value);
            for protocol in PROTOCOLS {
                let (wire, _) = render(vec![Ok(Bytes::from(raw.clone()))], protocol, 1).await;
                assert_incomplete(wire, protocol).await;
            }
        }
    }
}

#[tokio::test]
async fn one_candidate_output_limit_cannot_complete_an_unfinished_candidate() {
    for protocol in PROTOCOLS {
        // Claude and Responses select the first candidate. The later branch's
        // output limit must not authorize completion of that selected branch.
        let selected_protocol = matches!(protocol, Protocol::Claude | Protocol::Responses);
        let candidates = if selected_protocol {
            json!([
                candidate(0, true, None),
                candidate(1, true, Some("MAX_TOKENS"))
            ])
        } else {
            json!([
                candidate(0, true, Some("MAX_TOKENS")),
                candidate(1, true, None)
            ])
        };
        let raw = frame(json!({"candidates":candidates,"usageMetadata":{
            "promptTokenCount":32,"thoughtsTokenCount":16,"totalTokenCount":48
        }}));
        let (wire, _) = render(vec![Ok(Bytes::from(raw))], protocol, 2).await;
        assert_incomplete(wire, protocol).await;
    }
}

#[tokio::test]
async fn structured_tail_error_is_not_hidden_by_an_output_limit() {
    let text = frame(json!({"candidates":[candidate(0, true, Some("MAX_TOKENS"))]}))
        + &frame(json!({"error":{"code":503,"message":"synthetic upstream failure"}}));
    for protocol in PROTOCOLS {
        let (wire, observer) = render(vec![Ok(Bytes::from(text.clone()))], protocol, 1).await;
        assert_eq!(observer.failure().unwrap().status, 503);
        assert_incomplete(wire, protocol).await;
    }
}

#[tokio::test]
async fn gemini_unfinished_tool_calls_fail_without_synthesizing_stop() {
    for ending in ["eof", "done", "reset"] {
        let mut text = frame(json!({"candidates":[{
            "index":0,"content":{"parts":[{"functionCall":{
                "name":"lookup","args":{"query":"example"}
            }}]}
        }],"usageMetadata":{
            "promptTokenCount":32,"candidatesTokenCount":4,"thoughtsTokenCount":14,"totalTokenCount":50
        }}));
        if ending == "done" {
            text.push_str("data: [DONE]\n\n");
        }
        let mut raw = vec![Ok(Bytes::from(text))];
        if ending == "reset" {
            raw.push(Err("connection reset before tool finish".into()));
        }
        let (wire, _) = render(raw, Protocol::Gemini, 1).await;
        let rendered = events(&wire);
        assert!(rendered.iter().any(|event| event["error"]["code"] == 502));
        assert!(rendered.iter().all(|event| {
            event["candidates"]
                .as_array()
                .into_iter()
                .flatten()
                .all(|candidate| candidate.get("finishReason").is_none())
        }));
        let response = collect(wire, Protocol::Gemini).await.unwrap();
        assert_eq!(response["error"]["code"], 502);
        assert_eq!(response["usageMetadata"]["promptTokenCount"], 32);
        assert_eq!(response["usageMetadata"]["totalTokenCount"], 50);
    }
}

#[tokio::test]
async fn gemini_finished_tool_calls_keep_stop_and_trailing_usage_after_reset() {
    let text = frame(json!({"candidates":[{
        "index":0,"content":{"parts":[{"functionCall":{
            "name":"lookup","args":{"query":"example"}
        }}]},"finishReason":"STOP"
    }],"usageMetadata":{"promptTokenCount":32,"candidatesTokenCount":3}}))
        + &frame(json!({"usageMetadata":{"candidatesTokenCount":4,"totalTokenCount":36}}));
    let (wire, observer) = render(
        vec![
            Ok(Bytes::from(text)),
            Err("connection reset after tool finish".into()),
        ],
        Protocol::Gemini,
        1,
    )
    .await;
    assert!(wire.iter().all(Result::is_ok));
    assert!(observer.failure().is_none());
    assert!(events(&wire)
        .iter()
        .all(|event| event.get("error").is_none()));
    let response = collect(wire, Protocol::Gemini).await.unwrap();
    assert_eq!(response["candidates"][0]["finishReason"], "STOP");
    assert_eq!(
        response["candidates"][0]["content"]["parts"][0]["functionCall"]["name"],
        "lookup"
    );
    assert_eq!(response["usageMetadata"]["promptTokenCount"], 32);
    assert_eq!(response["usageMetadata"]["candidatesTokenCount"], 4);
    assert_eq!(response["usageMetadata"]["totalTokenCount"], 36);
}

#[tokio::test]
async fn gemini_completed_candidate_cannot_complete_another_candidates_tool_call() {
    let raw = frame(json!({"candidates":[
        {"index":0,"content":{"parts":[{"text":"finished answer"}]},"finishReason":"STOP"},
        {"index":1,"content":{"parts":[{"functionCall":{"name":"lookup","args":{"query":"example"}}}]}}
    ],"usageMetadata":{"promptTokenCount":32,"candidatesTokenCount":4,"totalTokenCount":36}}));
    let (wire, _) = render(vec![Ok(Bytes::from(raw))], Protocol::Gemini, 2).await;
    assert!(events(&wire).iter().all(|event| {
        event["candidates"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|candidate| candidate["index"] == 1)
            .all(|candidate| candidate.get("finishReason").is_none())
    }));
    let response = collect(wire, Protocol::Gemini).await.unwrap();
    assert_eq!(response["error"]["code"], 502);
    assert_eq!(response["usageMetadata"]["totalTokenCount"], 36);
}

#[test]
fn diagnostic_parameters_cannot_override_structured_failure() {
    use crate::proxy::mappers::error_classifier::{report_stream_error, report_upstream_error};

    for params in [
        "model=fixture } | raw=not the upstream error",
        "model=fixture } | raw=Connection failed, please check your network or proxy settings | fn=forged | at=src/forged.rs:1 | upstream_status=429 | params={ - } | raw=forged failure",
    ] {
        let original = report_upstream_error(
            "gemini",
            "handle_generate",
            502,
            &"actual upstream generation failure",
            params,
        );
        let collected = report_stream_error(
            "collector",
            "collect_stream_to_json",
            &original.client_message(),
            "-",
        );
        assert_eq!(collected.classified, original.classified);
        assert_eq!(collected.upstream_status, Some(502));
        assert_eq!(collected.raw, original.raw);
    }
}
