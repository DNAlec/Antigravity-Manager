use super::*;
use crate::proxy::mappers::{claude, openai};
use crate::proxy::pipeline::stream_error::{observe_response_errors, UpstreamStreamErrorObserver};
use futures::stream;
use std::io;

type Wire = Vec<Result<Bytes, String>>;
#[derive(Clone, Copy, Debug)]
enum Route {
    Chat,
    Intermediate,
    Claude,
}
const ROUTES: [Route; 3] = [Route::Chat, Route::Intermediate, Route::Claude];

fn frame(candidates: Value) -> Result<Bytes, String> {
    Ok(Bytes::from(format!(
        "data: {}\n\n",
        json!({"response":{"candidates":candidates,"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":7}}})
    )))
}
fn candidate(index: u64, text: &str, finish: Option<&str>) -> Value {
    let mut value = json!({"index":index,"content":{"role":"model","parts":[{"text":text}]}});
    if let Some(finish) = finish {
        value["finishReason"] = json!(finish);
    }
    value
}
fn events(wire: &Wire) -> Vec<Value> {
    wire.iter()
        .flat_map(|chunk| {
            String::from_utf8_lossy(chunk.as_ref().unwrap())
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .filter_map(|line| serde_json::from_str(line).ok())
                .collect::<Vec<_>>()
        })
        .collect()
}
async fn render(raw: Wire, route: Route) -> (Wire, UpstreamStreamErrorObserver) {
    render_source(Box::pin(stream::iter(raw)), route).await
}

async fn render_source(
    raw: Pin<Box<dyn Stream<Item = Result<Bytes, String>> + Send>>,
    route: Route,
) -> (Wire, UpstreamStreamErrorObserver) {
    let observer = if matches!(route, Route::Chat) {
        UpstreamStreamErrorObserver::default()
    } else {
        UpstreamStreamErrorObserver::default().with_selected_candidate()
    };
    let source = Box::pin(observe_response_errors(raw, observer.clone(), true));
    let wire = match route {
        Route::Chat => {
            create_openai_sse_stream(
                source,
                "gemini-flash".into(),
                "r32-chat".into(),
                1,
                None,
                true,
            )
            .collect()
            .await
        }
        Route::Intermediate => {
            create_openai_sse_stream_for_responses_with_selection(
                source,
                "gemini-flash".into(),
                "r32-intermediate".into(),
                1,
                None,
                true,
                None,
                None,
                observer.candidate_selection().unwrap(),
            )
            .collect()
            .await
        }
        Route::Claude => {
            claude::create_claude_sse_stream_with_selection(
                source,
                "r32".into(),
                "r32@example.test".into(),
                None,
                false,
                200_000,
                None,
                1,
                None,
                vec![],
                None,
                observer.candidate_selection().unwrap(),
            )
            .collect()
            .await
        }
    };
    (wire, observer)
}
async fn collect(wire: Wire, route: Route) -> Result<Value, String> {
    match route {
        Route::Chat | Route::Intermediate => {
            openai::collector::collect_stream_to_json(stream::iter(wire))
                .await
                .map(|v| serde_json::to_value(v).unwrap())
        }
        Route::Claude => claude::collect_stream_to_json(stream::iter(
            wire.into_iter()
                .map(|chunk| chunk.map_err(io::Error::other)),
        ))
        .await
        .map(|v| serde_json::to_value(v).unwrap()),
    }
}

#[tokio::test]
async fn r32_eof_done_and_reset_preserve_text_with_truncation_terminal() {
    let text = "Partial answer with HTTP 200 and {\"error\":503} as examples.";
    for ending in ["eof", "done", "reset"] {
        for route in ROUTES {
            let mut raw = vec![frame(json!([candidate(0, text, None)]))];
            if ending == "done" {
                raw.push(Ok(Bytes::from("data: [DONE]\n\n")));
            }
            if ending == "reset" {
                raw.push(Err("connection reset by peer".into()));
            }
            let (wire, observer) = render(raw, route).await;
            assert!(!events(&wire)
                .iter()
                .any(|event| event.get("error").is_some()));
            assert!(observer.failure().is_none());
            let result = collect(wire, route).await.unwrap();
            match route {
                Route::Claude => {
                    assert_eq!(result["content"][0]["text"], text);
                    assert_eq!(result["usage"]["output_tokens"], 7);
                    assert_eq!(result["stop_reason"], "max_tokens");
                }
                _ => {
                    assert_eq!(result["choices"][0]["message"]["content"], text);
                    assert_eq!(result["usage"]["completion_tokens"], 7);
                    assert_eq!(result["choices"][0]["finish_reason"], "length");
                }
            }
        }
    }
}

#[tokio::test]
async fn r32_completed_responses_keep_finish_and_trailing_usage_after_reset() {
    for (finish, chat_finish, claude_finish) in [
        ("STOP", "stop", "end_turn"),
        ("MAX_TOKENS", "length", "max_tokens"),
        ("IMAGE_RECITATION", "content_filter", "refusal"),
        ("RECITATION", "content_filter", "refusal"),
        ("IMAGE_SAFETY", "content_filter", "refusal"),
    ] {
        for route in ROUTES {
            let (wire, observer) = render(
                vec![
                    frame(json!([candidate(
                        0,
                        "Normal error:503 sample",
                        Some(finish)
                    )])),
                    Ok(Bytes::from(
                        "data: {\"usageMetadata\":{\"candidatesTokenCount\":9}}\n\n",
                    )),
                    Err("connection reset after model finish".into()),
                ],
                route,
            )
            .await;
            assert!(observer.failure().is_none(), "{finish}/{route:?}");
            assert!(!events(&wire)
                .iter()
                .any(|event| event.get("error").is_some()));
            let result = collect(wire, route).await.unwrap();
            match route {
                Route::Claude => {
                    assert_eq!(result["stop_reason"], claude_finish);
                    assert_eq!(result["usage"]["output_tokens"], 9);
                }
                _ => {
                    assert_eq!(result["choices"][0]["finish_reason"], chat_finish);
                    assert_eq!(result["usage"]["completion_tokens"], 9);
                }
            }
        }
    }
}

#[tokio::test]
async fn r32_chat_keeps_finished_candidates_and_truncates_unfinished_text() {
    for second_finish in [None, Some("STOP")] {
        for first_is_tool in [false, true] {
            let first = if first_is_tool {
                json!({"index":0,"content":{"role":"model","parts":[{"functionCall":{"name":"lookup","args":{"query":"safe"}}}]},"finishReason":"STOP"})
            } else {
                candidate(0, "A", Some("STOP"))
            };
            let raw = vec![
                frame(json!([first, candidate(1, "B", None)])),
                frame(json!([candidate(1, "_END", second_finish)])),
            ];
            let (wire, _) = render(raw, Route::Chat).await;
            assert!(!events(&wire)
                .iter()
                .any(|event| event.get("error").is_some()));
            let result = collect(wire, Route::Chat).await.unwrap();
            assert_eq!(
                result["choices"][0]["finish_reason"],
                if first_is_tool { "tool_calls" } else { "stop" }
            );
            if first_is_tool {
                assert_eq!(
                    result["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
                    "lookup"
                );
            }
            assert_eq!(result["choices"][1]["message"]["content"], "B_END");
            assert_eq!(
                result["choices"][1]["finish_reason"],
                if second_finish.is_some() {
                    "stop"
                } else {
                    "length"
                }
            );
        }
    }
}

#[tokio::test]
async fn r32_unfinished_tool_turn_cannot_be_published_as_executable_completion() {
    for route in ROUTES {
        let raw = vec![
            frame(json!([candidate(0, "Inspecting files", None)])),
            frame(
                json!([{"index":0,"content":{"role":"model","parts":[{"functionCall":{"name":"run_bash","args":{"command":"ls"}}}]}}]),
            ),
        ];
        let (wire, observer) = render(raw, route).await;
        let rendered = events(&wire);
        assert!(rendered
            .iter()
            .any(|event| event["error"]["code"] == "upstream_interrupted"));
        assert!(!rendered
            .iter()
            .any(
                |event| event.pointer("/choices/0/finish_reason") == Some(&json!("tool_calls"))
                    || event.pointer("/delta/stop_reason") == Some(&json!("tool_use"))
            ));
        let collected_error = collect(wire, route).await.unwrap_err();
        let failure = observer
            .failure_or_incomplete(Some(&collected_error))
            .unwrap();
        assert_eq!(failure.status, 502);
        assert!(!failure.replay_safe());
        let body: Value = serde_json::from_str(&failure.body).unwrap();
        assert_eq!(body["error"]["code"], "upstream_interrupted");
        for protocol in ["openai", "responses", "claude"] {
            let http_error =
                crate::proxy::handlers::common::build_dual_track_error_with_ua_and_lang(
                    protocol,
                    502,
                    "gemini-flash",
                    &failure.body,
                    None,
                    Some("en"),
                );
            assert_eq!(http_error["partial_response"], body["partial_response"]);
            assert_eq!(http_error["error"]["code"], "upstream_interrupted");
            assert_ne!(http_error["status"], "completed");
        }
    }
}

#[tokio::test]
async fn r32_truncated_json_does_not_manufacture_a_tool_call() {
    for route in ROUTES {
        let raw = vec![frame(json!([candidate(0,"Inspecting files",None)])),Ok(Bytes::from("data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"run_bash\",\"args\":{\"command\":"))];
        let (wire, _) = render(raw, route).await;
        let rendered = events(&wire);
        assert!(!rendered
            .iter()
            .any(
                |event| event.pointer("/content_block/type") == Some(&json!("tool_use"))
                    || event.pointer("/choices/0/delta/tool_calls").is_some()
            ));
        let result = collect(wire, route).await.unwrap();
        match route {
            Route::Claude => assert_eq!(result["stop_reason"], "max_tokens"),
            _ => assert_eq!(result["choices"][0]["finish_reason"], "length"),
        }
    }
}

#[tokio::test]
async fn r32_responses_stream_image_recitation_is_refusal_and_real_failure_stays_failed() {
    for (reason, expected_error) in [
        ("IMAGE_RECITATION", None),
        ("IMAGE_OTHER", Some("upstream_finish_error")),
        ("MALFORMED_FUNCTION_CALL", Some("upstream_finish_error")),
    ] {
        let observer = UpstreamStreamErrorObserver::default();
        let raw = vec![frame(json!([candidate(
            0,
            "Readable explanation",
            Some(reason)
        )]))];
        let source = Box::pin(observe_response_errors(
            stream::iter(raw),
            observer.clone(),
            false,
        ));
        let wire: Wire = create_codex_sse_stream(
            source,
            "gemini-flash".into(),
            "r32-responses".into(),
            1,
            1,
            format!("r32-{reason}"),
            None,
            false,
        )
        .collect()
        .await;
        let rendered = events(&wire);
        let terminal = rendered.last().unwrap();
        assert_eq!(terminal["type"], "response.incomplete");
        if let Some(error) = expected_error {
            assert_eq!(terminal["response"]["error"]["code"], error);
            assert_eq!(observer.failure().unwrap().status, 502);
        } else {
            assert_eq!(
                terminal["response"]["incomplete_details"]["reason"],
                "content_filter"
            );
            assert!(terminal["response"]["error"].is_null());
            assert!(observer.failure().is_none());
        }
    }
}

#[tokio::test]
async fn r32_text_truncation_keeps_usage_on_finish_when_usage_chunks_are_disabled() {
    for interrupted in [false, true] {
        let mut raw = vec![
            frame(json!([candidate(0, "Partial text", None)])),
            Ok(Bytes::from(
                "data: {\"usageMetadata\":{\"candidatesTokenCount\":9}}\n\n",
            )),
        ];
        if interrupted {
            raw.push(Err("connection reset".into()));
        }
        let wire: Wire = create_openai_sse_stream(
            Box::pin(stream::iter(raw)),
            "gemini-flash".into(),
            "r32-usage-disabled".into(),
            1,
            None,
            false,
        )
        .collect()
        .await;
        let rendered = events(&wire);
        let finish = rendered
            .iter()
            .find(|event| event.pointer("/choices/0/finish_reason") == Some(&json!("length")))
            .unwrap();
        assert_eq!(finish["usage"]["completion_tokens"], 9);
        assert!(rendered
            .iter()
            .all(|event| !event["choices"].as_array().is_some_and(Vec::is_empty)));
    }
}

#[tokio::test]
async fn r32_complete_unterminated_stop_before_reset_remains_successful() {
    for route in ROUTES {
        for tool in [false, true] {
            let first = if tool {
                json!({"index":0,"content":{"role":"model","parts":[{"functionCall":{"name":"lookup","args":{"query":"safe"}}}]}})
            } else {
                candidate(0, "Complete answer", None)
            };
            let raw = vec![frame(json!([first])),
                Ok(Bytes::from("data: {\"candidates\":[{\"index\":0,\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"candidatesTokenCount\":9}}")),
                Err("connection reset after full STOP event".into())];
            let (wire, _) = render(raw, route).await;
            assert!(!events(&wire)
                .iter()
                .any(|event| event.get("error").is_some()));
            let result = collect(wire, route).await.unwrap();
            match route {
                Route::Claude => {
                    assert_eq!(
                        result["stop_reason"],
                        if tool { "tool_use" } else { "end_turn" }
                    );
                    assert_eq!(result["usage"]["output_tokens"], 9);
                }
                _ => {
                    assert_eq!(
                        result["choices"][0]["finish_reason"],
                        if tool { "tool_calls" } else { "stop" }
                    );
                    assert_eq!(result["usage"]["completion_tokens"], 9);
                }
            }
        }
    }
}

#[tokio::test]
async fn r32_done_ends_pending_transport_without_manufacturing_model_completion() {
    for route in ROUTES {
        for tool in [false, true] {
            let first = if tool {
                json!({"index":0,"content":{"role":"model","parts":[{"functionCall":{"name":"lookup","args":{"query":"safe"}}}]}})
            } else {
                candidate(0, "Partial text", None)
            };
            let source = stream::iter(vec![
                frame(json!([first])),
                Ok(Bytes::from("data: [DONE]\n\n")),
            ])
            .chain(stream::pending());
            let (wire, _) = tokio::time::timeout(
                std::time::Duration::from_millis(500),
                render_source(Box::pin(source), route),
            )
            .await
            .expect("DONE must end transport without waiting for the idle timeout");
            let result = collect(wire, route).await;
            if tool {
                let error: Value = serde_json::from_str(&result.unwrap_err()).unwrap();
                assert_eq!(error["error"]["code"], "upstream_interrupted");
            } else {
                let result = result.unwrap();
                match route {
                    Route::Claude => assert_eq!(result["stop_reason"], "max_tokens"),
                    _ => assert_eq!(result["choices"][0]["finish_reason"], "length"),
                }
            }
        }
    }
}

async fn r34_render_text_api(
    raw: Pin<Box<dyn Stream<Item = Result<Bytes, String>> + Send>>,
    legacy: bool,
) -> (Wire, UpstreamStreamErrorObserver) {
    let observer = if legacy {
        UpstreamStreamErrorObserver::default()
    } else {
        UpstreamStreamErrorObserver::default().with_selected_candidate()
    };
    let source = Box::pin(observe_response_errors(raw, observer.clone(), false));
    let mapped = if legacy {
        create_legacy_sse_stream(source, "gemini-flash".into(), "r34-legacy".into(), 1)
    } else {
        create_codex_sse_stream_with_selection(
            source,
            "gemini-flash".into(),
            "r34-responses".into(),
            1,
            1,
            "r34-response".into(),
            None,
            false,
            None,
            Default::default(),
            observer.candidate_selection().unwrap(),
        )
    };
    (mapped.collect().await, observer)
}

fn r34_legacy_text(events: &[Value]) -> String {
    events
        .iter()
        .filter_map(|event| event.pointer("/choices/0/text").and_then(Value::as_str))
        .collect()
}

fn r34_assert_single_done(wire: &Wire) {
    let count = wire
        .iter()
        .map(|chunk| {
            String::from_utf8_lossy(chunk.as_ref().unwrap())
                .matches("data: [DONE]")
                .count()
        })
        .sum::<usize>();
    assert_eq!(count, 1);
}

#[tokio::test]
async fn r34_legacy_model_failures_preserve_partial_text_without_success_terminal() {
    for reason in [
        "OTHER",
        "NO_IMAGE",
        "IMAGE_OTHER",
        "MALFORMED_FUNCTION_CALL",
    ] {
        let text = "Unable to complete the requested analysis";
        let raw = vec![frame(json!([candidate(0, text, Some(reason))]))];
        let (wire, observer) = r34_render_text_api(Box::pin(stream::iter(raw)), true).await;
        let rendered = events(&wire);
        assert_eq!(r34_legacy_text(&rendered), text);
        assert!(rendered.iter().any(|event| {
            event["error"]["code"] == "upstream_finish_error"
                && event["error"]["finish_reason"] == reason
        }));
        assert!(!rendered.iter().any(|event| {
            event
                .pointer("/choices/0/finish_reason")
                .is_some_and(|value| !value.is_null())
        }));
        assert_eq!(observer.failure().unwrap().status, 502);
        r34_assert_single_done(&wire);
    }
}

#[tokio::test]
async fn r34_legacy_model_terminals_survive_reset_and_keep_sparse_usage() {
    for (reason, finish) in [
        ("STOP", "stop"),
        ("MAX_TOKENS", "length"),
        ("SAFETY", "content_filter"),
        ("IMAGE_RECITATION", "content_filter"),
    ] {
        for text in ["Complete answer with {\"error\":503} as ordinary text", ""] {
            let raw = vec![
                frame(json!([candidate(0, text, Some(reason))])),
                Ok(Bytes::from("data: {\"usageMetadata\":{\"candidatesTokenCount\":9,\"cachedContentTokenCount\":0}}\n\n")),
                Err("connection reset after finish".into()),
            ];
            let (wire, observer) = r34_render_text_api(Box::pin(stream::iter(raw)), true).await;
            let rendered = events(&wire);
            assert_eq!(r34_legacy_text(&rendered), text);
            assert!(rendered
                .iter()
                .any(|event| event["choices"][0]["finish_reason"] == finish));
            assert!(!rendered.iter().any(|event| event.get("error").is_some()));
            assert_eq!(rendered.last().unwrap()["usage"]["completion_tokens"], 9);
            assert!(observer.failure().is_none());
            r34_assert_single_done(&wire);
        }
    }
}

#[tokio::test]
async fn r34_legacy_eof_done_and_reset_truncate_text_but_do_not_complete_tools() {
    for ending in ["eof", "done", "reset"] {
        for tool in [false, true] {
            let mut first = candidate(0, "Partial answer", None);
            if tool {
                first["content"]["parts"]
                    .as_array_mut()
                    .unwrap()
                    .push(json!({"functionCall":{"name":"run_bash","args":{"command":"ls"}}}));
            }
            let mut raw = vec![frame(json!([first]))];
            if ending == "done" {
                raw.push(Ok(Bytes::from("data: [DONE]\n\n")));
            } else if ending == "reset" {
                raw.push(Err("connection reset".into()));
            }
            let (wire, _) = r34_render_text_api(Box::pin(stream::iter(raw)), true).await;
            let rendered = events(&wire);
            assert_eq!(r34_legacy_text(&rendered), "Partial answer");
            if tool {
                assert!(rendered
                    .iter()
                    .any(|event| event["error"]["code"] == "upstream_interrupted"));
                assert!(!rendered
                    .iter()
                    .any(|event| event["choices"][0]["finish_reason"] == "stop"));
            } else {
                assert!(!rendered.iter().any(|event| event.get("error").is_some()));
                assert_eq!(
                    rendered.last().unwrap()["choices"][0]["finish_reason"],
                    "length"
                );
                assert_eq!(rendered.last().unwrap()["usage"]["completion_tokens"], 7);
            }
            r34_assert_single_done(&wire);
        }
    }
}

#[tokio::test]
async fn r34_legacy_synthetic_and_unspecified_finishes_do_not_claim_completion() {
    for synthetic in [false, true] {
        let mut first = candidate(0, "Partial answer", Some("FINISH_REASON_UNSPECIFIED"));
        if synthetic {
            first["finishReason"] = json!("STOP");
            first["gatewayInterrupted"] = json!(true);
        }
        let (wire, _) =
            r34_render_text_api(Box::pin(stream::iter(vec![frame(json!([first]))])), true).await;
        let rendered = events(&wire);
        assert_eq!(
            rendered.last().unwrap()["choices"][0]["finish_reason"],
            "length"
        );
        assert!(!rendered
            .iter()
            .any(|event| event["choices"][0]["finish_reason"] == "stop"));
        r34_assert_single_done(&wire);
    }
}

#[tokio::test]
async fn r34_structured_upstream_errors_are_not_hidden_by_completed_or_partial_text() {
    for legacy in [false, true] {
        for finish in [None, Some("STOP"), Some("SAFETY")] {
            let raw = vec![
                frame(json!([candidate(0, "Readable answer", finish)])),
                Ok(Bytes::from("data: {\"error\":{\"code\":503,\"status\":\"UNAVAILABLE\",\"message\":\"provider unavailable\"}}\n\n")),
            ];
            let (wire, observer) = r34_render_text_api(Box::pin(stream::iter(raw)), legacy).await;
            let rendered = events(&wire);
            assert_eq!(observer.failure().unwrap().status, 503);
            if legacy {
                assert_eq!(r34_legacy_text(&rendered), "Readable answer");
                assert_eq!(rendered.last().unwrap()["error"]["upstream_status"], 503);
                assert!(!rendered
                    .iter()
                    .any(|event| event["choices"][0]["finish_reason"] == "length"));
                r34_assert_single_done(&wire);
            } else {
                assert_eq!(rendered.last().unwrap()["type"], "response.failed");
                assert_eq!(
                    rendered.last().unwrap()["response"]["error"]["upstream_status"],
                    503
                );
            }
        }
    }
}

#[tokio::test]
async fn r34_unterminated_structured_error_before_reset_survives_all_text_adapters() {
    let raw = vec![
        frame(json!([candidate(0, "Readable answer", Some("STOP"))])),
        Ok(Bytes::from("data: {\"error\":{\"code\":503,\"status\":\"UNAVAILABLE\",\"message\":\"provider unavailable\"}}")),
        Err("connection reset after error payload".into()),
    ];
    let mut rendered = Vec::new();
    for route in ROUTES {
        rendered.push(render(raw.clone(), route).await.0);
    }
    for legacy in [false, true] {
        rendered.push(
            r34_render_text_api(Box::pin(stream::iter(raw.clone())), legacy)
                .await
                .0,
        );
    }
    for wire in rendered {
        assert!(events(&wire).iter().any(|event| {
            event["error"]["upstream_status"] == 503 || event["upstream_status"] == 503
        }));
    }
}

#[tokio::test]
async fn r34_legacy_media_matches_nonstream_chat_content_and_multiplicity() {
    const PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
    let image = json!({"inlineData":{"mimeType":"image/png","data":PNG}});
    let cases = [
        vec![json!([image.clone(), image.clone()])],
        vec![
            json!([{"text":"BEFORE_IMAGE"},image]),
            json!([{"inline_data":{"mime_type":"image/png","data":PNG},"thoughtSignature":"late"},{"text":"AFTER_IMAGE"}]),
        ],
        vec![json!([
            {"fileData":{"mimeType":"application/pdf","fileUri":"https://example.test/result.pdf"}},
            {"file_data":{"mime_type":"audio/wav","file_uri":"gs://bucket/result.wav"}}
        ])],
    ];
    for (index, parts) in cases.into_iter().enumerate() {
        let last = parts.len() - 1;
        let raw: Wire = parts
            .into_iter()
            .enumerate()
            .map(|(i, parts)| {
                let mut c = json!({"index":0,"content":{"role":"model","parts":parts}});
                if i == last {
                    c["finishReason"] = json!("STOP");
                }
                frame(json!([c]))
            })
            .collect();
        let (wire, _) = r34_render_text_api(Box::pin(stream::iter(raw.clone())), true).await;
        let rendered = events(&wire);
        let text = r34_legacy_text(&rendered);
        let (chat, _) = render(raw, Route::Chat).await;
        let chat = collect(chat, Route::Chat).await.unwrap();
        assert_eq!(json!(text), chat["choices"][0]["message"]["content"]);
        match index {
            0 => assert_eq!(text.matches("![image]").count(), 2),
            1 => {
                assert_eq!(text.matches("![image]").count(), 1);
                assert!(text.starts_with("BEFORE_IMAGE") && text.ends_with("AFTER_IMAGE"));
            }
            _ => {
                assert!(text.contains("[file](<https://example.test/result.pdf>)"));
                assert!(text.contains("File reference: ` gs://bucket/result.wav `"));
            }
        }
        assert_eq!(
            rendered.last().unwrap()["choices"][0]["finish_reason"],
            "stop"
        );
        r34_assert_single_done(&wire);
    }
}

#[tokio::test]
async fn r34_responses_refusal_stays_content_filter_after_reset_or_unterminated_tail() {
    for reason in ["SAFETY", "RECITATION", "IMAGE_RECITATION"] {
        for visible in [false, true] {
            for ending in ["eof", "reset", "unterminated-reset"] {
                let mut c = candidate(0, "Readable explanation", Some(reason));
                if !visible {
                    c["content"]["parts"] = json!([]);
                }
                let mut raw = if ending == "unterminated-reset" {
                    vec![Ok(Bytes::from(format!(
                        "data: {}",
                        json!({"candidates":[c]})
                    )))]
                } else {
                    vec![frame(json!([c]))]
                };
                if ending != "eof" {
                    raw.push(Err("connection reset after refusal".into()));
                }
                let (wire, observer) =
                    r34_render_text_api(Box::pin(stream::iter(raw)), false).await;
                let rendered = events(&wire);
                let response = &rendered.last().unwrap()["response"];
                assert_eq!(
                    response["status"], "incomplete",
                    "{reason}/{visible}/{ending}"
                );
                assert_eq!(response["incomplete_details"]["reason"], "content_filter");
                assert!(response["error"].is_null());
                assert!(!rendered.iter().any(|event| event["type"] == "error"));
                assert!(observer.failure().is_none());
                if visible {
                    assert_eq!(
                        response["output"][0]["content"][0]["text"],
                        "Readable explanation"
                    );
                }
            }
        }
    }
}

#[tokio::test]
async fn r34_completed_text_apis_drain_usage_then_end_pending_transport() {
    for legacy in [false, true] {
        for reason in ["STOP", "SAFETY"] {
            let raw = stream::iter(vec![
                frame(json!([candidate(0, "Readable answer", Some(reason))])),
                Ok(Bytes::from(
                    "data: {\"usageMetadata\":{\"candidatesTokenCount\":9}}\n\n",
                )),
            ])
            .chain(stream::pending());
            let (wire, _) = tokio::time::timeout(
                std::time::Duration::from_secs(1),
                r34_render_text_api(Box::pin(raw), legacy),
            )
            .await
            .expect("model terminals must not wait for the idle timeout");
            let rendered = events(&wire);
            if legacy {
                assert_eq!(rendered.last().unwrap()["usage"]["completion_tokens"], 9);
                r34_assert_single_done(&wire);
            } else {
                assert_eq!(
                    rendered.last().unwrap()["response"]["usage"]["output_tokens"],
                    9
                );
                assert_eq!(
                    rendered.last().unwrap()["response"]["status"],
                    if reason == "STOP" {
                        "completed"
                    } else {
                        "incomplete"
                    }
                );
            }
        }
    }
}

async fn r36_claude_with_tools(raw: Wire, collecting: bool) -> (Wire, Value) {
    let request: claude::ClaudeRequest = serde_json::from_value(json!({
        "model":"gemini-2.5-flash", "max_tokens":64, "stream":!collecting,
        "messages":[{"role":"user","content":"Explain the Read tool syntax."}],
        "tools":[{"name":"Read","input_schema":{"type":"object","properties":{"file_path":{"type":"string"}},"required":["file_path"]}}]
    })).unwrap();
    let tools = request
        .tools
        .unwrap()
        .into_iter()
        .filter_map(|t| t.name)
        .collect();
    let observer = UpstreamStreamErrorObserver::default().with_selected_candidate();
    let source = Box::pin(observe_response_errors(
        stream::iter(raw),
        observer.clone(),
        collecting,
    ));
    let wire: Wire = claude::create_claude_sse_stream_with_selection(
        source,
        "r36".into(),
        "r36@example.test".into(),
        None,
        false,
        200_000,
        None,
        1,
        None,
        tools,
        None,
        observer.candidate_selection().unwrap(),
    )
    .collect()
    .await;
    let result = collect(wire.clone(), Route::Claude).await.unwrap();
    assert!(observer.failure_or_incomplete(None).is_none());
    (wire, result)
}

#[tokio::test]
async fn r36_claude_tool_like_text_is_preserved_across_part_boundaries() {
    let cases = [
        vec!["call:default_api:Read"],
        vec![r#"call:default_api:Read{"file_path":"/tmp/example.txt"}"#],
        vec![
            "call:default_api:Read",
            r#"{"file_path":"/tmp/example.txt"}"#,
        ],
        vec![
            "call:default_api:",
            r#"Read{"file_path":"/tmp/example.txt"}"#,
        ],
        vec!["call:default_api:Read is a syntax example, not a tool request."],
        vec![
            "call:default_api:Read",
            " is a syntax example, not a tool request.",
        ],
        vec![
            r#"call:default_api:Read{"file_path":"/tmp/example.txt"}"#,
            " is only a quoted example.",
        ],
    ];
    for parts in cases {
        for collecting in [false, true] {
            let raw = parts
                .iter()
                .enumerate()
                .map(|(i, text)| {
                    frame(json!([candidate(
                        0,
                        text,
                        (i + 1 == parts.len()).then_some("STOP")
                    )]))
                })
                .collect();
            let (wire, result) = r36_claude_with_tools(raw, collecting).await;
            assert_eq!(
                result["content"],
                json!([{"type":"text","text":parts.concat()}])
            );
            assert_eq!(result["stop_reason"], "end_turn");
            assert!(!events(&wire)
                .iter()
                .any(|event| event["content_block"]["type"] == "tool_use"));
        }
    }
}

#[tokio::test]
async fn r36_claude_structured_function_call_after_text_stays_executable() {
    for collecting in [false, true] {
        let raw = vec![
            frame(json!([candidate(0, "call:default_api:Read", None)])),
            frame(
                json!([{"index":0,"content":{"role":"model","parts":[{"functionCall":{
                "id":"read_r36","name":"Read","args":{"file_path":"/tmp/example.txt"}
            }}]},"finishReason":"STOP"}]),
            ),
        ];
        let (_, result) = r36_claude_with_tools(raw, collecting).await;
        assert_eq!(
            result["content"][0],
            json!({"type":"text","text":"call:default_api:Read"})
        );
        let calls: Vec<_> = result["content"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|block| block["type"] == "tool_use")
            .collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["id"], "read_r36");
        assert_eq!(calls[0]["input"], json!({"file_path":"/tmp/example.txt"}));
        assert_eq!(result["stop_reason"], "tool_use");
    }
}

fn r36_legacy_choices(wire: &Wire) -> std::collections::BTreeMap<u64, (String, Vec<String>)> {
    let mut choices = std::collections::BTreeMap::new();
    for event in events(wire) {
        if event.get("object").is_some() {
            assert_eq!(event["object"], "text_completion");
        }
        for choice in event["choices"].as_array().into_iter().flatten() {
            let output = choices
                .entry(choice["index"].as_u64().unwrap())
                .or_insert_with(|| (String::new(), Vec::new()));
            assert!(choice.get("delta").is_none());
            output.0.push_str(choice["text"].as_str().unwrap());
            if let Some(reason) = choice["finish_reason"].as_str() {
                output.1.push(reason.to_owned());
            }
        }
    }
    choices
}

#[tokio::test]
async fn r36_legacy_candidates_match_chat_for_same_frame_and_sparse_ordering() {
    let cases = [
        vec![frame(json!([
            candidate(0, "ALPHA", Some("STOP")),
            candidate(1, "BETA", Some("STOP"))
        ]))],
        vec![
            frame(json!([
                candidate(0, "ALPHA", None),
                candidate(1, "BETA", None)
            ])),
            frame(json!([candidate(1, "_FROM_BETA", Some("STOP"))])),
            frame(json!([candidate(0, "_FROM_ALPHA", Some("STOP"))])),
        ],
        vec![
            frame(json!([
                candidate(1, "BETA", None),
                candidate(0, "ALPHA", Some("STOP"))
            ])),
            frame(json!([candidate(1, "_FROM_BETA", Some("MAX_TOKENS"))])),
        ],
        vec![
            frame(json!([
                candidate(0, "ALPHA", Some("STOP")),
                candidate(1, "BETA", None)
            ])),
            Err("connection reset after the first candidate completed".into()),
        ],
    ];
    for raw in cases {
        let (wire, _) = r34_render_text_api(Box::pin(stream::iter(raw.clone())), true).await;
        let (chat_wire, _) = render(raw, Route::Chat).await;
        let chat = collect(chat_wire, Route::Chat).await.unwrap();
        let choices = r36_legacy_choices(&wire);
        assert_eq!(choices.len(), 2);
        for choice in chat["choices"].as_array().unwrap() {
            let actual = &choices[&choice["index"].as_u64().unwrap()];
            assert_eq!(actual.0, choice["message"]["content"].as_str().unwrap());
            assert_eq!(
                actual.1,
                vec![choice["finish_reason"].as_str().unwrap().to_owned()]
            );
        }
        assert!(!events(&wire)
            .iter()
            .any(|event| event.get("error").is_some()));
        r34_assert_single_done(&wire);
    }
}

#[tokio::test]
async fn r36_legacy_waits_for_requested_candidates_before_tail_drain() {
    let raw = async_stream::stream! {
        yield frame(json!([candidate(0,"ALPHA",Some("STOP"))]));
        tokio::time::sleep(std::time::Duration::from_millis(350)).await;
        yield frame(json!([candidate(1,"BETA",Some("STOP"))]));
        yield Ok(Bytes::from("data: {\"usageMetadata\":{\"candidatesTokenCount\":9}}\n\n"));
        std::future::pending::<()>().await;
    };
    let wire: Wire = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        create_legacy_sse_stream_with_candidate_count(
            Box::pin(raw),
            "gemini-flash".into(),
            "r36-delayed".into(),
            1,
            None,
            2,
        )
        .collect(),
    )
    .await
    .expect("completed requested candidates must drain and close");
    let choices = r36_legacy_choices(&wire);
    assert_eq!(choices[&0], ("ALPHA".to_owned(), vec!["stop".to_owned()]));
    assert_eq!(choices[&1], ("BETA".to_owned(), vec!["stop".to_owned()]));
    assert_eq!(
        events(&wire).last().unwrap()["usage"]["completion_tokens"],
        9
    );
    r34_assert_single_done(&wire);
}

#[tokio::test]
async fn r36_legacy_missing_requested_candidate_keeps_answer_and_reports_interruption() {
    let raw = vec![frame(json!([candidate(0, "ALPHA", Some("STOP"))]))];
    let wire: Wire = create_legacy_sse_stream_with_candidate_count(
        Box::pin(stream::iter(raw)),
        "gemini-flash".into(),
        "r36-missing".into(),
        1,
        None,
        2,
    )
    .collect()
    .await;
    assert_eq!(r36_legacy_choices(&wire)[&0].0, "ALPHA");
    assert!(events(&wire)
        .iter()
        .any(|event| event["error"]["code"] == "upstream_interrupted"));
    r34_assert_single_done(&wire);
}

#[tokio::test]
async fn r36_completed_candidates_only_accept_usage_and_synthetic_finish_stays_incomplete() {
    for legacy in [false, true] {
        let raw = vec![
            frame(json!([candidate(0, "ALPHA", Some("STOP"))])),
            frame(json!([candidate(0, "DUPLICATE_LATE_TEXT", Some("STOP"))])),
            Ok(Bytes::from(
                "data: {\"usageMetadata\":{\"candidatesTokenCount\":9}}\n\n",
            )),
        ];
        let wire = if legacy {
            r34_render_text_api(Box::pin(stream::iter(raw)), true)
                .await
                .0
        } else {
            render(raw, Route::Chat).await.0
        };
        if legacy {
            assert_eq!(
                r36_legacy_choices(&wire)[&0],
                ("ALPHA".to_owned(), vec!["stop".to_owned()])
            );
        } else {
            let collected = collect(wire.clone(), Route::Chat).await.unwrap();
            assert_eq!(collected["choices"][0]["message"]["content"], "ALPHA");
        }
        assert_eq!(
            events(&wire).last().unwrap()["usage"]["completion_tokens"],
            9
        );
        r34_assert_single_done(&wire);
    }
    let mut partial = candidate(0, "Partial answer", Some("STOP"));
    partial["gatewayInterrupted"] = json!(true);
    let (wire, _) = render(vec![frame(json!([partial]))], Route::Chat).await;
    let collected = collect(wire, Route::Chat).await.unwrap();
    assert_eq!(
        collected["choices"][0]["message"]["content"],
        "Partial answer"
    );
    assert_eq!(collected["choices"][0]["finish_reason"], "length");
}

#[tokio::test]
async fn r36_legacy_single_candidate_finish_keeps_usage_without_a_trailer() {
    for (reason, expected) in [
        ("STOP", "stop"),
        ("MAX_TOKENS", "length"),
        ("SAFETY", "content_filter"),
    ] {
        let raw = vec![frame(json!([candidate(0, "ANSWER", Some(reason))]))];
        let (wire, _) = r34_render_text_api(Box::pin(stream::iter(raw)), true).await;
        let rendered = events(&wire);
        let finish = rendered
            .iter()
            .find(|event| event["choices"][0]["finish_reason"] == expected)
            .unwrap();
        assert_eq!(finish["usage"]["prompt_tokens"], 10);
        assert_eq!(finish["usage"]["completion_tokens"], 7);
        assert_eq!(finish["usage"]["total_tokens"], 17);
        assert_eq!(
            r36_legacy_choices(&wire)[&0],
            ("ANSWER".to_owned(), vec![expected.to_owned()])
        );
        r34_assert_single_done(&wire);
    }
}

#[tokio::test]
async fn r36_finished_candidate_does_not_hide_later_model_failure() {
    for index in [0, 1] {
        for reason in ["OTHER", "NO_IMAGE"] {
            for legacy in [false, true] {
                let raw = vec![
                    frame(json!([
                        candidate(0, "ALPHA", Some("STOP")),
                        candidate(1, "BETA", Some("STOP"))
                    ])),
                    frame(json!([candidate(index, "IGNORED_LATE_TEXT", Some(reason))])),
                ];
                let wire = if legacy {
                    r34_render_text_api(Box::pin(stream::iter(raw)), true)
                        .await
                        .0
                } else {
                    render(raw, Route::Chat).await.0
                };
                let rendered = events(&wire);
                assert!(rendered
                    .iter()
                    .any(|event| event["error"]["code"] == "upstream_finish_error"
                        && event["error"]["finish_reason"] == reason));
                assert!(!rendered
                    .iter()
                    .any(|event| event.to_string().contains("IGNORED_LATE_TEXT")));
                if legacy {
                    let choices = r36_legacy_choices(&wire);
                    assert_eq!(choices[&0].0, "ALPHA");
                    assert_eq!(choices[&1].0, "BETA");
                } else {
                    assert!(collect(wire.clone(), Route::Chat).await.is_err());
                }
                r34_assert_single_done(&wire);
            }
        }
    }
}
