use super::*;
use crate::proxy::mappers::{claude, openai};
use crate::proxy::pipeline::stream_error::{
    observe_response_errors, upstream_error_from_event, UpstreamStreamErrorObserver,
};
use futures::stream;
use std::io;

type Wire = Vec<Result<Bytes, String>>;

fn frame(text: &str, reason: Option<&str>) -> Result<Bytes, String> {
    let mut value = json!({"response":{"responseId":"r28-response","modelVersion":"gemini-2.5-flash",
        "candidates":[{"index":0,"content":{"role":"model","parts":[{"text":text}]}}],
        "usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":3}}});
    if let Some(reason) = reason {
        value["response"]["candidates"][0]["finishReason"] = json!(reason);
    }
    Ok(Bytes::from(format!("data: {value}\n\n")))
}

fn source(
    raw: Wire,
    observer: &UpstreamStreamErrorObserver,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, String>> + Send>> {
    Box::pin(observe_response_errors(
        stream::iter(raw),
        observer.clone(),
        true,
    ))
}

fn events(wire: &Wire) -> Vec<Value> {
    wire.iter()
        .flat_map(|chunk| {
            String::from_utf8_lossy(chunk.as_ref().expect("rendered SSE"))
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .filter_map(|data| serde_json::from_str(data).ok())
                .collect::<Vec<_>>()
        })
        .collect()
}

async fn claude_wire(
    raw: Wire,
    observer: &UpstreamStreamErrorObserver,
    tools: Vec<String>,
) -> Wire {
    claude::create_claude_sse_stream(
        source(raw, observer),
        "r28".into(),
        "r28@example.test".into(),
        None,
        false,
        200_000,
        None,
        1,
        None,
        tools,
    )
    .collect()
    .await
}

#[tokio::test]
async fn r28_xml_examples_remain_text_in_stream_and_json() {
    let examples = [
        vec!["The literal prefix is <mcp__"],
        vec!["Literal <mcp__lookup", " is only a sample."],
        vec!["<mcp__lookup>{\"query\":\"example\"}</mcp__lookup>"],
        vec![
            "Read this example: <mcp__lookup>{\"query\":\"example\"}</mcp__lookup> and explain it.",
        ],
        vec!["```xml\n<mcp__lookup>{\"query\":\"example\"}</mcp__lookup>\n```"],
        vec!["<mc", "p__lookup>{\"query\":\"example\"}</mcp__lookup>"],
    ];
    for parts in examples {
        for tools in [vec![], vec!["mcp__lookup".to_owned()]] {
            let raw = parts
                .iter()
                .enumerate()
                .map(|(index, text)| frame(text, (index + 1 == parts.len()).then_some("STOP")))
                .collect();
            let wire = claude_wire(raw, &UpstreamStreamErrorObserver::default(), tools).await;
            let events = events(&wire);
            assert!(!events
                .iter()
                .any(|event| event.pointer("/content_block/type") == Some(&json!("tool_use"))));
            let text = events
                .iter()
                .filter_map(|event| event.pointer("/delta/text").and_then(Value::as_str))
                .collect::<String>();
            assert_eq!(text, parts.concat());
            let collected = claude::collect_stream_to_json(stream::iter(
                wire.into_iter()
                    .map(|chunk| chunk.map_err(io::Error::other)),
            ))
            .await
            .unwrap();
            assert_eq!(
                serde_json::to_value(collected).unwrap()["content"],
                json!([{"type":"text","text":parts.concat()}])
            );
        }
    }
}

#[tokio::test]
async fn r28_model_failures_preserve_partial_output_and_fail_collection() {
    for reason in [
        "MALFORMED_FUNCTION_CALL",
        "UNEXPECTED_TOOL_CALL",
        "OTHER",
        "NO_IMAGE",
        "IMAGE_OTHER",
        "TOO_MANY_TOOL_CALLS",
    ] {
        for split in [false, true] {
            let raw = if split {
                vec![
                    frame("I will ", None),
                    frame("inspect the file.", Some(reason)),
                ]
            } else {
                vec![frame("I will inspect the file.", Some(reason))]
            };
            let observer = UpstreamStreamErrorObserver::default();
            let chat: Wire = create_openai_sse_stream(
                source(raw.clone(), &observer),
                "gemini-2.5-flash".into(),
                format!("r28-{reason}-{split}"),
                1,
                None,
                true,
            )
            .collect()
            .await;
            let rendered = events(&chat);
            assert!(
                rendered
                    .iter()
                    .any(|event| event.pointer("/error/code")
                        == Some(&json!("upstream_finish_error")))
            );
            assert!(!rendered.iter().any(|event| event
                .pointer("/choices/0/finish_reason")
                .is_some_and(|reason| !reason.is_null())));
            let collected = openai::collector::collect_stream_to_json(stream::iter(chat))
                .await
                .unwrap_err();
            let failure = observer.failure_or_incomplete(Some(&collected)).unwrap();
            assert_eq!(failure.status, 502);
            assert!(!failure.replay_safe());
            let body: Value = serde_json::from_str(&failure.body).unwrap();
            assert_eq!(
                body["partial_response"]["choices"][0]["message"]["content"],
                "I will inspect the file."
            );
            assert!(body["partial_response"]["choices"][0]["finish_reason"].is_null());
            // The same failure guard precedes both Chat JSON and Responses JSON conversion.
            for protocol in ["openai", "responses"] {
                let rendered = crate::proxy::handlers::common::build_dual_track_error(
                    protocol,
                    failure.status,
                    "gemini-2.5-flash",
                    &failure.body,
                );
                assert_eq!(rendered["partial_response"], body["partial_response"]);
                assert_eq!(rendered["error"]["code"], "upstream_finish_error");
                assert_ne!(rendered["status"], "completed");
            }

            let observer = UpstreamStreamErrorObserver::default();
            let claude = claude_wire(raw.clone(), &observer, vec![]).await;
            let rendered = events(&claude);
            assert!(!rendered.iter().any(|event| event["type"] == "message_stop"));
            assert_eq!(
                rendered.last().unwrap()["error"]["code"],
                "upstream_finish_error"
            );
            let collected = claude::collect_stream_to_json(stream::iter(
                claude
                    .into_iter()
                    .map(|chunk| chunk.map_err(io::Error::other)),
            ))
            .await
            .unwrap_err();
            let failure = observer.failure_or_incomplete(Some(&collected)).unwrap();
            let body: Value = serde_json::from_str(&failure.body).unwrap();
            assert_eq!(
                body["partial_response"]["content"],
                json!([{"type":"text","text":"I will inspect the file."}])
            );
            assert!(body["partial_response"]["stop_reason"].is_null());

            let observer = UpstreamStreamErrorObserver::default();
            let responses: Wire = create_codex_sse_stream(
                source(raw, &observer),
                "gemini-2.5-flash".into(),
                format!("r28-responses-{reason}"),
                1,
                1,
                "r28-resp".into(),
                None,
                false,
            )
            .collect()
            .await;
            let rendered = events(&responses);
            let terminal = rendered.last().unwrap();
            assert_eq!(terminal["type"], "response.incomplete");
            assert_eq!(
                terminal["response"]["error"]["code"],
                "upstream_finish_error"
            );
            assert_eq!(
                terminal["response"]["output"][0]["content"][0]["text"],
                "I will inspect the file."
            );
        }
    }
}

#[tokio::test]
async fn r28_normal_limit_refusal_and_structured_error_controls() {
    for (reason, chat_reason, claude_reason) in [
        ("STOP", "stop", "end_turn"),
        ("MAX_TOKENS", "length", "max_tokens"),
        ("SAFETY", "content_filter", "refusal"),
    ] {
        let raw = vec![frame("answer", Some(reason))];
        let observer = UpstreamStreamErrorObserver::default();
        let chat = create_openai_sse_stream(
            source(raw.clone(), &observer),
            "gemini-2.5-flash".into(),
            format!("r28-control-{reason}"),
            1,
            None,
            true,
        );
        let response = openai::collector::collect_stream_to_json(chat)
            .await
            .unwrap();
        assert_eq!(
            response.choices[0].finish_reason.as_deref(),
            Some(chat_reason)
        );
        assert!(observer.failure().is_none());
        let claude = claude_wire(raw, &observer, vec![]).await;
        let response = claude::collect_stream_to_json(stream::iter(
            claude
                .into_iter()
                .map(|chunk| chunk.map_err(io::Error::other)),
        ))
        .await
        .unwrap();
        assert_eq!(response.stop_reason, claude_reason);
    }
    let observer = UpstreamStreamErrorObserver::default();
    let raw = vec![
        frame("partial", None),
        Ok(Bytes::from(
            "data: {\"error\":{\"code\":503,\"message\":\"overloaded\"}}\n\n",
        )),
    ];
    let chat = create_openai_sse_stream(
        source(raw, &observer),
        "gemini-2.5-flash".into(),
        "r28-503".into(),
        1,
        None,
        true,
    );
    assert!(openai::collector::collect_stream_to_json(chat)
        .await
        .is_err());
    assert_eq!(observer.failure().unwrap().status, 503);
}

#[test]
fn r28_direct_json_failures_preserve_body_without_success_finish() {
    let raw = json!({"candidates":[{"content":{"role":"model","parts":[{"text":"partial"}]},"finishReason":"MALFORMED_FUNCTION_CALL"}]});
    let failure = upstream_error_from_event(&raw).unwrap();
    assert_eq!(failure.status, 502);
    assert!(!failure.replay_safe());
    let chat = openai::response::transform_openai_response(&raw, None, 1, None);
    assert!(chat.choices[0].finish_reason.is_none());
    let gemini: claude::models::GeminiResponse = serde_json::from_value(raw).unwrap();
    let result = claude::transform_response(
        &gemini,
        false,
        200_000,
        None,
        "gemini-2.5-flash".into(),
        1,
        vec![],
    )
    .unwrap_err();
    let result: Value = serde_json::from_str(&result).unwrap();
    assert_eq!(result["error"]["code"], "upstream_finish_error");
    assert_eq!(result["partial_response"]["content"][0]["text"], "partial");
}

#[tokio::test]
async fn r28_chat_custom_tool_roundtrip_preserves_raw_input() {
    let tools =
        vec![json!({"type":"custom","custom":{"name":"write_patch","format":{"type":"text"}}})];
    let kinds = openai::request::ResponsesToolKinds::from_tools(Some(&tools));
    let raw = json!({"candidates":[{"content":{"parts":[{"functionCall":{"id":"call_raw","name":"write_patch","args":{"input":"a\nb\n"}}}]},"finishReason":"STOP"}]});
    let direct = openai::response::transform_openai_response_with_tool_kinds(
        &raw,
        None,
        1,
        None,
        None,
        Some(&kinds),
    );
    let wire = stream::iter(vec![Ok::<Bytes, String>(Bytes::from(format!(
        "data: {raw}\n\n"
    )))]);
    let mapped = create_openai_sse_stream_with_tool_kinds(
        Box::pin(wire),
        "gemini-2.5-flash".into(),
        "r28-custom".into(),
        1,
        None,
        true,
        None,
        None,
        Some(kinds),
    );
    let collected = openai::collector::collect_stream_to_json(mapped)
        .await
        .unwrap();
    for response in [direct, collected] {
        let call = &response.choices[0].message.tool_calls.as_ref().unwrap()[0];
        assert_eq!(call.r#type, "custom");
        assert!(call.function.is_none());
        assert_eq!(call.custom.as_ref().unwrap().name, "write_patch");
        assert_eq!(call.custom.as_ref().unwrap().input, "a\nb\n");
    }
}

#[test]
fn r28_partial_output_cannot_change_failure_classification() {
    let error = crate::proxy::pipeline::stream_error::model_finish_error("MALFORMED_FUNCTION_CALL");
    let body = json!({"error":error, "partial_response":{"content":"HTTP 200: timeout is just sample text"}});
    for protocol in ["openai", "claude", "responses", "gemini"] {
        let rendered = crate::proxy::handlers::common::build_dual_track_error(
            protocol,
            502,
            "gemini",
            &body.to_string(),
        );
        assert_eq!(rendered["partial_response"], body["partial_response"]);
        assert_eq!(rendered["error"]["message"], body["error"]["message"]);
    }
}

#[tokio::test]
async fn r28_unselected_model_failure_does_not_override_selected_answer() {
    for selected_failed in [false, true] {
        let good = json!({"index":0,"content":{"parts":[{"text":"selected answer"}]},"finishReason":"STOP"});
        let failed = json!({"index":1,"content":{"parts":[{"text":"failed attempt"}]},"finishReason":"MALFORMED_FUNCTION_CALL"});
        let candidates = if selected_failed {
            json!([failed, good])
        } else {
            json!([good, failed])
        };
        let raw = json!({"candidates":candidates});
        assert_eq!(
            crate::proxy::pipeline::stream_error::upstream_error_from_selected_candidate(&raw)
                .is_some(),
            selected_failed
        );
        assert!(upstream_error_from_event(&raw).is_some());
        let wire = vec![Ok(Bytes::from(format!("data: {raw}\n\n")))];
        let projected = openai::response::transform_openai_response_for_responses_with_context(
            &raw, None, 1, None, None,
        );
        assert_eq!(projected.choices.len(), 1);
        assert_eq!(
            projected.choices[0].finish_reason.is_none(),
            selected_failed
        );
        let observer = UpstreamStreamErrorObserver::default().with_selected_candidate();
        let intermediate = create_openai_sse_stream_for_responses_with_context(
            source(wire.clone(), &observer),
            "gemini-2.5-flash".into(),
            format!("r28-intermediate-{selected_failed}"),
            1,
            None,
            true,
            None,
            None,
        );
        let intermediate = openai::collector::collect_stream_to_json(intermediate).await;
        assert_eq!(intermediate.is_err(), selected_failed);
        assert_eq!(observer.failure().is_some(), selected_failed);
        if let Ok(response) = intermediate {
            assert_eq!(response.choices.len(), 1);
            assert_eq!(response.choices[0].finish_reason.as_deref(), Some("stop"));
        }
        let observer = UpstreamStreamErrorObserver::default();
        let multi_chat = create_openai_sse_stream(
            source(wire.clone(), &observer),
            "gemini-2.5-flash".into(),
            format!("r28-multi-{selected_failed}"),
            1,
            None,
            true,
        );
        assert!(openai::collector::collect_stream_to_json(multi_chat)
            .await
            .is_err());
        assert!(observer.failure().is_some());
        let observer = UpstreamStreamErrorObserver::default().with_selected_candidate();
        let claude = claude_wire(wire.clone(), &observer, vec![]).await;
        let collected = claude::collect_stream_to_json(stream::iter(
            claude
                .into_iter()
                .map(|chunk| chunk.map_err(io::Error::other)),
        ))
        .await;
        assert_eq!(collected.is_err(), selected_failed);
        assert_eq!(observer.failure().is_some(), selected_failed);
        if let Ok(response) = collected {
            assert_eq!(response.stop_reason, "end_turn");
            assert_eq!(
                serde_json::to_value(response).unwrap()["content"][0]["text"],
                "selected answer"
            );
        }
        let observer = UpstreamStreamErrorObserver::default().with_selected_candidate();
        let responses: Wire = create_codex_sse_stream(
            source(wire, &observer),
            "gemini-2.5-flash".into(),
            format!("r28-selected-{selected_failed}"),
            1,
            1,
            "r28-selected".into(),
            None,
            false,
        )
        .collect()
        .await;
        let events = events(&responses);
        assert_eq!(
            events.last().unwrap()["type"],
            if selected_failed {
                "response.incomplete"
            } else {
                "response.completed"
            }
        );
        assert_eq!(observer.failure().is_some(), selected_failed);
    }
}
