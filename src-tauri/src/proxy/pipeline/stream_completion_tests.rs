use crate::proxy::{
    mappers::{
        claude::create_claude_sse_stream_with_context,
        gemini::{collector, streaming::GeminiStreamHandler},
        openai::streaming::*,
    },
    pipeline::*,
    thinking_store::*,
};
use base64::Engine;
use bytes::Bytes;
use futures::StreamExt;
use serde_json::{json, Value};
fn sig(c: u8) -> String {
    let mut raw = vec![18, 48];
    raw.extend([c; 48]);
    base64::engine::general_purpose::STANDARD.encode(raw)
}
fn user() -> Value {
    json!({"role":"user","parts":[{"text":"Original"}]})
}
fn frame(v: Value) -> Result<Bytes, String> {
    Ok(Bytes::from(format!("data: {v}\n\n")))
}
fn raw(
    items: Vec<Value>,
) -> std::pin::Pin<Box<dyn futures::Stream<Item = Result<Bytes, String>> + Send>> {
    Box::pin(futures::stream::iter(items.into_iter().map(frame)))
}
fn native(s: &str, model: &str, buffer: bool, n: usize) -> GeminiStreamHandler {
    GeminiStreamHandler::new_with_context(
        None,
        s.into(),
        model.into(),
        1,
        compute_causal_anchor(Some(&user())),
        false,
        None,
        "a".into(),
        model.into(),
        Some(compute_response_text_anchor(&[user()])),
    )
    .with_candidate_request(n, buffer)
}
#[tokio::test]
async fn retry_history() {
    crate::modules::proxy_db::init_db().unwrap();
    let model = "gemini-2.5-flash";
    let u = user();
    for proto in ["gemini", "chat", "legacy", "claude", "responses"] {
        for failure in [
            "none",
            "503",
            "transport",
            "eof",
            "MALFORMED_FUNCTION_CALL",
            "SAFETY",
        ] {
            let prior = failure != "none";
            let sid = format!("r19_{proto}_{prior}_{}", uuid::Uuid::new_v4());
            for fail in [true, false] {
                if fail && !prior {
                    continue;
                }
                let mut v = json!({"candidates":[{"index":0,"content":{"parts":[{"thought":true,"text":if fail{"bad reasoning"}else{"good reasoning"}},{"text":"Done.","thoughtSignature":if fail{sig(b'A')}else{sig(b'B')}}]}}]});
                if !fail {
                    v["candidates"][0]["finishReason"] = json!("STOP");
                } else if failure == "MALFORMED_FUNCTION_CALL" || failure == "SAFETY" {
                    v["candidates"][0]["finishReason"] = json!(failure);
                }
                let mut items = vec![v];
                if fail && failure == "503" {
                    items.push(json!({"error":{"code":503,"message":"upstream failed"}}));
                }
                let s = raw(items);
                let s: std::pin::Pin<
                    Box<dyn futures::Stream<Item = Result<Bytes, String>> + Send>,
                > = if fail && failure == "transport" {
                    Box::pin(s.chain(futures::stream::iter([Err("connection reset".to_owned())])))
                } else {
                    s
                };
                let anchor = Some(compute_response_text_anchor(&[u.clone()]));
                let stream = match proto {
                    "chat" => create_openai_sse_stream_with_context(
                        s,
                        model.into(),
                        sid.clone(),
                        1,
                        None,
                        false,
                        Some(compute_causal_anchor(Some(&u))),
                        anchor,
                    ),
                    "legacy" => create_legacy_sse_stream_with_context(
                        s,
                        model.into(),
                        sid.clone(),
                        1,
                        anchor,
                    ),
                    "claude" => create_claude_sse_stream_with_context(
                        s,
                        "trace".into(),
                        "test".into(),
                        Some(sid.clone()),
                        false,
                        1000,
                        None,
                        1,
                        None,
                        vec![],
                        anchor,
                    ),
                    "responses" => create_codex_sse_stream_with_context(
                        s,
                        model.into(),
                        sid.clone(),
                        1,
                        1,
                        format!("{sid}-{fail}"),
                        None,
                        false,
                        anchor,
                        Default::default(),
                    ),
                    _ => run_stream_lifecycle(
                        s,
                        native(&sid, model, false, 1),
                        StreamLifecycleConfig::new("gemini", "r19", "test".into()),
                    ),
                };
                let mut stream = stream;
                let mut wire = String::new();
                while let Some(chunk) = stream.next().await {
                    let chunk = match chunk {
                        Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
                        Err(error) => error,
                    };
                    if !fail
                        && [
                            "\"finishReason\":\"STOP\"",
                            "\"finish_reason\":\"stop\"",
                            "\"type\":\"message_stop\"",
                            "\"type\":\"response.completed\"",
                        ]
                        .iter()
                        .any(|marker| chunk.contains(marker))
                    {
                        assert_eq!(
                            crate::modules::proxy_db::load_thinking_records(&sid)
                                .unwrap()
                                .len(),
                            1,
                            "{proto}: history must be committed before terminal bytes"
                        );
                    }
                    wire.push_str(&chunk);
                }
                if fail && failure == "503" {
                    assert!(wire.contains("503"), "{proto}: {wire}");
                }
                if fail {
                    assert!(
                        crate::modules::proxy_db::load_thinking_records(&sid)
                            .unwrap()
                            .is_empty(),
                        "{proto} {failure}: failed output stored"
                    );
                    assert!(
                        crate::proxy::SignatureCache::global()
                            .get_session_signature(&sid)
                            .is_none(),
                        "{proto} {failure}: failed signature published"
                    );
                }
            }
            let rows = crate::modules::proxy_db::load_thinking_records(&sid).unwrap();
            let mut hist = vec![
                u.clone(),
                json!({"role":"model","parts":[{"text":"Done."}]}),
                json!({"role":"user","parts":[{"text":"next"}]}),
            ];
            finalize_request_text_context(&sid, &mut hist, model, true, false);
            assert_eq!(
                rows.len(),
                1,
                "{proto}: only successful output may enter SQLite"
            );
            assert!(
                !hist[1].to_string().contains(&sig(b'A')),
                "{proto}: failed signature restored"
            );
            assert!(
                hist[1].to_string().contains(&sig(b'B')),
                "{proto}: successful signature missing: {}",
                hist[1]
            );
            eprintln!(
                "RETRY proto={proto} failed-first={prior} db={} A={}, B={} final={}",
                rows.len(),
                hist[1].to_string().contains(&sig(b'A')),
                hist[1].to_string().contains(&sig(b'B')),
                hist[1]
            );
        }
    }
}
#[tokio::test]
async fn media_only() {
    for key in ["fileData", "file_data", "inlineData", "text"] {
        for buffered in [true, false] {
            let sid = format!("media_{key}_{buffered}");
            let part = match key {
                "fileData" => {
                    json!({"fileData":{"fileUri":"https://files.example/output.png","mimeType":"image/png"}})
                }
                "file_data" => {
                    json!({"file_data":{"file_uri":"https://files.example/output.png","mime_type":"image/png"}})
                }
                "inlineData" => json!({"inlineData":{"data":"cG5n","mimeType":"image/png"}}),
                _ => json!({"text":"ok"}),
            };
            let items = vec![
                json!({"response":{"candidates":[{"content":{"parts":[part]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":4}}}),
            ];
            let s = run_stream_lifecycle(
                raw(items),
                native(&sid, "gemini-2.5-flash", buffered, 1),
                StreamLifecycleConfig::new("gemini", "r19", "test".into()),
            );
            let chunks = s.collect::<Vec<_>>().await;
            let errs = chunks.iter().filter(|x| x.is_err()).count();
            let result = if buffered {
                collector::collect_stream_to_json(futures::stream::iter(chunks), &sid)
                    .await
                    .map(|v| v.to_string())
                    .unwrap_or_else(|e| format!("ERROR {e}"))
            } else {
                chunks
                    .into_iter()
                    .map(|c| match c {
                        Ok(b) => String::from_utf8_lossy(&b).into_owned(),
                        Err(e) => format!("ERROR {e}"),
                    })
                    .collect()
            };
            assert_eq!(errs, 0, "{key} buffered={buffered}: {result}");
            assert!(
                !result.contains("ERROR"),
                "{key} buffered={buffered}: {result}"
            );
            if key.starts_with("file") {
                assert!(result.contains("https://files.example/output.png"));
            }
            eprintln!("MEDIA key={key} buffered={buffered} errs={errs} result={result}");
        }
    }
}
#[tokio::test]
async fn id_shapes() {
    let calls = [
        json!({"functionCall":{"name":"lookup","args":{"i":0}}}),
        json!({"functionCall":{"name":"lookup","args":{"i":1}}}),
    ];
    for shape in [
        "same",
        "split",
        "wrapped",
        "explicit_same",
        "explicit_split",
    ] {
        let mut items = if shape == "same" || shape == "explicit_same" {
            vec![json!({"candidates":[{"content":{"parts":calls},"finishReason":"STOP"}]})]
        } else {
            vec![
                json!({"candidates":[{"content":{"parts":[calls[0]]}}]}),
                json!({"candidates":[{"content":{"parts":[calls[1]]},"finishReason":"STOP"}]}),
            ]
        };
        if shape.starts_with("explicit") {
            let mut i = 0;
            for x in &mut items {
                for p in x["candidates"][0]["content"]["parts"]
                    .as_array_mut()
                    .unwrap()
                {
                    p["functionCall"]["id"] = json!(format!("real{i}"));
                    i += 1;
                }
            }
        }
        if shape == "wrapped" {
            items = items.into_iter().map(|v| json!({"response":v})).collect();
        }
        let sid = format!("ids_{shape}");
        let s = run_stream_lifecycle(
            raw(items),
            native(&sid, "claude-sonnet-4-6", true, 1),
            StreamLifecycleConfig::new("gemini", "r19", "test".into()),
        );
        let result = collector::collect_stream_to_json(s, &sid).await.unwrap();
        let ids = result["candidates"][0]["content"]["parts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x["functionCall"].get("id").cloned())
            .collect::<Vec<_>>();
        assert_eq!(ids.len(), 2, "{shape}");
        assert!(ids.iter().all(Option::is_some), "{shape}: {ids:?}");
        assert_ne!(ids[0], ids[1], "{shape}");
        eprintln!("IDS shape={shape} ids={ids:?}");
    }
}
#[tokio::test]
async fn cumulative_images() {
    let img = json!({"fileData":{"mimeType":"image/png","fileUri":"https://example.test/a.png"}});
    let png = json!({"inlineData":{"mimeType":"image/png","data":"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"}});
    for mode in ["single", "cumulative", "separate"] {
        let mut items = vec![
            json!({"candidates":[{"index":0,"content":{"parts":[png.clone()]}},{"index":1,"content":{"parts":[img.clone()]}}]}),
        ];
        if mode == "single" {
            for c in items[0]["candidates"].as_array_mut().unwrap() {
                c["finishReason"] = json!("STOP")
            }
        } else {
            let mut second = items[0].clone();
            for c in second["candidates"].as_array_mut().unwrap() {
                c["finishReason"] = json!("STOP");
                if mode == "separate" {
                    c.as_object_mut().unwrap().remove("content");
                }
            }
            items.push(second);
        }
        let sid = format!("cumul_{mode}");
        let s = run_stream_lifecycle(
            raw(items),
            native(&sid, "gemini-3.8-flash", true, 2),
            StreamLifecycleConfig::new("gemini", "r19", "test".into()),
        );
        let resp = collector::collect_stream_to_json(s, &sid).await.unwrap();
        let counts = resp["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v["content"]["parts"].as_array().unwrap().len())
            .collect::<Vec<_>>();
        assert_eq!(counts, vec![1, 1], "{mode}: {resp}");
        if let Ok(dir) = std::env::var("R19_BILLING_INPUT") {
            std::fs::write(format!("{dir}/{mode}.json"), resp.to_string()).unwrap();
        }
        eprintln!("CUMUL mode={mode} counts={counts:?}");
    }
}
#[tokio::test]
async fn preserving_thought_parts() {
    for mode in ["unsigned", "signed", "metadata", "snake"] {
        for buffered in [true, false] {
            let sid = format!("r19_segments_{buffered}_{mode}_{}", std::process::id());
            let mut a = json!({"thought":true,"text":"first "});
            let mut b = json!({"thought":true,"text":"second"});
            match mode {
                "signed" => {
                    a["thoughtSignature"] = json!(sig(b'A'));
                    b["thoughtSignature"] = json!(sig(b'B'));
                }
                "snake" => {
                    b["thought_signature"] = json!(sig(b'B'));
                }
                "metadata" => {
                    a["thoughtSignature"] = json!(sig(b'A'));
                    b["videoMetadata"] = json!({"startOffset":"1s"});
                }
                _ => (),
            }
            let expected_b = b.clone();
            let mut items = vec![];
            for p in [a, b] {
                items
                    .push(json!({"response":{"candidates":[{"index":0,"content":{"parts":[p]}}]}}));
            }
            items.push(json!({"response":{"candidates":[{"index":0,"content":{"parts":[{"text":"Done."}]},"finishReason":"STOP"}]}}));
            let s = run_stream_lifecycle(
                raw(items),
                native(&sid, "gemini-2.5-flash", buffered, 1),
                StreamLifecycleConfig::new("gemini", "r19", "test".into()),
            );
            let chunks = s.collect::<Vec<_>>().await;
            let resp = if buffered {
                collector::collect_stream_to_json(futures::stream::iter(chunks), &sid)
                    .await
                    .unwrap()
                    .to_string()
            } else {
                chunks
                    .into_iter()
                    .map(|r| String::from_utf8_lossy(&r.unwrap()).into_owned())
                    .collect()
            };
            if mode != "unsigned" && !(mode == "snake" && buffered) {
                assert_eq!(
                    resp.matches("\"thought\":true").count(),
                    2,
                    "{mode}: {resp}"
                );
            }
            if mode == "signed" || mode == "snake" {
                assert!(resp.contains(&sig(b'B')), "{mode}: {resp}");
            }
            if mode == "signed" || mode == "metadata" {
                assert!(resp.contains(&sig(b'A')), "{mode}: {resp}");
            }
            if mode == "metadata" {
                assert!(resp.contains("startOffset"));
            }
            eprintln!("THOUGHT mode={mode} buffered={buffered} count={} A={} B={} ref2={} second_metadata={} wire={resp}",resp.matches("\"thought\":true").count(),resp.contains(&sig(b'A')),resp.contains(&sig(b'B')),expected_b,resp.contains("startOffset"));
        }
    }
}

#[tokio::test]
async fn cumulative_media_keeps_same_frame_multiplicity() {
    let image = json!({"inlineData":{"mimeType":"image/png","data":"cG5n"}});
    let mut frame = json!({"candidates":[{"index":0,"content":{"parts":[image,image]}}]});
    let first = frame.clone();
    frame["candidates"][0]["finishReason"] = json!("STOP");
    let result = collector::collect_stream_to_json(raw(vec![first, frame]), "r19-duplicates")
        .await
        .unwrap();
    assert_eq!(
        result["candidates"][0]["content"]["parts"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
}

#[test]
fn generated_call_ids_survive_cumulative_wrapped_frames() {
    let mut ids = crate::proxy::mappers::gemini::wrapper::ResponseToolIds::default();
    let call = json!({"functionCall":{"name":"lookup","args":{"i":1}}});
    let mut first = json!({"response":{"candidates":[{"content":{"parts":[call.clone()]}}]}});
    ids.inject(&mut first, "claude-sonnet");
    let mut second = json!({"response":{"candidates":[{"content":{"parts":[call.clone(),call]}}]}});
    ids.inject(&mut second, "claude-sonnet");
    assert_eq!(
        first["response"]["candidates"][0]["content"]["parts"][0]["functionCall"]["id"],
        second["response"]["candidates"][0]["content"]["parts"][0]["functionCall"]["id"]
    );
    assert_ne!(
        second["response"]["candidates"][0]["content"]["parts"][0]["functionCall"]["id"],
        second["response"]["candidates"][0]["content"]["parts"][1]["functionCall"]["id"]
    );
}

#[tokio::test]
async fn cumulative_media_identity_ignores_aliases_key_order_and_metadata() {
    let first =
        serde_json::from_str::<Value>(r#"{"inlineData":{"mimeType":"image/png","data":"cG5n"}}"#)
            .unwrap();
    let second = serde_json::from_str::<Value>(
        r#"{"inline_data":{"data":"cG5n","mime_type":"image/png"},"thoughtSignature":"trailing"}"#,
    )
    .unwrap();
    let frames = vec![
        json!({"candidates":[{"content":{"parts":[first]}}]}),
        json!({"candidates":[{"content":{"parts":[second]},"finishReason":"STOP"}]}),
    ];
    let result = collector::collect_stream_to_json(raw(frames), "r19-alias-media")
        .await
        .unwrap();
    assert_eq!(
        result["candidates"][0]["content"]["parts"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        result["candidates"][0]["content"]["parts"][0]["thoughtSignature"],
        "trailing"
    );
}

#[test]
fn cumulative_tool_ids_ignore_argument_key_order() {
    let mut ids = crate::proxy::mappers::gemini::wrapper::ResponseToolIds::default();
    let mut first: Value = serde_json::from_str(r#"{"candidates":[{"content":{"parts":[{"functionCall":{"name":"lookup","args":{"a":1,"b":2}}}]}}]}"#).unwrap();
    let mut second: Value = serde_json::from_str(r#"{"candidates":[{"content":{"parts":[{"functionCall":{"args":{"b":2,"a":1},"name":"lookup","id":null}}]}}]}"#).unwrap();
    ids.inject(&mut first, "claude");
    ids.inject(&mut second, "claude");
    assert_eq!(
        first["candidates"][0]["content"]["parts"][0]["functionCall"]["id"],
        second["candidates"][0]["content"]["parts"][0]["functionCall"]["id"]
    );
}

#[tokio::test]
async fn buffered_error_terminal_never_publishes_reasoning_history() {
    crate::modules::proxy_db::init_db().unwrap();
    let session = format!("r19-invalid-terminal-{}", uuid::Uuid::new_v4());
    let frame = json!({"candidates":[{"content":{"parts":[{"thought":true,"text":"failed reasoning"},{"text":"partial", "thoughtSignature":sig(b'A')}]},"finishReason":"MALFORMED_FUNCTION_CALL"}]});
    collector::collect_stream_to_json(raw(vec![frame]), &session)
        .await
        .unwrap();
    assert!(crate::modules::proxy_db::load_thinking_records(&session)
        .unwrap()
        .is_empty());
    assert!(crate::proxy::SignatureCache::global()
        .get_session_signature(&session)
        .is_none());
}

async fn check_thinking_only_completion(proto: &str) {
    crate::modules::proxy_db::init_db().unwrap();
    for reason in ["STOP", "MAX_TOKENS"] {
        for split in [false, true] {
            let session = format!("r19-thinking-only-{proto}-{}", uuid::Uuid::new_v4());
            let thought = json!({"thought":true,"text":"Signed reasoning without an answer.","thoughtSignature":sig(b'A')});
            let mut frames = vec![json!({"candidates":[{"content":{"parts":[thought]}}]})];
            if split {
                frames.push(json!({"candidates":[{"finishReason":reason}]}));
            } else {
                frames[0]["candidates"][0]["finishReason"] = json!(reason);
            }
            let anchor = Some(compute_response_text_anchor(&[user()]));
            let stream = match proto {
                "claude" => create_claude_sse_stream_with_context(
                    raw(frames),
                    "trace".into(),
                    "test".into(),
                    Some(session.clone()),
                    false,
                    1000,
                    None,
                    1,
                    None,
                    vec![],
                    anchor,
                ),
                "chat" => create_openai_sse_stream_with_context(
                    raw(frames),
                    "gemini-2.5-flash".into(),
                    session.clone(),
                    1,
                    None,
                    false,
                    Some(compute_causal_anchor(Some(&user()))),
                    anchor,
                ),
                "legacy" => create_legacy_sse_stream_with_context(
                    raw(frames),
                    "gemini-2.5-flash".into(),
                    session.clone(),
                    1,
                    anchor,
                ),
                "responses" => create_codex_sse_stream_with_context(
                    raw(frames),
                    "gemini-2.5-flash".into(),
                    session.clone(),
                    1,
                    1,
                    format!("response-{session}"),
                    None,
                    false,
                    anchor,
                    Default::default(),
                ),
                "gemini" => run_stream_lifecycle(
                    raw(frames),
                    native(&session, "gemini-2.5-flash", false, 1),
                    StreamLifecycleConfig::new("gemini", "r19", "test".into()),
                ),
                _ => unreachable!(),
            };
            let chunks = stream.collect::<Vec<_>>().await;
            let wire = chunks
                .iter()
                .filter_map(|chunk| chunk.as_ref().ok())
                .map(|chunk| String::from_utf8_lossy(chunk).into_owned())
                .collect::<String>();
            if matches!(proto, "claude" | "chat" | "gemini") {
                assert!(
                    chunks.iter().any(Result::is_err),
                    "{proto} {reason} split={split}: expected incomplete stream: {wire}"
                );
                assert!(!wire.contains("message_stop") && !wire.contains("data: [DONE]"));
            } else if proto == "responses" {
                assert!(wire.contains("response.incomplete"), "{reason}: {wire}");
            } else {
                // Legacy renders reasoning as ordinary completion text, so this
                // is a deliverable answer and must retain its successful history.
                assert!(chunks.iter().all(Result::is_ok));
                assert!(
                    wire.contains("Signed reasoning without an answer.")
                        && wire.contains("data: [DONE]")
                );
            }
            let rows = crate::modules::proxy_db::load_thinking_records(&session).unwrap();
            let signature = crate::proxy::SignatureCache::global().get_session_signature(&session);
            if proto == "legacy" {
                assert_eq!(rows.len(), 1, "{proto} {reason} split={split}");
                assert!(signature.is_some());
            } else {
                assert!(
                    rows.is_empty(),
                    "{proto} {reason} split={split}: incomplete output stored"
                );
                assert!(
                    signature.is_none(),
                    "{proto} {reason} split={split}: incomplete signature published"
                );
            }
        }
    }
}

#[tokio::test]
async fn thinking_only_claude_failure_does_not_publish_history() {
    check_thinking_only_completion("claude").await;
}

#[tokio::test]
async fn thinking_only_chat_failure_does_not_publish_history() {
    check_thinking_only_completion("chat").await;
}

#[tokio::test]
async fn thinking_only_responses_incomplete_does_not_publish_history() {
    check_thinking_only_completion("responses").await;
}

#[tokio::test]
async fn thinking_only_legacy_success_retains_visible_reasoning_history() {
    check_thinking_only_completion("legacy").await;
}

#[tokio::test]
async fn thinking_only_gemini_failure_does_not_publish_history() {
    check_thinking_only_completion("gemini").await;
}

#[tokio::test]
async fn thinking_only_gemini_candidate_cannot_borrow_sibling_content() {
    crate::modules::proxy_db::init_db().unwrap();
    for buffered in [false, true] {
        let session = format!("r19-thinking-only-sibling-{}", uuid::Uuid::new_v4());
        let frames = vec![json!({"candidates":[
            {"index":0,"content":{"parts":[{"thought":true,"text":"Undeliverable thought.","thoughtSignature":sig(b'A')}]},"finishReason":"STOP"},
            {"index":1,"content":{"parts":[{"thought":true,"text":"Successful thought.","thoughtSignature":sig(b'B')},{"text":"Done."}]},"finishReason":"STOP"}
        ]})];
        let stream = run_stream_lifecycle(
            raw(frames),
            native(&session, "gemini-2.5-flash", buffered, 2),
            StreamLifecycleConfig::new("gemini", "r19", "test".into()),
        );
        let chunks = stream.collect::<Vec<_>>().await;
        assert!(chunks.iter().all(Result::is_ok));
        if buffered {
            collector::collect_stream_to_json(futures::stream::iter(chunks), &session)
                .await
                .unwrap();
        }
        let rows = crate::modules::proxy_db::load_thinking_records(&session).unwrap();
        assert_eq!(
            rows.len(),
            1,
            "buffered={buffered}: undeliverable candidate stored"
        );
        assert_eq!(rows[0].signature.as_deref(), Some(sig(b'B').as_str()));
        assert!(crate::proxy::SignatureCache::global()
            .get_session_signature(&session)
            .is_none());
    }
}

#[tokio::test]
async fn signed_gemini_text_tool_and_media_outputs_retain_history() {
    crate::modules::proxy_db::init_db().unwrap();
    for visible in [
        json!({"text":"Done."}),
        json!({"functionCall":{"name":"inspect","id":"call_visible","args":{}}}),
        json!({"inlineData":{"mimeType":"image/png","data":"cG5n"}}),
        json!({"fileData":{"mimeType":"image/png","fileUri":"https://files.example/output.png"}}),
    ] {
        for buffered in [false, true] {
            for split in [false, true] {
                let session = format!("r19-visible-history-{}", uuid::Uuid::new_v4());
                let thought = json!({"thought":true,"text":"Successful signed reasoning.","thoughtSignature":sig(b'B')});
                let mut frames =
                    vec![json!({"candidates":[{"content":{"parts":[thought,visible]}}]})];
                if split {
                    frames.push(json!({"candidates":[{"finishReason":"STOP"}]}));
                } else {
                    frames[0]["candidates"][0]["finishReason"] = json!("STOP");
                }
                let stream = run_stream_lifecycle(
                    raw(frames),
                    native(&session, "gemini-2.5-flash", buffered, 1),
                    StreamLifecycleConfig::new("gemini", "r19", "test".into()),
                );
                let chunks = stream.collect::<Vec<_>>().await;
                assert!(chunks.iter().all(Result::is_ok), "{visible}");
                if buffered {
                    collector::collect_stream_to_json(futures::stream::iter(chunks), &session)
                        .await
                        .unwrap();
                }
                let rows = crate::modules::proxy_db::load_thinking_records(&session).unwrap();
                assert_eq!(
                    rows.len(),
                    1,
                    "buffered={buffered} split={split}: {visible}"
                );
                assert_eq!(rows[0].signature.as_deref(), Some(sig(b'B').as_str()));
                assert!(crate::proxy::SignatureCache::global()
                    .get_session_signature(&session)
                    .is_some());
            }
        }
    }
}
