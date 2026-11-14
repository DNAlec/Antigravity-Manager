//! Exercise the real protocol adapters and collectors with visible media.

use crate::proxy::mappers::{claude, openai};
use bytes::Bytes;
use futures::{stream, Stream, StreamExt};
use serde_json::{json, Value};
use std::{io, pin::Pin};

const MODEL: &str = "gemini-2.5-flash";
const PNG: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC";
const IMAGE_URL: &str = "https://example.test/output.png";
const TEXT: &str = "Here is the result.";
type Wire = Vec<Result<Bytes, String>>;

fn source(
    payloads: &[Value],
    interrupted: bool,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, String>> + Send>> {
    let mut frames: Wire = payloads
        .iter()
        .map(|payload| {
            Ok(Bytes::from(format!(
                "data: {}\n\n",
                json!({"response": payload})
            )))
        })
        .collect();
    if interrupted {
        frames.push(Err("connection reset by peer".into()));
    }
    Box::pin(stream::iter(frames))
}

async fn outputs(payloads: &[Value], interrupted: bool) -> [Wire; 3] {
    let chat = openai::streaming::create_openai_sse_stream(
        source(payloads, interrupted),
        MODEL.into(),
        uuid::Uuid::new_v4().to_string(),
        1,
        None,
        true,
    )
    .collect()
    .await;
    let responses = openai::streaming::create_codex_sse_stream(
        source(payloads, interrupted),
        MODEL.into(),
        uuid::Uuid::new_v4().to_string(),
        1,
        1,
        format!("resp_{}", uuid::Uuid::new_v4()),
        None,
        false,
    )
    .collect()
    .await;
    let claude = claude::create_claude_sse_stream(
        source(payloads, interrupted),
        "media-test".into(),
        "media@example.test".into(),
        None,
        false,
        200_000,
        None,
        1,
        None,
        vec![],
    )
    .collect()
    .await;
    [chat, responses, claude]
}

fn events(wire: &Wire) -> Vec<Value> {
    wire.iter()
        .filter_map(|item| item.as_ref().ok())
        .flat_map(|bytes| {
            String::from_utf8_lossy(bytes)
                .lines()
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .filter_map(|line| {
            line.strip_prefix("data: ")
                .and_then(|data| serde_json::from_str(data).ok())
        })
        .collect()
}

fn visible_delta(route: usize, events: &[Value]) -> String {
    events
        .iter()
        .flat_map(|event| match route {
            0 => event["choices"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|choice| choice["delta"]["content"].as_str())
                .collect::<Vec<_>>(),
            1 if event["type"] == "response.output_text.delta" => {
                event["delta"].as_str().into_iter().collect()
            }
            2 if event["type"] == "content_block_delta" => {
                event["delta"]["text"].as_str().into_iter().collect()
            }
            _ => vec![],
        })
        .collect()
}

fn text_blocks(content: &Value) -> String {
    content
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|block| block["text"].as_str())
        .collect()
}

fn payload(parts: Vec<Value>) -> Value {
    json!({
        "responseId": "media-response", "modelVersion": MODEL,
        "candidates": [{"index": 0, "content": {"role": "model", "parts": parts}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 5, "thoughtsTokenCount": 2, "totalTokenCount": 107}
    })
}

#[tokio::test]
async fn response_media_public_entries_preserve_images_files_and_aliases() {
    for (media, marker, label) in [
        (
            json!({"inlineData":{"mimeType":"image/png","data":PNG}}),
            PNG,
            "![image](<data:",
        ),
        (
            json!({"fileData":{"mimeType":"image/png","fileUri":IMAGE_URL}}),
            IMAGE_URL,
            "![image](<https:",
        ),
        (
            json!({"text":"","inline_data":{"mime_type":"image/png","data":PNG}}),
            PNG,
            "![image](<data:",
        ),
        (
            json!({"text":"","file_data":{"mime_type":"application/pdf","file_uri":"https://example.test/report.pdf"}}),
            "report.pdf",
            "[file](<https:",
        ),
        (
            json!({"inlineData":{"mimeType":"audio/wav","data":"AQID"}}),
            "AQID",
            "[file](<data:",
        ),
        (
            json!({"inline_data":{"data":"AQID"}}),
            "AQID",
            "[file](<data:application/octet-stream",
        ),
        (
            json!({"fileData":{"mimeType":"application/pdf","fileUri":"gs://bucket/report.pdf"}}),
            "gs://bucket/report.pdf",
            "File reference:",
        ),
    ] {
        for with_text in [false, true] {
            let mut parts = vec![];
            if with_text {
                parts.push(json!({"text":TEXT}));
            }
            parts.push(media.clone());
            let input = payload(parts);
            let [chat, responses, claude_wire] = outputs(&[input.clone()], false).await;
            let mut visible = vec![];
            for (route, wire) in [&chat, &responses, &claude_wire].into_iter().enumerate() {
                assert!(wire.iter().all(Result::is_ok), "route {route}: {wire:?}");
                let parsed = events(wire);
                visible.push(visible_delta(route, &parsed));
                match route {
                    0 => {
                        assert!(parsed
                            .iter()
                            .any(|e| e["choices"][0]["finish_reason"] == "stop"));
                        assert!(parsed.iter().any(|e| e["usage"]["prompt_tokens"] == 100
                            && e["usage"]["completion_tokens"] == 7));
                    }
                    1 => {
                        let completed = parsed
                            .iter()
                            .find(|e| e["type"] == "response.completed")
                            .unwrap();
                        assert_eq!(completed["response"]["usage"]["input_tokens"], 100);
                        assert_eq!(completed["response"]["usage"]["output_tokens"], 7);
                        let final_text: String = completed["response"]["output"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .filter(|item| item["type"] == "message")
                            .map(|item| text_blocks(&item["content"]))
                            .collect();
                        assert_eq!(final_text, visible[route]);
                    }
                    2 => {
                        assert_eq!(
                            parsed
                                .iter()
                                .filter(|e| e["type"] == "message_stop")
                                .count(),
                            1
                        );
                        assert!(parsed.iter().any(|e| e["type"] == "message_delta"
                            && e["delta"]["stop_reason"] == "end_turn"));
                    }
                    _ => unreachable!(),
                }
            }
            let collected = openai::collector::collect_stream_to_json(stream::iter(chat))
                .await
                .unwrap();
            let collected = serde_json::to_value(collected).unwrap();
            visible.push(
                collected["choices"][0]["message"]["content"]
                    .as_str()
                    .unwrap()
                    .into(),
            );
            let direct = serde_json::to_value(openai::response::transform_openai_response(
                &input, None, 1, None,
            ))
            .unwrap();
            visible.push(
                direct["choices"][0]["message"]["content"]
                    .as_str()
                    .unwrap()
                    .into(),
            );
            let collected = claude::collect_stream_to_json(stream::iter(
                claude_wire.into_iter().map(|v| v.map_err(io::Error::other)),
            ))
            .await
            .unwrap();
            visible.push(text_blocks(
                &serde_json::to_value(collected).unwrap()["content"],
            ));
            let typed = serde_json::from_value::<claude::models::GeminiResponse>(input).unwrap();
            let direct =
                claude::transform_response(&typed, false, 200_000, None, MODEL.into(), 1, vec![])
                    .unwrap();
            visible.push(text_blocks(
                &serde_json::to_value(direct).unwrap()["content"],
            ));
            for (route, text) in visible.iter().enumerate() {
                assert_eq!(text.matches(marker).count(), 1, "route {route}: {text}");
                assert!(text.contains(label), "route {route}: {text}");
                assert_eq!(
                    text.matches(TEXT).count(),
                    usize::from(with_text),
                    "route {route}: {text}"
                );
                assert_eq!(text, &visible[0], "route {route} differs from Chat stream");
            }
        }
    }
}

#[tokio::test]
async fn response_media_streams_keep_multiplicity_candidate_identity_and_usage_trailers() {
    let media = json!({"fileData":{"mimeType":"image/png","fileUri":IMAGE_URL}});
    let alias = json!({"file_data":{"file_uri":IMAGE_URL,"mime_type":"image/png"}, "thoughtSignature":"later"});
    let input = [
        json!({"candidates":[{"index":"1","content":{"parts":[media,media]}}]}),
        json!({"candidates":[{"index":1,"content":{"parts":[alias,alias]}}]}),
        json!({"candidates":[{"content":{"parts":[media]}}]}),
        json!({"candidates":[{"index":1,"content":{"parts":[alias,alias,alias]}}]}),
        json!({"candidates":[{"index":1,"finishReason":"STOP"},{"index":0,"finishReason":"STOP"}],
            "usageMetadata":{"promptTokenCount":100}}),
        json!({"usageMetadata":{"candidatesTokenCount":5,"thoughtsTokenCount":2,"totalTokenCount":107}}),
    ];
    for (route, wire) in outputs(&input, false).await.iter().enumerate() {
        assert!(wire.iter().all(Result::is_ok), "route {route}: {wire:?}");
        let parsed = events(wire);
        let text = visible_delta(route, &parsed);
        assert_eq!(text.matches(IMAGE_URL).count(), 4, "route {route}: {text}");
        let usage = match route {
            0 => &parsed
                .iter()
                .rev()
                .find(|e| e["usage"].is_object())
                .unwrap()["usage"],
            1 => &parsed
                .iter()
                .find(|e| e["type"] == "response.completed")
                .unwrap()["response"]["usage"],
            2 => &parsed
                .iter()
                .find(|e| e["type"] == "message_delta")
                .unwrap()["usage"],
            _ => unreachable!(),
        };
        assert_eq!(
            usage[if route == 0 {
                "prompt_tokens"
            } else {
                "input_tokens"
            }],
            100
        );
        assert_eq!(
            usage[if route == 0 {
                "completion_tokens"
            } else {
                "output_tokens"
            }],
            7
        );
    }
}

#[tokio::test]
async fn response_media_does_not_turn_a_real_interruption_into_success() {
    let mut input = payload(vec![
        json!({"fileData":{"mimeType":"image/png","fileUri":IMAGE_URL}}),
    ]);
    input["candidates"][0]
        .as_object_mut()
        .unwrap()
        .remove("finishReason");
    for (route, wire) in outputs(&[input], true).await.iter().enumerate() {
        let parsed = events(wire);
        assert_eq!(visible_delta(route, &parsed).matches(IMAGE_URL).count(), 1);
        assert!(
            !parsed.iter().any(|e| e["type"] == "response.completed"
                || e["choices"][0]["finish_reason"] == "stop"
                || e["delta"]["stop_reason"] == "end_turn"),
            "route {route}: {parsed:?}"
        );
        assert!(
            wire.iter().any(Result::is_err)
                || parsed.iter().any(|e| e["error"].is_object()
                    || e["type"] == "response.incomplete"
                    || e["type"] == "response.failed"
                    || e["type"] == "error"
                    || e["choices"][0]["finish_reason"] == "length"
                    || e["delta"]["stop_reason"] == "max_tokens"),
            "route {route} must report the interruption: {parsed:?}"
        );
        let usage = match route {
            0 => &parsed
                .iter()
                .rev()
                .find(|e| e["usage"].is_object())
                .unwrap()["usage"],
            1 => &parsed
                .iter()
                .find(|e| e["type"] == "response.incomplete" || e["type"] == "response.failed")
                .unwrap()["response"]["usage"],
            2 => &parsed
                .iter()
                .find(|e| e["type"] == "message_delta")
                .unwrap()["usage"],
            _ => unreachable!(),
        };
        assert_eq!(
            usage[if route == 0 {
                "prompt_tokens"
            } else {
                "input_tokens"
            }],
            100
        );
        assert_eq!(
            usage[if route == 0 {
                "completion_tokens"
            } else {
                "output_tokens"
            }],
            7
        );
    }
}
