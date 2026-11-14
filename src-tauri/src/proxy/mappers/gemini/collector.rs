// Gemini Stream Collector
// Used for auto-converting streaming responses to JSON for non-streaming requests

use bytes::Bytes;
use futures::StreamExt;
use serde_json::{json, Value};
use std::collections::BTreeMap;

use crate::proxy::common::response_media::media_identity;
use crate::proxy::pipeline::usage::merge_cumulative_usage;
use crate::proxy::SignatureCache; // Assuming this is available at crate root or re-exported

/// Collects a Gemini SSE stream into a complete Gemini Response Value
/// ALSO performs signature caching side-effect
pub async fn collect_stream_to_json<S, E>(stream: S, session_id: &str) -> Result<Value, String>
where
    S: futures::Stream<Item = Result<Bytes, E>> + Unpin,
    E: std::fmt::Display,
{
    collect_stream_to_json_with_anchor_and_count(stream, session_id, None, 1).await
}

pub async fn collect_stream_to_json_with_anchor<S, E>(
    stream: S,
    session_id: &str,
    anchor: Option<&str>,
) -> Result<Value, String>
where
    S: futures::Stream<Item = Result<Bytes, E>> + Unpin,
    E: std::fmt::Display,
{
    collect_stream_to_json_with_anchor_and_count(stream, session_id, anchor, 1).await
}

pub async fn collect_stream_to_json_with_anchor_and_count<S, E>(
    stream: S,
    session_id: &str,
    anchor: Option<&str>,
    message_count: usize,
) -> Result<Value, String>
where
    S: futures::Stream<Item = Result<Bytes, E>> + Unpin,
    E: std::fmt::Display,
{
    collect_stream_to_json_with_context(stream, session_id, anchor, message_count, None).await
}

#[allow(clippy::too_many_arguments)]
pub async fn collect_stream_to_json_with_context<S, E>(
    stream: S,
    session_id: &str,
    anchor: Option<&str>,
    message_count: usize,
    text_context_anchor: Option<String>,
) -> Result<Value, String>
where
    S: futures::Stream<Item = Result<Bytes, E>> + Unpin,
    E: std::fmt::Display,
{
    // Route an unterminated final data line through the same content/usage path.
    let mut stream = stream.chain(futures::stream::iter([Ok(Bytes::from("\n"))]));
    let mut prompt_feedback = None;
    let mut blocked_prompt = false;
    let mut collected_response = json!({});
    let mut grouped: BTreeMap<u64, (Value, Vec<Value>)> = BTreeMap::new();
    let mut media_positions =
        BTreeMap::<u64, std::collections::HashMap<(bool, [u8; 32]), Vec<usize>>>::new();
    let mut tool_calls = crate::proxy::common::response_tools::ToolCallTracker::default();
    let mut tool_positions = std::collections::HashMap::<usize, usize>::new();
    let mut usage_metadata: Option<Value> = None;
    let mut stream_error: Option<Value> = None;
    let mut line_buffer = bytes::BytesMut::new();

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| {
            crate::proxy::mappers::error_classifier::report_stream_error(
                "gemini-collector",
                "collect_stream_to_json_with_anchor",
                &e,
                format!("session={}", session_id),
            )
            .client_message()
        })?;

        line_buffer.extend_from_slice(&chunk);

        while let Some(pos) = line_buffer.iter().position(|&b| b == b'\n') {
            let line_raw = line_buffer.split_to(pos + 1);
            let line_str = String::from_utf8_lossy(&line_raw);
            let line = line_str.trim();
            if line.starts_with("data: ") {
                let json_part = line.trim_start_matches("data: ").trim();
                if json_part == "[DONE]" {
                    continue;
                }

                if let Ok(mut json) = serde_json::from_str::<Value>(json_part) {
                    // Unwrap v1internal response wrapper similar to handler
                    let actual_data =
                        if let Some(inner) = json.get_mut("response").map(|v| v.take()) {
                            inner
                        } else {
                            json
                        };

                    // Counters can accompany the terminal error itself.
                    if let Some(usage) = actual_data.get("usageMetadata") {
                        merge_cumulative_usage(&mut usage_metadata, usage);
                    }

                    // Check for error payload (e.g. 504 Deadline Exceeded, 503 Overloaded, 429)
                    if let Some(err) = actual_data.get("error") {
                        let err_val = if actual_data.as_object().is_some_and(|m| m.len() == 1) {
                            actual_data
                        } else {
                            json!({ "error": err })
                        };
                        stream_error = Some(err_val);
                        break;
                    }

                    if let Some(feedback) = actual_data.get("promptFeedback") {
                        let blocking = feedback
                            .get("blockReason")
                            .and_then(Value::as_str)
                            .is_some_and(|reason| {
                                !reason.is_empty() && reason != "BLOCK_REASON_UNSPECIFIED"
                            });
                        if blocking || !blocked_prompt {
                            prompt_feedback = Some(feedback.clone());
                        }
                        blocked_prompt |= blocking;
                    }

                    // Capture Content & Signature
                    tool_calls.begin_frame();
                    if let Some(candidates) =
                        actual_data.get("candidates").and_then(|c| c.as_array())
                    {
                        let Some(indices) = super::candidate_indices(candidates) else {
                            stream_error = Some(
                                json!({"error":{"code":502,"status":"INTERNAL","message":"Invalid or duplicate Gemini candidate index"}}),
                            );
                            break;
                        };
                        for (index, candidate) in indices.into_iter().zip(candidates) {
                            let (meta, content_parts) = grouped.entry(index).or_insert_with(|| {
                                (
                                    json!({"index":index,"content":{"role":"model"}}),
                                    Vec::new(),
                                )
                            });
                            if let Some(fields) = candidate.as_object() {
                                for (k, v) in fields {
                                    if k == "content" {
                                        if let Some(content) = v.as_object() {
                                            for (k, v) in content {
                                                if k != "parts" {
                                                    meta["content"][k] = v.clone();
                                                }
                                            }
                                        }
                                    } else if k != "index" {
                                        meta[k] = v.clone();
                                    }
                                }
                            }
                            if let Some(parts) = candidate
                                .get("content")
                                .and_then(|c| c.get("parts"))
                                .and_then(|p| p.as_array())
                            {
                                // Retain occurrence positions across frames so text trailers do
                                // not repeatedly hash every already-collected image payload.
                                let prior_media = media_positions.entry(index).or_default();
                                let mut frame_media = std::collections::HashMap::new();
                                for part in parts {
                                    if let Some(call) = part.get("functionCall") {
                                        let tracked = tool_calls.track(index, call);
                                        if let Some(&position) = tool_positions.get(&tracked.index)
                                        {
                                            if let (Some(existing), Some(update)) = (
                                                content_parts[position].as_object_mut(),
                                                part.as_object(),
                                            ) {
                                                for (key, value) in update {
                                                    if key == "functionCall" {
                                                        if let (Some(call), Some(fields)) = (
                                                            existing
                                                                .get_mut(key)
                                                                .and_then(Value::as_object_mut),
                                                            value.as_object(),
                                                        ) {
                                                            // Omitted call fields are metadata trailers;
                                                            // supplied args replace the previous snapshot.
                                                            call.extend(fields.clone());
                                                            continue;
                                                        }
                                                    }
                                                    existing.insert(key.clone(), value.clone());
                                                }
                                            }
                                            continue;
                                        }
                                        tool_positions.insert(tracked.index, content_parts.len());
                                    }
                                    if super::has_media(part) {
                                        let key = media_identity(part);
                                        let count = frame_media.entry(key).or_insert(0usize);
                                        *count += 1;
                                        if let Some(position) = prior_media
                                            .get(&key)
                                            .and_then(|positions| positions.get(*count - 1))
                                        {
                                            merge_media_metadata(
                                                &mut content_parts[*position],
                                                part,
                                            );
                                            continue;
                                        }
                                        prior_media
                                            .entry(key)
                                            .or_default()
                                            .push(content_parts.len());
                                    }
                                    // Collect part
                                    if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                                        let is_part_thought = part
                                            .get("thought")
                                            .and_then(|v| v.as_bool())
                                            .unwrap_or(false);

                                        if let Some(last) = content_parts.last_mut() {
                                            let is_last_thought = last
                                                .get("thought")
                                                .and_then(|v| v.as_bool())
                                                .unwrap_or(false);

                                            if !is_part_thought
                                                && !is_last_thought
                                                && last.get("text").is_some()
                                                && plain_text_mergeable(last)
                                                && plain_text_mergeable(part)
                                            {
                                                // Merge only plain unsigned text; signatures and
                                                // provider metadata belong to their original part.
                                                if let Some(last_text) =
                                                    last.get_mut("text").and_then(|v| v.as_str())
                                                {
                                                    let new_text = format!("{}{}", last_text, text);
                                                    if let Some(obj) = last.as_object_mut() {
                                                        obj.insert(
                                                            "text".to_string(),
                                                            Value::String(new_text),
                                                        );
                                                    } else {
                                                        *last = json!({ "text": new_text });
                                                    }
                                                    continue;
                                                }
                                            } else if is_part_thought
                                                && is_last_thought
                                                && last.get("text").is_some()
                                                && plain_text_mergeable(last)
                                                && thinking_fragment_mergeable(part)
                                            {
                                                // Merge only unsigned thinking fragments [FIX #3697]
                                                if let Some(last_text) =
                                                    last.get_mut("text").and_then(|v| v.as_str())
                                                {
                                                    let new_text = format!("{}{}", last_text, text);
                                                    if let Some(obj) = last.as_object_mut() {
                                                        obj.insert(
                                                            "text".to_string(),
                                                            Value::String(new_text),
                                                        );
                                                        if let Some(sig) =
                                                            part.get("thoughtSignature").or_else(
                                                                || part.get("thought_signature"),
                                                            )
                                                        {
                                                            obj.insert(
                                                                "thoughtSignature".to_string(),
                                                                sig.clone(),
                                                            );
                                                        }
                                                    } else {
                                                        let mut obj = serde_json::Map::new();
                                                        obj.insert(
                                                            "thought".to_string(),
                                                            Value::Bool(true),
                                                        );
                                                        obj.insert(
                                                            "text".to_string(),
                                                            Value::String(new_text),
                                                        );
                                                        if let Some(sig) = part
                                                            .get("thoughtSignature")
                                                            .or_else(|| {
                                                                part.get("thought_signature")
                                                            })
                                                            .or_else(|| {
                                                                last.get("thoughtSignature")
                                                            })
                                                        {
                                                            obj.insert(
                                                                "thoughtSignature".to_string(),
                                                                sig.clone(),
                                                            );
                                                        }
                                                        *last = Value::Object(obj);
                                                    }
                                                    continue;
                                                }
                                            }
                                        }
                                        content_parts.push(part.clone());
                                    } else {
                                        // Other parts (images, thoughts without text, function calls), just push
                                        content_parts.push(part.clone());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if stream_error.is_some() {
            break;
        }
    }

    // Preserve measured work on failure without publishing partial content or
    // committing it as a completed model response.
    if let Some(mut err_val) = stream_error {
        crate::proxy::pipeline::stream_error::attach_usage_metadata(
            &mut err_val,
            usage_metadata.as_ref(),
        );
        return Ok(err_val);
    }

    // Stream finished without content parts or finish reason -> premature interruption
    if grouped.values().all(|(m, p)| {
        p.is_empty()
            && m.get("finishReason")
                .or_else(|| m.get("finish_reason"))
                .is_none()
    }) && !blocked_prompt
    {
        return Err(
            crate::proxy::mappers::error_classifier::report_stream_error(
                "gemini-collector",
                "collect_stream_to_json_with_anchor",
                &"stream terminated prematurely without content or finish reason",
                format!("session={}", session_id),
            )
            .client_message(),
        );
    }

    // Aggregation and history use the same candidate boundaries. A branch
    // selection has no session-wide signature authority until it is replayed.
    let singleton = grouped.len() == 1;
    if grouped.len() > 1 {
        SignatureCache::global().mark_multi_candidate_turn(session_id, message_count.max(1));
    }
    let mut candidates = Vec::with_capacity(grouped.len());
    for (_, (mut candidate, parts)) in grouped {
        if blocked_prompt
            && parts.is_empty()
            && candidate
                .get("finishReason")
                .or_else(|| candidate.get("finish_reason"))
                .is_none()
        {
            continue;
        }
        if parts.iter().any(super::has_visible_content)
            && super::has_successful_candidate_finish(&candidate)
        {
            if singleton {
                for part in &parts {
                    if let Some(sig) = part
                        .get("thoughtSignature")
                        .or_else(|| part.get("thought_signature"))
                        .and_then(Value::as_str)
                    {
                        SignatureCache::global().cache_session_signature(
                            session_id,
                            sig.to_owned(),
                            message_count.max(1),
                        );
                    }
                }
            }
            crate::proxy::thinking_store::capture_gemini_parts_with_context(
                session_id,
                &parts,
                anchor.unwrap_or("root"),
                text_context_anchor.as_deref(),
            );
        }
        candidate["content"]["parts"] = Value::Array(parts);
        candidates.push(candidate);
    }
    collected_response["candidates"] = Value::Array(candidates);
    if let Some(feedback) = prompt_feedback {
        collected_response["promptFeedback"] = feedback;
    }
    if let Some(usage) = usage_metadata {
        collected_response["usageMetadata"] = usage;
    }

    Ok(collected_response)
}

fn merge_media_metadata(previous: &mut Value, current: &Value) {
    let (Some(previous), Some(current)) = (previous.as_object_mut(), current.as_object()) else {
        return;
    };
    for (key, value) in current {
        match key.as_str() {
            "inlineData" | "inline_data" | "fileData" | "file_data" => {}
            "thoughtSignature" | "thought_signature" => {
                previous.remove("thought_signature");
                previous.insert("thoughtSignature".into(), value.clone());
            }
            _ => {
                previous.insert(key.clone(), value.clone());
            }
        }
    }
}

// An unsigned incremental thought may acquire its closing signature. Once a
// signed block is closed, subsequent thought text starts a separate block.
fn thinking_fragment_mergeable(part: &Value) -> bool {
    part.as_object().is_some_and(|object| {
        object.keys().all(|key| {
            matches!(
                key.as_str(),
                "text" | "thought" | "thoughtSignature" | "thought_signature"
            )
        })
    })
}

// Native signed parts must round-trip without moving their signature onto
// concatenated text. Keep other provider metadata on its original part too.
fn plain_text_mergeable(part: &Value) -> bool {
    part.as_object()
        .is_some_and(|object| object.keys().all(|key| key == "text" || key == "thought"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use std::io;

    #[tokio::test]
    async fn numeric_string_indices_aggregate_with_numeric_continuations() {
        let frames = [
            json!({"response":{"candidates":[{"index":"1","content":{"parts":[{"text":"second "}]}},{"index":"0","content":{"parts":[{"text":"first "}]}}]}}),
            json!({"candidates":[{"index":0,"content":{"parts":[{"text":"answer"}]},"finishReason":"STOP"},{"index":1,"content":{"parts":[{"text":"answer"}]},"finishReason":"STOP"}]}),
        ];
        let source = stream::iter(
            frames.map(|frame| Ok::<_, io::Error>(Bytes::from(format!("data: {frame}\n\n")))),
        );
        let response = collect_stream_to_json(source, "numeric-string-collector")
            .await
            .unwrap();
        assert_eq!(response["candidates"].as_array().unwrap().len(), 2);
        for (index, expected) in [(0, "first answer"), (1, "second answer")] {
            assert_eq!(response["candidates"][index]["index"], index);
            assert_eq!(
                response["candidates"][index]["content"]["parts"][0]["text"],
                expected
            );
            assert_eq!(response["candidates"][index]["finishReason"], "STOP");
        }
    }

    #[tokio::test]
    async fn signed_plain_text_parts_keep_their_original_boundaries_at_eof() {
        for signed in [
            json!({"text":"second","thoughtSignature":"signed-fragment"}),
            json!({"text":"second","thought_signature":"signed-fragment"}),
            json!({"text":"","thoughtSignature":"signature-only"}),
            json!({"text":"second","providerMetadata":{"anchor":7}}),
        ] {
            for newline in ["\n\n", ""] {
                let first =
                    json!({"candidates":[{"content":{"parts":[{"text":"first"},signed.clone()]}}]});
                let last = json!({"candidates":[{"content":{"parts":[{"text":"third"}]},"finishReason":"STOP"}]});
                let wire = format!("data: {first}\n\ndata: {last}{newline}");
                let source = stream::iter(
                    wire.as_bytes()
                        .chunks(3)
                        .map(|bytes| Ok::<_, io::Error>(Bytes::copy_from_slice(bytes)))
                        .collect::<Vec<_>>(),
                );
                let response = collect_stream_to_json(source, "signed-parts")
                    .await
                    .unwrap();
                assert_eq!(
                    response["candidates"][0]["content"]["parts"],
                    json!([
                        {"text":"first"},signed,{"text":"third"}
                    ])
                );
                assert_eq!(response["candidates"][0]["finishReason"], "STOP");
            }
        }
    }

    #[tokio::test]
    async fn blocked_prompt_json_preserves_feedback_usage_and_has_no_fake_candidate() {
        for newline in ["\n\n", ""] {
            let blocked = json!({"response":{"promptFeedback":{"blockReason":"SAFETY","futureField":7},
                "usageMetadata":{"promptTokenCount":100,"cachedContentTokenCount":80}}});
            let trailer = json!({"usageMetadata":{"candidatesTokenCount":0},"promptFeedback":{"blockReason":"BLOCK_REASON_UNSPECIFIED"}});
            let wire = format!("data: {blocked}\n\ndata: {trailer}{newline}");
            let response = collect_stream_to_json(
                stream::iter(vec![Ok::<_, io::Error>(Bytes::from(wire))]),
                "blocked-prompt",
            )
            .await
            .unwrap();
            assert_eq!(
                response["promptFeedback"],
                blocked["response"]["promptFeedback"]
            );
            assert_eq!(response["candidates"], json!([]));
            assert_eq!(
                response["usageMetadata"],
                json!({"promptTokenCount":100,"cachedContentTokenCount":80,"candidatesTokenCount":0})
            );
        }
        let response = collect_stream_to_json(
            stream::iter(vec![Ok::<_, io::Error>(Bytes::from(
                "data: {\"promptFeedback\":{\"blockReason\":\"BLOCK_REASON_UNSPECIFIED\"}}\n\n",
            ))]),
            "default-feedback",
        )
        .await;
        assert!(response.is_err());
    }

    #[tokio::test]
    async fn error_frames_keep_sparse_usage_zero_and_null_with_or_without_final_newline() {
        for trailing_newline in [false, true] {
            let initial = json!({"response": {
                "candidates":[{"content":{"parts":[{"text":"partial"}]}}],
                "usageMetadata":{"promptTokenCount":100,"cachedContentTokenCount":80,"candidatesTokenCount":1,"futureUsageField":{"value":7}}
            }});
            let error = json!({"response": {
                "error":{"code":503,"status":"UNAVAILABLE","message":"overloaded"},
                "usageMetadata":{"promptTokenCount":null,"cachedContentTokenCount":0,"candidatesTokenCount":5}
            }});
            let wire = format!(
                "data: {initial}\n\ndata: {error}{}",
                if trailing_newline { "\n\n" } else { "" }
            );
            let chunks = wire
                .as_bytes()
                .chunks(3)
                .map(|chunk| Ok::<_, io::Error>(Bytes::copy_from_slice(chunk)))
                .collect::<Vec<_>>();
            let response = collect_stream_to_json(stream::iter(chunks), "error-usage-test")
                .await
                .unwrap();
            assert_eq!(response["error"]["code"], 503);
            assert!(response.get("candidates").is_none());
            assert_eq!(
                response["usageMetadata"],
                json!({
                    "promptTokenCount":100,"cachedContentTokenCount":0,"candidatesTokenCount":5,"futureUsageField":{"value":7}
                })
            );
        }
    }

    #[tokio::test]
    async fn r26_collect_tool_metadata_preserves_omitted_args_and_replaces_new_snapshots() {
        let frames = [
            json!({"candidates":[{"content":{"parts":[{"functionCall":{"id":"same","name":"inspect","args":{"path":"first","stale":true}}}]}}]}),
            json!({"candidates":[{"content":{"parts":[{"functionCall":{"id":"same","args":{"path":"updated"}}}]}}]}),
            json!({"candidates":[{"content":{"parts":[{"functionCall":{"id":"same"},"thoughtSignature":"late"}]},"finishReason":"STOP"}]}),
        ];
        let source = stream::iter(
            frames
                .into_iter()
                .map(|frame| Ok::<_, io::Error>(Bytes::from(format!("data: {frame}\n\n")))),
        );
        let response = collect_stream_to_json(source, "r26-tool-metadata")
            .await
            .unwrap();
        assert_eq!(
            response["candidates"][0]["content"]["parts"],
            json!([
                {"functionCall":{"id":"same","name":"inspect","args":{"path":"updated"}},"thoughtSignature":"late"}
            ])
        );
    }

    #[tokio::test]
    async fn r26_collect_cumulative_tools_preserves_multiplicity_candidates_and_metadata() {
        let anonymous = json!({"functionCall":{"name":"inspect","args":{"path":"src"}}});
        let first_id = json!({"functionCall":{"id":"first","name":"inspect","args":{}}});
        let second_id = json!({"functionCall":{"id":"second","name":"inspect","args":{}}});
        let mut signed = first_id.clone();
        signed["thoughtSignature"] = json!("late-signature");
        let frames = [
            json!({"candidates":[
                {"index":2,"content":{"parts":[anonymous.clone(),anonymous.clone(),first_id.clone(),second_id.clone()]}},
                {"index":5,"content":{"parts":[first_id.clone()]}}
            ]}),
            json!({"candidates":[
                {"index":5,"content":{"parts":[first_id]},"finishReason":"STOP"},
                {"index":2,"content":{"parts":[anonymous.clone(),anonymous,signed,second_id]},"finishReason":"STOP"}
            ]}),
        ];
        let source = stream::iter(
            frames
                .into_iter()
                .map(|frame| Ok::<_, io::Error>(Bytes::from(format!("data: {frame}\n\n")))),
        );
        let response = collect_stream_to_json(source, "r26-tool-collector")
            .await
            .unwrap();
        let candidates = response["candidates"].as_array().unwrap();
        let parts = candidates[0]["content"]["parts"].as_array().unwrap();
        assert_eq!(candidates[0]["index"], 2);
        assert_eq!(parts.len(), 4, "same-frame independent calls must survive");
        assert!(parts[0]["functionCall"].get("id").is_none());
        assert!(parts[1]["functionCall"].get("id").is_none());
        assert_eq!(parts[2]["functionCall"]["id"], "first");
        assert_eq!(parts[2]["thoughtSignature"], "late-signature");
        assert_eq!(parts[3]["functionCall"]["id"], "second");
        assert_eq!(candidates[1]["index"], 5);
        assert_eq!(
            candidates[1]["content"]["parts"].as_array().unwrap().len(),
            1
        );
    }

    #[tokio::test]
    async fn test_collect_sparse_cumulative_usage_preserves_cache_and_honors_zero() {
        for explicit_zero in [false, true] {
            let mut final_usage = json!({"candidatesTokenCount": 5, "totalTokenCount": 105});
            if explicit_zero {
                final_usage["cachedContentTokenCount"] = json!(0);
            }
            let frames = vec![
                json!({"response": {
                    "candidates": [{"content": {"parts": [{"text": "Hello"}]}}],
                    "usageMetadata": {
                        "promptTokenCount": 100,
                        "cachedContentTokenCount": 80,
                        "candidatesTokenCount": 3,
                        "futureUsageField": {"value": 7}
                    }
                }}),
                json!({"response": {
                    "candidates": [{"finishReason": "STOP"}],
                    "usageMetadata": final_usage
                }}),
            ];
            let byte_stream =
                stream::iter(frames.into_iter().map(|frame| {
                    Ok::<Bytes, io::Error>(Bytes::from(format!("data: {frame}\n\n")))
                }));
            let response = collect_stream_to_json(byte_stream, "sparse-usage-test")
                .await
                .unwrap();
            let usage = &response["usageMetadata"];
            assert_eq!(usage["promptTokenCount"], 100);
            assert_eq!(
                usage["cachedContentTokenCount"],
                if explicit_zero { 0 } else { 80 }
            );
            assert_eq!(usage["candidatesTokenCount"], 5);
            assert_eq!(usage["totalTokenCount"], 105);
            assert_eq!(usage["futureUsageField"]["value"], 7);
        }
    }

    #[tokio::test]
    async fn test_collect_simple_text_response() {
        let sse_data = vec![
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hello\"}],\"role\":\"model\"},\"index\":0}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\" world!\"}],\"role\":\"model\"},\"finishReason\":\"STOP\",\"index\":0}],\"usageMetadata\":{\"totalTokenCount\":10}}\n\n",
            "data: [DONE]\n\n",
        ];

        let byte_stream = stream::iter(
            sse_data
                .into_iter()
                .map(|s| Ok::<Bytes, io::Error>(Bytes::from(s))),
        );

        let result = collect_stream_to_json(byte_stream, "test_session").await;
        assert!(result.is_ok());

        let resp = result.unwrap();
        let parts = resp["candidates"][0]["content"]["parts"]
            .as_array()
            .unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0]["text"], "Hello world!");
        assert_eq!(resp["candidates"][0]["finishReason"], "STOP");
        assert_eq!(resp["usageMetadata"]["totalTokenCount"], 10);
    }

    #[tokio::test]
    async fn test_collect_error_payload_propagation_504() {
        let sse_data = vec![
            "data: {\"error\":{\"code\":504,\"message\":\"stream idle timeout\",\"status\":\"DEADLINE_EXCEEDED\"}}\n\n",
            "data: [DONE]\n\n",
        ];

        let byte_stream = stream::iter(
            sse_data
                .into_iter()
                .map(|s| Ok::<Bytes, io::Error>(Bytes::from(s))),
        );

        let result = collect_stream_to_json(byte_stream, "test_session_504").await;
        assert!(result.is_ok());

        let resp = result.unwrap();
        assert!(resp.get("error").is_some());
        assert_eq!(resp["error"]["code"], 504);
        assert_eq!(resp["error"]["status"], "DEADLINE_EXCEEDED");
    }

    #[tokio::test]
    async fn test_collect_error_payload_with_v1internal_wrapper() {
        let sse_data = vec![
            "data: {\"response\":{\"error\":{\"code\":503,\"message\":\"model overloaded\",\"status\":\"UNAVAILABLE\"}}}\n\n",
            "data: [DONE]\n\n",
        ];

        let byte_stream = stream::iter(
            sse_data
                .into_iter()
                .map(|s| Ok::<Bytes, io::Error>(Bytes::from(s))),
        );

        let result = collect_stream_to_json(byte_stream, "test_session_503").await;
        assert!(result.is_ok());

        let resp = result.unwrap();
        assert!(resp.get("error").is_some());
        assert_eq!(resp["error"]["code"], 503);
        assert_eq!(resp["error"]["status"], "UNAVAILABLE");
    }

    #[tokio::test]
    async fn test_collect_premature_empty_stream_fails() {
        let sse_data = vec!["data: [DONE]\n\n"];

        let byte_stream = stream::iter(
            sse_data
                .into_iter()
                .map(|s| Ok::<Bytes, io::Error>(Bytes::from(s))),
        );

        let result = collect_stream_to_json(byte_stream, "test_session_empty").await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("stream terminated prematurely"));
    }

    #[tokio::test]
    async fn test_collect_multibyte_chunk_split() {
        let cyrillic_word = "Привет, мир! 🚀";
        let chunk_data = format!(
            "data: {{\"candidates\":[{{\"content\":{{\"parts\":[{{\"text\":\"{}\"}}],\"role\":\"model\"}},\"finishReason\":\"STOP\",\"index\":0}}]}}\n\ndata: [DONE]\n\n",
            cyrillic_word
        );

        let bytes = chunk_data.into_bytes();
        let chunk_size = 13;
        let mut chunks = Vec::new();
        for chunk in bytes.chunks(chunk_size) {
            chunks.push(Ok::<Bytes, io::Error>(Bytes::copy_from_slice(chunk)));
        }

        let byte_stream = stream::iter(chunks);
        let result = collect_stream_to_json(byte_stream, "test_session_multibyte").await;
        assert!(result.is_ok());

        let resp = result.unwrap();
        let parts = resp["candidates"][0]["content"]["parts"]
            .as_array()
            .unwrap();
        assert_eq!(parts[0]["text"], cyrillic_word);
    }

    #[tokio::test]
    async fn test_collect_thinking_chunks_merged() {
        let chunk1 = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Thinking part 1. \",\"thought\":true}],\"role\":\"model\"},\"index\":0}]}\n\n";
        let chunk2 = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Thinking part 2.\",\"thought\":true,\"thoughtSignature\":\"sig_valid_1234567890\"}],\"role\":\"model\"},\"index\":0}]}\n\n";
        let chunk3 = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Final answer.\"}],\"role\":\"model\"},\"finishReason\":\"STOP\",\"index\":0}]}\n\n";

        let chunks = vec![
            Ok::<Bytes, io::Error>(Bytes::from(chunk1)),
            Ok::<Bytes, io::Error>(Bytes::from(chunk2)),
            Ok::<Bytes, io::Error>(Bytes::from(chunk3)),
        ];

        let byte_stream = stream::iter(chunks);
        let result = collect_stream_to_json(byte_stream, "test_session_thinking_merge").await;
        assert!(result.is_ok());

        let resp = result.unwrap();
        let parts = resp["candidates"][0]["content"]["parts"]
            .as_array()
            .unwrap();

        assert_eq!(
            parts.len(),
            2,
            "Expected 2 parts: 1 merged thinking part and 1 answer part, got: {:?}",
            parts
        );
        assert_eq!(parts[0]["text"], "Thinking part 1. Thinking part 2.");
        assert_eq!(parts[0]["thought"], true);
        assert_eq!(parts[0]["thoughtSignature"], "sig_valid_1234567890");
        assert_eq!(parts[1]["text"], "Final answer.");
    }
}
