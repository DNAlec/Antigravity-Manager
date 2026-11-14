// OpenAI Handler
use axum::{
    body::Body, extract::Json, extract::State, http::StatusCode, response::IntoResponse,
    response::Response,
};
use base64::Engine as _;
use bytes::Bytes;
use serde_json::{json, Value};
use std::sync::Arc;
use tracing::{debug, error, info}; // Import Engine trait for encode method

use crate::proxy::mappers::gemini::SUMMARY_REQUEST_TIMEOUT_SECS;
use crate::proxy::mappers::openai::{
    transform_openai_request, transform_openai_request_with_session, OpenAIContent,
    OpenAIContentBlock, OpenAIMessage, OpenAIRequest, OpenAIResponse,
};
use crate::proxy::pipeline::timing::{TimingBuilderExt, TimingHeaders};
// use crate::proxy::upstream::client::UpstreamClient; // 通过 state 获取
use crate::proxy::debug_logger;
use crate::proxy::pipeline::{
    extract_image_generation_diagnostic, response_has_inline_image_data,
    stream_chunk_has_image_data,
};
use crate::proxy::server::AppState;
use crate::proxy::upstream::client::mask_email;

const MAX_RETRY_ATTEMPTS: usize = 3;
const MAX_INPUT_IMAGE_BYTES: usize = 20 * 1024 * 1024;
const MAX_TOTAL_INPUT_IMAGE_BYTES: usize = 32 * 1024 * 1024;
const CODEX_VISIBLE_THOUGHT_MESSAGE_PREFIX: &str = "msg_thought_";
use super::common::{
    apply_retry_strategy, next_rotation_attempt, should_rotate_account, FailureStatusTracker,
    RequestRetryState, RetryStrategy,
};
use crate::modules::account;
use crate::proxy::common::client_adapter::CLIENT_ADAPTERS; // [NEW] Adapter Registry
use crate::proxy::session_manager::SessionManager;
use axum::http::HeaderMap;
use std::collections::VecDeque;
use std::io;
use tokio::task::JoinSet;
use tokio::time::Duration;

/// A peek timeout stops this attempt even when the mapper has buffered only
/// usage. Preserve those raw counters before dropping the unfinished stream.
fn record_nonstream_peek_timeout(
    protocol: &str,
    budget_secs: u64,
    observer: &crate::proxy::pipeline::stream_error::UpstreamStreamErrorObserver,
    failures: &mut FailureStatusTracker,
    retry_state: &mut RequestRetryState,
    mode: crate::proxy::sticky_config::SchedulingMode,
    cache_bound: bool,
) -> Option<Value> {
    crate::proxy::pipeline::terminal::record_first_byte_timeout(protocol, budget_secs, failures);
    retry_state.preserve_on_stop(mode, cache_bound);
    observer
        .failure_or_incomplete(Some("Timeout waiting for first data"))
        .and_then(|failure| failure.usage)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct NormalizedInputImage {
    mime_type: String,
    base64_data: String,
    decoded_len: usize,
}

fn validate_input_image_limits(
    _image_count: usize,
    image_bytes: usize,
    total_bytes: usize,
) -> Result<(), String> {
    if image_bytes > MAX_INPUT_IMAGE_BYTES {
        return Err(format!(
            "Input image is too large: maximum decoded size is {} bytes",
            MAX_INPUT_IMAGE_BYTES
        ));
    }
    let max_total = crate::proxy::config::get_multimodal_config()
        .max_total_image_mb
        .max(1)
        .saturating_mul(1024 * 1024);
    if total_bytes > max_total {
        return Err(format!(
            "Total input image data is too large: maximum decoded size is {} bytes",
            max_total
        ));
    }
    Ok(())
}

fn normalized_image_from_bytes(
    bytes: &[u8],
    mime_type: &str,
    image_count: usize,
    total_bytes: usize,
) -> Result<NormalizedInputImage, String> {
    let next_total = total_bytes.saturating_add(bytes.len());
    validate_input_image_limits(image_count, bytes.len(), next_total)?;
    Ok(NormalizedInputImage {
        mime_type: mime_type.to_string(),
        base64_data: base64::engine::general_purpose::STANDARD.encode(bytes),
        decoded_len: bytes.len(),
    })
}

fn parse_image_data_url(
    data_url: &str,
    image_count: usize,
    total_bytes: usize,
) -> Result<NormalizedInputImage, String> {
    let (mime_type, encoded) = parse_image_data_url_parts(data_url)?;

    let max_encoded_len = MAX_INPUT_IMAGE_BYTES.div_ceil(3).saturating_mul(4);
    if encoded.len() > max_encoded_len {
        return Err(format!(
            "Input image is too large: maximum decoded size is {} bytes",
            MAX_INPUT_IMAGE_BYTES
        ));
    }
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|_| "Input image contains invalid base64 data".to_string())?;
    normalized_image_from_bytes(&decoded, mime_type, image_count, total_bytes)
}

fn parse_image_data_url_parts(data_url: &str) -> Result<(&str, &str), String> {
    let (metadata, encoded) = data_url
        .strip_prefix("data:")
        .and_then(|value| value.split_once(','))
        .ok_or_else(|| "Input image must be a base64 data:image URL".to_string())?;
    let mut metadata_parts = metadata.split(';');
    let mime_type = metadata_parts.next().unwrap_or_default();
    if !mime_type.starts_with("image/") || mime_type.len() <= "image/".len() {
        return Err("Input image data URL must use an image MIME type".to_string());
    }
    if !metadata_parts.any(|part| part.eq_ignore_ascii_case("base64")) {
        return Err("Input image data URL must be base64 encoded".to_string());
    }
    if encoded.is_empty() {
        return Err("Input image data URL is empty".to_string());
    }
    Ok((mime_type, encoded))
}

fn parse_generation_input_images(
    image: Option<&Value>,
) -> Result<Vec<NormalizedInputImage>, String> {
    let Some(image) = image else {
        return Ok(Vec::new());
    };

    let urls: Vec<&str> = match image {
        Value::String(url) => vec![url.as_str()],
        Value::Array(urls) if urls.is_empty() => {
            return Err("Input image array must not be empty".to_string())
        }
        Value::Array(urls) => urls
            .iter()
            .map(|url| {
                url.as_str()
                    .ok_or_else(|| "Every input image must be a base64 data:image URL".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?,
        _ => return Err("Input image must be a string or an array of strings".to_string()),
    };

    validate_input_image_limits(urls.len(), 0, 0)?;
    let mut images = Vec::with_capacity(urls.len());
    let mut total_bytes = 0;
    for url in urls {
        let image = parse_image_data_url(url, images.len() + 1, total_bytes)?;
        total_bytes = total_bytes.saturating_add(image.decoded_len);
        images.push(image);
    }
    Ok(images)
}

fn generation_image_size_param(body: &Value) -> Result<Option<&str>, String> {
    for key in ["image_size", "imageSize"] {
        let Some(value) = body.get(key) else {
            continue;
        };
        if value.is_null() {
            continue;
        }
        return value
            .as_str()
            .map(Some)
            .ok_or_else(|| "Invalid image_size: expected one of 1K, 2K, 4K, or auto".to_string());
    }
    Ok(None)
}

fn is_edit_image_field(name: &str) -> bool {
    name == "image"
        || name == "image[]"
        || name.strip_prefix("image").is_some_and(|suffix| {
            !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())
        })
}

fn edit_size_input<'a>(aspect_ratio: Option<&'a str>, size: Option<&'a str>) -> Option<&'a str> {
    aspect_ratio
        .filter(|value| {
            crate::proxy::mappers::common_utils::image_aspect_ratio_from_size(value).is_some()
        })
        .or_else(|| {
            size.filter(|value| {
                crate::proxy::mappers::common_utils::image_aspect_ratio_from_size(value).is_some()
            })
        })
}

fn image_account_selection_target(model_to_use: &str) -> &str {
    model_to_use
}

fn image_inline_part(image: &NormalizedInputImage) -> Value {
    json!({
        "inlineData": {
            "mimeType": image.mime_type,
            "data": image.base64_data
        }
    })
}

fn build_image_contents(
    prompt: String,
    input_images: &[NormalizedInputImage],
    mask: Option<&NormalizedInputImage>,
) -> Vec<Value> {
    let mut parts = Vec::with_capacity(1 + input_images.len() + usize::from(mask.is_some()));
    parts.push(json!({ "text": prompt }));
    for (index, image) in input_images.iter().enumerate() {
        parts.push(image_inline_part(image));
        if index == 0 {
            if let Some(mask) = mask {
                parts.push(image_inline_part(mask));
            }
        }
    }
    if input_images.is_empty() {
        if let Some(mask) = mask {
            parts.push(image_inline_part(mask));
        }
    }
    parts
}

fn build_image_edit_body(
    project_id: String,
    resolved_model: &str,
    contents_parts: Vec<Value>,
    image_config: Value,
) -> Value {
    json!({
        "project": project_id,
        "requestId": format!("img-edit-{}", uuid::Uuid::new_v4()),
        "model": resolved_model,
        "userAgent": "antigravity",
        "requestType": "image_gen",
        "request": {
            "contents": [{
                "role": "user",
                "parts": contents_parts
            }],
            "generationConfig": {
                "candidateCount": 1,
                "imageConfig": image_config,
                "maxOutputTokens": 8192,
                "stopSequences": [],
                "temperature": 1.0,
                "topP": 0.95,
                "topK": 40
            },
            "safetySettings": [
                { "category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF" },
                { "category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF" },
                { "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF" },
                { "category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF" },
            ]
        }
    })
}

/// Return true only when a streamed chunk contains an actual error event.
///
/// Responses lifecycle envelopes legitimately contain `"error": null`, and
/// assistant text may also mention the word "error". A substring search would
/// misclassify both as failures and rotate otherwise healthy accounts.
fn stream_chunk_has_error_event(bytes: &[u8]) -> bool {
    crate::proxy::pipeline::stream_peek::has_openai_error_event(bytes)
}

fn responses_input_item_type(item: &Value) -> &str {
    item.get("type")
        .and_then(Value::as_str)
        .or_else(|| item.get("role").and_then(Value::as_str).map(|_| "message"))
        .unwrap_or("")
}

fn take_responses_input_items(body: &mut Value) -> Vec<Value> {
    match body.as_object_mut().and_then(|obj| obj.remove("input")) {
        Some(Value::Array(items)) => items,
        // Responses accepts a string as shorthand for a user message. Normalize
        // it before merging history so both the current request and stored delta
        // retain the same input, including on previous_response_id continuations.
        Some(Value::String(text)) => vec![json!({"role": "user", "content": text})],
        _ => Vec::new(),
    }
}

fn responses_item_tool_name(item: &Value) -> Option<String> {
    let name = item
        .get("name")
        .and_then(Value::as_str)
        .or_else(|| {
            item.get("function")
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
        })
        .or_else(|| item.get("tool_name").and_then(Value::as_str))
        .or_else(|| {
            item.get("tool")
                .and_then(|t| t.get("name"))
                .and_then(Value::as_str)
        })?;
    Some(match item.get("namespace").and_then(Value::as_str) {
        Some(ns) => crate::proxy::mappers::openai::request::qualify_namespace_tool_name(ns, name),
        None => name.to_owned(),
    })
}

fn push_responses_content_part(
    mut part: Value,
    text_parts: &mut Vec<String>,
    media_parts: &mut Vec<Value>,
) -> Result<(), Value> {
    let part_type = part
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    if let Some(Value::String(text)) = part.as_object_mut().and_then(|obj| obj.remove("text")) {
        text_parts.push(text);
        Ok(())
    } else if part_type == "input_image" {
        if let Some(Value::String(image_url)) =
            part.as_object_mut().and_then(|obj| obj.remove("image_url"))
        {
            media_parts.push(json!({
                "type": "image_url",
                "image_url": { "url": image_url }
            }));
            Ok(())
        } else {
            Err(part)
        }
    } else if part_type == "input_file" {
        if let Some(obj) = part.as_object_mut() {
            obj.remove("type");
        }
        media_parts.push(json!({"type":"file","file":part}));
        Ok(())
    } else if part_type == "image_url" {
        if let Some(image_url) = part.as_object_mut().and_then(|obj| obj.remove("image_url")) {
            media_parts.push(json!({
                "type": "image_url",
                "image_url": image_url
            }));
            Ok(())
        } else {
            Err(part)
        }
    } else if matches!(part_type.as_str(), "input_audio" | "audio") {
        if let Some(input_audio) = part
            .as_object_mut()
            .and_then(|obj| obj.remove("input_audio"))
        {
            media_parts.push(json!({
                "type": "input_audio",
                "input_audio": input_audio
            }));
            Ok(())
        } else {
            Err(part)
        }
    } else if part_type == "audio_url" {
        if let Some(audio_url) = part.as_object_mut().and_then(|obj| obj.remove("audio_url")) {
            media_parts.push(json!({
                "type": "audio_url",
                "audio_url": audio_url
            }));
            Ok(())
        } else {
            Err(part)
        }
    } else {
        Err(part)
    }
}

fn responses_content_parts(content: Option<Value>) -> (Vec<String>, Vec<Value>, Vec<Value>) {
    let mut text_parts = Vec::new();
    let mut media_parts = Vec::new();
    let mut unhandled_parts = Vec::new();

    match content {
        Some(Value::String(text)) => text_parts.push(text),
        Some(Value::Array(parts)) => {
            for part in parts {
                if let Err(part) =
                    push_responses_content_part(part, &mut text_parts, &mut media_parts)
                {
                    unhandled_parts.push(part);
                }
            }
        }
        Some(part @ Value::Object(_)) => {
            if let Err(part) = push_responses_content_part(part, &mut text_parts, &mut media_parts)
            {
                unhandled_parts.push(part);
            }
        }
        Some(other) => unhandled_parts.push(other),
        None => {}
    }

    (text_parts, media_parts, unhandled_parts)
}

// Keep one ordered sequence for assistant refusals, text, images and audio.
// A plain-text-only message retains the legacy string representation.
fn responses_message_content(item: &mut Value, marker: Option<String>) -> (Value, String, bool) {
    let raw = item.as_object_mut().and_then(|obj| obj.remove("content"));
    let parts = match raw {
        Some(Value::Array(parts)) => parts,
        Some(part) => vec![part],
        None => Vec::new(),
    };
    let mut ordered = Vec::new();
    let mut text = Vec::new();
    let mut run_start = 0;
    let mut has_media = false;
    let flush = |run: &[String], out: &mut Vec<Value>| {
        if !run.is_empty() {
            let text = prefix_with_step_marker(None, run.join("\n"));
            if !text.is_empty() {
                out.push(json!({"type":"text","text":text}));
            }
        }
    };
    for part in parts {
        match part {
            Value::String(s) => text.push(s),
            mut part if part["type"] == "refusal" => {
                if let Some(Value::String(s)) =
                    part.as_object_mut().and_then(|m| m.remove("refusal"))
                {
                    text.push(s);
                }
            }
            part => {
                let mut media = Vec::new();
                let _ = push_responses_content_part(part, &mut text, &mut media);
                if !media.is_empty() {
                    has_media = true;
                    flush(&text[run_start..], &mut ordered);
                    run_start = text.len();
                    ordered.append(&mut media);
                }
            }
        }
    }
    let joined = text.join("\n");
    let content = if has_media {
        flush(&text[run_start..], &mut ordered);
        Value::Array(ordered)
    } else {
        Value::String(prefix_with_step_marker(marker, joined.clone()))
    };
    (content, joined, has_media)
}

fn responses_tool_output_parts(item: &mut Value) -> (String, Vec<Value>) {
    let Some(output) = item.as_object_mut().and_then(|obj| obj.remove("output")) else {
        return (String::new(), Vec::new());
    };
    let mut output = output;
    let content = output
        .as_object_mut()
        .and_then(|obj| obj.remove("content"))
        .unwrap_or(output);

    match content {
        Value::String(text) => (text, Vec::new()),
        content @ Value::Array(_) => {
            let (text_parts, media_parts, unhandled) = responses_content_parts(Some(content));
            if text_parts.is_empty() && media_parts.is_empty() {
                (Value::Array(unhandled).to_string(), Vec::new())
            } else {
                (text_parts.join("\n"), media_parts)
            }
        }
        content @ Value::Object(_) if content.get("type").is_some() => {
            let (text_parts, media_parts, unhandled) = responses_content_parts(Some(content));
            if text_parts.is_empty() && media_parts.is_empty() {
                (
                    unhandled
                        .into_iter()
                        .next()
                        .unwrap_or(Value::Null)
                        .to_string(),
                    Vec::new(),
                )
            } else {
                (text_parts.join("\n"), media_parts)
            }
        }
        _ => (content.to_string(), Vec::new()),
    }
}

fn decoded_base64_len(encoded: &str) -> Result<usize, String> {
    let mut decoder = base64::read::DecoderReader::new(
        encoded.as_bytes(),
        &base64::engine::general_purpose::STANDARD,
    );
    let mut sink = io::sink();
    usize::try_from(
        io::copy(&mut decoder, &mut sink)
            .map_err(|_| "Input image contains invalid base64 data".to_string())?,
    )
    .map_err(|_| "Input image is too large".to_string())
}

fn validate_openai_file(
    file: &crate::proxy::mappers::openai::models::OpenAIInputFile,
) -> Result<(), String> {
    let mapped = file.to_gemini_part().ok_or_else(|| {
        "File input requires one valid inline document or public file URL".to_owned()
    })?;
    if let Some(data) = mapped["inlineData"]["data"].as_str() {
        if data.len() > MAX_INPUT_IMAGE_BYTES.div_ceil(3).saturating_mul(4) {
            return Err("Input document exceeds the media size limit".to_owned());
        }
        let decoded = decoded_base64_len(data)
            .map_err(|_| "Input document contains invalid base64 data".to_owned())?;
        if decoded > MAX_INPUT_IMAGE_BYTES {
            return Err("Input document exceeds the media size limit".to_owned());
        }
    }
    Ok(())
}

fn validate_openai_file_inputs(request: &OpenAIRequest) -> Result<(), String> {
    for message in &request.messages {
        if let Some(crate::proxy::mappers::openai::models::OpenAIContent::Array(parts)) =
            &message.content
        {
            for part in parts {
                if let crate::proxy::mappers::openai::models::OpenAIContentBlock::File { file } =
                    part
                {
                    validate_openai_file(file)?;
                }
            }
        }
    }
    Ok(())
}

fn validate_responses_image_data_url(
    data_url: &str,
    image_count: usize,
    total_bytes: usize,
) -> Result<usize, String> {
    let (_, encoded) = parse_image_data_url_parts(data_url)?;
    validate_input_image_limits(image_count, 0, total_bytes)?;
    let max_encoded_len = MAX_INPUT_IMAGE_BYTES.div_ceil(3).saturating_mul(4);
    if encoded.len() > max_encoded_len {
        return Err(format!(
            "Input image is too large: maximum decoded size is {} bytes",
            MAX_INPUT_IMAGE_BYTES
        ));
    }
    let decoded_len = decoded_base64_len(encoded)?;
    validate_input_image_limits(
        image_count,
        decoded_len,
        total_bytes.saturating_add(decoded_len),
    )?;
    Ok(decoded_len)
}

fn validate_responses_input_image_limits(input: Option<&Value>) -> Result<(), String> {
    let sliding_window_enabled =
        crate::proxy::config::get_multimodal_config().enable_sliding_window;
    validate_responses_input_image_limits_with_mode(input, sliding_window_enabled)
}

fn validate_responses_input_image_limits_with_mode(
    input: Option<&Value>,
    sliding_window_enabled: bool,
) -> Result<(), String> {
    fn visit(
        value: &Value,
        count: &mut usize,
        total: &mut usize,
        sliding_window_enabled: bool,
    ) -> Result<(), String> {
        match value {
            Value::Array(values) => {
                for value in values {
                    visit(value, count, total, sliding_window_enabled)?;
                }
            }
            Value::Object(values) => {
                if matches!(
                    values.get("type").and_then(Value::as_str),
                    Some("input_image") | Some("image_url")
                ) {
                    let image_url = values.get("image_url").and_then(|value| {
                        value
                            .as_str()
                            .or_else(|| value.get("url").and_then(Value::as_str))
                    });
                    if let Some(data_url) = image_url {
                        if !data_url.starts_with("data:") {
                            return Ok(());
                        }
                        *count = count.saturating_add(1);
                        // Pipeline First: 当启用多模态保鲜滑窗时，入口仅校验单张图片合法性与单图 20MB 上限，
                        // 累积历史图片总容量交由 InboundThinkingPipeline 滑窗与 LRU 内存兜底统一治理，
                        // 防止长程 Agent 多轮看图历史在进入流水线裁剪前被入口 32MB 门禁提前误杀。
                        let accumulated = if sliding_window_enabled { 0 } else { *total };
                        let decoded_len =
                            validate_responses_image_data_url(data_url, *count, accumulated)?;
                        if !sliding_window_enabled {
                            *total = total.saturating_add(decoded_len);
                        }
                    }
                    return Ok(());
                }
                for value in values.values() {
                    visit(value, count, total, sliding_window_enabled)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    let mut count = 0;
    let mut total = 0;
    if let Some(input) = input {
        visit(input, &mut count, &mut total, sliding_window_enabled)?;
    }
    Ok(())
}

fn historical_media_placeholder(value: &serde_json::Map<String, Value>) -> Option<Value> {
    crate::proxy::http_session_store::media_placeholder(value)
}

fn history_without_inline_media(value: &Value) -> Value {
    match value {
        Value::Array(v) => Value::Array(v.iter().map(history_without_inline_media).collect()),
        Value::Object(m) => historical_media_placeholder(m).unwrap_or_else(|| {
            Value::Object(
                m.iter()
                    .map(|(k, v)| (k.clone(), history_without_inline_media(v)))
                    .collect(),
            )
        }),
        _ => value.clone(),
    }
}

fn into_history_without_inline_media(value: Value) -> Value {
    // For the owned path, drop actual media without cloning the payload. A
    // data URL in a text field (notably custom_tool_call.input) remains text.
    fn bound(value: &mut Value) {
        match value {
            Value::Array(values) => values.iter_mut().for_each(bound),
            Value::Object(values) => {
                if let Some(placeholder) = historical_media_placeholder(values) {
                    *value = placeholder;
                } else {
                    values.values_mut().for_each(bound);
                }
            }
            _ => {}
        }
    }
    let mut value = value;
    bound(&mut value);
    value
}

async fn save_session_unless_response_cancelled<F>(
    mut ack_tx: tokio::sync::oneshot::Sender<()>,
    save: F,
) where
    F: std::future::Future<Output = ()>,
{
    let saved = tokio::select! {
        biased;
        _ = ack_tx.closed() => false,
        _ = save => true,
    };
    if saved {
        let _ = ack_tx.send(());
    }
}

fn build_responses_tool_output_content(text: String, mut media_parts: Vec<Value>) -> Value {
    if media_parts.is_empty() {
        return Value::String(text);
    }

    let mut content = Vec::with_capacity(media_parts.len() + usize::from(!text.is_empty()));
    if !text.is_empty() {
        content.push(json!({ "type": "text", "text": text }));
    }
    content.append(&mut media_parts);
    Value::Array(content)
}

fn debug_value_without_inline_data(value: &Value) -> Value {
    match value {
        Value::String(text)
            if text.starts_with("data:image/") || text.starts_with("data:audio/") =>
        {
            Value::String(format!("[inline data omitted: {} chars]", text.len()))
        }
        Value::Array(values) => {
            Value::Array(values.iter().map(debug_value_without_inline_data).collect())
        }
        Value::Object(values) => {
            let is_inline_data = values.get("mimeType").and_then(Value::as_str).is_some()
                && values.get("data").and_then(Value::as_str).is_some();
            Value::Object(
                values
                    .iter()
                    .map(|(key, value)| {
                        let value = if is_inline_data && key == "data" {
                            Value::String(format!(
                                "[inline data omitted: {} chars]",
                                value.as_str().map(str::len).unwrap_or_default()
                            ))
                        } else {
                            debug_value_without_inline_data(value)
                        };
                        (key.clone(), value)
                    })
                    .collect(),
            )
        }
        _ => value.clone(),
    }
}

#[derive(Default)]
struct JsonByteCounter(usize);

impl std::io::Write for JsonByteCounter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0 += bytes.len();
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn serialized_json_len(value: &Value) -> usize {
    let mut counter = JsonByteCounter::default();
    serde_json::to_writer(&mut counter, value)
        .map(|_| counter.0)
        .unwrap_or_default()
}

fn rewrite_terminal_assistant_prefill(messages: &mut [Value]) -> bool {
    let Some(last_message) = messages.last_mut() else {
        return false;
    };

    if last_message.get("role").and_then(Value::as_str) != Some("assistant") {
        return false;
    }

    let has_nonempty_plain_text = last_message
        .get("content")
        .and_then(Value::as_str)
        .is_some_and(|content| !content.trim().is_empty());
    if !has_nonempty_plain_text {
        return false;
    }

    let has_tool_calls = match last_message.get("tool_calls") {
        None | Some(Value::Null) => false,
        Some(Value::Array(tool_calls)) => !tool_calls.is_empty(),
        Some(_) => true,
    };
    if has_tool_calls {
        return false;
    }

    let Some(message) = last_message.as_object_mut() else {
        return false;
    };
    message.insert("role".to_string(), Value::String("user".to_string()));
    true
}

fn is_tool_history_message(message: &Value) -> bool {
    match message.get("role").and_then(Value::as_str) {
        Some("tool" | "function") => true,
        Some("assistant") => message
            .get("tool_calls")
            .and_then(Value::as_array)
            .is_some_and(|tool_calls| !tool_calls.is_empty()),
        _ => false,
    }
}

fn drop_leading_orphan_tool_history(messages: &mut Vec<Value>) -> usize {
    let conversation_start = messages
        .iter()
        .position(|message| message.get("role").and_then(Value::as_str) != Some("system"))
        .unwrap_or(messages.len());
    let orphan_count = messages[conversation_start..]
        .iter()
        .take_while(|message| is_tool_history_message(message))
        .count();

    if orphan_count > 0 {
        messages.drain(conversation_start..conversation_start + orphan_count);
    }
    orphan_count
}

fn openai_content_text(content: &OpenAIContent) -> String {
    match content {
        OpenAIContent::String(text) => text.clone(),
        OpenAIContent::Array(parts) => parts
            .iter()
            .filter_map(|part| match part {
                OpenAIContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect(),
    }
}

fn responses_usage_value(chat_response: &OpenAIResponse) -> Value {
    chat_response
        .usage
        .as_ref()
        .map(|usage| usage.to_responses_usage_value())
        .unwrap_or_else(|| {
            json!({
                "input_tokens": 0,
                "input_tokens_details": { "cached_tokens": 0 },
                "output_tokens": 0,
                "output_tokens_details": { "reasoning_tokens": 0 },
                "total_tokens": 0
            })
        })
}

/// 剥离最外层 Markdown 代码块围栏（如 ```json ... ```），确保 Codex Desktop 原生 JSON.parse 正常解析。
/// 严格执行整包匹配（starts_with 与 ends_with），绝不在正文中进行贪婪查找或截断，彻底避免误杀。
pub(crate) fn unwrap_outer_code_fence_if_structured(text: &str) -> String {
    let trimmed = text.trim();
    // 1. 若本身已经是合法 JSON，直接原样返回
    if serde_json::from_str::<serde_json::Value>(trimmed).is_ok() {
        return trimmed.to_string();
    }

    // 2. 仅当整段输出被 ``` ... ``` 严格包裹时执行整包剥离
    if trimmed.starts_with("```") && trimmed.ends_with("```") && trimmed.len() >= 6 {
        if let Some(first_newline) = trimmed.find('\n') {
            let inner = trimmed[first_newline + 1..trimmed.len() - 3].trim();
            if serde_json::from_str::<serde_json::Value>(inner).is_ok() {
                return inner.to_string();
            }
        }
    }

    // 3. 其余任何情况（包括常规对话文本、内嵌 Markdown 代码块等）均保持原样返回
    text.to_string()
}

#[inline]
pub(crate) fn is_request_structured_output(req: &OpenAIRequest) -> bool {
    req.response_format
        .as_ref()
        .is_some_and(|f| f.r#type == "json_schema" || f.r#type == "json_object")
        || req.text.as_ref().and_then(|t| t.get("format")).is_some()
}

#[cfg(test)]
fn convert_chat_response_to_responses(
    chat_response: &OpenAIResponse,
    is_structured_output: bool,
) -> Value {
    convert_chat_response_to_responses_with_tools(
        chat_response,
        is_structured_output,
        &Default::default(),
    )
}

fn convert_chat_response_to_responses_with_tools(
    chat_response: &OpenAIResponse,
    is_structured_output: bool,
    tool_kinds: &crate::proxy::mappers::openai::request::ResponsesToolKinds,
) -> Value {
    let mut output = Vec::new();

    for choice in &chat_response.choices {
        if let Some(reasoning) = choice
            .message
            .reasoning_content
            .as_deref()
            .filter(|reasoning| !reasoning.trim().is_empty())
        {
            output.push(json!({
                "id": format!("rs_{}", uuid::Uuid::new_v4().simple()),
                "type": "reasoning",
                "status": "completed",
                "summary": [{ "type": "summary_text", "text": reasoning }]
            }));
        }

        let raw_text = choice
            .message
            .content
            .as_ref()
            .map(openai_content_text)
            .unwrap_or_default();
        let cleaned_text = if is_structured_output {
            unwrap_outer_code_fence_if_structured(&raw_text)
        } else {
            raw_text
        };
        let refusal = choice
            .message
            .refusal
            .as_deref()
            .filter(|refusal| !refusal.is_empty());
        if !cleaned_text.is_empty() || refusal.is_some() {
            let mut content = Vec::new();
            if !cleaned_text.is_empty() {
                content.push(json!({
                    "type": "output_text",
                    "text": cleaned_text,
                    "annotations": []
                }));
            }
            if let Some(refusal) = refusal {
                content.push(json!({ "type": "refusal", "refusal": refusal }));
            }
            output.push(json!({
                "id": format!("msg_{}", uuid::Uuid::new_v4().simple()),
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": content
            }));
        }

        for tool_call in choice.message.tool_calls.iter().flatten() {
            let Some(function) = tool_call.function.as_ref() else {
                tracing::warn!(
                    tool_call_id = %tool_call.id,
                    "[Responses Compat] Skipping tool call without function payload"
                );
                continue;
            };
            let call_id = tool_call.call_id.as_deref().unwrap_or(&tool_call.id);
            let args = serde_json::from_str::<Value>(&function.arguments).ok();
            output.push(
                crate::proxy::mappers::openai::streaming::responses_tool_call_item(
                    &function.name,
                    args.as_ref(),
                    &function.arguments,
                    &format!("fc_{}", uuid::Uuid::new_v4().simple()),
                    call_id,
                    tool_kinds,
                ),
            );
        }
    }

    // The internal Chat stream already carries the upstream terminal reason.
    // Preserve it when returning Responses JSON, just as the Responses stream
    // does, rather than turning a truncated/refused answer into a completion.
    let refused = chat_response
        .choices
        .iter()
        .any(|choice| choice.finish_reason.as_deref() == Some("content_filter"));
    let length_limited = chat_response
        .choices
        .iter()
        .any(|choice| choice.finish_reason.as_deref() == Some("length"));
    let missing_finish = chat_response.choices.iter().any(|choice| {
        choice
            .finish_reason
            .as_deref()
            .filter(|reason| !reason.is_empty())
            .is_none()
    });
    let missing_actionable_output = !output.iter().any(|item| {
        matches!(
            item["type"].as_str(),
            Some("message" | "function_call" | "custom_tool_call")
        )
    });
    let incomplete = refused || length_limited || missing_finish || missing_actionable_output;
    let incomplete_details = if incomplete {
        json!({"reason": if refused {
            "content_filter"
        } else if length_limited {
            "max_output_tokens"
        } else {
            "interrupted"
        }})
    } else {
        Value::Null
    };
    let error = if refused {
        Value::Null
    } else if missing_actionable_output {
        json!({"code":"empty_response","message":"Gemini stream ended without a final assistant message or tool call."})
    } else if missing_finish {
        json!({"code":"upstream_interrupted","message":"Gemini stream ended without finishReason."})
    } else {
        Value::Null
    };

    json!({
        "id": format!("resp_{}", uuid::Uuid::new_v4().simple()),
        "object": "response",
        "type": "response",
        "created_at": chrono::Utc::now().timestamp(),
        "status": if incomplete { "incomplete" } else { "completed" },
        "incomplete_details": incomplete_details,
        "error": error,
        "output": output,
        "model": chat_response.model,
        "usage": responses_usage_value(chat_response)
    })
}

#[cfg(test)]
mod responses_conversion_terminal_tests {
    use super::convert_chat_response_to_responses;
    use crate::proxy::mappers::openai::{models::OpenAIResponse, streaming};
    use bytes::Bytes;
    use futures::{Stream, StreamExt};
    use serde_json::{json, Value};
    use std::pin::Pin;

    fn chat_response(finish_reason: Value, message: Value) -> OpenAIResponse {
        serde_json::from_value(json!({
            "id":"chat_test", "object":"chat.completion", "created":1, "model":"gemini-pro",
            "choices":[{"index":0,"message":message,"finish_reason":finish_reason}],
            "usage":{"prompt_tokens":100,"completion_tokens":7,"total_tokens":107,
                "prompt_tokens_details":{"cached_tokens":80},
                "completion_tokens_details":{"reasoning_tokens":2}}
        }))
        .unwrap()
    }

    #[test]
    fn responses_json_terminal_reasons_preserve_output_and_usage() {
        for (finish, expected_status, expected_reason) in [
            ("stop", "completed", Value::Null),
            ("tool_calls", "completed", Value::Null),
            ("length", "incomplete", json!("max_output_tokens")),
            ("content_filter", "incomplete", json!("content_filter")),
        ] {
            for with_tool in [false, true] {
                let mut message =
                    json!({"role":"assistant","content":"answer","reasoning_content":"thought"});
                if with_tool {
                    message["tool_calls"] = json!([{"id":"call_test","type":"function",
                        "function":{"name":"lookup","arguments":"{\"key\":1}"}}]);
                }
                let response = convert_chat_response_to_responses(
                    &chat_response(json!(finish), message),
                    false,
                );
                assert_eq!(response["status"], expected_status, "{finish}/{with_tool}");
                assert_eq!(response["incomplete_details"]["reason"], expected_reason);
                assert!(response["error"].is_null());
                assert_eq!(response["output"][0]["summary"][0]["text"], "thought");
                assert_eq!(response["output"][1]["content"][0]["text"], "answer");
                if with_tool {
                    assert_eq!(response["output"][2]["call_id"], "call_test");
                    assert_eq!(response["output"][2]["arguments"], "{\"key\":1}");
                }
                assert_eq!(
                    response["usage"],
                    json!({"input_tokens":100,
                    "input_tokens_details":{"cached_tokens":80},"output_tokens":7,
                    "output_tokens_details":{"reasoning_tokens":2},"total_tokens":107})
                );
            }
        }
    }

    #[test]
    fn responses_json_retains_refusal_without_inventing_content() {
        for refusal in [None, Some("provider refusal")] {
            let mut message = json!({"role":"assistant","content":null});
            if let Some(refusal) = refusal {
                message["refusal"] = json!(refusal);
            }
            let response = convert_chat_response_to_responses(
                &chat_response(json!("content_filter"), message),
                false,
            );
            assert_eq!(response["status"], "incomplete");
            assert_eq!(response["incomplete_details"]["reason"], "content_filter");
            assert!(response["error"].is_null());
            if let Some(refusal) = refusal {
                assert_eq!(
                    response["output"][0]["content"],
                    json!([{"type":"refusal","refusal":refusal}])
                );
            } else {
                assert_eq!(response["output"], json!([]));
            }
        }
    }

    #[test]
    fn responses_json_missing_finish_or_answer_is_incomplete() {
        let cases = [
            (
                Value::Null,
                json!({"role":"assistant","content":"partial"}),
                "upstream_interrupted",
            ),
            (
                json!(""),
                json!({"role":"assistant","content":"partial"}),
                "upstream_interrupted",
            ),
            (
                json!("stop"),
                json!({"role":"assistant","content":null,"reasoning_content":"thought"}),
                "empty_response",
            ),
        ];
        for (finish, message, error) in cases {
            let response =
                convert_chat_response_to_responses(&chat_response(finish, message), false);
            assert_eq!(response["status"], "incomplete");
            assert_eq!(response["incomplete_details"]["reason"], "interrupted");
            assert_eq!(response["error"]["code"], error);
        }
    }

    type Source = Pin<Box<dyn Stream<Item = Result<Bytes, String>> + Send>>;
    fn source(frames: Vec<Value>) -> Source {
        Box::pin(futures::stream::iter(
            frames
                .into_iter()
                .map(|value| Ok(Bytes::from(format!("data: {value}\n\n")))),
        ))
    }

    #[tokio::test]
    async fn responses_media_survives_stream_buffering_and_direct_json() {
        const PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
        let media_cases = [
            (
                json!({"inlineData":{"mimeType":"image/png","data":PNG}}),
                PNG,
            ),
            (
                json!({"fileData":{"mimeType":"image/png","fileUri":"https://files.example/answer.png"}}),
                "https://files.example/answer.png",
            ),
            (
                json!({"fileData":{"mimeType":"application/pdf","fileUri":"https://files.example/report.pdf"}}),
                "https://files.example/report.pdf",
            ),
        ];
        for (media, expected) in media_cases {
            for with_text in [false, true] {
                let mut parts = Vec::new();
                if with_text {
                    parts.push(json!({"text":"The generated result follows."}));
                }
                parts.push(media.clone());
                let response = json!({
                    "candidates":[{"index":0,"content":{"role":"model","parts":parts},"finishReason":"STOP"}],
                    "usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":7,"totalTokenCount":107}
                });
                let direct_chat =
                    crate::proxy::mappers::openai::response::transform_openai_response(
                        &response, None, 1, None,
                    );
                let buffered_chat =
                    crate::proxy::mappers::openai::collector::collect_stream_to_json(
                        streaming::create_openai_sse_stream(
                            source(vec![response.clone()]),
                            "gemini-pro".into(),
                            "media-parity-buffered".into(),
                            1,
                            None,
                            true,
                        ),
                    )
                    .await
                    .unwrap();
                let mut responses_stream = streaming::create_codex_sse_stream(
                    source(vec![response]),
                    "gemini-pro".into(),
                    "media-parity-stream".into(),
                    1,
                    1,
                    "resp_media_parity".into(),
                    None,
                    false,
                );
                let mut terminal = None;
                while let Some(chunk) = responses_stream.next().await {
                    for line in String::from_utf8_lossy(&chunk.unwrap()).lines() {
                        let Some(data) = line.strip_prefix("data: ") else {
                            continue;
                        };
                        let Ok(event) = serde_json::from_str::<Value>(data) else {
                            continue;
                        };
                        assert_ne!(event["type"], "error", "{event}");
                        if event["type"] == "response.completed" {
                            terminal = Some(event["response"].clone());
                        }
                    }
                }
                for result in [
                    convert_chat_response_to_responses(&direct_chat, false),
                    convert_chat_response_to_responses(&buffered_chat, false),
                    terminal.expect("media response must complete"),
                ] {
                    assert_eq!(result["status"], "completed", "{result}");
                    let output = result["output"].to_string();
                    assert!(output.contains(expected), "missing media: {output}");
                    if with_text {
                        assert!(output.contains("The generated result follows."), "{output}");
                    }
                    assert_eq!(result["usage"]["input_tokens"], 100);
                    assert_eq!(result["usage"]["output_tokens"], 7);
                }
            }
        }
    }

    #[tokio::test]
    async fn responses_json_matches_stream_for_text_tool_refusal_and_usage_tail() {
        let tool = json!({"functionCall":{"name":"lookup","args":{"key":1},"id":"call_test"}});
        let cases = [
            (
                "stop",
                json!({"candidates":[{"content":{"parts":[{"text":"answer"}]},"finishReason":"STOP"}]}),
            ),
            (
                "tool_calls",
                json!({"candidates":[{"content":{"parts":[tool.clone()]},"finishReason":"STOP"}]}),
            ),
            (
                "length",
                json!({"candidates":[{"content":{"parts":[{"text":"partial"}]},"finishReason":"MAX_TOKENS"}]}),
            ),
            (
                "length",
                json!({"candidates":[{"content":{"parts":[tool.clone()]},"finishReason":"MAX_TOKENS"}]}),
            ),
            (
                "content_filter",
                json!({"candidates":[{"finishReason":"SAFETY"}]}),
            ),
            (
                "content_filter",
                json!({"candidates":[{"content":{"parts":[tool]},"finishReason":"SAFETY"}]}),
            ),
            (
                "content_filter",
                json!({"candidates":[{"content":{"parts":[{"text":"thought","thought":true}]},"finishReason":"RECITATION"}]}),
            ),
            (
                "content_filter",
                json!({"candidates":[{"finishReason":"MODEL_ARMOR"}]}),
            ),
            (
                "content_filter",
                json!({"promptFeedback":{"blockReason":"OTHER"}}),
            ),
        ];
        for (finish, candidate) in cases {
            for cache in [80, 0] {
                let mut candidate = candidate.clone();
                candidate["usageMetadata"] = json!({"promptTokenCount":100,"cachedContentTokenCount":80,
                    "candidatesTokenCount":5,"thoughtsTokenCount":2});
                let frames = vec![
                    candidate,
                    json!({"usageMetadata":{"cachedContentTokenCount":cache}}),
                ];
                let chat_stream = streaming::create_openai_sse_stream(
                    source(frames.clone()),
                    "gemini-pro".into(),
                    "terminal-json".into(),
                    1,
                    None,
                    true,
                );
                let chat =
                    crate::proxy::mappers::openai::collector::collect_stream_to_json(chat_stream)
                        .await
                        .unwrap();
                assert_eq!(chat.choices[0].finish_reason.as_deref(), Some(finish));
                let converted = convert_chat_response_to_responses(&chat, false);
                let mut responses = streaming::create_codex_sse_stream(
                    source(frames),
                    "gemini-pro".into(),
                    "terminal-stream".into(),
                    1,
                    1,
                    "resp_terminal".into(),
                    None,
                    false,
                );
                let mut terminal = None;
                while let Some(chunk) = responses.next().await {
                    for line in String::from_utf8_lossy(&chunk.unwrap()).lines() {
                        if let Some(data) = line.strip_prefix("data: ") {
                            if let Ok(event) = serde_json::from_str::<Value>(data) {
                                if matches!(
                                    event["type"].as_str(),
                                    Some(
                                        "response.completed"
                                            | "response.incomplete"
                                            | "response.failed"
                                    )
                                ) {
                                    terminal = Some(event["response"].clone());
                                }
                            }
                        }
                    }
                }
                let terminal = terminal.expect("Responses terminal frame");
                for key in ["status", "incomplete_details", "error", "usage"] {
                    assert_eq!(
                        converted[key], terminal[key],
                        "{finish}/cache={cache}/{key}"
                    );
                }
                assert_eq!(
                    converted["usage"]["input_tokens_details"]["cached_tokens"],
                    cache
                );
                assert_eq!(converted["usage"]["output_tokens"], 7);
                let json_types: Vec<_> = converted["output"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|item| item["type"].clone())
                    .collect();
                let stream_types: Vec<_> = terminal["output"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|item| item["type"].clone())
                    .collect();
                assert_eq!(json_types, stream_types);
            }
        }
    }
}

/// 仅识别客户端本地私有展示项（如本地渲染的思考块），绝不误杀伴随工具调用的真实过程进度说明。
/// 只有明确以 `msg_thought_` 为 ID 前缀或历史遗留以 `**Thinking**` 开头的消息才被视为 transcript-only。
/// 真实的 `phase="commentary"` 消息包含正文说明，必须作为上下文历史保留以保证模型少样本进度输出范式。
fn is_codex_transcript_only_assistant_message(item: &Value, text: &str) -> bool {
    if responses_input_item_type(item) != "message"
        || item.get("role").and_then(Value::as_str) != Some("assistant")
    {
        return false;
    }

    item.get("id")
        .and_then(Value::as_str)
        .is_some_and(|id| id.starts_with(CODEX_VISIBLE_THOUGHT_MESSAGE_PREFIX))
        || text.trim_start().starts_with("**Thinking**")
}

#[cfg(test)]
mod stream_peek_tests {
    use super::build_image_contents;
    use super::build_image_edit_body;
    use super::convert_chat_response_to_responses;
    use super::convert_codex_to_openai_request;
    use super::drop_leading_orphan_tool_history;
    use super::edit_size_input;
    use super::extract_image_generation_diagnostic;
    use super::generation_image_size_param;
    use super::history_without_inline_media;
    use super::image_account_selection_target;
    use super::into_history_without_inline_media;
    use super::is_codex_transcript_only_assistant_message;
    use super::is_edit_image_field;
    use super::is_responses_api_path;
    use super::parse_generation_input_images;
    use super::response_has_inline_image_data;
    use super::responses_input_item_type;
    use super::responses_message_content;
    use super::responses_routing_session_id;
    use super::rewrite_terminal_assistant_prefill;
    use super::save_session_unless_response_cancelled;
    use super::stream_chunk_has_error_event;
    use super::stream_chunk_has_image_data;
    use super::validate_input_image_limits;
    use super::validate_responses_image_data_url;
    use super::validate_responses_input_image_limits_with_mode;
    use super::{MAX_INPUT_IMAGE_BYTES, MAX_TOTAL_INPUT_IMAGE_BYTES};
    use crate::proxy::mappers::openai::{transform_openai_request, OpenAIRequest};
    use serde_json::{json, Value};

    #[test]
    fn responses_routing_identity_follows_the_response_chain() {
        let first = responses_routing_session_id(None, None, None, "resp-root-a");
        let second = responses_routing_session_id(None, None, None, "resp-root-b");
        assert_eq!(first, "resp-root-a");
        assert_eq!(second, "resp-root-b");
        assert_ne!(first, second);

        let continued =
            responses_routing_session_id(None, Some("resp-parent"), Some(&first), "resp-child");
        let branch =
            responses_routing_session_id(None, Some("resp-parent"), Some(&first), "resp-branch");
        assert_eq!(continued, first);
        assert_eq!(branch, first);
        assert_eq!(
            responses_routing_session_id(
                Some("client-session"),
                Some("resp-parent"),
                Some(&first),
                "resp-child"
            ),
            "client-session"
        );
        assert_eq!(
            responses_routing_session_id(None, Some("resp-missing"), None, "resp-child"),
            "resp-missing"
        );
    }

    #[test]
    fn responses_store_defaults_to_enabled_and_honors_explicit_false() {
        assert!(super::responses_store_enabled(&json!({})));
        assert!(super::responses_store_enabled(&json!({"store": true})));
        assert!(super::responses_store_enabled(&json!({"store": null})));
        assert!(!super::responses_store_enabled(&json!({"store": false})));
    }

    #[test]
    fn responses_api_path_matches_v1_and_root_aliases() {
        assert!(is_responses_api_path("/v1/responses"));
        assert!(is_responses_api_path("/responses"));
        assert!(is_responses_api_path("/responses/compact"));
        assert!(!is_responses_api_path("/v1/chat/completions"));
        assert!(!is_responses_api_path("/v1/completions"));
        assert!(!is_responses_api_path("/v1beta/models"));
    }

    #[test]
    fn responses_created_with_null_error_is_not_an_error_event() {
        let chunk = br#"event: response.created
data: {"type":"response.created","response":{"status":"in_progress","error":null}}

"#;
        assert!(!stream_chunk_has_error_event(chunk));
    }

    #[test]
    fn response_failed_is_an_error_event() {
        let chunk = br#"event: response.failed
data: {"type":"response.failed","response":{"status":"failed","error":{"code":"upstream_error"}}}

"#;
        assert!(stream_chunk_has_error_event(chunk));
    }

    #[test]
    fn legacy_top_level_error_is_an_error_event() {
        let chunk = br#"data: {"error":{"message":"quota exceeded"}}

"#;
        assert!(stream_chunk_has_error_event(chunk));
    }

    #[test]
    fn normal_text_containing_error_is_not_an_error_event() {
        let chunk = br#"data: {"type":"response.output_text.delta","delta":"The JSON key is called \"error\"."}

"#;
        assert!(!stream_chunk_has_error_event(chunk));
    }

    #[test]
    fn task_image_success_requires_nonempty_payload() {
        let empty = json!({
            "response": {"candidates": [{"content": {"parts": [{"inlineData": {"data": ""}}]}}]}
        });
        let image = json!({
            "response": {"candidates": [{"content": {"parts": [{"inlineData": {"data": "AQ=="}}]}}]}
        });
        assert!(!response_has_inline_image_data(&empty));
        assert!(response_has_inline_image_data(&image));

        let empty_chunk =
            br#"data: {"choices":[{"delta":{"content":"![image](data:image/png;base64,)"}}]}

"#;
        let image_chunk =
            br#"data: {"choices":[{"delta":{"content":"![image](data:image/png;base64,AQ==)"}}]}

"#;
        assert!(!stream_chunk_has_image_data(empty_chunk));
        assert!(stream_chunk_has_image_data(image_chunk));
    }

    #[test]
    fn test_extract_image_generation_diagnostic() {
        use reqwest::StatusCode;

        // 1. promptFeedback 安全策略拦截
        let prompt_feedback = json!({
            "promptFeedback": {
                "blockReason": "SAFETY"
            }
        });
        let diag = extract_image_generation_diagnostic(&prompt_feedback);
        assert_eq!(diag.status, StatusCode::BAD_REQUEST);
        assert!(diag.is_safety_or_policy);
        assert!(diag.message.contains("blockReason: SAFETY"));

        // 2. candidate finishReason 安全策略拦截
        let cand_safety = json!({
            "candidates": [{
                "finishReason": "SAFETY"
            }]
        });
        let diag = extract_image_generation_diagnostic(&cand_safety);
        assert_eq!(diag.status, StatusCode::BAD_REQUEST);
        assert!(diag.is_safety_or_policy);
        assert!(diag.message.contains("finishReason: SAFETY"));

        // 3. candidate 文本拒答部件
        let text_refusal = json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": "I cannot create images of real people."
                    }]
                },
                "finishReason": "STOP"
            }]
        });
        let diag = extract_image_generation_diagnostic(&text_refusal);
        assert_eq!(diag.status, StatusCode::BAD_REQUEST);
        assert!(diag.is_safety_or_policy);
        assert!(diag
            .message
            .contains("I cannot create images of real people."));

        // 4. 空候选集
        let empty_cand = json!({
            "candidates": []
        });
        let diag = extract_image_generation_diagnostic(&empty_cand);
        assert_eq!(diag.status, StatusCode::BAD_GATEWAY);
        assert!(!diag.is_safety_or_policy);
        assert!(diag.message.contains("empty candidates array"));

        // 5. 嵌套 upstream error
        let upstream_err = json!({
            "error": {
                "message": "Resource exhausted"
            }
        });
        let diag = extract_image_generation_diagnostic(&upstream_err);
        assert_eq!(diag.status, StatusCode::BAD_GATEWAY);
        assert!(!diag.is_safety_or_policy);
        assert!(diag.message.contains("Resource exhausted"));
    }

    #[test]
    fn responses_compat_accepts_optional_type_and_string_or_array_content() {
        let string_message = json!({
            "role": "system",
            "content": "Follow the system instructions."
        });
        let array_message = json!({
            "role": "user",
            "content": [{"type": "input_text", "text": "Continue planning."}]
        });

        assert_eq!(responses_input_item_type(&string_message), "message");
        assert_eq!(
            responses_message_content(&mut string_message.clone(), None).0,
            "Follow the system instructions."
        );
        assert_eq!(
            responses_message_content(&mut array_message.clone(), None).0,
            "Continue planning."
        );

        let converted = convert_codex_to_openai_request(json!({
            "input": [
                string_message,
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Previous output."}]
                },
                array_message
            ]
        }));

        assert_eq!(
            converted["messages"],
            json!([
                {"role": "system", "content": "Follow the system instructions."},
                {"role": "assistant", "content": "Previous output."},
                {"role": "user", "content": "Continue planning."}
            ])
        );
    }

    #[test]
    fn chat_responses_compat_preserves_multiturn_roles_in_input_array() {
        let converted = convert_codex_to_openai_request(json!({
            "model": "gemini-2.5-pro",
            "instructions": "Stay in character.",
            "input": [
                {"role": "user", "content": "Who are you?"},
                {"role": "assistant", "content": "I am Character A."},
                {"role": "user", "content": "Describe the room."},
                {"role": "assistant", "content": "There is a table."},
                {"role": "user", "content": "What color is the window?"}
            ]
        }));

        assert_eq!(
            converted["messages"],
            json!([
                {"role": "system", "content": "Stay in character."},
                {"role": "user", "content": "Who are you?"},
                {"role": "assistant", "content": "I am Character A."},
                {"role": "user", "content": "Describe the room."},
                {"role": "assistant", "content": "There is a table."},
                {"role": "user", "content": "What color is the window?"}
            ])
        );
        assert!(converted.get("input").is_none());
        assert!(converted.get("instructions").is_none());

        let request: OpenAIRequest =
            serde_json::from_value(converted).expect("converted Chat request");
        let (upstream, _, _, _) =
            transform_openai_request(&request, "test-project", "gemini-2.5-pro", None).unwrap();
        let roles = upstream["request"]["contents"]
            .as_array()
            .expect("Gemini contents")
            .iter()
            .map(|content| content["role"].as_str().expect("content role"))
            .collect::<Vec<_>>();
        assert_eq!(roles, ["user", "model", "user", "model", "user"]);
    }

    #[test]
    fn chat_responses_compat_keeps_string_input() {
        let converted = convert_codex_to_openai_request(json!({
            "model": "gemini-2.5-pro",
            "instructions": "Answer briefly.",
            "input": "What is the next step?"
        }));

        assert_eq!(
            converted["messages"],
            json!([
                {"role": "system", "content": "Answer briefly."},
                {"role": "user", "content": "What is the next step?"}
            ])
        );
    }

    #[test]
    fn responses_tool_output_image_is_sent_as_inline_data() {
        let converted = convert_codex_to_openai_request(json!({
            "model": "gemini-3.7-flash-high",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Generate an image."}]
                },
                {
                    "type": "function_call",
                    "call_id": "call_image",
                    "name": "view_image",
                    "arguments": "{}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_image",
                    "output": [
                        {"type": "input_text", "text": "image generated"},
                        {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="}
                    ]
                }
            ]
        }));
        assert!(converted.get("input").is_none());
        let request: OpenAIRequest =
            serde_json::from_value(converted).expect("valid OpenAI request");

        let (upstream, _, _, _) =
            transform_openai_request(&request, "project", "gemini-3.7-flash-high", None).unwrap();
        let parts = upstream["request"]["contents"]
            .as_array()
            .expect("contents")
            .iter()
            .flat_map(|content| content["parts"].as_array().expect("content parts").iter())
            .collect::<Vec<_>>();
        let function_response = parts
            .iter()
            .find(|part| part.get("functionResponse").is_some())
            .expect("function response");
        let inline_data = function_response["functionResponse"]["parts"]
            .as_array()
            .expect("tool-owned media")
            .iter()
            .find(|part| part.get("inlineData").is_some())
            .expect("inline image data");
        assert_eq!(function_response["functionResponse"]["id"], "call_image");

        assert_eq!(
            function_response["functionResponse"]["response"]["output"],
            "image generated"
        );
        assert!(!function_response.to_string().contains("data:image/"));
        assert_eq!(inline_data["inlineData"]["mimeType"], "image/png");
        assert_eq!(inline_data["inlineData"]["data"], "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==");
    }

    #[test]
    fn pure_image_tool_history_keeps_small_placeholder() {
        let history = json!({
            "type": "function_call_output",
            "call_id": "call_image",
            "output": [{
                "type": "input_image",
                "image_url": "data:image/png;base64,AQ=="
            }]
        });
        let expected = json!({
            "type": "function_call_output",
            "call_id": "call_image",
            "output": [{
                "type": "input_text",
                "text": "[historical image omitted]"
            }]
        });

        let borrowed = history_without_inline_media(&history);
        let moved = into_history_without_inline_media(history);
        assert_eq!(borrowed, moved);
        let mut visible = moved;
        let digest = visible["output"][0]
            .as_object_mut()
            .unwrap()
            .remove("_am_media_digest")
            .unwrap();
        assert_eq!(digest.as_str().unwrap().len(), 64);
        assert_eq!(visible, expected);
        assert!(!borrowed.to_string().contains("base64"));
    }

    #[test]
    fn responses_data_url_metadata_and_byte_limits() {
        assert_eq!(
            validate_responses_image_data_url("data:image/png;BASE64,AQ==", 1, 0)
                .expect("uppercase base64 token"),
            1
        );
        let encoded = "AAAA".repeat(MAX_INPUT_IMAGE_BYTES / 3 + 1);
        assert_eq!(
            validate_responses_image_data_url(&format!("data:image/png;BASE64,{encoded}"), 1, 0)
                .unwrap_err(),
            format!(
                "Input image is too large: maximum decoded size is {} bytes",
                MAX_INPUT_IMAGE_BYTES
            )
        );
        assert!(validate_responses_image_data_url(
            "data:image/png;BASE64,AQ==",
            1,
            MAX_TOTAL_INPUT_IMAGE_BYTES
        )
        .unwrap_err()
        .starts_with("Total input image data is too large"));
        assert!(
            validate_input_image_limits(100, 2 * 1024 * 1024, MAX_TOTAL_INPUT_IMAGE_BYTES).is_ok()
        );
        assert!(
            validate_input_image_limits(100, 2 * 1024 * 1024, MAX_TOTAL_INPUT_IMAGE_BYTES + 1)
                .is_err()
        );

        // 当开启多模态保鲜滑窗时，多张合法图片累积超过 32MB 应在入口放行交由 Pipeline 裁剪；
        // 当关闭滑窗时，入口应严格拦截累积超限。
        let chunk_b64 = "AAAA".repeat((12 * 1024 * 1024) / 3); // 单张 12MB (< 20MB 单图上限)
        let data_url = format!("data:image/png;base64,{chunk_b64}");
        let multi_turn_input = json!([
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": [{ "type": "input_image", "image_url": data_url.clone() }]
            },
            {
                "type": "function_call_output",
                "call_id": "call_2",
                "output": [{ "type": "input_image", "image_url": data_url.clone() }]
            },
            {
                "type": "function_call_output",
                "call_id": "call_3",
                "output": [{ "type": "input_image", "image_url": data_url }]
            }
        ]);
        assert!(
            validate_responses_input_image_limits_with_mode(Some(&multi_turn_input), true).is_ok(),
            "Sliding window enabled should defer cumulative historical total to InboundThinkingPipeline"
        );
        assert!(
            validate_responses_input_image_limits_with_mode(Some(&multi_turn_input), false)
                .unwrap_err()
                .starts_with("Total input image data is too large"),
            "Sliding window disabled must enforce cumulative limit at ingress"
        );
    }

    #[tokio::test]
    async fn cancelled_response_drops_pending_session_save() {
        let response_id = format!("resp-cancelled-{}", uuid::Uuid::new_v4());
        let save_response_id = response_id.clone();
        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let save_task = tokio::spawn(save_session_unless_response_cancelled(ack_tx, async move {
            entered_tx.send(()).expect("save future entered");
            std::future::pending::<()>().await;
            crate::proxy::http_session_store::save_session_delta(
                save_response_id,
                None,
                vec![json!({"id": "cancelled-user"})],
                Vec::new(),
                String::new(),
                "gemini-pro-agent".to_string(),
                "routing-cancelled".to_string(),
            )
            .await;
        }));

        entered_rx.await.expect("save future is waiting");
        drop(ack_rx);
        tokio::time::timeout(std::time::Duration::from_secs(1), save_task)
            .await
            .expect("cancelled save task exits")
            .expect("save task");
        assert!(crate::proxy::http_session_store::get_session(&response_id)
            .await
            .is_none());
    }

    #[test]
    fn responses_compat_rewrites_only_terminal_plain_text_assistant_prefill() {
        let mut plain_text = vec![json!({
            "role": "assistant",
            "content": "Choose the next action."
        })];
        assert!(rewrite_terminal_assistant_prefill(&mut plain_text));
        assert_eq!(plain_text[0]["role"], "user");

        let unchanged_messages = [
            json!({"role": "assistant", "content": "   "}),
            json!({
                "role": "assistant",
                "content": [{
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AA=="}
                }]
            }),
            json!({
                "role": "assistant",
                "content": "Call a tool.",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "query_memory", "arguments": "{}"}
                }]
            }),
        ];
        for original in unchanged_messages {
            let mut messages = vec![original.clone()];
            assert!(!rewrite_terminal_assistant_prefill(&mut messages));
            assert_eq!(messages[0], original);
        }

        let mut non_terminal = vec![
            json!({"role": "assistant", "content": "Earlier output."}),
            json!({"role": "user", "content": "Latest input."}),
        ];
        assert!(!rewrite_terminal_assistant_prefill(&mut non_terminal));
        assert_eq!(non_terminal[0]["role"], "assistant");

        let converted = convert_codex_to_openai_request(json!({
            "input": [{
                "role": "assistant",
                "content": "Choose the next action."
            }]
        }));
        assert_eq!(
            converted["messages"],
            json!([{"role": "user", "content": "Choose the next action."}])
        );
    }

    #[test]
    fn responses_compat_drops_only_leading_orphan_tool_history() {
        let mut messages = vec![
            json!({"role": "system", "content": "Plan safely."}),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_orphan",
                    "type": "function",
                    "function": {"name": "reply", "arguments": "{}"}
                }]
            }),
            json!({
                "role": "tool",
                "tool_call_id": "call_orphan",
                "content": "done"
            }),
            json!({"role": "user", "content": "Latest message"}),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_valid",
                    "type": "function",
                    "function": {"name": "query_memory", "arguments": "{}"}
                }]
            }),
            json!({
                "role": "tool",
                "tool_call_id": "call_valid",
                "content": "result"
            }),
        ];

        assert_eq!(drop_leading_orphan_tool_history(&mut messages), 2);
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[2]["tool_calls"][0]["id"], "call_valid");
        assert_eq!(messages[3]["tool_call_id"], "call_valid");

        let mut ordinary_history = vec![
            json!({"role": "system", "content": "Plan safely."}),
            json!({"role": "assistant", "content": "Earlier answer"}),
            json!({"role": "user", "content": "Latest message"}),
        ];
        let original = ordinary_history.clone();
        assert_eq!(drop_leading_orphan_tool_history(&mut ordinary_history), 0);
        assert_eq!(ordinary_history, original);
    }

    #[test]
    fn responses_compat_emits_standard_text_reasoning_and_function_calls() {
        let chat_response = serde_json::from_value(json!({
            "id": "chatcmpl_test",
            "object": "chat.completion",
            "created": 1,
            "model": "gemini-3.6-flash-high",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Visible answer",
                    "reasoning_content": "Internal analysis",
                    "tool_calls": [{
                        "id": "call_reply",
                        "type": "function",
                        "function": {
                            "name": "reply",
                            "arguments": "{\"msg_id\":\"1\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }))
        .expect("valid OpenAI response fixture");

        let response = convert_chat_response_to_responses(&chat_response, false);
        assert_eq!(response["object"], "response");
        assert_eq!(response["output"][0]["type"], "reasoning");
        assert_eq!(
            response["output"][0]["summary"][0]["text"],
            "Internal analysis"
        );
        assert_eq!(response["output"][1]["type"], "message");
        assert_eq!(response["output"][1]["content"][0]["type"], "output_text");
        assert_eq!(
            response["output"][1]["content"][0]["text"],
            "Visible answer"
        );
        assert_eq!(response["output"][2]["type"], "function_call");
        assert_eq!(response["output"][2]["call_id"], "call_reply");
        assert_eq!(response["output"][2]["name"], "reply");
        assert!(response["output"][1]["content"][0]["text"]
            .as_str()
            .is_some_and(|text| !text.contains("Internal analysis")));
    }

    #[test]
    fn test_unwrap_outer_code_fence_safety() {
        use super::unwrap_outer_code_fence_if_structured;

        // 1. 裸 JSON：安全透传，原样返回
        let bare = r#"{"title": "Bug Fix", "description": "Fixes issue #3710"}"#;
        assert_eq!(unwrap_outer_code_fence_if_structured(bare), bare);

        // 2. 被 ```json ... ``` 严格包裹：整包解壳
        let wrapped_json =
            "```json\n{\"title\": \"Bug Fix\", \"description\": \"Fixes issue #3710\"}\n```";
        assert_eq!(unwrap_outer_code_fence_if_structured(wrapped_json), bare);

        // 3. 被 ``` ... ``` 严格包裹（无语言标识）：整包解壳
        let wrapped_plain =
            "```\n{\"title\": \"Bug Fix\", \"description\": \"Fixes issue #3710\"}\n```";
        assert_eq!(unwrap_outer_code_fence_if_structured(wrapped_plain), bare);

        // 4. 【零误杀保障】常规代码块：非 JSON 代码绝不破坏
        let non_json = "```rust\nfn main() { println!(\"Hello\"); }\n```";
        assert_eq!(unwrap_outer_code_fence_if_structured(non_json), non_json);

        // 5. 【零误杀保障】字段内部包含反引号代码块的合法 JSON：绝不被内部反引号误杀
        let inner_with_backticks = r#"{"title":"Summary","code":"```ts\nconst x = 1;\n```"}"#;
        assert_eq!(
            unwrap_outer_code_fence_if_structured(inner_with_backticks),
            inner_with_backticks
        );

        // 6. 前后带有自然语言说明（非整包围栏包裹）：解壳函数保持原样返回
        let conversational = "Here is the result:\n```json\n{\"a\":1}\n```\nHope it helps!";
        assert_eq!(
            unwrap_outer_code_fence_if_structured(conversational),
            conversational
        );
    }

    #[test]
    fn test_convert_chat_response_to_responses_structured_vs_normal() {
        use super::convert_chat_response_to_responses;

        let wrapped_json =
            "```json\n{\"title\": \"Issue 3710\", \"description\": \"Structured outputs\"}\n```";
        let bare_json = "{\"title\": \"Issue 3710\", \"description\": \"Structured outputs\"}";

        let chat_response = serde_json::from_value(json!({
            "id": "chatcmpl_title",
            "object": "chat.completion",
            "created": 1,
            "model": "gemini-3.8-flash-high",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": wrapped_json,
                    "reasoning_content": null
                },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15 }
        }))
        .expect("chat response");

        // is_structured_output = true: 剥离整包外层围栏，输出可供客户端原生 JSON.parse 解析的裸 JSON
        let structured_val = convert_chat_response_to_responses(&chat_response, true);
        let out_text = structured_val["output"][0]["content"][0]["text"]
            .as_str()
            .unwrap();
        assert_eq!(out_text, bare_json);
        assert!(serde_json::from_str::<serde_json::Value>(out_text).is_ok());

        // is_structured_output = false (普通聊天会话): 100% 原始呈现，绝不剥离或丢弃
        let normal_val = convert_chat_response_to_responses(&chat_response, false);
        assert_eq!(
            normal_val["output"][0]["content"][0]["text"]
                .as_str()
                .unwrap(),
            wrapped_json
        );
    }

    #[test]
    fn test_responses_api_text_format_and_pipeline_tool_suppression() {
        use super::convert_codex_to_openai_request;
        use crate::proxy::mappers::openai::models::OpenAIRequest;
        use crate::proxy::mappers::openai::request::transform_openai_request_with_session;
        use crate::proxy::pipeline::InboundThinkingPipeline;

        let input = json!({
            "model": "gpt-6-luna",
            "reasoning": { "effort": "low" },
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "codex_output_schema",
                    "strict": true,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "title": { "type": "string" },
                            "description": { "type": "string" }
                        },
                        "required": ["title", "description"],
                        "additionalProperties": false
                    }
                }
            },
            "tools": [{
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read file contents",
                    "parameters": { "type": "object" }
                }
            }],
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": "Summarize this session."
                }
            ]
        });

        let converted = convert_codex_to_openai_request(input);
        let openai_req: OpenAIRequest =
            serde_json::from_value(converted).expect("deserialize OpenAIRequest");

        let (mut gemini_payload, _, _, _) = transform_openai_request_with_session(
            &openai_req,
            "proj-test",
            "gemini-3.8-flash-high",
            None,
            "sess-test",
            Some("sess-test"),
            true,
            false,
        )
        .unwrap();

        let inner_req = &mut gemini_payload["request"];

        // 1. 验证适配器层提取 text.format 并映射为 responseMimeType 与 responseSchema
        assert_eq!(
            inner_req["generationConfig"]["responseMimeType"],
            "application/json"
        );
        assert_eq!(
            inner_req["generationConfig"]["responseSchema"]["type"],
            "object"
        );
        // 2. 验证 low effort 将 thinkingBudget 约束为 1024
        assert_eq!(
            inner_req["generationConfig"]["thinkingConfig"]["thinkingBudget"],
            1024
        );

        // 3. 验证 Pipeline 统一拓扑对齐与工具抑制 (Pipeline First)
        InboundThinkingPipeline::align_google_request_prefix_topology_with_model(
            inner_req,
            "gemini-3.8-flash-high",
            None,
        )
        .unwrap();
        assert!(
            inner_req.get("tools").is_none(),
            "tools must be suppressed by pipeline when responseSchema is active"
        );
    }

    #[test]
    fn identifies_codex_transcript_only_assistant_messages() {
        let thought = json!({
            "type": "message",
            "id": "msg_thought_abc_0",
            "role": "assistant",
            "phase": "commentary",
            "content": [{"type": "output_text", "text": "thinking"}],
        });
        let normal_commentary = json!({
            "type": "message",
            "role": "assistant",
            "phase": "commentary",
            "content": [{"type": "output_text", "text": "progress"}],
        });
        let contaminated_final = json!({
            "type": "message",
            "role": "assistant",
            "phase": "final_answer",
            "content": [{"type": "output_text", "text": "**Thinking**\n\nlegacy thought"}],
        });
        let clean_final = json!({
            "type": "message",
            "role": "assistant",
            "phase": "final_answer",
            "content": [{"type": "output_text", "text": "done"}],
        });

        assert!(is_codex_transcript_only_assistant_message(
            &thought, "thinking"
        ));
        assert!(!is_codex_transcript_only_assistant_message(
            &normal_commentary,
            "progress"
        ));
        assert!(is_codex_transcript_only_assistant_message(
            &contaminated_final,
            "**Thinking**\n\nlegacy thought"
        ));
        assert!(!is_codex_transcript_only_assistant_message(
            &clean_final,
            "done"
        ));
    }

    #[test]
    fn test_codex_process_commentary_is_retained_in_request_messages() {
        let input = json!({
            "model": "gemini-2.5-pro",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "检查系统故障"}]
                },
                {
                    "type": "message",
                    "id": "msg_thought_123",
                    "role": "assistant",
                    "phase": "commentary",
                    "content": [{"type": "output_text", "text": "private thought"}]
                },
                {
                    "type": "message",
                    "id": "msg_comm_123",
                    "role": "assistant",
                    "phase": "commentary",
                    "content": [{"type": "output_text", "text": "概览显示 HTTP 502，接下来检查网关连接。"}]
                },
                {
                    "type": "function_call",
                    "id": "call_inspect",
                    "name": "inspect_case",
                    "arguments": "{\"case\":\"case_1\"}"
                }
            ]
        });

        let converted = convert_codex_to_openai_request(input);
        let messages = converted["messages"].as_array().expect("messages array");

        // 验证：user 消息存在
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"], "检查系统故障");

        // 验证：私有思考 msg_thought_123 被成功过滤，未进入 messages
        assert!(messages
            .iter()
            .all(|m| m.get("content").and_then(Value::as_str) != Some("private thought")));

        // 验证：普通进度 commentary 消息被成功保留
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(
            messages[1]["content"],
            "概览显示 HTTP 502，接下来检查网关连接。"
        );

        // 验证：工具调用正常跟随
        assert_eq!(messages[2]["role"], "assistant");
        assert_eq!(
            messages[2]["tool_calls"][0]["function"]["name"],
            "inspect_case"
        );
    }

    #[test]
    fn generation_image_extension_adds_single_and_array_images_in_order() {
        let single = json!("data:image/png;base64,AQ==");
        let single_images = parse_generation_input_images(Some(&single)).expect("single data URL");
        let single_parts = build_image_contents("prompt".to_string(), &single_images, None);
        assert_eq!(single_parts.len(), 2);
        assert_eq!(single_parts[1]["inlineData"]["data"], "AQ==");

        let multiple = json!(["data:image/png;base64,AQ==", "data:image/jpeg;base64,Ag=="]);
        let multiple_images =
            parse_generation_input_images(Some(&multiple)).expect("ordered data URL array");
        let multiple_parts = build_image_contents("prompt".to_string(), &multiple_images, None);
        assert_eq!(multiple_parts.len(), 3);
        assert_eq!(multiple_parts[1]["inlineData"]["data"], "AQ==");
        assert_eq!(multiple_parts[2]["inlineData"]["data"], "Ag==");
    }

    #[test]
    fn generation_without_image_remains_text_only() {
        let images =
            parse_generation_input_images(None).expect("absent image is standard text-to-image");
        let parts = build_image_contents("prompt".to_string(), &images, None);
        assert_eq!(parts, vec![json!({"text": "prompt"})]);
    }

    #[test]
    fn generation_image_extension_rejects_invalid_or_remote_inputs() {
        let invalid_inputs = [
            json!("https://example.invalid/image.png"),
            json!("data:text/plain;base64,AQ=="),
            json!("data:image/png;base64,not-base64"),
            json!([]),
            json!(["data:image/png;base64,AQ==", 2]),
        ];
        for input in invalid_inputs {
            assert!(parse_generation_input_images(Some(&input)).is_err());
        }

        let oversized = Value::Array(vec![json!(format!(
            "data:image/png;base64,{}",
            "AAAA".repeat(MAX_INPUT_IMAGE_BYTES / 3 + 1)
        ))]);
        assert!(parse_generation_input_images(Some(&oversized)).is_err());
        assert!(validate_input_image_limits(1, MAX_INPUT_IMAGE_BYTES + 1, 0).is_err());
        assert!(validate_input_image_limits(1, 1, MAX_TOTAL_INPUT_IMAGE_BYTES + 1).is_err());

        assert!(generation_image_size_param(&json!({"imageSize": 4})).is_err());
    }

    #[test]
    fn edit_image_fields_preserve_all_supported_forms_in_arrival_order() {
        let field_names = [
            "image",
            "image",
            "image[]",
            "image[]",
            "image1",
            "image2",
            "imageSize",
            "image_size",
            "imageReference",
        ];
        let accepted: Vec<(usize, &str)> = field_names
            .iter()
            .enumerate()
            .filter(|(_, name)| is_edit_image_field(name))
            .map(|(index, name)| (index, *name))
            .collect();
        assert_eq!(
            accepted,
            vec![
                (0, "image"),
                (1, "image"),
                (2, "image[]"),
                (3, "image[]"),
                (4, "image1"),
                (5, "image2"),
            ]
        );
    }

    #[test]
    fn edit_aspect_ratio_priority_preserves_suffix_without_explicit_size() {
        use crate::proxy::mappers::common_utils::try_parse_image_config_with_params;

        let suffix_input = edit_size_input(None, None);
        let (suffix_config, _) = try_parse_image_config_with_params(
            "gemini-3.1-flash-image-16x9",
            suffix_input,
            None,
            None,
        )
        .expect("model suffix config");
        assert_eq!(suffix_config["aspectRatio"], "16:9");

        let explicit_input = edit_size_input(Some("4:3"), Some("1280x720"));
        let (explicit_config, _) = try_parse_image_config_with_params(
            "gemini-3.1-flash-image-16x9",
            explicit_input,
            None,
            None,
        )
        .expect("explicit aspect ratio config");
        assert_eq!(explicit_config["aspectRatio"], "4:3");
    }

    #[test]
    fn edit_flash_model_is_used_for_account_selection_and_resolved_upstream_body() {
        use crate::proxy::mappers::common_utils::try_parse_image_config_with_params;

        let (image_config, model_to_use) =
            try_parse_image_config_with_params("gemini-3.1-flash-image", None, None, None)
                .expect("flash edit model config");
        assert_eq!(
            image_account_selection_target(&model_to_use),
            "gemini-3.1-flash-image"
        );

        let body = build_image_edit_body(
            "project".to_string(),
            "account-resolved-image-model",
            json!([{"text": "prompt"}])
                .as_array()
                .cloned()
                .expect("parts"),
            image_config,
        );
        assert_eq!(body["model"], "account-resolved-image-model");
    }
}

#[cfg(test)]
mod variant_tests {
    use crate::proxy::common::variant_mapping;
    use crate::proxy::mappers::openai::models::ThinkingConfig;

    #[test]
    fn openai_opus_preserves_client_budget_when_present() {
        let client_budget = Some(32_768);
        let spec = variant_mapping::resolve("claude-opus-4-6-thinking", client_budget)
            .expect("Claude Opus 4.6 thinking must resolve");
        let request_thinking = ThinkingConfig {
            thinking_type: Some("enabled".to_string()),
            budget_tokens: Some(spec.effective_thinking_budget(client_budget)),
            effort: None,
        };

        assert_eq!(request_thinking.budget_tokens, client_budget);
    }

    #[test]
    fn openai_opus_falls_back_to_spec_budget_when_client_budget_is_absent() {
        let client_budget = None;
        let spec = variant_mapping::resolve("claude-opus-4-6-thinking", client_budget)
            .expect("Claude Opus 4.6 thinking must resolve");
        let request_thinking = ThinkingConfig {
            thinking_type: Some("enabled".to_string()),
            budget_tokens: Some(spec.effective_thinking_budget(client_budget)),
            effort: None,
        };

        assert_eq!(request_thinking.budget_tokens, Some(1_024));
    }
}

fn codex_ledger_from_body(
    body: &Value,
) -> (
    Option<crate::proxy::mappers::openai::interaction_ledger::InteractionLedger>,
    VecDeque<String>,
) {
    let ledger = crate::proxy::mappers::openai::interaction_ledger::build_codex_interaction_ledger(
        body.get("input"),
        body.get("instructions").and_then(|v| v.as_str()),
        body.get("session_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        body.get("previous_response_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
    );

    let mut markers = VecDeque::new();
    if let Some(ledger) = &ledger {
        for step in ledger.turns.iter().flat_map(|turn| turn.steps.iter()) {
            if step.raw_item.get("type").and_then(|v| v.as_str()) != Some("instructions") {
                markers.push_back(
                    crate::proxy::mappers::openai::interaction_ledger::step_marker(step),
                );
            }
        }
    }

    (ledger, markers)
}

fn responses_routing_session_id(
    explicit_session_id: Option<&str>,
    previous_response_id: Option<&str>,
    stored_routing_session_id: Option<&str>,
    response_id: &str,
) -> String {
    explicit_session_id
        .filter(|id| !id.is_empty())
        .or(stored_routing_session_id.filter(|id| !id.is_empty()))
        .or(previous_response_id.filter(|id| !id.is_empty()))
        .unwrap_or(response_id)
        .to_string()
}

fn strip_codex_step_markers(content: &str) -> String {
    if !content.contains("[codex-turn:") {
        return content.to_string();
    }
    let had_trailing_newline = content.ends_with('\n');
    let mut cleaned = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("[codex-turn:")
            && trimmed.contains(" step:")
            && trimmed.contains(" type:")
            && trimmed.ends_with(']')
        {
            continue;
        }
        cleaned.push(line);
    }
    let mut res = cleaned.join("\n");
    if had_trailing_newline && !res.ends_with('\n') {
        res.push('\n');
    }
    res
}

fn prefix_with_step_marker(_marker: Option<String>, content: String) -> String {
    // Step markers are useful in debug ledgers, but must not be visible to
    // Gemini. If they enter the prompt, the model learns to emit them as text.
    strip_codex_step_markers(&content)
}

pub async fn handle_chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap, // [CHANGED] Extract headers
    upstream_recorder: Option<
        axum::extract::Extension<crate::proxy::monitor::UpstreamRequestBodyHolder>,
    >,
    user_identity: Option<
        axum::extract::Extension<crate::proxy::middleware::auth::UserTokenIdentity>,
    >,
    Json(mut body): Json<Value>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let clean_start = std::time::Instant::now();
    validate_openai_tool_schemas(body["tools"].as_array().map(Vec::as_slice))
        .map_err(|message| (StatusCode::BAD_REQUEST, message))?;
    // [NEW] Check for Image Model Redirection
    let model_name = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_lowercase();
    // [FIX] Only redirect non-native image aliases (dall-e / midjourney) to the
    // images-generations shim. Native Gemini image models (gemini-3-pro-image*) must
    // flow through the normal pipeline (transform_openai_request -> resolve_request_config),
    // which correctly sets requestType=image_gen, imageConfig (size/aspect ratio), sessionId,
    // structured requestId and per-account dynamic model resolution — matching the official
    // Antigravity client. The old shim dropped `size` and built a divergent upstream body,
    // which caused image generation to silently fail for gemini-3-pro-image.
    if (model_name.contains("image")
        || model_name.contains("dall-e")
        || model_name.contains("midjourney"))
        && !model_name.contains("gemini")
    {
        tracing::info!(
            "[ChatRedirection] Redirecting model {} to image generations",
            model_name
        );
        return intercept_chat_to_image(
            state,
            headers,
            user_identity.map(|identity| identity.token_id.clone()),
            body,
            &model_name,
        )
        .await;
    }

    // [MODEL GATE] 入站模型闸门：缺失/不存在 → 404 model not exist；全池无配额 → 502 no quota accounts
    if let Err(rejection) = crate::proxy::handlers::common::enforce_model_gate(
        &state,
        Some(body.get("model").and_then(|v| v.as_str()).unwrap_or("")),
        "openai",
        upstream_recorder.as_ref().map(|recorder| &recorder.0),
    )
    .await
    {
        return Ok(rejection);
    }

    let debug_cfg = state.debug_logging.read().await.clone();
    let original_body = debug_logger::is_enabled(&debug_cfg).then(|| {
        crate::proxy::payload_audit::reorder_payload_fields(&debug_value_without_inline_data(&body))
    });

    // [FIX] 自动检测并转换 Responses 格式
    // 如果请求包含 instructions 或 input 但没有 messages，则认为是 Responses 格式。
    // 必须复用完整转换器，不能把 input 数组序列化成一条 user 文本，否则会
    // 丢失 system/user/assistant 的角色边界，导致 Gemini 重放整段历史而不是
    // 继续最新一轮对话。
    let is_responses_format = !body.get("messages").is_some()
        && (body.get("instructions").is_some() || body.get("input").is_some());

    if is_responses_format {
        normalize_responses_plaintext(&mut body)
            .map_err(|(_, _, message)| (StatusCode::BAD_REQUEST, message.to_owned()))?;
        debug!("Detected Responses API format, converting with structured role preservation");
        body = convert_codex_to_openai_request(body);
    }

    let normalized_interaction_ledger = body.get("_interaction_ledger").cloned();
    let session_hint = json!({
        "session_id": body.get("session_id").filter(|v| v.is_string()),
        "sessionId": body.get("sessionId").filter(|v| v.is_string()),
        "conversation_id": body.get("conversation_id").filter(|v| v.is_string()),
        "chat_id": body.get("chat_id").filter(|v| v.is_string()),
        "thread_id": body.get("thread_id").filter(|v| v.is_string()),
        "client_session_id": body.get("client_session_id").filter(|v| v.is_string()),
        "metadata": body.get("metadata").filter(|v| v.is_object()),
    });
    let mut openai_req: OpenAIRequest = serde_json::from_value(body)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid request: {}", e)))?;

    validate_openai_file_inputs(&openai_req)
        .map_err(|message| (StatusCode::BAD_REQUEST, message))?;

    // Safety: Ensure messages is not empty
    if openai_req.messages.is_empty() {
        debug!("Received request with empty messages, injecting fallback...");
        openai_req
            .messages
            .push(crate::proxy::mappers::openai::OpenAIMessage {
                role: "user".to_string(),
                content: Some(crate::proxy::mappers::openai::OpenAIContent::String(
                    " ".to_string(),
                )),
                reasoning_content: None,
                signature: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
                refusal: None,
            });
    }

    let clean_ms = clean_start.elapsed().as_micros() as f64 / 1000.0;
    let mut norm_ms = 0.0f64;
    let mut think_fill_ms = 0.0f64;
    // TTFT 单源口径：仅记录“首个上游数据块”；流式发头时写 -1。
    let mut ttft_ms = crate::proxy::pipeline::stream_peek::TTFT_UNKNOWN_MS;
    let mut headers_ms = 0.0f64;
    let mut fallback_overhead_ms = 0.0f64;

    let trace_id = format!("req_{}", chrono::Utc::now().timestamp_subsec_millis());
    info!(
        "[{}] OpenAI Chat Request: {} | {} messages | stream: {}",
        trace_id,
        openai_req.model,
        openai_req.messages.len(),
        openai_req.stream
    );
    let mut force_rotate = false;

    if debug_logger::is_enabled(&debug_cfg) {
        if let Some(ledger) = normalized_interaction_ledger {
            let payload = json!({
                "kind": "normalized_interaction_ledger",
                "protocol": "openai",
                "trace_id": trace_id.clone(),
                "original_model": openai_req.model.clone(),
                "interaction_ledger": ledger,
            });
            debug_logger::write_exchange_payload(
                &debug_cfg,
                Some(&trace_id),
                "normalized_interaction_ledger",
                &payload,
            )
            .await;
        }

        // [FIX] 使用原始 body 副本记录日志，确保不丢失任何字段
        let original_payload = json!({
            "kind": "original_request",
            "protocol": "openai",
            "trace_id": trace_id,
            "original_model": openai_req.model,
            "request": original_body.as_ref(),
        });
        debug_logger::write_exchange_payload(
            &debug_cfg,
            Some(&trace_id),
            "original_request",
            &original_payload,
        )
        .await;
    }

    // [NEW] Detect Client Adapter
    let client_adapter = CLIENT_ADAPTERS
        .iter()
        .find(|a| a.matches(&headers))
        .cloned();
    if client_adapter.is_some() {
        debug!("[{}] Client Adapter detected", trace_id);
    }

    // [Variant] Resolve canonical model + variant → real model + real params.
    // Replace the client's model/thinking/max_tokens with verified real values so the
    // forwarded request matches the expected upstream format. OpenCode encodes the variant as
    // thinking.budget_tokens; we infer the tier from its magnitude.
    let model_lower = openai_req.model.to_lowercase();
    let is_v3_or_above = crate::proxy::model_specs::is_gemini_v3_or_above(&openai_req.model);
    let is_explicit_tier_model = model_lower.ends_with("-high")
        || model_lower.ends_with("-medium")
        || model_lower.ends_with("-low")
        || model_lower.ends_with("-extra-low");

    let client_switch = crate::proxy::pipeline::extract_client_thinking_switch(
        openai_req
            .thinking
            .as_ref()
            .and_then(|t| t.thinking_type.as_deref()),
        openai_req
            .thinking
            .as_ref()
            .and_then(|t| t.budget_tokens.map(|b| b as u64))
            .or_else(|| {
                openai_req
                    .reasoning
                    .as_ref()
                    .and_then(|r| r.max_tokens.map(|b| b as u64))
            }),
        openai_req
            .reasoning_effort
            .as_deref()
            .or_else(|| {
                openai_req
                    .reasoning
                    .as_ref()
                    .and_then(|r| r.effort.as_deref())
            })
            .or_else(|| {
                openai_req
                    .thinking
                    .as_ref()
                    .and_then(|t| t.effort.as_deref())
            }),
    );

    let raw_client_budget = openai_req
        .thinking
        .as_ref()
        .and_then(|t| t.budget_tokens)
        .or_else(|| openai_req.reasoning.as_ref().and_then(|r| r.max_tokens));

    // 客户端直控模式已下线：高世代/显式档位模型一律清理客户端预算，由流水线按官方结构体决定
    let client_budget = if is_v3_or_above || is_explicit_tier_model {
        if let Some(ref mut t) = openai_req.thinking {
            t.budget_tokens = None; // 清理客户端 budget_tokens，防止污染
        }
        if let Some(ref mut r) = openai_req.reasoning {
            r.max_tokens = None;
        }
        None
    } else {
        raw_client_budget
    };
    let effective_budget_hint = if is_explicit_tier_model || is_v3_or_above {
        None
    } else {
        client_budget
    };

    let effort_hint: Option<String> = openai_req
        .reasoning_effort
        .as_ref()
        .map(|s| s.to_string())
        .or_else(|| openai_req.reasoning.as_ref().and_then(|r| r.effort.clone()))
        .or_else(|| openai_req.thinking.as_ref().and_then(|t| t.effort.clone()));
    let canonical_model = openai_req.model.clone();
    // [ROUTE SINGLE ENTRY] 路由只由 router 决定；variant 仅根据已路由的物理模型解析 spec。
    openai_req.model = {
        let mapping = state.custom_mapping.read().await;
        crate::proxy::common::model_mapping::resolve_model_route_with_effort(
            &openai_req.model,
            &mapping,
            effort_hint.as_deref(),
        )
    };
    let variant_spec = crate::proxy::common::variant_mapping::resolve_spec_exact(
        &openai_req.model,
        effective_budget_hint,
    );
    if let Some(spec) = variant_spec {
        tracing::info!(
            "[{}] [Variant] canonical='{}' effort={:?} budget_hint={:?} -> real_model='{}' budget={} maxOut={}",
            trace_id,
            canonical_model,
            effort_hint,
            effective_budget_hint,
            spec.id,
            spec.thinking_budget,
            spec.max_output_tokens
        );
        openai_req.model = spec.id.to_string();
        if !spec.supports_thinking() {
            // 仅官方明确声明“不支持思考”的模型才丢掉工具（此前把 budget=0 误判为不支持思考）
            openai_req.thinking = None;
            openai_req.tools = None;
            openai_req.tool_choice = None;
        } else {
            openai_req.thinking = Some(crate::proxy::mappers::openai::models::ThinkingConfig {
                thinking_type: Some("enabled".to_string()),
                budget_tokens: spec.budget_tokens_option(None),
                effort: effort_hint.clone(),
            });
        }
        // 尊重客户端显式 max_tokens，只做官方硬顶收敛（不再无条件抬到官方上限）
        if let Some(client_max) = openai_req.max_tokens {
            // 单点硬顶在流水线：`configure_inbound_thinking` 会按官方 max_output_tokens
            // 收敛客户端显式值，这里只负责把客户端值写进统一 IR。
            openai_req.max_tokens = Some(client_max);
        }
    }

    let client_tool_names =
        crate::proxy::mappers::openai::request::extract_client_tool_names(&openai_req.tools);

    // 1. 获取 UpstreamClient (Clone handle)
    let upstream = state.upstream.clone();
    let image_scheduler = state.image_scheduler.clone();
    let request_timeout = state.request_timeout;
    let token_manager = state.token_manager;
    let pool_size = token_manager.len();
    // [FIX #3485] 自适应多账号池与单账号退避最大重试次数 (单账号3次，多账号整池两轮)
    let max_attempts = crate::proxy::handlers::common::calculate_max_retry_attempts(pool_size);

    let mut last_error = String::new();
    let mut terminal_stream_usage: Option<serde_json::Value> = None;
    let mut last_email: Option<String> = None;
    let mut retry_state = RequestRetryState::default();
    let mut retry_credentials: Option<crate::proxy::token_manager::LeasedCredentials> = None;
    let mut session_lease = None;
    let mut attempted_accounts: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    let mut image_permit = None;
    let mut failure_statuses = FailureStatusTracker::default();
    let mut used_attempts = 0;
    let mut retried_without_thinking = false;
    let mut heal_flags = crate::proxy::pipeline::AutoHealFlags::default();

    // 2. 模型路由解析 (移到循环外以支持在所有路径返回 X-Mapped-Model)
    // [2026-10-29] 四协议统一抽象：显式关闭思考（thinking.type=disabled / budget=0 / effort=none）
    // 归一为路由意图 "none"，由路由层按官方档位数组降为最低档。
    let route_effort =
        crate::proxy::pipeline::routing_effort(client_switch.is_disabled(), effort_hint.as_deref());
    let mapped_model = crate::proxy::common::model_mapping::resolve_model_route_with_effort(
        &openai_req.model,
        &*state.custom_mapping.read().await,
        route_effort,
    );
    let anchor = SessionManager::openai_content_anchor(&openai_req);
    let session_scope = crate::proxy::thinking_store::SessionScope::resolve(
        &headers,
        Some(&session_hint),
        None,
        anchor,
        user_identity
            .as_ref()
            .map(|identity| identity.token_id.as_str()),
    );
    let store_key = session_scope.store_key.clone();
    let affinity_key = session_scope.affinity_key.clone();
    openai_req.session_id = Some(store_key.clone());
    let client_session_id = session_scope.client_id.clone();

    'attempts: while let Some(attempt) = next_rotation_attempt(
        &mut used_attempts,
        max_attempts,
        retry_credentials.is_some(),
    ) {
        // [Pipeline Terminal] 全局预算耗尽：统一记终态、补 last_error、直接退出轮换
        if crate::proxy::pipeline::terminal::gate_exhausted(
            "openai",
            attempt + 1,
            &mut failure_statuses,
            &mut last_error,
        ) {
            break;
        }
        retry_state.begin_attempt();

        let norm_start = std::time::Instant::now();
        // 将 OpenAI 工具转为 Value 数组以便探测联网
        let tools_val: Option<Vec<Value>> = openai_req
            .tools
            .as_ref()
            .map(|list| list.iter().cloned().collect());
        let config = crate::proxy::mappers::common_utils::resolve_request_config(
            &openai_req.model,
            &mapped_model,
            &tools_val,
            None, // size (not used in handler, transform_openai_request handles it)
            None, // quality
            None, // image_size
            None, // body
        );

        // 4. 获取 Token (使用准确的 request_type)
        // 关键：在重试尝试时根据 force_rotate 决定是否轮换账号
        let (access_token, project_id, email, account_id, _wait_ms, selected_session_lease) =
            if let Some(credentials) = retry_credentials.take() {
                credentials
            } else if config.request_type == "image_gen" {
                drop(image_permit.take());
                match token_manager
                    .get_image_token_with_lease(
                        force_rotate,
                        Some(&affinity_key),
                        &mapped_model,
                        &image_scheduler,
                        request_timeout,
                    )
                    .await
                {
                    Ok((access_token, project_id, email, account_id, wait_ms, permit, lease)) => {
                        image_permit = Some(permit);
                        (access_token, project_id, email, account_id, wait_ms, lease)
                    }
                    Err((status, message)) => {
                        failure_statuses.record(status);
                        last_error = message;
                        break;
                    }
                }
            } else {
                let mut token_result = token_manager
                    .get_token_filtered_with_retry_lease(
                        &config.request_type,
                        force_rotate,
                        Some(&affinity_key),
                        &mapped_model,
                        &attempted_accounts,
                        &mut session_lease,
                    )
                    .await;

                if let Err(ref e) = token_result {
                    if crate::proxy::handlers::common::is_transient_token_error(e) {
                        tracing::warn!(
                            "Token acquisition transient error ({}), retrying once under scheduling policy...",
                            e
                        );
                        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                        token_result = token_manager
                            .get_token_filtered_with_retry_lease(
                                &config.request_type,
                                crate::proxy::pipeline::rotation::token_retry_force_rotate(
                                    force_rotate,
                                    token_manager.get_scheduling_mode().await,
                                ),
                                Some(&affinity_key),
                                &mapped_model,
                                &attempted_accounts,
                                &mut session_lease,
                            )
                            .await;
                    }
                }

                match token_result {
                    Ok(t) => t,
                    Err(e) => {
                        retry_state.preserve_token_error(&e);
                        if !failure_statuses.is_empty()
                            && !crate::proxy::session_bindings::is_session_binding_changed_error(&e)
                        {
                            tracing::warn!(
                            "[{}] All available accounts attempted or exhausted for this request, breaking retry loop to return upstream error",
                            trace_id
                        );
                            break;
                        }
                        return Ok(retry_state.finish_error_response(
                            crate::proxy::handlers::common::build_token_error_response_with_usage(
                                "openai",
                                mapped_model.as_str(),
                                None,
                                &e,
                                terminal_stream_usage.as_ref(),
                            ),
                        ));
                    }
                }
            };

        session_lease = selected_session_lease;

        let in_flight_guard = Arc::new(token_manager.acquire_in_flight(&account_id));

        // [NEW v4.1.29] 获取完整 Token 对象用于动态规格查询
        let proxy_token = token_manager.get_token_by_id(&account_id);
        let mapped_model = token_manager
            .resolve_dynamic_model_for_account(&account_id, &mapped_model)
            .await;

        last_email = Some(email.clone());
        info!("✓ Using account: {} (type: {})", email, config.request_type);

        // 4. 转换请求 (返回内容包含 session_id, message_count, prefix_hash)
        let tf_start = std::time::Instant::now();
        let (mut gemini_body, _routed_session, message_count, _prefix_hash) =
            transform_openai_request_with_session(
                &openai_req,
                &project_id,
                &mapped_model,
                proxy_token.as_ref(),
                &affinity_key,
                None,
                false,
                retried_without_thinking,
            )
            .map_err(|message| (StatusCode::BAD_REQUEST, message))?;
        heal_flags.reapply(&mut gemini_body);
        let session_id = store_key.clone();
        // [Signature-Retry] 400 签名错误后的重试：流水线统一去签名
        if retried_without_thinking {
            crate::proxy::pipeline::InboundThinkingPipeline::strip_signatures_for_retry(
                &mut gemini_body,
                Some(session_id.as_str()),
                &mapped_model,
            );
        }
        let tf_micros = tf_start.elapsed().as_micros() as u64;
        let norm_total_micros = norm_start.elapsed().as_micros() as u64;
        norm_ms = norm_total_micros.saturating_sub(tf_micros) as f64 / 1000.0;
        think_fill_ms = tf_micros as f64 / 1000.0;
        let _ =
            crate::proxy::mappers::context_manager::ContextManager::apply_post_transit_context_mgmt(
                &mut gemini_body,
                &mapped_model,
            );
        crate::proxy::mappers::prompt_sanitizer::PromptSanitizer::sanitize_gemini_payload(
            &mut gemini_body,
        );
        crate::proxy::mappers::common_utils::ensure_gemini_payload_ends_with_user(&mut gemini_body);
        // Resolve text history only after all request normalization, and use the
        // same visible prefix when recording the response from this attempt.
        let thinking_enabled =
            crate::proxy::pipeline::qos::payload_enables_thinking(&gemini_body, None);
        let contents_path = if gemini_body.get("request").is_some() {
            "/request/contents"
        } else {
            "/contents"
        };
        let response_text_context = gemini_body
            .pointer_mut(contents_path)
            .and_then(Value::as_array_mut)
            .map(|contents| {
                crate::proxy::thinking_store::finalize_request_text_context(
                    &session_id,
                    contents,
                    &mapped_model,
                    thinking_enabled,
                    retried_without_thinking,
                )
            });
        let estimated_input_tokens = crate::proxy::pipeline::estimate_tokens(&gemini_body);
        if let Some(ref recorder) = upstream_recorder {
            recorder.set_value(&gemini_body);
        }
        let gemini_body_for_debug = debug_logger::is_enabled(&debug_cfg)
            .then(|| debug_value_without_inline_data(&gemini_body));

        if debug_logger::is_enabled(&debug_cfg) {
            let payload = json!({
                "kind": "v1internal_request",
                "protocol": "openai",
                "trace_id": trace_id,
                "original_model": openai_req.model,
                "mapped_model": mapped_model,
                "request_type": config.request_type,
                "attempt": attempt,
                "v1internal_request": gemini_body_for_debug.as_ref(),
            });
            debug_logger::write_exchange_payload(
                &debug_cfg,
                Some(&trace_id),
                "v1internal_request",
                &payload,
            )
            .await;
        }

        let actual_request_type = gemini_body
            .get("requestType")
            .and_then(|v| v.as_str())
            .unwrap_or("none (standard)");
        info!(
            "[{}] Upstream request ready -> model: {}, requestType: {}",
            trace_id, mapped_model, actual_request_type
        );
        debug!(
            "[OpenAI-Request] Transformed Gemini body: {} bytes",
            serialized_json_len(&gemini_body)
        );

        // 5. 发送请求
        let client_wants_stream = openai_req.stream;
        let force_stream_internally = !client_wants_stream;
        let actual_stream = client_wants_stream || force_stream_internally;

        if force_stream_internally {
            debug!(
                "[{}] 🔄 Auto-converting non-stream request to stream for better quota",
                trace_id
            );
        }

        let method = if actual_stream {
            "streamGenerateContent"
        } else {
            "generateContent"
        };
        let query_string = if actual_stream { Some("alt=sse") } else { None };

        // [FIX #1522] Inject Anthropic Beta Headers for Claude models (OpenAI path)
        let mut extra_headers = std::collections::HashMap::new();
        extra_headers.insert("x-session-id".to_string(), client_session_id.clone());
        if mapped_model.to_lowercase().contains("claude") {
            extra_headers.insert(
                "anthropic-beta".to_string(),
                "claude-code-20250219".to_string(),
            );
            tracing::debug!(
                "[{}] Injected Anthropic beta headers for Claude model (via OpenAI)",
                trace_id
            );
        }

        let preceding_turn_anchor = gemini_body
            .get("request")
            .and_then(|r| r.get("contents"))
            .or_else(|| gemini_body.get("contents"))
            .and_then(|c| c.as_array())
            .and_then(|a| a.last())
            .cloned();
        let causal_anchor =
            crate::proxy::thinking_store::compute_causal_anchor(preceding_turn_anchor.as_ref());

        let upstream_req_start = std::time::Instant::now();
        let call_result = match upstream
            .call_v1_internal_with_headers(
                method,
                &access_token,
                gemini_body.clone(),
                query_string,
                extra_headers.clone(),
                Some(account_id.as_str()),
            )
            .await
        {
            Ok(r) => r,
            Err(e) => {
                last_error = e.clone();
                drop(image_permit.take());
                // 终态（首字节 / 上传 / 全局预算 / 重放上限）都不是账号故障：
                // 不冷却、不跨账号重放；判定、状态码与观测统一由 pipeline::terminal 给出。
                if let Some(handling) = crate::proxy::pipeline::terminal::apply_terminal(
                    &e,
                    &mut failure_statuses,
                    "openai",
                    attempt + 1,
                    max_attempts,
                    upstream_req_start.elapsed().as_secs_f64(),
                    |err| token_manager.mark_account_transient_failure(&account_id, 30, err),
                ) {
                    debug!(
                        "OpenAI Request failed on attempt {}/{} (terminal={}): {}",
                        attempt + 1,
                        max_attempts,
                        handling.terminal.as_str(),
                        crate::proxy::upstream::client::strip_internal_markers(&e)
                    );
                    // 终态都不是账号故障：解绑 / 换号 / 原地重放统一由 pipeline 按调度模式裁决。
                    // CacheFirst 下文本请求绝不解绑 / 换号；Balance 下传输层断连先原地重连一次。
                    let terminal_plan = retry_state.plan_terminal(
                        &account_id,
                        handling.disposition,
                        token_manager.get_scheduling_mode().await,
                        config.request_type != "image_gen",
                    );
                    if terminal_plan.abandon_sticky_account {
                        token_manager.abandon_session_lease(&mut session_lease);
                        tracing::debug!(
                            "Unbound session {} from account {} due to terminal {}",
                            affinity_key,
                            email,
                            handling.terminal.as_str()
                        );
                    }
                    match terminal_plan.next {
                        crate::proxy::pipeline::TerminalNextStep::Stop => break,
                        crate::proxy::pipeline::TerminalNextStep::RetrySameAccount => {
                            tracing::info!(
                                "Upstream terminal {} on account {} (attempt {}/{}): keeping session {} bound and retrying the same account in-place to preserve KV cache",
                                handling.terminal.as_str(),
                                email,
                                attempt + 1,
                                max_attempts,
                                affinity_key
                            );
                            retry_credentials = Some((
                                access_token.clone(),
                                project_id.clone(),
                                email.clone(),
                                account_id.clone(),
                                0,
                                session_lease.clone(),
                            ));
                            continue;
                        }
                        crate::proxy::pipeline::TerminalNextStep::Rotate => {}
                    }
                    crate::proxy::pipeline::rotation::record_attempted_account(
                        &mut attempted_accounts,
                        &account_id,
                        pool_size,
                        &mut force_rotate,
                    );
                    continue;
                }

                // [Gateway Self-Answer Guard] 模型不存在 / 无可用账号：不罚号、不轮换
                if let Some((status, reason)) =
                    crate::proxy::common::model_gate::self_answer_disposition(&e)
                {
                    failure_statuses.record(status);
                    last_error = reason.to_string();
                    break;
                }

                // 非终态（连接失败 / 调度异常）：冷却该账号并继续轮换
                failure_statuses.record(StatusCode::BAD_GATEWAY);
                if let Some(strategy) = retry_state.cache_first_failure(
                    &account_id,
                    502,
                    &e,
                    None,
                    &token_manager.get_sticky_config().await,
                    config.request_type != "image_gen",
                ) {
                    if apply_retry_strategy(strategy, attempt, max_attempts, 502, &trace_id).await {
                        retry_credentials = Some((
                            access_token.clone(),
                            project_id.clone(),
                            email.clone(),
                            account_id.clone(),
                            0,
                            session_lease.clone(),
                        ));
                        continue;
                    }
                    break;
                }
                token_manager.mark_account_transient_failure(&account_id, 30, &e);
                crate::proxy::pipeline::rotation::record_attempted_account(
                    &mut attempted_accounts,
                    &account_id,
                    pool_size,
                    &mut force_rotate,
                );
                debug!(
                    "OpenAI Request failed on attempt {}/{}: {}",
                    attempt + 1,
                    max_attempts,
                    e
                );
                continue;
            }
        };

        let effective_attempt_start = call_result.effective_attempt_start;
        let mut fallback_attempts = call_result.fallback_attempts.clone();
        fallback_overhead_ms = call_result.fallback_overhead_s() * 1000.0;
        let attempts_b64 =
            crate::proxy::handlers::common::encode_attempts_header(&fallback_attempts);

        // [NEW] 记录端点降级日志到 debug 文件
        if call_result.has_fallback() && debug_logger::is_enabled(&debug_cfg) {
            let fallback_entries: Vec<Value> = call_result
                .fallback_attempts
                .iter()
                .map(|a| {
                    json!({
                        "endpoint_url": a.endpoint_url,
                        "status": a.status,
                        "error": a.error,
                    })
                })
                .collect();
            let payload = json!({
                "kind": "endpoint_fallback",
                "protocol": "openai",
                "trace_id": trace_id,
                "original_model": openai_req.model,
                "mapped_model": mapped_model,
                "attempt": attempt,
                "account": mask_email(&email),
                "fallback_attempts": fallback_entries,
            });
            debug_logger::write_debug_payload(
                &debug_cfg,
                Some(&trace_id),
                "endpoint_fallback",
                &payload,
            )
            .await;
        }

        let response = call_result.response;
        // [NEW] 提取实际请求的上游端点 URL，用于日志记录和排查
        let upstream_url = response.url().to_string();
        let status = response.status();
        // HTTP failures and pre-output SSE errors share the same recovery policy.
        let (status, retry_after, error_text) = 'upstream_response: {
            if !status.is_success() {
                let retry_after = response
                    .headers()
                    .get("Retry-After")
                    .and_then(|header| header.to_str().ok())
                    .map(str::to_string);
                let error_text = response
                    .text()
                    .await
                    .unwrap_or_else(|_| format!("HTTP {}", status));
                break 'upstream_response (status, retry_after, error_text);
            }

            let track_image_success = config.request_type == "image_gen";
            let mut session_confirmation = if !track_image_success {
                // HTTP acceptance still confirms account health in all scheduling
                // modes; only cache warmth waits for body validation.
                token_manager.mark_account_success(&account_id);
                token_manager.accept_session_response(&mut session_lease)
            } else {
                None
            };
            // 5. 处理流式 vs 非流式
            if actual_stream {
                use axum::body::Body;
                use axum::response::Response;
                use futures::StreamExt;

                let meta = json!({
                    "protocol": "openai",
                    "trace_id": trace_id,
                    "original_model": openai_req.model,
                    "mapped_model": mapped_model,
                    "request_type": config.request_type,
                    "attempt": attempt,
                    "status": status.as_u16(),
                    "upstream_url": upstream_url,
                });
                let gemini_stream = debug_logger::wrap_stream_with_debug(
                    crate::proxy::pipeline::usage_diagnostics::observe_stream(
                        response.bytes_stream(),
                        crate::proxy::pipeline::usage_diagnostics::UsageAttempt::current(
                            &config.final_model,
                            &account_id,
                            "openai_chat",
                        ),
                    ),
                    debug_cfg.clone(),
                    trace_id.clone(),
                    "upstream_response",
                    meta,
                );

                // [Auto-Heal] 纯思考空回复流式自愈门禁 (Pipeline First)
                let auto_heal_ctx = crate::proxy::pipeline::auto_heal::ThinkingAutoHealContext {
                    upstream: upstream.clone(),
                    method,
                    access_token: access_token.clone(),
                    original_body: gemini_body.clone(),
                    query_string,
                    extra_headers: extra_headers.clone(),
                    account_id: Some(account_id.clone()),
                    trace_id: trace_id.clone(),
                    budget: crate::proxy::pipeline::budget::current(),
                };
                let gemini_stream =
                    crate::proxy::pipeline::auto_heal::wrap_stream_with_empty_thinking_auto_heal(
                        Box::pin(gemini_stream),
                        auto_heal_ctx,
                    );
                // Preserve structured upstream errors before protocol conversion.
                let stream_errors =
                    crate::proxy::pipeline::stream_error::UpstreamStreamErrorObserver::default();
                let gemini_stream = Box::pin(
                    crate::proxy::pipeline::stream_error::observe_response_errors(
                        gemini_stream,
                        stream_errors.clone(),
                        !client_wants_stream,
                    ),
                );
                let gemini_stream = Box::pin(
                    crate::proxy::token_manager::confirm_session_response_stream(
                        gemini_stream,
                        stream_errors.clone(),
                        session_confirmation.take(),
                    ),
                );

                // [P1 FIX] Enhanced Peek logic to handle heartbeats and slow start
                // Pre-read until we find meaningful content, skip heartbeats
                use crate::proxy::mappers::openai::streaming::create_openai_sse_stream_with_context;
                // Internal streaming is an implementation detail of JSON responses;
                // the collector needs the usage trailer even without stream_options.
                let include_usage = !client_wants_stream
                    || openai_req
                        .stream_options
                        .as_ref()
                        .map(|o| o.include_usage)
                        .unwrap_or(false);
                let mut openai_stream = create_openai_sse_stream_with_context(
                    gemini_stream,
                    openai_req.model.clone(),
                    session_id.clone(),
                    message_count,
                    Some(client_tool_names.clone()),
                    include_usage,
                    Some(causal_anchor),
                    response_text_context.clone(),
                );

                let mut first_data_chunk = None;
                let mut peek_failed = false;

                // [Pipeline StreamPeek] 流式请求不再阻塞等待首包：直接返回响应头，
                // 由 run_stream_lifecycle 的 InitialTtft 负责首字宽限与心跳保活
                // （与 Claude / Gemini 同构，避免长 prefill 期间客户端看护狗先断开）。
                // 仅非流式请求需要在返回前拿到首块。
                headers_ms = effective_attempt_start.elapsed().as_micros() as f64 / 1000.0;
                if !client_wants_stream {
                    let is_thinking = crate::proxy::pipeline::qos::payload_enables_thinking(
                        &gemini_body,
                        Some(&mapped_model),
                    ) || openai_req.thinking.is_some()
                        || openai_req.reasoning.is_some()
                        || openai_req.reasoning_effort.is_some();
                    let peek_timeout_secs = crate::proxy::pipeline::budget::clamp_secs(
                        crate::proxy::pipeline::stream_peek::peek_budget_secs(
                            false,
                            config.request_type == "image_gen",
                            is_thinking,
                            None,
                        ),
                    );
                    match crate::proxy::pipeline::stream_peek::peek_first_data(
                        &mut openai_stream,
                        peek_timeout_secs,
                        |bytes| {
                            crate::proxy::pipeline::stream_peek::is_heartbeat(
                                bytes,
                                crate::proxy::pipeline::stream_peek::StreamDialect::OpenAi,
                            )
                        },
                        |bytes| stream_chunk_has_error_event(bytes),
                    )
                    .await
                    {
                        crate::proxy::pipeline::stream_peek::PeekOutcome::Data {
                            bytes, ..
                        } => {
                            ttft_ms = effective_attempt_start.elapsed().as_micros() as f64 / 1000.0;
                            crate::proxy::handlers::common::report_qos_ttft(
                                &token_manager,
                                &account_id,
                                Some(affinity_key.as_str()),
                                ttft_ms,
                                is_thinking,
                                Some(estimated_input_tokens),
                                session_lease.as_ref(),
                            );
                            if let Some(effective_item) =
                                fallback_attempts.iter_mut().find(|a| a.is_effective)
                            {
                                effective_item.duration_s = ttft_ms / 1000.0;
                            }
                            let _ = crate::proxy::monitor::CURRENT_UPSTREAM_CAPTURE.try_with(
                                |holder| {
                                    holder.set_attempts(fallback_attempts.clone());
                                },
                            );
                            first_data_chunk = Some(bytes);
                        }
                        crate::proxy::pipeline::stream_peek::PeekOutcome::Timeout {
                            budget_secs,
                            ..
                        } => {
                            // 首包预算耗尽不是账号故障：不跨账号重放，直接 504
                            last_error = format!("First chunk timeout after {}s", budget_secs);
                            terminal_stream_usage = record_nonstream_peek_timeout(
                                "openai",
                                budget_secs,
                                &stream_errors,
                                &mut failure_statuses,
                                &mut retry_state,
                                token_manager.get_scheduling_mode().await,
                                config.request_type != "image_gen",
                            );
                            break 'attempts;
                        }
                        crate::proxy::pipeline::stream_peek::PeekOutcome::ErrorEvent { .. } => {
                            last_error = "Error event during peek".to_string();
                            peek_failed = true;
                        }
                        crate::proxy::pipeline::stream_peek::PeekOutcome::StreamError {
                            error,
                            ..
                        } => {
                            last_error = format!("Stream error during peek: {}", error);
                            peek_failed = true;
                        }
                        crate::proxy::pipeline::stream_peek::PeekOutcome::Eof { .. } => {
                            last_error = "Empty response stream during peek".to_string();
                            peek_failed = true;
                        }
                    }
                } else {
                    // 流式：发头时 TTFT 未知（-1）；真实值由首块探针上报 QoS。
                    ttft_ms = crate::proxy::pipeline::stream_peek::TTFT_UNKNOWN_MS;
                }

                if peek_failed {
                    if let Some(failure) = stream_errors.failure_or_incomplete(Some(&last_error)) {
                        let failure_status =
                            StatusCode::from_u16(failure.status).unwrap_or(StatusCode::BAD_GATEWAY);
                        if failure.replay_safe() {
                            break 'upstream_response (failure_status, None, failure.body);
                        }
                        // Output has already been consumed: classify the error, but do not replay.
                        retry_state.preserve_http_stop(
                            failure.status,
                            &failure.body,
                            &token_manager.get_sticky_config().await,
                            config.request_type != "image_gen",
                        );
                        failure_statuses.record(failure_status);
                        terminal_stream_usage = failure.usage.clone();
                        last_error = failure.body;
                        break 'attempts;
                    }
                    failure_statuses.record(StatusCode::BAD_GATEWAY);
                    if let Some(strategy) = retry_state.cache_first_failure(
                        &account_id,
                        502,
                        &last_error,
                        None,
                        &token_manager.get_sticky_config().await,
                        config.request_type != "image_gen",
                    ) {
                        if apply_retry_strategy(strategy, attempt, max_attempts, 502, &trace_id)
                            .await
                        {
                            retry_credentials = Some((
                                access_token.clone(),
                                project_id.clone(),
                                email.clone(),
                                account_id.clone(),
                                0,
                                session_lease.clone(),
                            ));
                            continue 'attempts;
                        }
                        break 'attempts;
                    }
                    force_rotate = true;
                    continue 'attempts;
                }

                // Combine first chunk with remaining stream (only when peek ran)
                let combined_stream: std::pin::Pin<
                    Box<dyn futures::Stream<Item = Result<Bytes, String>> + Send>,
                > = if let Some(first) = first_data_chunk {
                    Box::pin(
                        futures::stream::once(async move { Ok::<Bytes, String>(first) })
                            .chain(openai_stream),
                    )
                } else {
                    openai_stream
                };

                // [NEW] 针对 OpenAI 流增加 300 秒空闲超时保护
                let image_permit_for_stream = image_permit.take();
                let track_image_success = config.request_type == "image_gen";
                let image_success_manager = token_manager.clone();
                let image_success_account = account_id.clone();
                let image_success_model = mapped_model.clone();
                let in_flight_guard_for_stream = Arc::clone(&in_flight_guard);
                let combined_stream = async_stream::stream! {
                    let _in_flight = in_flight_guard_for_stream;
                    let _image_permit = image_permit_for_stream;
                    let mut s = Box::pin(combined_stream);
                    let mut saw_image_data = false;
                    let mut stream_failed = false;

                    loop {
                        match tokio::time::timeout(std::time::Duration::from_secs(300), s.next()).await {
                            Ok(Some(Ok(bytes))) => {
                                if stream_chunk_has_error_event(&bytes) {
                                    stream_failed = true;
                                }
                                if track_image_success && stream_chunk_has_image_data(&bytes) {
                                    saw_image_data = true;
                                }
                                yield Ok::<Bytes, String>(bytes);
                            }
                            Ok(Some(Err(error))) => {
                                stream_failed = true;
                                yield Err::<Bytes, String>(error);
                                break;
                            }
                            Ok(None) => break,
                            Err(_) => {
                                tracing::error!("[OpenAI-SSE] Idle timeout after 300s, terminating stream");
                                stream_failed = true;
                                yield Err::<Bytes, String>("Stream idle timeout after 300s".to_string());
                                break;
                            }
                        }
                    }

                    if track_image_success && saw_image_data && !stream_failed {
                        image_success_manager.mark_account_success(&image_success_account);
                        image_success_manager
                            .clear_persisted_live_limit(
                                &image_success_account,
                                Some(&image_success_model),
                            );
                    }
                };
                // [QoS] 流式：以首个真实数据块上报 TTFT；非流式已由 peek 上报
                let ttft_seen = std::sync::Arc::new(std::sync::OnceLock::<f64>::new());
                let ttft_seen_cb = ttft_seen.clone();
                let combined_stream: std::pin::Pin<
                    Box<dyn futures::Stream<Item = Result<Bytes, String>> + Send>,
                > = if client_wants_stream {
                    let ttft_start = effective_attempt_start;
                    let ttft_tm = token_manager.clone();
                    let ttft_account = account_id.clone();
                    let ttft_session = affinity_key.clone();
                    let ttft_lease = session_lease.clone();
                    let ttft_is_thinking = crate::proxy::pipeline::qos::payload_enables_thinking(
                        &gemini_body,
                        Some(&mapped_model),
                    ) || effort_hint.is_some()
                        || openai_req.thinking.is_some()
                        || openai_req.reasoning.is_some()
                        || openai_req.reasoning_effort.is_some();
                    let ttft_tokens = estimated_input_tokens;
                    Box::pin(
                        crate::proxy::pipeline::stream_peek::wrap_stream_with_ttft_probe(
                            combined_stream,
                            ttft_start,
                            |bytes: &[u8]| {
                                crate::proxy::pipeline::stream_peek::is_heartbeat(
                                    bytes,
                                    crate::proxy::pipeline::stream_peek::StreamDialect::OpenAi,
                                )
                            },
                            move |ttft| {
                                let _ = ttft_seen_cb.set(ttft);
                                crate::proxy::handlers::common::report_qos_ttft(
                                    &ttft_tm,
                                    &ttft_account,
                                    Some(ttft_session.as_str()),
                                    ttft,
                                    ttft_is_thinking,
                                    Some(ttft_tokens),
                                    ttft_lease.as_ref(),
                                );
                            },
                        ),
                    )
                } else {
                    Box::pin(combined_stream)
                };
                let converted_meta = json!({
                    "protocol": "openai",
                    "trace_id": trace_id,
                    "stage": "converted_codex_response",
                    "original_model": openai_req.model,
                    "mapped_model": mapped_model,
                    "request_type": config.request_type,
                    "attempt": attempt,
                    "status": status.as_u16(),
                    "upstream_url": upstream_url,
                });
                let combined_stream = debug_logger::wrap_stream_with_debug(
                    Box::pin(combined_stream),
                    debug_cfg.clone(),
                    trace_id.clone(),
                    "converted_codex_response",
                    converted_meta,
                );

                if client_wants_stream {
                    // [MULTI-TURN] 保存本次对话的 messages 到 session store（/v1/chat/completions）
                    {
                        let save_msgs = openai_req
                            .messages
                            .iter()
                            .map(|m| {
                                let content_str = match &m.content {
                                    Some(crate::proxy::mappers::openai::OpenAIContent::String(
                                        s,
                                    )) => s.clone(),
                                    _ => String::new(),
                                };
                                json!({"role": m.role, "content": content_str})
                            })
                            .collect::<Vec<_>>();
                        let chat_response_id =
                            format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
                        let entry = crate::proxy::http_session_store::HttpSessionEntry {
                            input_items: save_msgs,
                            instructions: String::new(),
                            model: openai_req.model.clone(),
                            last_accessed: std::time::Instant::now(),
                        };
                        let rid = chat_response_id.clone();
                        tokio::spawn(async move {
                            crate::proxy::http_session_store::save_session(rid, entry).await;
                        });
                    }
                    // 客户端请求流式，返回 SSE
                    let body = Body::from_stream(combined_stream);
                    return Ok(Response::builder()
                        .header("Content-Type", "text/event-stream")
                        .header("Cache-Control", "no-cache")
                        .header("Connection", "keep-alive")
                        .header("X-Accel-Buffering", "no")
                        .header("X-Account-Email", &email)
                        .header("X-Mapped-Model", &mapped_model)
                        .header("X-Upstream-Model", config.final_model.as_str())
                        .header("X-Session-Id", &client_session_id)
                        .header("X-Antigravity-Session-Id", &client_session_id)
                        .timing_stream(
                            &TimingHeaders::new(
                                clean_ms,
                                norm_ms,
                                think_fill_ms,
                                ttft_ms,
                                headers_ms,
                                fallback_overhead_ms,
                                attempts_b64.as_deref().unwrap_or(""),
                            ),
                            crate::proxy::pipeline::timing::StreamTtftCapture(ttft_seen.clone()),
                        )
                        .body(body)
                        .unwrap()
                        .into_response());
                } else {
                    // 客户端请求非流式，但内部强制转为流式
                    // 收集流数据并聚合为 JSON
                    use crate::proxy::mappers::openai::collector::collect_stream_to_json;

                    let collected_response = collect_stream_to_json(combined_stream).await;
                    if let Some(failure) = stream_errors.failure_or_incomplete(
                        collected_response.as_ref().err().map(String::as_str),
                    ) {
                        let failure_status =
                            StatusCode::from_u16(failure.status).unwrap_or(StatusCode::BAD_GATEWAY);
                        if failure.replay_safe() {
                            break 'upstream_response (failure_status, None, failure.body);
                        }
                        // Output has already been consumed: classify the error, but do not replay.
                        retry_state.preserve_http_stop(
                            failure.status,
                            &failure.body,
                            &token_manager.get_sticky_config().await,
                            config.request_type != "image_gen",
                        );
                        failure_statuses.record(failure_status);
                        terminal_stream_usage = failure.usage.clone();
                        last_error = failure.body;
                        break 'attempts;
                    }
                    match collected_response {
                        Ok(full_response) => {
                            info!("[{}] ✓ Stream collected and converted to JSON", trace_id);
                            if debug_logger::is_enabled(&debug_cfg) {
                                let converted_response = serde_json::to_value(&full_response)
                                    .unwrap_or_else(
                                        |e| json!({ "serialization_error": e.to_string() }),
                                    );
                                let payload = json!({
                                    "kind": "exchange_summary",
                                    "protocol": "openai",
                                    "trace_id": trace_id,
                                    "original_codex_request": original_body.as_ref(),
                                    "gemini_request": gemini_body_for_debug.as_ref(),
                                    "converted_codex_response": converted_response,
                                    "gemini_raw_response_ref": "see upstream_response file with the same trace_id",
                                });
                                debug_logger::write_exchange_payload(
                                    &debug_cfg,
                                    Some(&trace_id),
                                    "exchange_summary",
                                    &payload,
                                )
                                .await;
                            }
                            return Ok(Response::builder()
                                .status(StatusCode::OK)
                                .header("Content-Type", "application/json")
                                .header("X-Account-Email", email.as_str())
                                .header("X-Mapped-Model", mapped_model.as_str())
                                .header("X-Upstream-Model", full_response.model.as_str())
                                .header("X-Session-Id", client_session_id.as_str())
                                .header("X-Antigravity-Session-Id", client_session_id.as_str())
                                .timing(&TimingHeaders::new(
                                    clean_ms,
                                    norm_ms,
                                    think_fill_ms,
                                    ttft_ms,
                                    headers_ms,
                                    fallback_overhead_ms,
                                    attempts_b64.as_deref().unwrap_or(""),
                                ))
                                .body(Body::from(
                                    serde_json::to_string(&full_response).unwrap_or_default(),
                                ))
                                .unwrap()
                                .into_response());
                        }
                        Err(e) => {
                            error!("[{}] Stream collection error: {}", trace_id, e);
                            retry_state.preserve_on_stop(
                                token_manager.get_scheduling_mode().await,
                                config.request_type != "image_gen",
                            );
                            return Ok(retry_state.finish_error_response(
                                (
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    format!("Stream collection error: {}", e),
                                )
                                    .into_response(),
                            ));
                        }
                    }
                }
            }

            // QoS/TTFT 已由 peek 在首个数据块上报；这里不能再用“整条响应完成时间”覆盖
            if let Some(effective_item) = fallback_attempts.iter_mut().find(|a| a.is_effective) {
                effective_item.duration_s = ttft_ms / 1000.0;
            }
            let _ = crate::proxy::monitor::CURRENT_UPSTREAM_CAPTURE.try_with(|holder| {
                holder.set_attempts(fallback_attempts.clone());
            });
            let gemini_resp: Value =
                match crate::proxy::pipeline::usage_diagnostics::read_json_response(
                    response,
                    crate::proxy::pipeline::usage_diagnostics::UsageAttempt::current(
                        &config.final_model,
                        &account_id,
                        "openai_chat_json",
                    ),
                )
                .await
                {
                    Ok(value) => value,
                    Err(e) => {
                        retry_state.preserve_on_stop(
                            token_manager.get_scheduling_mode().await,
                            config.request_type != "image_gen",
                        );
                        return Ok(retry_state.finish_error_response(
                            (StatusCode::BAD_GATEWAY, format!("Parse error: {}", e))
                                .into_response(),
                        ));
                    }
                };

            if let Some(confirmation) = session_confirmation.as_mut() {
                confirmation.observe_json(&gemini_resp);
            }

            crate::proxy::thinking_store::capture_gemini_response_with_context(
                &session_id,
                &gemini_resp,
                None,
                response_text_context.as_deref(),
            );

            // [CACHE] 从 Gemini 响应中提取缓存信息，关闭反馈循环
            // 兼容两种格式: cachedContentTokenCount (旧), total_cached_tokens (新)
            if let Some(usage) = gemini_resp.get("usageMetadata") {
                let cached = usage
                    .get("total_cached_tokens")
                    .or_else(|| usage.get("cachedContentTokenCount"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                if cached > 0 {
                    let cm = crate::proxy::cache_manager::global_cache_manager();
                    cm.record_implicit_hit(&_prefix_hash);
                    // [CACHE] 分层统计日志
                    let stats = cm.get_layer_stats();
                    tracing::info!(
                        "[Cache-Opt] Implicit cache HIT: prefix_hash={} cached_tokens={} | L1(SI): {}/{}, L2(Tools): {}/{}, L3(Prefix): {}/{}",
                        &_prefix_hash[.._prefix_hash.len().min(16)],
                        cached,
                        stats.si_hits, stats.si_total,
                        stats.tools_hits, stats.tools_total,
                        stats.prefix_hits, stats.prefix_total,
                    );
                }
            }

            let openai_response =
                crate::proxy::mappers::openai::response::transform_openai_response_with_context(
                    &gemini_resp,
                    Some(&session_id),
                    message_count,
                    Some(&client_tool_names),
                    response_text_context.clone(),
                );
            if debug_logger::is_enabled(&debug_cfg) {
                let converted_response = serde_json::to_value(&openai_response)
                    .unwrap_or_else(|e| json!({ "serialization_error": e.to_string() }));
                let payload = json!({
                    "kind": "exchange_summary",
                    "protocol": "openai",
                    "trace_id": trace_id,
                    "original_codex_request": original_body.as_ref(),
                    "gemini_request": gemini_body_for_debug.as_ref(),
                    "gemini_raw_response": gemini_resp,
                    "converted_codex_response": converted_response,
                });
                debug_logger::write_exchange_payload(
                    &debug_cfg,
                    Some(&trace_id),
                    "exchange_summary",
                    &payload,
                )
                .await;
            }
            return Ok(Response::builder()
                .status(StatusCode::OK)
                .header("Content-Type", "application/json")
                .header("X-Account-Email", email.as_str())
                .header("X-Mapped-Model", mapped_model.as_str())
                .header("X-Upstream-Model", config.final_model.as_str())
                .header("X-Session-Id", client_session_id.as_str())
                .header("X-Antigravity-Session-Id", client_session_id.as_str())
                .timing_core(&TimingHeaders::new(
                    clean_ms,
                    norm_ms,
                    think_fill_ms,
                    ttft_ms,
                    headers_ms,
                    0.0,
                    "",
                ))
                .body(Body::from(
                    serde_json::to_string(&openai_response).unwrap_or_default(),
                ))
                .unwrap()
                .into_response());
        };

        failure_statuses.record(status);
        let status_code = status.as_u16();
        last_error = format!("HTTP {}: {}", status_code, error_text);
        let classification = crate::proxy::pipeline::UpstreamClassification::classify(
            status_code,
            &error_text,
            retry_after.as_deref(),
        );
        let scheduling = token_manager.get_sticky_config().await;

        // [Pipeline Auto-Heal] 503 过载 / 400 tool_choice 冲突 / 400 损坏图片：
        // 三者均是对已转出的 Gemini 报文的原地修复，统一在同账号重试（换号只会重蹈覆车）。
        if let Some(heal) = crate::proxy::pipeline::auto_heal::try_auto_heal_with_retry_policy(
            &classification,
            status_code,
            &error_text,
            retry_after.as_deref(),
            &mut gemini_body,
            &mut heal_flags,
            &mut retry_state,
            &account_id,
            &scheduling,
            config.request_type != "image_gen",
        ) {
            force_rotate = false;
            crate::proxy::stage_trace::StageTrace::alert_current(
                "self_heal",
                format!(
                    "protocol=openai kind={} status={} account={}",
                    heal.kind, status_code, email
                ),
            );
            if !apply_retry_strategy(
                RetryStrategy::GraceRetry(std::time::Duration::from_millis(heal.delay_ms)),
                attempt,
                max_attempts,
                status_code,
                &trace_id,
            )
            .await
            {
                break;
            }
            if config.request_type != "image_gen" {
                retry_credentials = Some((
                    access_token.clone(),
                    project_id.clone(),
                    email.clone(),
                    account_id.clone(),
                    0,
                    session_lease.clone(),
                ));
            }
            continue;
        }

        // CacheFirst policy is shared across protocols and runs before account penalties.
        if let Some(strategy) = retry_state.cache_first_failure(
            &account_id,
            status_code,
            &error_text,
            retry_after.as_deref(),
            &token_manager.get_sticky_config().await,
            config.request_type != "image_gen",
        ) {
            if apply_retry_strategy(strategy, attempt, max_attempts, status_code, &trace_id).await {
                retry_credentials = Some((
                    access_token.clone(),
                    project_id.clone(),
                    email.clone(),
                    account_id.clone(),
                    0,
                    session_lease.clone(),
                ));
                continue;
            }
            break;
        }

        // [New] 打印错误报文日志
        tracing::error!(
            "[OpenAI-Upstream] Error Response {}: {}",
            status_code,
            error_text
        );
        if debug_logger::is_enabled(&debug_cfg) {
            let payload = json!({
                "kind": "upstream_response_error",
                "protocol": "openai",
                "trace_id": trace_id,
                "original_model": openai_req.model,
                "mapped_model": mapped_model,
                "request_type": config.request_type,
                "attempt": attempt,
                "status": status_code,
                "upstream_url": upstream_url,
                "account": mask_email(&email),
                "error_text": error_text,
            });
            debug_logger::write_debug_payload(
                &debug_cfg,
                Some(&trace_id),
                "upstream_response_error",
                &payload,
            )
            .await;
        }

        // [Grace Policy] SchedulingMode 不参与此裁决：429 微间隙 GraceRetry 属于
        // “账号池保护”（Fixes #3329/#3409），单账号等待已由 RequestRetryState 限制为
        // 每账号一次，不存在无界等待。Claude 因历史原因仍按调度模式决定是否原地等待。

        // 确定重试策略：传入当前 attempt 与 pool_size，执行智能自适应裁决
        let strategy = retry_state.determine_strategy_adaptive(
            &account_id,
            status_code,
            &error_text,
            retry_after.as_deref(),
            retried_without_thinking,
            attempt,
            pool_size,
        );
        // 统一流水线决策判定：协议无关的限流与错误判定

        if classification.is_model_not_found() {
            // [Pipeline Policy] UA 门禁短路 + 单模型临时熔断 + 是否换号的决策已下沉
            let custom_ua = state.upstream.get_user_agent_override().await;
            let can_rotate = crate::proxy::pipeline::policy::apply_model_not_found_with_ua(
                &mapped_model,
                status_code,
                &error_text,
                custom_ua.as_deref(),
                attempt,
                pool_size,
                || token_manager.mark_model_unsupported(&account_id, &mapped_model, Some(900)),
            );

            if can_rotate {
                tracing::warn!(
                    "[{}] 上游报错模型不可用 (HTTP {})，已标记账号 {} 对模型 [{}] 临时熔断，继续换号重试 ({}/{})...",
                    trace_id, status_code, email, mapped_model, attempt, pool_size
                );
                force_rotate = true;
                continue;
            }

            tracing::warn!(
                "[{}] Pipeline: Target model [{}] not found on upstream (HTTP {}). Pool exhausted without account-level lockout.",
                trace_id, mapped_model, status_code
            );
            let dual_err = crate::proxy::handlers::common::build_dual_track_error_with_ua(
                "openai",
                status_code,
                &mapped_model,
                &error_text,
                custom_ua.as_deref(),
            );
            return Ok((
                StatusCode::from_u16(status_code).unwrap_or(StatusCode::NOT_FOUND),
                [
                    ("X-Account-Email", email.as_str()),
                    ("X-Mapped-Model", mapped_model.as_str()),
                ],
                Json(dual_err),
            )
                .into_response());
        }

        if classification.is_thought_signature_error() {
            if !retried_without_thinking {
                retried_without_thinking = true;
                tracing::warn!(
                    "[{}] Pipeline: Thinking signature error detected on upstream (HTTP {}). Surgically purging corrupted signatures and retrying on same account.",
                    trace_id, status_code
                );
                // 1. 精准定向净化 ThinkingStore 中的异构污染签名（保留思考文本与健康签名）
                session_scope.purge_signatures(&mapped_model);
                // 2. 请求体不做特化：重试轮次由流水线统一将思考块降级为正文并清理签名
                // 3. 保持同一账号原地重试
                force_rotate = false;
                continue;
            } else {
                tracing::warn!(
                    "[{}] Pipeline: Thinking signature error persisted after retry without thinking. Terminating retry loop.",
                    trace_id
                );
            }
        }

        let should_mark_limited = classification.should_lock_account_on_attempt(
            status_code,
            &error_text,
            retry_after.as_deref(),
            attempt,
        );
        let needs_quota_refresh = if should_mark_limited {
            token_manager
                .mark_rate_limited_fast(
                    &email,
                    status_code,
                    retry_after.as_deref(),
                    &error_text,
                    Some(&mapped_model),
                )
                .await
        } else {
            false
        };
        if !matches!(&strategy, RetryStrategy::GraceRetry(_)) {
            drop(image_permit.take());
        }
        if needs_quota_refresh {
            token_manager
                .refresh_quota_lock_after_fast_mark(&email, Some(&mapped_model))
                .await;
        }

        if classification.abandons_sticky_account() {
            token_manager.abandon_session_lease(&mut session_lease);
        }

        // [FIX] 403 时优先检测 VALIDATION_REQUIRED 并设置 is_forbidden / validation_block 状态，确保及时提取 URL 与更新 UI
        if status_code == 403 {
            if let Some(acc_id) = token_manager.get_account_id_by_email(&email) {
                if error_text.contains("VALIDATION_REQUIRED")
                    || error_text.contains("verify your account")
                    || error_text.contains("Verify your account")
                    || error_text.contains("validation_url")
                {
                    tracing::warn!(
                        "[OpenAI] VALIDATION_REQUIRED detected on account {}, temporarily blocking",
                        email
                    );
                    let block_minutes = 10i64;
                    let block_until = chrono::Utc::now().timestamp() + (block_minutes * 60);

                    if let Err(e) = token_manager
                        .set_validation_block_public(&acc_id, block_until, &error_text)
                        .await
                    {
                        tracing::error!("Failed to set validation block: {}", e);
                    }
                } else {
                    // 设置 is_forbidden 状态并持久化（同时传入请求模型与映射模型，彻底消除动态改写盲区）
                    if let Err(e) = token_manager
                        .set_forbidden_with_models(
                            &acc_id,
                            &error_text,
                            Some(&mapped_model),
                            Some(&config.final_model),
                        )
                        .await
                    {
                        tracing::error!("Failed to set forbidden status: {}", e);
                    }
                }
            }
        }

        // 执行退避
        if apply_retry_strategy(
            strategy.clone(),
            attempt,
            max_attempts,
            status_code,
            &trace_id,
        )
        .await
        {
            if matches!(strategy, RetryStrategy::GraceRetry(_)) {
                retry_credentials = Some((
                    access_token.clone(),
                    project_id.clone(),
                    email.clone(),
                    account_id.clone(),
                    0,
                    session_lease.clone(),
                ));
            }
            // [NEW] Apply Client Adapter "let_it_crash" strategy
            if let Some(adapter) = &client_adapter {
                if adapter.let_it_crash() && attempt > 0 {
                    tracing::warn!(
                        "[OpenAI] let_it_crash active: Aborting retries after attempt {}",
                        attempt
                    );
                    break;
                }
            }

            // 判断是否需要轮换账号
            if !should_rotate_account(status_code, Some(&strategy)) {
                debug!(
                    "[{}] Keeping same account for status {} (Grace Retry or Server Issue)",
                    trace_id, status_code
                );
                force_rotate = false;
            } else {
                crate::proxy::pipeline::rotation::record_attempted_account(
                    &mut attempted_accounts,
                    &account_id,
                    pool_size,
                    &mut force_rotate,
                );
            }

            tracing::warn!(
                "OpenAI Upstream {} on {} attempt {}/{}, rotating account",
                status_code,
                email,
                attempt + 1,
                max_attempts
            );
            continue;
        }

        // [FIX session-1M] 上游按 sessionId 在服务端累计会话输入,长工具循环会把累计推过 1M,
        // 之后该 sessionId 的所有请求都 400 "input token count exceeds ... 1048576"。
        // 给 (账号, 对话) 的 sessionId 升代并立即重试:新 sessionId = 上游全新会话,对话无感恢复。
        if status_code == 400 && error_text.contains("exceeds the maximum number of tokens") {
            let generation = crate::proxy::common::session::bump_session(&account_id, &store_key);
            tracing::warn!(
                "[OpenAI] Upstream session token accumulation exceeded 1M on account {}. sessionId bumped to generation {}, retrying with a fresh upstream session.",
                email, generation
            );
            continue; // 重试:下一轮 transform 时读取新代数,派生全新 sessionId
        }

        // 404 等由于模型配置或路径错误的 HTTP 异常，直接报错返回双轨制友好报文，不进行无效轮换
        error!(
            "OpenAI Upstream non-retryable error {} on account {}: {}",
            status_code, email, error_text
        );
        let dual_err = crate::proxy::handlers::common::build_dual_track_error(
            "openai",
            status_code,
            &mapped_model,
            &error_text,
        );
        return Ok((
            status,
            [
                ("X-Account-Email", email.as_str()),
                ("X-Mapped-Model", mapped_model.as_str()),
            ],
            Json(dual_err),
        )
            .into_response());
    }

    // 所有尝试均失败：仅当全部结构化失败状态均为 429 时返回 429
    let final_status = failure_statuses.final_status();
    let headers = crate::proxy::handlers::common::build_token_error_headers(
        Some(mapped_model.as_str()),
        last_email.as_deref(),
        &last_error,
    );

    let mut dual_err = crate::proxy::handlers::common::build_dual_track_error(
        "openai",
        final_status.as_u16(),
        &mapped_model,
        &last_error,
    );

    crate::proxy::pipeline::stream_error::attach_error_usage(
        &mut dual_err,
        "openai",
        terminal_stream_usage.as_ref(),
    );
    Ok(retry_state.finish_error_response((final_status, headers, Json(dual_err)).into_response()))
}

/// Only requests without a cursor start a new Responses conversation.
async fn load_responses_parent(
    id: Option<&str>,
    strict: bool,
) -> Result<
    Option<(
        crate::proxy::http_session_store::HttpSessionEntry,
        crate::proxy::http_session_store::SessionParent,
    )>,
    Value,
> {
    let Some(id) = id else {
        return Ok(None);
    };
    let parent = crate::proxy::http_session_store::get_session_with_parent(id).await;
    if strict && parent.is_none() {
        return Err(
            json!({"error":{"type":"invalid_request_error","code":"previous_response_not_found",
            "param":"previous_response_id","message":format!("Previous response '{}' was not found",id)}}),
        );
    }
    Ok(parent)
}

fn validate_openai_tool_schemas(tools: Option<&[Value]>) -> Result<(), String> {
    for tool in tools.unwrap_or_default() {
        if tool["type"] == "namespace" {
            validate_openai_tool_schemas(tool["tools"].as_array().map(Vec::as_slice))?;
        } else if let Some(schema) = tool
            .get("parameters")
            .or_else(|| tool["function"].get("parameters"))
        {
            crate::proxy::common::json_schema::validate_tool_schema(schema)?;
        }
    }
    Ok(())
}

// Foreign reasoning may carry both readable context and provider-encrypted
// state. Prefer the readable representation when adapting it to Gemini.
fn responses_plaintext_summary(item: &Value) -> Option<String> {
    for key in ["summary", "text", "thought", "content"] {
        let value = &item[key];
        let text = if let Some(text) = value.as_str() {
            text.to_owned()
        } else if let Some(parts) = value.as_array() {
            parts
                .iter()
                .filter_map(|part| part.as_str().or_else(|| part["text"].as_str()))
                .filter(|text| !text.trim().is_empty())
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            continue;
        };
        if !text.trim().is_empty() {
            return Some(text);
        }
    }
    None
}

fn responses_reasoning_message(item: &Value) -> Value {
    let text = responses_plaintext_summary(item).unwrap_or_default();
    let signature = item
        .get("thoughtSignature")
        .or_else(|| item.get("thought_signature"))
        .or_else(|| item.get("signature"))
        .and_then(Value::as_str)
        .filter(|signature| {
            crate::proxy::thinking_store::is_real_signature(signature)
                && (crate::proxy::thinking_store::is_likely_gemini_signature(signature)
                    || crate::proxy::thinking_store::is_claude_signature(signature))
        });
    if let Some(signature) = signature {
        json!({"role":"assistant", "content":"", "reasoning_content":text, "thoughtSignature":signature})
    } else {
        // An external summary is readable history, not a signed Gemini thought.
        // Plain content survives non-thinking targets and is injected only once.
        json!({"role":"assistant", "content":text})
    }
}

fn normalize_responses_plaintext(
    body: &mut Value,
) -> Result<(), (&'static str, &'static str, &'static str)> {
    // Validate before changing input: failed requests must not consume history.
    validate_responses_continuation(body)?;
    let Some(items) = body.get_mut("input").and_then(Value::as_array_mut) else {
        return Ok(());
    };
    let mut normalized = Vec::with_capacity(items.len());
    for mut item in std::mem::take(items) {
        match item["type"].as_str() {
            Some("reasoning") if responses_plaintext_summary(&item).is_some() => {
                if let Some(fields) = item.as_object_mut() {
                    fields.remove("encrypted_content");
                }
                normalized.push(item);
            }
            Some("compaction_summary") => {
                let text = responses_plaintext_summary(&item);
                let mut marker = json!({"type": "compaction_summary"});
                if let Some(id) = item.get("id") {
                    marker["id"] = id.clone();
                }
                // Keep the existing reset signal until history is merged. The
                // readable summary must be a separate item because reset
                // markers themselves are intentionally removed by the store.
                normalized.push(marker);
                if let Some(text) = text {
                    normalized.push(json!({"role":"user", "content":text}));
                }
            }
            _ => normalized.push(item),
        }
    }
    *items = normalized;
    Ok(())
}

fn validate_responses_continuation(
    body: &Value,
) -> Result<(), (&'static str, &'static str, &'static str)> {
    if body
        .get("previous_response_id")
        .is_some_and(|v| !v.is_null() && !v.is_string())
    {
        return Err((
            "invalid_type",
            "previous_response_id",
            "previous_response_id must be a string or null",
        ));
    }
    if body["input"].as_array().is_some_and(|items| {
        items.iter().any(|item| {
            item["type"] == "compaction"
                || (item["type"] == "compaction_summary"
                    && responses_plaintext_summary(item).is_none()
                    && ["encrypted_content", "summary", "text", "thought", "content"]
                        .iter()
                        .filter_map(|key| item.get(key))
                        .any(|value| match value {
                            Value::Null => false,
                            Value::String(text) => !text.trim().is_empty(),
                            Value::Array(parts) => !parts.is_empty(),
                            Value::Object(fields) => !fields.is_empty(),
                            _ => true,
                        }))
        })
    }) {
        return Err(("unsupported_compaction", "input", "Cannot restore a Responses compaction payload; send the compacted plaintext transcript instead"));
    }
    // Validate only Responses protocol containers. Business tool arguments may
    // legitimately contain fields named type, encrypted_content or file_id.
    fn unsupported_content(value: &Value) -> bool {
        match value {
            Value::Array(parts) => parts.iter().any(unsupported_content),
            Value::Object(part) => {
                if part.get("type").is_some_and(|t| t == "input_file") {
                    if part.get("file_id").is_some_and(|id| !id.is_null()) {
                        return true;
                    }
                    let Ok(file) = serde_json::from_value::<
                        crate::proxy::mappers::openai::models::OpenAIInputFile,
                    >(value.clone()) else {
                        return true;
                    };
                    validate_openai_file(&file).is_err()
                } else {
                    part.get("type").is_some_and(|t| t == "input_image")
                        && part.get("file_id").is_some_and(|id| !id.is_null())
                }
            }
            _ => false,
        }
    }
    if let Some(items) = body["input"].as_array() {
        for item in items {
            if item["type"] == "reasoning"
                && item
                    .get("encrypted_content")
                    .is_some_and(|v| !v.is_null() && v != "")
                && responses_plaintext_summary(item).is_none()
            {
                return Err((
                    "unsupported_reasoning",
                    "input",
                    "Cannot restore external encrypted reasoning; send plaintext reasoning instead",
                ));
            }
            let content = match item["type"].as_str() {
                Some("function_call_output" | "custom_tool_call_output") => {
                    item["output"].get("content").unwrap_or(&item["output"])
                }
                None | Some("message") => &item["content"],
                _ => item,
            };
            if unsupported_content(content) {
                return Err(("unsupported_file_input", "input", "File input cannot be represented: supply one valid inline base64 document or public file URL; external file IDs are not supported"));
            }
        }
    }
    fn unique_tools(
        tools: &[Value],
        namespaces: &mut Vec<String>,
        names: &mut std::collections::HashSet<String>,
    ) -> bool {
        for tool in tools {
            if tool["type"] == "namespace" {
                if let (Some(ns), Some(children)) =
                    (tool["name"].as_str(), tool["tools"].as_array())
                {
                    namespaces.push(ns.to_owned());
                    let unique = unique_tools(children, namespaces, names);
                    namespaces.pop();
                    if !unique {
                        return false;
                    }
                }
            } else if let Some(name) = tool["function"]["name"]
                .as_str()
                .or_else(|| tool["name"].as_str())
            {
                let qualified = namespaces.iter().rev().fold(name.to_owned(), |child, ns| {
                    crate::proxy::mappers::openai::request::qualify_namespace_tool_name(ns, &child)
                });
                if !names.insert(qualified) {
                    return false;
                }
            }
        }
        true
    }
    if body["tools"]
        .as_array()
        .is_some_and(|tools| !unique_tools(tools, &mut Vec::new(), &mut Default::default()))
    {
        return Err((
            "ambiguous_tool_name",
            "tools",
            "Tool names must remain unique after namespace qualification",
        ));
    }
    if validate_openai_tool_schemas(body["tools"].as_array().map(Vec::as_slice)).is_err() {
        return Err((
            "unsupported_tool_schema",
            "tools",
            "Tool schema is unsupported or exceeds the schema resource budget",
        ));
    }
    Ok(())
}

fn responses_store_enabled(body: &Value) -> bool {
    body.get("store").and_then(Value::as_bool) != Some(false)
}

#[inline]
pub(crate) fn is_responses_api_path(path: &str) -> bool {
    path == "/v1/responses" || path == "/responses" || path == "/responses/compact"
}

/// 处理 Legacy Completions API (/v1/completions)
/// 将 Prompt 转换为 Chat Message 格式，复用 handle_chat_completions
pub async fn handle_completions(
    axum::extract::OriginalUri(uri): axum::extract::OriginalUri,
    State(state): State<AppState>,
    headers: HeaderMap,
    upstream_recorder: Option<
        axum::extract::Extension<crate::proxy::monitor::UpstreamRequestBodyHolder>,
    >,
    user_identity: Option<
        axum::extract::Extension<crate::proxy::middleware::auth::UserTokenIdentity>,
    >,
    Json(mut body): Json<Value>,
) -> Response {
    let clean_start = std::time::Instant::now();
    debug!(
        "Received /v1/completions or /v1/responses payload: {} bytes",
        serialized_json_len(&body)
    );
    let debug_cfg = state.debug_logging.read().await.clone();
    let original_body = debug_logger::is_enabled(&debug_cfg).then(|| {
        crate::proxy::payload_audit::reorder_payload_fields(&debug_value_without_inline_data(&body))
    });
    let is_responses_api = is_responses_api_path(uri.path());
    let is_codex_style = body.get("input").is_some()
        || body.get("instructions").is_some()
        || (is_responses_api && body.get("previous_response_id").is_some());
    if is_responses_api {
        if let Err((code, param, message)) = normalize_responses_plaintext(&mut body) {
            return (StatusCode::BAD_REQUEST, Json(json!({"error":{"type":"invalid_request_error","code":code,"param":param,"message":message}}))).into_response();
        }
    }
    let store_response = responses_store_enabled(&body);

    // [MODEL GATE] /v1/completions 与 /responses 系列同样必须在选号前完成模型校验：
    // 不存在 → 404 model not exist；全池无配额 → 502 no quota accounts。
    // 两者均为网关自答：直接返回，不罚号、不解绑粘性、不进入上游重试。
    // 未携带 model 的请求不在此处拦截（交由协议自身校验）。
    if let Some(requested_model) = body.get("model").and_then(|v| v.as_str()) {
        if let Err(rejection) = crate::proxy::handlers::common::enforce_model_gate(
            &state,
            Some(requested_model),
            if is_responses_api {
                "openai_responses"
            } else {
                "openai_completions"
            },
            upstream_recorder.as_ref().map(|recorder| &recorder.0),
        )
        .await
        {
            return rejection;
        }
    }

    // [MULTI-TURN] 支持 previous_response_id 链式历史恢复
    // 当客户端通过 HTTP POST /v1/responses 传入 previous_response_id 时，
    // 从服务器端 session store 取出上一轮的历史，合并到本轮的 input 中
    let previous_response_id = body
        .get("previous_response_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let mut explicit_session_id =
        crate::proxy::thinking_store::stable_session_winner(&headers, Some(&body), None);
    let response_id_for_save = format!("resp-{}", uuid::Uuid::new_v4());
    let http_tool_call_cache: std::collections::HashMap<String, serde_json::Value> =
        std::collections::HashMap::new();
    let mut session_parent = None;
    let mut stored_routing_session_id = None;
    let mut session_delta_input = Vec::new();
    if is_codex_style {
        let existing_input = take_responses_input_items(&mut body);
        let selected =
            match load_responses_parent(previous_response_id.as_deref(), is_responses_api).await {
                Ok(selected) => selected,
                Err(error) => return (StatusCode::BAD_REQUEST, Json(error)).into_response(),
            };
        let existing_input = match crate::proxy::http_session_store::resolve_input_references(
            selected.as_ref().map(|(session, _)| session.input_items.as_slice()).unwrap_or_default(),
            existing_input,
        ) {
            Ok(input) => input,
            Err(message) => return (StatusCode::BAD_REQUEST, Json(json!({"error":{"type":"invalid_request_error","code":"item_reference_not_found","param":"input","message":message}}))).into_response(),
        };
        let merged = if let Some((session, parent)) = selected {
            if explicit_session_id.is_none() {
                explicit_session_id = parent.scope_identity().map(str::to_owned);
            }
            stored_routing_session_id = Some(parent.routing_session_id().to_string());
            let prepared = crate::proxy::http_session_store::prepare_session_input_with_storage(
                session.input_items,
                existing_input,
                &http_tool_call_cache,
                store_response,
            );
            session_delta_input = prepared.delta;
            if store_response && !prepared.reset_parent {
                session_parent = Some(parent);
            }
            if let Some(obj) = body.as_object_mut() {
                if !obj.contains_key("instructions") && !session.instructions.is_empty() {
                    obj.insert("instructions".to_string(), json!(session.instructions));
                }
                if !obj.contains_key("model") && !session.model.is_empty() {
                    obj.insert("model".to_string(), json!(session.model));
                }
            }
            tracing::debug!(
                "[MultiTurn] Restored session from prev_id={:?}, {} items in history",
                previous_response_id,
                prepared.merged.len()
            );
            prepared.merged
        } else {
            if store_response {
                session_delta_input = existing_input.clone();
            }
            existing_input
        };

        if let Some(obj) = body.as_object_mut() {
            obj.insert("input".to_string(), Value::Array(merged));
        }
        if let Err(message) = validate_responses_input_image_limits(body.get("input")) {
            return (StatusCode::BAD_REQUEST, message).into_response();
        }
    }
    let routing_session_id = responses_routing_session_id(
        explicit_session_id.as_deref(),
        previous_response_id.as_deref(),
        stored_routing_session_id.as_deref(),
        &response_id_for_save,
    );
    let signature_read_key = previous_response_id.clone();

    let mut bounded_session_input = None;

    // 1. Convert Payload to Messages (Shared Chat Format)
    if is_codex_style {
        let instructions = body
            .get("instructions")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let (interaction_ledger, mut step_markers) = codex_ledger_from_body(&body);
        let input_items = body
            .as_object_mut()
            .and_then(|obj| obj.remove("input"))
            .and_then(|value| match value {
                Value::Array(items) => Some(items),
                _ => None,
            })
            .unwrap_or_default();
        bounded_session_input = store_response.then(|| {
            session_delta_input
                .drain(..)
                .map(into_history_without_inline_media)
                .filter(|item| !item.is_null())
                .collect()
        });

        let mut messages = Vec::new();

        // System Instructions
        if !instructions.is_empty() {
            messages.push(json!({ "role": "system", "content": instructions }));
        }

        let mut call_id_to_name = std::collections::HashMap::new();
        let mut skipped_incomplete_custom_call_ids = std::collections::HashSet::new();

        // Pass 1: Build Call ID to Name Map
        {
            for item in &input_items {
                let item_type = responses_input_item_type(&item).to_string();
                if item_type == "custom_tool_call"
                    && item.get("status").and_then(|v| v.as_str()) == Some("incomplete")
                {
                    if let Some(call_id) = item
                        .get("call_id")
                        .and_then(|v| v.as_str())
                        .or_else(|| item.get("id").and_then(|v| v.as_str()))
                    {
                        skipped_incomplete_custom_call_ids.insert(call_id.to_string());
                    }
                    continue;
                }
                match item_type.as_str() {
                    "function_call" | "custom_tool_call" | "local_shell_call"
                    | "web_search_call" => {
                        let call_id = item
                            .get("call_id")
                            .and_then(|v| v.as_str())
                            .or_else(|| item.get("id").and_then(|v| v.as_str()))
                            .unwrap_or("unknown");

                        let name = if item_type == "local_shell_call" {
                            "shell".to_string()
                        } else if item_type == "web_search_call" {
                            "google_search".to_string()
                        } else {
                            responses_item_tool_name(item).unwrap_or_else(|| "unknown".to_string())
                        };

                        call_id_to_name.insert(call_id.to_string(), name.to_string());
                        tracing::debug!("Mapped call_id {} to name {}", call_id, name);
                    }
                    _ => {}
                }
            }
        }

        // Pass 2: Map durable conversation items to Gemini messages. Visible
        // assistant commentary stays in Codex's local transcript and must not
        // be replayed as model history.
        {
            for mut item in input_items {
                let item_type = responses_input_item_type(&item).to_string();
                let step_marker = step_markers.pop_front();
                if item_type == "custom_tool_call"
                    && item.get("status").and_then(|v| v.as_str()) == Some("incomplete")
                {
                    continue;
                }
                match item_type.as_str() {
                    "message" => {
                        let role = item
                            .get("role")
                            .and_then(Value::as_str)
                            .unwrap_or("user")
                            .to_string();
                        let (content, joined_text, has_media) =
                            responses_message_content(&mut item, step_marker);
                        if is_codex_transcript_only_assistant_message(&item, &joined_text) {
                            continue;
                        }

                        let reasoning_content = item
                            .get("reasoning_content")
                            .or_else(|| item.get("thought"))
                            .and_then(Value::as_str)
                            .map(str::to_string);
                        let signature = item
                            .get("thoughtSignature")
                            .or_else(|| item.get("thought_signature"))
                            .or_else(|| item.get("signature"))
                            .and_then(Value::as_str)
                            .map(str::to_string);

                        // 若为 assistant 角色且没有任何实际正文、图像或思考元数据，属于纯空占位消息，予以过滤
                        if role == "assistant"
                            && joined_text.trim().is_empty()
                            && !has_media
                            && reasoning_content.is_none()
                            && signature.is_none()
                        {
                            continue;
                        }

                        let mut message = json!({"role":role,"content":content});

                        if let Some(rc) = reasoning_content {
                            if let Some(obj) = message.as_object_mut() {
                                obj.insert("reasoning_content".to_string(), json!(rc));
                            }
                        }
                        if let Some(sig) = signature {
                            if let Some(obj) = message.as_object_mut() {
                                obj.insert("thoughtSignature".to_string(), json!(sig));
                            }
                        }

                        messages.push(message);
                    }
                    "reasoning" => {
                        messages.push(responses_reasoning_message(&item));
                    }
                    "function_call" | "custom_tool_call" | "local_shell_call"
                    | "web_search_call" => {
                        let mut name = responses_item_tool_name(&item)
                            .unwrap_or_else(|| "unknown".to_string());
                        let mut args_str = item
                            .get("arguments")
                            .and_then(|v| v.as_str())
                            .unwrap_or("{}")
                            .to_string();
                        let call_id = item
                            .get("call_id")
                            .and_then(|v| v.as_str())
                            .or_else(|| item.get("id").and_then(|v| v.as_str()))
                            .unwrap_or("unknown");

                        // Handle native shell calls
                        if item_type == "custom_tool_call" {
                            if let Some(input) = item.get("input").and_then(|v| v.as_str()) {
                                args_str = serde_json::to_string(&json!({ "input": input }))
                                    .unwrap_or_else(|_| "{}".to_string());
                            }
                        } else if item_type == "local_shell_call" {
                            name = "shell".to_string();
                            if let Some(action) = item.get("action") {
                                if let Some(exec) = action.get("exec") {
                                    let mut args_obj = serde_json::Map::new();
                                    if let Some(cmd) = exec.get("command") {
                                        let cmd_val = if cmd.is_string() {
                                            json!([cmd])
                                        } else {
                                            cmd.clone()
                                        };
                                        args_obj.insert("command".to_string(), cmd_val);
                                    }
                                    if let Some(wd) =
                                        exec.get("working_directory").or(exec.get("workdir"))
                                    {
                                        args_obj.insert("workdir".to_string(), wd.clone());
                                    }
                                    args_str = serde_json::to_string(&args_obj)
                                        .unwrap_or("{}".to_string());
                                }
                            }
                        } else if item_type == "web_search_call" {
                            name = "google_search".to_string();
                            if let Some(action) = item.get("action") {
                                let mut args_obj = serde_json::Map::new();
                                if let Some(q) = action.get("query") {
                                    args_obj.insert("query".to_string(), q.clone());
                                }
                                args_str =
                                    serde_json::to_string(&args_obj).unwrap_or("{}".to_string());
                            }
                        }

                        let tool_sig = item
                            .get("thoughtSignature")
                            .or_else(|| item.get("thought_signature"))
                            .or_else(|| item.get("signature"))
                            .and_then(Value::as_str)
                            .map(str::to_string);

                        let mut tc_obj = json!({
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": args_str
                            }
                        });
                        if let Some(ref s) = tool_sig {
                            tc_obj["thoughtSignature"] = json!(s);
                        }

                        let mut message = json!({
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [ tc_obj ]
                        });
                        if let Some(ref s) = tool_sig {
                            message["thoughtSignature"] = json!(s);
                        }
                        messages.push(message);
                    }
                    "function_call_output" | "custom_tool_call_output" => {
                        let call_id = item
                            .get("call_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown")
                            .to_string();
                        if item_type == "custom_tool_call_output"
                            && skipped_incomplete_custom_call_ids.contains(&call_id)
                        {
                            tracing::warn!(
                                "Skipping output for incomplete custom tool call {}",
                                call_id
                            );
                            continue;
                        }
                        let (mut output_str, output_media) = responses_tool_output_parts(&mut item);

                        let name = if let Some(name) = call_id_to_name.get(&call_id).cloned() {
                            name
                        } else if item_type == "custom_tool_call_output" {
                            tracing::warn!(
                                "Skipping orphan custom_tool_call_output for unknown call_id {}",
                                call_id
                            );
                            continue;
                        } else {
                            tracing::warn!(
                                "Unknown function_call_output tool name for call_id {}, defaulting to 'shell'",
                                call_id
                            );
                            "shell".to_string()
                        };

                        output_str = prefix_with_step_marker(step_marker, output_str);
                        let output_content =
                            build_responses_tool_output_content(output_str, output_media);

                        messages.push(json!({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": name,
                            "content": output_content
                        }));
                    }
                    _ => {}
                }
            }
        }
        if let Some(obj) = body.as_object_mut() {
            obj.insert("messages".to_string(), Value::Array(messages));
            if let Some(ledger) = interaction_ledger {
                obj.insert("_interaction_ledger".to_string(), json!(ledger));
            }
        }
    } else if let Some(prompt_val) = body.get("prompt") {
        // Legacy OpenAI Style: prompt -> Chat
        let prompt_str = match prompt_val {
            Value::String(s) => s.clone(),
            Value::Array(arr) => arr
                .iter()
                .filter_map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
            _ => prompt_val.to_string(),
        };
        let messages = json!([ { "role": "user", "content": prompt_str } ]);
        if let Some(obj) = body.as_object_mut() {
            obj.remove("prompt");
            obj.insert("messages".to_string(), messages);
        }
    }

    // 2. Reuse handle_chat_completions logic (wrapping with custom handler or direct call)
    // Actually, due to SSE handling differences (Codex uses different event format), we replicate the loop here or abstract it.
    // For now, let's replicate the core loop but with Codex specific SSE mapping.

    // [Fix Phase 2] Backport normalization logic from handle_chat_completions
    // Handle "instructions" + "input" (Codex style) -> system + user messages
    // This is critical because `transform_openai_request` expects `messages` to be populated.

    // [FIX] 检查是否已经有 messages (被第一次标准化处理过)
    let has_codex_fields = body.get("instructions").is_some() || body.get("input").is_some();
    let already_normalized = body
        .get("messages")
        .and_then(|m| m.as_array())
        .map(|arr| !arr.is_empty())
        .unwrap_or(false);

    // 只有在未标准化时才进行简单转换
    if has_codex_fields && !already_normalized {
        tracing::debug!("[Codex] Performing simple normalization (messages not yet populated)");

        let mut messages = Vec::new();

        // instructions -> system message
        if let Some(inst) = body.get("instructions").and_then(|v| v.as_str()) {
            if !inst.is_empty() {
                messages.push(json!({
                    "role": "system",
                    "content": inst
                }));
            }
        }

        // input -> user message (支持对象数组形式的对话历史)
        if let Some(input) = body.get("input") {
            if let Some(s) = input.as_str() {
                messages.push(json!({
                    "role": "user",
                    "content": s
                }));
            } else if let Some(arr) = input.as_array() {
                // 判断是消息对象数组还是简单的内容块/字符串数组
                let is_message_array = arr
                    .first()
                    .and_then(|v| v.as_object())
                    .map(|obj| obj.contains_key("role") || obj.contains_key("type"))
                    .unwrap_or(false);

                if is_message_array {
                    // 深度识别：像处理 messages 一样处理 input 数组，并自动映射 Responses API 的工具流
                    for item in arr {
                        if let Some(obj) = item.as_object() {
                            let item_type = responses_input_item_type(item);
                            if !item_type.is_empty() {
                                match item_type {
                                    "message" => {
                                        let role = obj
                                            .get("role")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("user");
                                        let content =
                                            obj.get("content").cloned().unwrap_or(json!(""));
                                        messages.push(json!({ "role": role, "content": content }));
                                    }
                                    "function_call" | "custom_tool_call" => {
                                        let call_id = obj
                                            .get("call_id")
                                            .or_else(|| obj.get("id"))
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("");
                                        let name =
                                            obj.get("name").and_then(|v| v.as_str()).unwrap_or("");
                                        let mut arguments = obj
                                            .get("arguments")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("")
                                            .to_string();
                                        if item_type == "custom_tool_call" {
                                            if let Some(input) =
                                                obj.get("input").and_then(|v| v.as_str())
                                            {
                                                arguments = serde_json::to_string(
                                                    &json!({ "input": input }),
                                                )
                                                .unwrap_or_else(|_| "{}".to_string());
                                            }
                                        }
                                        messages.push(json!({
                                            "role": "assistant",
                                            "content": "",
                                            "tool_calls": [{
                                                "id": if call_id.is_empty() { "call_unknown" } else { call_id },
                                                "type": "function",
                                                "function": { "name": name, "arguments": arguments },
                                            }],
                                        }));
                                    }
                                    "function_call_output" | "custom_tool_call_output" => {
                                        let call_id = obj
                                            .get("call_id")
                                            .or_else(|| obj.get("tool_call_id"))
                                            .or_else(|| obj.get("id"))
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("");
                                        let output_value =
                                            obj.get("output").cloned().unwrap_or(json!(""));
                                        let output_str = if let Some(s) = output_value.as_str() {
                                            s.to_string()
                                        } else {
                                            output_value.to_string()
                                        };
                                        messages.push(json!({
                                            "role": "tool",
                                            "tool_call_id": call_id,
                                            "content": output_str,
                                        }));
                                    }
                                    _ => {
                                        messages.push(item.clone());
                                    }
                                }
                                continue;
                            }
                        }
                        messages.push(item.clone());
                    }
                } else {
                    // 降级处理：传统的字符串或混合内容拼接
                    let content = arr
                        .iter()
                        .map(|v| {
                            if let Some(s) = v.as_str() {
                                s.to_string()
                            } else if v.is_object() {
                                v.to_string()
                            } else {
                                "".to_string()
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("\n");

                    if !content.is_empty() {
                        messages.push(json!({
                            "role": "user",
                            "content": content
                        }));
                    }
                }
            } else {
                let content = input.to_string();
                if !content.is_empty() {
                    messages.push(json!({
                        "role": "user",
                        "content": content
                    }));
                }
            };
        }

        if let Some(obj) = body.as_object_mut() {
            tracing::debug!(
                "[Codex] Injecting normalized messages: {} messages",
                messages.len()
            );
            obj.insert("messages".to_string(), Value::Array(messages));
        }
    } else if already_normalized {
        tracing::debug!(
            "[Codex] Skipping normalization (messages already populated by first pass)"
        );
    }

    if is_codex_style {
        if let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) {
            let dropped = drop_leading_orphan_tool_history(messages);
            if dropped > 0 {
                tracing::warn!(
                    dropped_messages = dropped,
                    "[Responses Compat] Dropped leading orphan tool history"
                );
            }
            if rewrite_terminal_assistant_prefill(messages) {
                tracing::debug!(
                    "[Responses Compat] Rewrote terminal assistant text prefill as user input"
                );
            }
        }
    }

    // [FIX] 在 openai_req 反序列化之前，从 body 中捕获原始 input 和 instructions
    // 用于后续 session 保存时，保留完整的工具调用历史（而非从 openai_req.messages 重建丢失信息）
    let normalized_interaction_ledger = body.get("_interaction_ledger").cloned();
    let (session_save_input, session_save_instructions) = if let Some(obj) = body.as_object_mut() {
        let input = bounded_session_input.take().unwrap_or_default();
        obj.remove("input");
        let instructions = obj
            .remove("instructions")
            .and_then(|value| match value {
                Value::String(text) => Some(text),
                _ => None,
            })
            .unwrap_or_default();
        (
            input,
            if store_response {
                instructions
            } else {
                String::new()
            },
        )
    } else {
        (Vec::new(), String::new())
    };

    let mut openai_req: OpenAIRequest = match serde_json::from_value(body) {
        Ok(req) => req,
        Err(e) => {
            return (StatusCode::BAD_REQUEST, format!("Invalid request: {}", e)).into_response();
        }
    };

    if let Err(message) = validate_openai_file_inputs(&openai_req) {
        return (StatusCode::BAD_REQUEST, message).into_response();
    }

    // Safety: Inject empty message if needed
    if openai_req.messages.is_empty() {
        openai_req
            .messages
            .push(crate::proxy::mappers::openai::OpenAIMessage {
                role: "user".to_string(),
                content: Some(crate::proxy::mappers::openai::OpenAIContent::String(
                    " ".to_string(),
                )),
                reasoning_content: None,
                signature: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
                refusal: None,
            });
    }

    // [NEW v4.2.0] Context Management & Reasoning Replay
    let anchor = SessionManager::openai_content_anchor(&openai_req);
    // 粘性身份只用客户端显式会话号。网关生成的 response id 和每轮
    // previous_response_id 只给会话库串历史，不进入 SessionScope。
    let session_hint = match explicit_session_id.as_deref() {
        Some(id) => json!({ "session_id": id }),
        None => json!({}),
    };
    let session_scope = crate::proxy::thinking_store::SessionScope::resolve(
        &headers,
        Some(&session_hint),
        None,
        anchor,
        user_identity
            .as_ref()
            .map(|identity| identity.token_id.as_str()),
    );
    let store_key = session_scope.store_key.clone();
    let affinity_key = session_scope.affinity_key.clone();
    openai_req.session_id = Some(store_key.clone());
    let session_id_str = store_key.clone();
    let client_session_id = session_scope.client_id.clone();
    let signature_session_id_str = session_id_str.clone();

    let client_tool_names =
        crate::proxy::mappers::openai::request::extract_client_tool_names(&openai_req.tools);
    let responses_tool_kinds = if is_codex_style {
        crate::proxy::mappers::openai::request::ResponsesToolKinds::from_tools(
            openai_req.tools.as_deref(),
        )
    } else {
        Default::default()
    };

    // Server-authoritative thinking: do NOT prefill messages.reasoning_content from
    // SignatureCache. OpenAI mapping ignores client/cached reasoning text and fills
    // placeholders via ThinkingStore hydrate + finalize instead.

    let effort_hint = openai_req
        .reasoning_effort
        .as_deref()
        .or_else(|| {
            openai_req
                .reasoning
                .as_ref()
                .and_then(|r| r.effort.as_deref())
        })
        .or_else(|| {
            openai_req
                .thinking
                .as_ref()
                .and_then(|t| t.effort.as_deref())
        });

    let route_effort = crate::proxy::pipeline::routing_effort(
        crate::proxy::pipeline::extract_client_thinking_switch(
            openai_req
                .thinking
                .as_ref()
                .and_then(|t| t.thinking_type.as_deref()),
            openai_req
                .thinking
                .as_ref()
                .and_then(|t| t.budget_tokens.map(|b| b as u64))
                .or_else(|| {
                    openai_req
                        .reasoning
                        .as_ref()
                        .and_then(|r| r.max_tokens.map(|b| b as u64))
                }),
            effort_hint,
        )
        .is_disabled(),
        effort_hint,
    );
    let mapped_model = crate::proxy::common::model_mapping::resolve_model_route_with_effort(
        &openai_req.model,
        &*state.custom_mapping.read().await,
        route_effort,
    );
    let trace_id = format!("req_{}", chrono::Utc::now().timestamp_subsec_millis());
    if debug_logger::is_enabled(&debug_cfg) {
        if let Some(ledger) = normalized_interaction_ledger {
            let payload = json!({
                "kind": "normalized_interaction_ledger",
                "protocol": "openai",
                "trace_id": trace_id.clone(),
                "request_path": uri.path(),
                "original_model": openai_req.model.clone(),
                "interaction_ledger": ledger,
            });
            debug_logger::write_exchange_payload(
                &debug_cfg,
                Some(&trace_id),
                "normalized_interaction_ledger",
                &payload,
            )
            .await;
        }
    }
    let token_manager = state.token_manager.clone();

    let assistant_turn_index = openai_req
        .messages
        .iter()
        .filter(|m| m.role == "assistant")
        .count();

    let upstream = state.upstream.clone();
    let pool_size = token_manager.len();
    // [FIX #3485] 自适应多账号池与单账号退避最大重试次数 (单账号3次，多账号整池两轮)
    let max_attempts = crate::proxy::handlers::common::calculate_max_retry_attempts(pool_size);

    let mut last_error = String::new();
    let mut terminal_stream_usage: Option<serde_json::Value> = None;
    let mut last_email: Option<String> = None;
    let mut retry_state = RequestRetryState::default();
    let mut retry_credentials: Option<crate::proxy::token_manager::LeasedCredentials> = None;
    let mut session_lease = None;
    let mut attempted_accounts: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    let mut failure_statuses = FailureStatusTracker::default();
    let mut used_attempts = 0;

    let clean_ms = clean_start.elapsed().as_micros() as f64 / 1000.0;
    // TTFT 单源口径：仅记录“首个上游数据块”；流式发头时写 -1。
    let mut ttft_ms = crate::proxy::pipeline::stream_peek::TTFT_UNKNOWN_MS;
    let mut headers_ms = 0.0f64;
    let mut fallback_overhead_ms = 0.0f64;

    if debug_logger::is_enabled(&debug_cfg) {
        let payload = json!({
            "kind": "original_request",
            "protocol": "openai",
            "trace_id": trace_id,
            "request_path": uri.path(),
            "request": original_body.as_ref(),
        });
        debug_logger::write_exchange_payload(
            &debug_cfg,
            Some(&trace_id),
            "original_request",
            &payload,
        )
        .await;
    }

    let mut force_rotate = false;
    let mut retried_without_thinking = false;
    let mut heal_flags = crate::proxy::pipeline::AutoHealFlags::default();

    'attempts: while let Some(attempt) = next_rotation_attempt(
        &mut used_attempts,
        max_attempts,
        retry_credentials.is_some(),
    ) {
        // [Pipeline Terminal] 全局预算耗尽：统一记终态、补 last_error、直接退出轮换
        if crate::proxy::pipeline::terminal::gate_exhausted(
            "openai",
            attempt + 1,
            &mut failure_statuses,
            &mut last_error,
        ) {
            break;
        }
        retry_state.begin_attempt();

        let norm_start = std::time::Instant::now();
        // 3. 模型配置解析
        // 将 OpenAI 工具转为 Value 数组以便探测联网
        let tools_val: Option<Vec<Value>> = openai_req
            .tools
            .as_ref()
            .map(|list| list.iter().cloned().collect());
        let config = crate::proxy::mappers::common_utils::resolve_request_config(
            &openai_req.model,
            &mapped_model,
            &tools_val,
            None, // size
            None, // quality
            None, // image_size
            None, // body
        );

        let session_id = Some(affinity_key.as_str());

        let (access_token, project_id, email, account_id, _wait_ms, selected_session_lease) =
            if let Some(credentials) = retry_credentials.take() {
                credentials
            } else {
                let mut token_result = token_manager
                    .get_token_filtered_with_retry_lease(
                        &config.request_type,
                        force_rotate,
                        session_id,
                        &mapped_model,
                        &attempted_accounts,
                        &mut session_lease,
                    )
                    .await;

                if let Err(ref e) = token_result {
                    if crate::proxy::handlers::common::is_transient_token_error(e) {
                        tracing::warn!(
                            "Token acquisition transient error ({}), retrying once under scheduling policy...",
                            e
                        );
                        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                        token_result = token_manager
                            .get_token_filtered_with_retry_lease(
                                &config.request_type,
                                crate::proxy::pipeline::rotation::token_retry_force_rotate(
                                    force_rotate,
                                    token_manager.get_scheduling_mode().await,
                                ),
                                session_id,
                                &mapped_model,
                                &attempted_accounts,
                                &mut session_lease,
                            )
                            .await;
                    }
                }

                match token_result {
                    Ok(t) => t,
                    Err(e) => {
                        retry_state.preserve_token_error(&e);
                        if !failure_statuses.is_empty()
                            && !crate::proxy::session_bindings::is_session_binding_changed_error(&e)
                        {
                            tracing::warn!(
                            "[{}] All available accounts attempted or exhausted for this request, breaking retry loop to return upstream error",
                            trace_id
                        );
                            break;
                        }
                        let protocol = if is_responses_api {
                            "responses"
                        } else {
                            "openai"
                        };
                        return retry_state.finish_error_response(
                            crate::proxy::handlers::common::build_token_error_response_with_usage(
                                protocol,
                                mapped_model.as_str(),
                                None,
                                &e,
                                terminal_stream_usage.as_ref(),
                            ),
                        );
                    }
                }
            };

        session_lease = selected_session_lease;

        let in_flight_guard = Arc::new(token_manager.acquire_in_flight(&account_id));

        let mapped_model = token_manager
            .resolve_dynamic_model_for_account(&account_id, &mapped_model)
            .await;

        last_email = Some(email.clone());

        info!("✓ Using account: {} (type: {})", email, config.request_type);

        let proxy_token = token_manager.get_token_by_id(&account_id);
        let tf_start = std::time::Instant::now();
        let (mut gemini_body, _routed_session, message_count, _prefix_hash) =
            match if is_responses_api {
                transform_openai_request_with_session(
                    &openai_req,
                    &project_id,
                    &mapped_model,
                    proxy_token.as_ref(),
                    &affinity_key,
                    signature_read_key.as_deref(),
                    true, // is_responses_api
                    retried_without_thinking,
                )
            } else {
                transform_openai_request_with_session(
                    &openai_req,
                    &project_id,
                    &mapped_model,
                    proxy_token.as_ref(),
                    &affinity_key,
                    None,
                    false,
                    retried_without_thinking,
                )
            } {
                Ok(mapped) => mapped,
                Err(message) => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(json!({
                            "error": {"type": "invalid_request_error", "message": message}
                        })),
                    )
                        .into_response()
                }
            };
        heal_flags.reapply(&mut gemini_body);
        let session_id = session_id_str.clone();
        // [Signature-Retry] 400 签名错误后的重试：流水线统一去签名
        if retried_without_thinking {
            crate::proxy::pipeline::InboundThinkingPipeline::strip_signatures_for_retry(
                &mut gemini_body,
                Some(session_id.as_str()),
                &mapped_model,
            );
        }
        let tf_micros = tf_start.elapsed().as_micros() as u64;
        let norm_total_micros = norm_start.elapsed().as_micros() as u64;
        let norm_ms = norm_total_micros.saturating_sub(tf_micros) as f64 / 1000.0;
        let think_fill_ms = tf_micros as f64 / 1000.0;
        let _ =
            crate::proxy::mappers::context_manager::ContextManager::apply_post_transit_context_mgmt(
                &mut gemini_body,
                &mapped_model,
            );
        crate::proxy::mappers::prompt_sanitizer::PromptSanitizer::sanitize_gemini_payload(
            &mut gemini_body,
        );
        crate::proxy::mappers::common_utils::ensure_gemini_payload_ends_with_user(&mut gemini_body);
        // Resolve text history only after all request normalization, and use the
        // same visible prefix when recording the response from this attempt.
        let thinking_enabled =
            crate::proxy::pipeline::qos::payload_enables_thinking(&gemini_body, None);
        let contents_path = if gemini_body.get("request").is_some() {
            "/request/contents"
        } else {
            "/contents"
        };
        let response_text_context = gemini_body
            .pointer_mut(contents_path)
            .and_then(Value::as_array_mut)
            .map(|contents| {
                crate::proxy::thinking_store::finalize_request_text_context(
                    &session_id,
                    contents,
                    &mapped_model,
                    thinking_enabled,
                    retried_without_thinking,
                )
            });
        let estimated_input_tokens = crate::proxy::pipeline::estimate_tokens(&gemini_body);
        if let Some(ref recorder) = upstream_recorder {
            recorder.set_value(&gemini_body);
        }
        let gemini_body_for_debug = debug_logger::is_enabled(&debug_cfg)
            .then(|| debug_value_without_inline_data(&gemini_body));
        if debug_logger::is_enabled(&debug_cfg) {
            let payload = json!({
                "kind": "v1internal_request",
                "protocol": "openai",
                "trace_id": trace_id,
                "request_path": uri.path(),
                "original_model": openai_req.model,
                "mapped_model": mapped_model,
                "request_type": config.request_type,
                "attempt": attempt,
                "v1internal_request": gemini_body_for_debug.as_ref(),
            });
            debug_logger::write_exchange_payload(
                &debug_cfg,
                Some(&trace_id),
                "v1internal_request",
                &payload,
            )
            .await;
        }

        // [DEBUG v4.2.0] Detailed size analysis of Gemini request body
        if let Some(contents) = gemini_body.get("contents").and_then(|c| c.as_array()) {
            let mut sizes = Vec::new();
            for (idx, msg) in contents.iter().enumerate() {
                let role = msg
                    .get("role")
                    .and_then(|r| r.as_str())
                    .unwrap_or("unknown");
                sizes.push(format!(
                    "msg_{}[{}]: {} chars",
                    idx,
                    role,
                    serialized_json_len(msg)
                ));
            }

            let system_instruction_len = gemini_body
                .get("request")
                .and_then(|r| r.get("systemInstruction"))
                .map(serialized_json_len)
                .unwrap_or(0);

            let tools_len = gemini_body
                .get("request")
                .and_then(|r| r.get("tools"))
                .map(serialized_json_len)
                .unwrap_or(0);

            tracing::info!(
                "[Codex-Token-Analysis] Total parts: {}. SystemInstruction: {} chars, Tools: {} chars. Content sizes: {:?}",
                contents.len(),
                system_instruction_len,
                tools_len,
                sizes
            );
        }

        // [AUTO-CONVERSION] For Legacy/Codex as well
        let client_wants_stream = openai_req.stream;
        let force_stream_internally = !client_wants_stream;
        let list_response = client_wants_stream || force_stream_internally;
        let method = if list_response {
            "streamGenerateContent"
        } else {
            "generateContent"
        };
        let query_string = if list_response { Some("alt=sse") } else { None };

        let mut extra_headers = std::collections::HashMap::new();
        extra_headers.insert("x-session-id".to_string(), client_session_id.clone());

        let preceding_turn_anchor = gemini_body
            .get("request")
            .and_then(|r| r.get("contents"))
            .or_else(|| gemini_body.get("contents"))
            .and_then(|c| c.as_array())
            .and_then(|a| a.last())
            .cloned();
        let causal_anchor =
            crate::proxy::thinking_store::compute_causal_anchor(preceding_turn_anchor.as_ref());

        let upstream_req_start = std::time::Instant::now();
        let call_result = match upstream
            .call_v1_internal_with_headers(
                method,
                &access_token,
                gemini_body.clone(),
                query_string,
                extra_headers.clone(),
                Some(account_id.as_str()),
            )
            .await
        {
            Ok(r) => r,
            Err(e) => {
                last_error = e.clone();
                // 终态（首字节 / 上传 / 全局预算 / 重放上限）都不是账号故障：
                // 不冷却、不跨账号重放；判定、状态码与观测统一由 pipeline::terminal 给出。
                if let Some(handling) = crate::proxy::pipeline::terminal::apply_terminal(
                    &e,
                    &mut failure_statuses,
                    "openai_responses",
                    attempt + 1,
                    max_attempts,
                    upstream_req_start.elapsed().as_secs_f64(),
                    |err| token_manager.mark_account_transient_failure(&account_id, 30, err),
                ) {
                    debug!(
                        "Codex Request failed on attempt {}/{} (terminal={}): {}",
                        attempt + 1,
                        max_attempts,
                        handling.terminal.as_str(),
                        crate::proxy::upstream::client::strip_internal_markers(&e)
                    );
                    // 终态都不是账号故障：解绑 / 换号 / 原地重放统一由 pipeline 按调度模式裁决。
                    // CacheFirst 下文本请求绝不解绑 / 换号；Balance 下传输层断连先原地重连一次。
                    let terminal_plan = retry_state.plan_terminal(
                        &account_id,
                        handling.disposition,
                        token_manager.get_scheduling_mode().await,
                        config.request_type != "image_gen",
                    );
                    if terminal_plan.abandon_sticky_account {
                        token_manager.abandon_session_lease(&mut session_lease);
                        tracing::debug!(
                            "Unbound session {} from account {} due to terminal {}",
                            affinity_key,
                            email,
                            handling.terminal.as_str()
                        );
                    }
                    match terminal_plan.next {
                        crate::proxy::pipeline::TerminalNextStep::Stop => break,
                        crate::proxy::pipeline::TerminalNextStep::RetrySameAccount => {
                            tracing::info!(
                                "Upstream terminal {} on account {} (attempt {}/{}): keeping session {} bound and retrying the same account in-place to preserve KV cache",
                                handling.terminal.as_str(),
                                email,
                                attempt + 1,
                                max_attempts,
                                affinity_key
                            );
                            retry_credentials = Some((
                                access_token.clone(),
                                project_id.clone(),
                                email.clone(),
                                account_id.clone(),
                                0,
                                session_lease.clone(),
                            ));
                            continue;
                        }
                        crate::proxy::pipeline::TerminalNextStep::Rotate => {}
                    }
                    crate::proxy::pipeline::rotation::record_attempted_account(
                        &mut attempted_accounts,
                        &account_id,
                        pool_size,
                        &mut force_rotate,
                    );
                    continue;
                }

                // [Gateway Self-Answer Guard] 模型不存在 / 无可用账号：不罚号、不轮换
                if let Some((status, reason)) =
                    crate::proxy::common::model_gate::self_answer_disposition(&e)
                {
                    failure_statuses.record(status);
                    last_error = reason.to_string();
                    break;
                }

                // 非终态（连接失败 / 调度异常）：冷却该账号并继续轮换
                failure_statuses.record(StatusCode::BAD_GATEWAY);
                if let Some(strategy) = retry_state.cache_first_failure(
                    &account_id,
                    502,
                    &e,
                    None,
                    &token_manager.get_sticky_config().await,
                    config.request_type != "image_gen",
                ) {
                    if apply_retry_strategy(strategy, attempt, max_attempts, 502, &trace_id).await {
                        retry_credentials = Some((
                            access_token.clone(),
                            project_id.clone(),
                            email.clone(),
                            account_id.clone(),
                            0,
                            session_lease.clone(),
                        ));
                        continue;
                    }
                    break;
                }
                token_manager.mark_account_transient_failure(&account_id, 30, &e);
                crate::proxy::pipeline::rotation::record_attempted_account(
                    &mut attempted_accounts,
                    &account_id,
                    pool_size,
                    &mut force_rotate,
                );
                debug!(
                    "Codex Request failed on attempt {}/{}: {}",
                    attempt + 1,
                    max_attempts,
                    e
                );
                continue;
            }
        };

        let effective_attempt_start = call_result.effective_attempt_start;
        let mut fallback_attempts = call_result.fallback_attempts.clone();
        fallback_overhead_ms = call_result.fallback_overhead_s() * 1000.0;
        let attempts_b64 =
            crate::proxy::handlers::common::encode_attempts_header(&fallback_attempts);

        let response = call_result.response;
        let upstream_url = response.url().to_string();
        let status = response.status();
        // HTTP failures and pre-output SSE errors share the same recovery policy.
        let (status, retry_after, error_text) = 'upstream_response: {
            if !status.is_success() {
                let retry_after = response
                    .headers()
                    .get("Retry-After")
                    .and_then(|header| header.to_str().ok())
                    .map(str::to_string);
                let error_text = response
                    .text()
                    .await
                    .unwrap_or_else(|_| format!("HTTP {}", status));
                break 'upstream_response (status, retry_after, error_text);
            }

            let track_image_success = config.request_type == "image_gen";
            let mut session_confirmation = if !track_image_success {
                // HTTP acceptance still confirms account health in all scheduling
                // modes; only cache warmth waits for body validation.
                token_manager.mark_account_success(&account_id);
                token_manager.accept_session_response(&mut session_lease)
            } else {
                None
            };

            if list_response {
                use axum::body::Body;
                use axum::response::Response;
                use futures::StreamExt;

                let upstream_meta = json!({
                    "protocol": "openai",
                    "trace_id": trace_id,
                    "request_path": uri.path(),
                    "original_model": openai_req.model,
                    "mapped_model": mapped_model,
                    "request_type": config.request_type,
                    "attempt": attempt,
                    "status": status.as_u16(),
                    "upstream_url": upstream_url,
                });
                let gemini_stream = debug_logger::wrap_stream_with_debug(
                    crate::proxy::pipeline::usage_diagnostics::observe_stream(
                        response.bytes_stream(),
                        crate::proxy::pipeline::usage_diagnostics::UsageAttempt::current(
                            &config.final_model,
                            &account_id,
                            "openai_responses",
                        ),
                    ),
                    debug_cfg.clone(),
                    trace_id.clone(),
                    "upstream_response",
                    upstream_meta,
                );

                // [Auto-Heal] 纯思考空回复流式自愈门禁 (Pipeline First)
                let auto_heal_ctx = crate::proxy::pipeline::auto_heal::ThinkingAutoHealContext {
                    upstream: upstream.clone(),
                    method: "streamGenerateContent",
                    access_token: access_token.clone(),
                    original_body: gemini_body.clone(),
                    query_string: Some("alt=sse"),
                    extra_headers: extra_headers.clone(),
                    account_id: Some(account_id.clone()),
                    trace_id: trace_id.clone(),
                    budget: crate::proxy::pipeline::budget::current(),
                };
                let gemini_stream =
                    crate::proxy::pipeline::auto_heal::wrap_stream_with_empty_thinking_auto_heal(
                        Box::pin(gemini_stream),
                        auto_heal_ctx,
                    );
                // Preserve structured upstream errors before protocol conversion.
                let stream_errors =
                    crate::proxy::pipeline::stream_error::UpstreamStreamErrorObserver::default();
                let gemini_stream = Box::pin(
                    crate::proxy::pipeline::stream_error::observe_response_errors(
                        gemini_stream,
                        stream_errors.clone(),
                        !client_wants_stream,
                    ),
                );
                let gemini_stream = Box::pin(
                    crate::proxy::token_manager::confirm_session_response_stream(
                        gemini_stream,
                        stream_errors.clone(),
                        session_confirmation.take(),
                    ),
                );

                // DECISION: Which stream to create?
                // If client wants stream: give them what they asked (Legacy/Codex SSE).
                // If forced stream: use Chat SSE + Collector, because our collector works on Chat format
                // and we already have logic to convert Chat JSON -> Legacy JSON.

                if client_wants_stream {
                    let mut session_completion_rx = None;
                    let mut openai_stream = if is_codex_style {
                        use crate::proxy::mappers::openai::streaming::create_codex_sse_stream_with_context;
                        let completion_tx = if store_response {
                            let (tx, rx) = tokio::sync::oneshot::channel();
                            session_completion_rx = Some(rx);
                            Some(tx)
                        } else {
                            None
                        };
                        create_codex_sse_stream_with_context(
                            gemini_stream,
                            openai_req.model.clone(),
                            session_id_str.clone(),
                            message_count,
                            assistant_turn_index,
                            response_id_for_save.clone(),
                            completion_tx,
                            store_response,
                            response_text_context.clone(),
                            responses_tool_kinds.clone(),
                        )
                    } else {
                        use crate::proxy::mappers::openai::streaming::create_legacy_sse_stream_with_context;
                        create_legacy_sse_stream_with_context(
                            gemini_stream,
                            openai_req.model.clone(),
                            session_id,
                            message_count,
                            response_text_context.clone(),
                        )
                    };

                    // [Pipeline StreamPeek] 流式响应直接返回：由 run_stream_lifecycle 的 InitialTtft
                    // 负责首字宽限与心跳保活，不再阻塞等待首包（与 Claude / Gemini 同构）。
                    headers_ms = effective_attempt_start.elapsed().as_micros() as f64 / 1000.0;
                    ttft_ms = crate::proxy::pipeline::stream_peek::TTFT_UNKNOWN_MS;
                    // [QoS] Responses 流式与 chat 同口径：首个真实数据块上报 TTFT
                    let ttft_seen = std::sync::Arc::new(std::sync::OnceLock::<f64>::new());
                    let ttft_seen_cb = ttft_seen.clone();
                    let ttft_start = effective_attempt_start;
                    let ttft_tm = token_manager.clone();
                    let ttft_account = account_id.clone();
                    let ttft_session = affinity_key.clone();
                    let ttft_lease = session_lease.clone();
                    let ttft_is_thinking = crate::proxy::pipeline::qos::payload_enables_thinking(
                        &gemini_body,
                        Some(&mapped_model),
                    ) || openai_req.thinking.is_some()
                        || openai_req.reasoning.is_some()
                        || openai_req.reasoning_effort.is_some();
                    let ttft_tokens = estimated_input_tokens;
                    let combined_stream =
                        crate::proxy::pipeline::stream_peek::wrap_stream_with_ttft_probe(
                            openai_stream,
                            ttft_start,
                            |bytes: &[u8]| {
                                crate::proxy::pipeline::stream_peek::is_heartbeat(
                                    bytes,
                                    crate::proxy::pipeline::stream_peek::StreamDialect::OpenAi,
                                )
                            },
                            move |ttft| {
                                let _ = ttft_seen_cb.set(ttft);
                                crate::proxy::handlers::common::report_qos_ttft(
                                    &ttft_tm,
                                    &ttft_account,
                                    Some(ttft_session.as_str()),
                                    ttft,
                                    ttft_is_thinking,
                                    Some(ttft_tokens),
                                    ttft_lease.as_ref(),
                                );
                            },
                        );
                    let converted_meta = json!({
                        "protocol": "openai",
                        "trace_id": trace_id,
                        "stage": "converted_codex_response",
                        "request_path": uri.path(),
                        "original_model": openai_req.model,
                        "mapped_model": mapped_model,
                        "request_type": config.request_type,
                        "attempt": attempt,
                        "status": status.as_u16(),
                        "upstream_url": upstream_url,
                    });
                    let combined_stream = debug_logger::wrap_stream_with_debug(
                        Box::pin(combined_stream),
                        debug_cfg.clone(),
                        trace_id.clone(),
                        "converted_codex_response",
                        converted_meta,
                    );

                    // Save completed output and recoverable max-token truncations;
                    // the converter excludes transport failures and empty output.
                    if let Some(completion_rx) = session_completion_rx {
                        let save_parent = session_parent;
                        let save_input = session_save_input;
                        let save_instructions = session_save_instructions;
                        let save_model = openai_req.model.clone();
                        let rid = response_id_for_save.clone();
                        let scope_identity = explicit_session_id.clone();
                        tokio::spawn(async move {
                            if let Ok((outputs, ack_tx)) = completion_rx.await {
                                let outputs = outputs
                                    .into_iter()
                                    .map(into_history_without_inline_media)
                                    .collect();
                                save_session_unless_response_cancelled(
                                    ack_tx,
                                    crate::proxy::http_session_store::save_session_delta_with_identity(
                                        rid,
                                        save_parent,
                                        save_input,
                                        outputs,
                                        save_instructions,
                                        save_model,
                                        routing_session_id,
                                        scope_identity,
                                    ),
                                )
                                .await;
                            }
                        });
                    }
                    let in_flight_guard_for_stream = Arc::clone(&in_flight_guard);
                    let combined_stream = async_stream::stream! {
                        let _in_flight = in_flight_guard_for_stream;
                        let mut s = Box::pin(combined_stream);
                        while let Some(item) = s.next().await {
                            yield item;
                        }
                    };
                    return Response::builder()
                        .header("Content-Type", "text/event-stream")
                        .header("Cache-Control", "no-cache")
                        .header("Connection", "keep-alive")
                        .header("X-Account-Email", &email)
                        .header("X-Mapped-Model", &mapped_model)
                        .header("X-Upstream-Model", config.final_model.as_str())
                        .header("X-Session-Id", &session_scope.client_id)
                        .header("X-Antigravity-Session-Id", &session_scope.client_id)
                        .timing(&TimingHeaders::new(
                            clean_ms,
                            norm_ms,
                            think_fill_ms,
                            ttft_ms,
                            headers_ms,
                            fallback_overhead_ms,
                            attempts_b64.as_deref().unwrap_or(""),
                        ))
                        .body(Body::from_stream(combined_stream))
                        .unwrap()
                        .into_response();
                } else {
                    // Forced Stream Internal -> Convert to Legacy JSON
                    // Use CHAT SSE Stream (so Collector can parse it)
                    use crate::proxy::mappers::openai::streaming::create_openai_sse_stream_with_context;
                    // Note: We use create_openai_sse_stream regardless of is_codex_style here,
                    // because we just want the content aggregation which chat stream does well.
                    let mut openai_stream = create_openai_sse_stream_with_context(
                        gemini_stream,
                        openai_req.model.clone(),
                        session_id_str.clone(),
                        message_count,
                        Some(client_tool_names.clone()),
                        true,
                        Some(causal_anchor),
                        response_text_context.clone(),
                    );

                    // [Pipeline StreamPeek] 内部强制流（非流式客户端）：返回前仍需首包，
                    // 统一走 pipeline::stream_peek；超时属于时间问题（不跨账号重放），其余错误仍可换号。
                    headers_ms = effective_attempt_start.elapsed().as_micros() as f64 / 1000.0;
                    let mut first_data_chunk = None;
                    let mut peek_failed = false;
                    let is_thinking = crate::proxy::pipeline::qos::payload_enables_thinking(
                        &gemini_body,
                        Some(&mapped_model),
                    ) || openai_req.thinking.is_some()
                        || openai_req.reasoning.is_some()
                        || openai_req.reasoning_effort.is_some();
                    let peek_timeout_secs = crate::proxy::pipeline::budget::clamp_secs(
                        crate::proxy::pipeline::stream_peek::peek_budget_secs(
                            false,
                            config.request_type == "image_gen",
                            is_thinking,
                            Some(estimated_input_tokens),
                        ),
                    );
                    match crate::proxy::pipeline::stream_peek::peek_first_data(
                        &mut openai_stream,
                        peek_timeout_secs,
                        |bytes| {
                            crate::proxy::pipeline::stream_peek::is_heartbeat(
                                bytes,
                                crate::proxy::pipeline::stream_peek::StreamDialect::OpenAi,
                            )
                        },
                        |bytes| stream_chunk_has_error_event(bytes),
                    )
                    .await
                    {
                        crate::proxy::pipeline::stream_peek::PeekOutcome::Data {
                            bytes, ..
                        } => {
                            ttft_ms = effective_attempt_start.elapsed().as_micros() as f64 / 1000.0;
                            // 非流式 Responses：因为删除了 tail 上报，这里必须补一次真实首块样本
                            crate::proxy::handlers::common::report_qos_ttft(
                                &token_manager,
                                &account_id,
                                Some(affinity_key.as_str()),
                                ttft_ms,
                                is_thinking,
                                Some(estimated_input_tokens),
                                session_lease.as_ref(),
                            );
                            if let Some(effective_item) =
                                fallback_attempts.iter_mut().find(|a| a.is_effective)
                            {
                                effective_item.duration_s = ttft_ms / 1000.0;
                            }
                            let _ = crate::proxy::monitor::CURRENT_UPSTREAM_CAPTURE.try_with(
                                |holder| {
                                    holder.set_attempts(fallback_attempts.clone());
                                },
                            );
                            first_data_chunk = Some(bytes);
                        }
                        crate::proxy::pipeline::stream_peek::PeekOutcome::Timeout {
                            budget_secs,
                            ..
                        } => {
                            last_error = format!("Timeout peek internal ({}s)", budget_secs);
                            terminal_stream_usage = record_nonstream_peek_timeout(
                                "openai_responses",
                                budget_secs,
                                &stream_errors,
                                &mut failure_statuses,
                                &mut retry_state,
                                token_manager.get_scheduling_mode().await,
                                config.request_type != "image_gen",
                            );
                            break 'attempts;
                        }
                        crate::proxy::pipeline::stream_peek::PeekOutcome::ErrorEvent { .. } => {
                            last_error = "Error event in internal stream".to_string();
                            peek_failed = true;
                        }
                        crate::proxy::pipeline::stream_peek::PeekOutcome::StreamError {
                            error,
                            ..
                        } => {
                            last_error = format!("Internal stream error: {}", error);
                            peek_failed = true;
                        }
                        crate::proxy::pipeline::stream_peek::PeekOutcome::Eof { .. } => {
                            last_error = "Empty internal stream".to_string();
                            peek_failed = true;
                        }
                    }

                    if peek_failed {
                        if let Some(failure) =
                            stream_errors.failure_or_incomplete(Some(&last_error))
                        {
                            let failure_status = StatusCode::from_u16(failure.status)
                                .unwrap_or(StatusCode::BAD_GATEWAY);
                            if failure.replay_safe() {
                                break 'upstream_response (failure_status, None, failure.body);
                            }
                            // Output has already been consumed: classify the error, but do not replay.
                            retry_state.preserve_http_stop(
                                failure.status,
                                &failure.body,
                                &token_manager.get_sticky_config().await,
                                config.request_type != "image_gen",
                            );
                            failure_statuses.record(failure_status);
                            terminal_stream_usage = failure.usage.clone();
                            last_error = failure.body;
                            break 'attempts;
                        }
                        failure_statuses.record(StatusCode::BAD_GATEWAY);
                        if let Some(strategy) = retry_state.cache_first_failure(
                            &account_id,
                            502,
                            &last_error,
                            None,
                            &token_manager.get_sticky_config().await,
                            config.request_type != "image_gen",
                        ) {
                            if apply_retry_strategy(strategy, attempt, max_attempts, 502, &trace_id)
                                .await
                            {
                                retry_credentials = Some((
                                    access_token.clone(),
                                    project_id.clone(),
                                    email.clone(),
                                    account_id.clone(),
                                    0,
                                    session_lease.clone(),
                                ));
                                continue 'attempts;
                            }
                            break 'attempts;
                        }
                        force_rotate = true;
                        continue 'attempts;
                    }

                    let combined_stream = futures::stream::once(async move {
                        Ok::<Bytes, String>(first_data_chunk.unwrap())
                    })
                    .chain(openai_stream);
                    let converted_meta = json!({
                        "protocol": "openai",
                        "trace_id": trace_id,
                        "stage": "converted_codex_response",
                        "request_path": uri.path(),
                        "original_model": openai_req.model,
                        "mapped_model": mapped_model,
                        "request_type": config.request_type,
                        "attempt": attempt,
                        "status": status.as_u16(),
                        "upstream_url": upstream_url,
                    });
                    let combined_stream = debug_logger::wrap_stream_with_debug(
                        Box::pin(combined_stream),
                        debug_cfg.clone(),
                        trace_id.clone(),
                        "converted_codex_response",
                        converted_meta,
                    );

                    // Collect
                    use crate::proxy::mappers::openai::collector::collect_stream_to_json;
                    let collected_response = collect_stream_to_json(combined_stream).await;
                    if let Some(failure) = stream_errors.failure_or_incomplete(
                        collected_response.as_ref().err().map(String::as_str),
                    ) {
                        let failure_status =
                            StatusCode::from_u16(failure.status).unwrap_or(StatusCode::BAD_GATEWAY);
                        if failure.replay_safe() {
                            break 'upstream_response (failure_status, None, failure.body);
                        }
                        // Output has already been consumed: classify the error, but do not replay.
                        retry_state.preserve_http_stop(
                            failure.status,
                            &failure.body,
                            &token_manager.get_sticky_config().await,
                            config.request_type != "image_gen",
                        );
                        failure_statuses.record(failure_status);
                        terminal_stream_usage = failure.usage.clone();
                        last_error = failure.body;
                        break 'attempts;
                    }
                    match collected_response {
                        Ok(chat_resp) => {
                            let is_responses_api = is_responses_api_path(uri.path());

                            if is_responses_api {
                                let is_structured = is_request_structured_output(&openai_req);
                                let mut resp = convert_chat_response_to_responses_with_tools(
                                    &chat_resp,
                                    is_structured,
                                    &responses_tool_kinds,
                                );
                                resp["id"] = json!(response_id_for_save.clone());
                                let outputs = resp
                                    .get("output")
                                    .and_then(Value::as_array)
                                    .cloned()
                                    .unwrap_or_default()
                                    .into_iter()
                                    .map(into_history_without_inline_media)
                                    .collect();
                                if store_response {
                                    crate::proxy::http_session_store::save_session_delta_with_identity(
                                        response_id_for_save.clone(),
                                        session_parent,
                                        session_save_input,
                                        outputs,
                                        session_save_instructions,
                                        openai_req.model.clone(),
                                        routing_session_id.clone(),
                                        explicit_session_id.clone(),
                                    )
                                    .await;
                                }
                                if debug_logger::is_enabled(&debug_cfg) {
                                    let payload = json!({
                                        "kind": "exchange_summary",
                                        "protocol": "openai",
                                        "trace_id": trace_id,
                                        "request_path": uri.path(),
                                        "original_codex_request": original_body.as_ref(),
                                        "gemini_request": gemini_body_for_debug.as_ref(),
                                        "converted_codex_response": resp.clone(),
                                        "gemini_raw_response_ref": "see upstream_response file with the same trace_id",
                                    });
                                    debug_logger::write_exchange_payload(
                                        &debug_cfg,
                                        Some(&trace_id),
                                        "exchange_summary",
                                        &payload,
                                    )
                                    .await;
                                }

                                return Response::builder()
                                    .status(StatusCode::OK)
                                    .header("Content-Type", "application/json")
                                    .header("X-Account-Email", email.as_str())
                                    .header("X-Mapped-Model", mapped_model.as_str())
                                    .header("X-Upstream-Model", config.final_model.as_str())
                                    .header("X-Session-Id", session_scope.client_id.as_str())
                                    .header(
                                        "X-Antigravity-Session-Id",
                                        session_scope.client_id.as_str(),
                                    )
                                    .timing(&TimingHeaders::new(
                                        clean_ms,
                                        norm_ms,
                                        think_fill_ms,
                                        ttft_ms,
                                        headers_ms,
                                        fallback_overhead_ms,
                                        attempts_b64.as_deref().unwrap_or(""),
                                    ))
                                    .body(Body::from(
                                        serde_json::to_string(&resp).unwrap_or_default(),
                                    ))
                                    .unwrap()
                                    .into_response();
                            }

                            // NOW: Convert Chat Response -> Legacy Response (Same logic as below)
                            let choices = chat_resp
                                .choices
                                .iter()
                                .map(|c| {
                                    let mut text = match &c.message.content {
                                        Some(
                                            crate::proxy::mappers::openai::OpenAIContent::String(s),
                                        ) => s.clone(),
                                        _ => "".to_string(),
                                    };
                                    if let Some(ref reasoning) = c.message.reasoning_content {
                                        if !reasoning.is_empty() {
                                            text = format!("{}\n\n{}", reasoning, text);
                                        }
                                    }
                                    json!({
                                        "text": text,
                                        "index": c.index,
                                        "logprobs": null,
                                        "finish_reason": c.finish_reason
                                    })
                                })
                                .collect::<Vec<_>>();

                            let legacy_resp = json!({
                                "id": chat_resp.id,
                                "object": "text_completion",
                                "created": chat_resp.created,
                                "model": chat_resp.model,
                                "choices": choices,
                                "usage": chat_resp.usage
                            });
                            if debug_logger::is_enabled(&debug_cfg) {
                                let payload = json!({
                                    "kind": "exchange_summary",
                                    "protocol": "openai",
                                    "trace_id": trace_id,
                                    "request_path": uri.path(),
                                    "original_codex_request": original_body.as_ref(),
                                    "gemini_request": gemini_body_for_debug.as_ref(),
                                    "converted_codex_response": legacy_resp.clone(),
                                    "gemini_raw_response_ref": "see upstream_response file with the same trace_id",
                                });
                                debug_logger::write_exchange_payload(
                                    &debug_cfg,
                                    Some(&trace_id),
                                    "exchange_summary",
                                    &payload,
                                )
                                .await;
                            }

                            return Response::builder()
                                .status(StatusCode::OK)
                                .header("Content-Type", "application/json")
                                .header("X-Account-Email", email.as_str())
                                .header("X-Mapped-Model", mapped_model.as_str())
                                .header("X-Upstream-Model", config.final_model.as_str())
                                .header("X-Session-Id", session_scope.client_id.as_str())
                                .header(
                                    "X-Antigravity-Session-Id",
                                    session_scope.client_id.as_str(),
                                )
                                .timing(&TimingHeaders::new(
                                    clean_ms,
                                    norm_ms,
                                    think_fill_ms,
                                    ttft_ms,
                                    headers_ms,
                                    fallback_overhead_ms,
                                    attempts_b64.as_deref().unwrap_or(""),
                                ))
                                .body(Body::from(
                                    serde_json::to_string(&legacy_resp).unwrap_or_default(),
                                ))
                                .unwrap()
                                .into_response();
                        }
                        Err(e) => {
                            retry_state.preserve_on_stop(
                                token_manager.get_scheduling_mode().await,
                                config.request_type != "image_gen",
                            );
                            return retry_state.finish_error_response(
                                (
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    format!("Stream collection error: {}", e),
                                )
                                    .into_response(),
                            );
                        }
                    }
                }
            }

            headers_ms = effective_attempt_start.elapsed().as_micros() as f64 / 1000.0;
            // 直接 JSON 路径：响应头与 body 同一响应，TTFT 按响应头时间等价记录；
            // 若已有 peek 得到的首块值则保留。
            if ttft_ms < 0.0 {
                ttft_ms = headers_ms;
            }
            if let Some(effective_item) = fallback_attempts.iter_mut().find(|a| a.is_effective) {
                effective_item.duration_s = headers_ms / 1000.0;
            }
            let _ = crate::proxy::monitor::CURRENT_UPSTREAM_CAPTURE.try_with(|holder| {
                holder.set_attempts(fallback_attempts.clone());
            });
            let gemini_resp: Value =
                match crate::proxy::pipeline::usage_diagnostics::read_json_response(
                    response,
                    crate::proxy::pipeline::usage_diagnostics::UsageAttempt::current(
                        &config.final_model,
                        &account_id,
                        "openai_responses_json",
                    ),
                )
                .await
                {
                    Ok(json) => json,
                    Err(e) => {
                        retry_state.preserve_on_stop(
                            token_manager.get_scheduling_mode().await,
                            config.request_type != "image_gen",
                        );
                        return retry_state.finish_error_response(
                            (
                                StatusCode::BAD_GATEWAY,
                                [("X-Mapped-Model", mapped_model.as_str())],
                                format!("Parse error: {}", e),
                            )
                                .into_response(),
                        );
                    }
                };

            if let Some(confirmation) = session_confirmation.as_mut() {
                confirmation.observe_json(&gemini_resp);
            }

            crate::proxy::thinking_store::capture_gemini_response_with_context(
                &session_id_str,
                &gemini_resp,
                None,
                response_text_context.as_deref(),
            );

            let chat_resp =
                crate::proxy::mappers::openai::response::transform_openai_response_with_context(
                    &gemini_resp,
                    Some(&signature_session_id_str),
                    1,
                    Some(&client_tool_names),
                    response_text_context.clone(),
                );

            let is_responses_api = is_responses_api_path(uri.path());

            if is_responses_api {
                let is_structured = is_request_structured_output(&openai_req);
                let resp = convert_chat_response_to_responses_with_tools(
                    &chat_resp,
                    is_structured,
                    &responses_tool_kinds,
                );
                if debug_logger::is_enabled(&debug_cfg) {
                    let payload = json!({
                        "kind": "exchange_summary",
                        "protocol": "openai",
                        "trace_id": trace_id,
                        "request_path": uri.path(),
                        "original_codex_request": original_body.as_ref(),
                        "gemini_request": gemini_body_for_debug.as_ref(),
                        "gemini_raw_response": gemini_resp.clone(),
                        "converted_codex_response": resp.clone(),
                    });
                    debug_logger::write_exchange_payload(
                        &debug_cfg,
                        Some(&trace_id),
                        "exchange_summary",
                        &payload,
                    )
                    .await;
                }

                return Response::builder()
                    .status(StatusCode::OK)
                    .header("Content-Type", "application/json")
                    .header("X-Account-Email", email.as_str())
                    .header("X-Mapped-Model", mapped_model.as_str())
                    .header("X-Upstream-Model", config.final_model.as_str())
                    .header("X-Session-Id", session_scope.client_id.as_str())
                    .header("X-Antigravity-Session-Id", session_scope.client_id.as_str())
                    .timing_core(&TimingHeaders::new(
                        clean_ms,
                        norm_ms,
                        think_fill_ms,
                        ttft_ms,
                        headers_ms,
                        0.0,
                        "",
                    ))
                    .body(Body::from(serde_json::to_string(&resp).unwrap_or_default()))
                    .unwrap()
                    .into_response();
            }

            // Map Chat Response -> Legacy Completions Response
            let choices = chat_resp.choices.iter().map(|c| {
                json!({
                    "text": match &c.message.content {
                        Some(crate::proxy::mappers::openai::OpenAIContent::String(s)) => s.clone(),
                        _ => "".to_string()
                    },
                    "index": c.index,
                    "logprobs": null,
                    "finish_reason": c.finish_reason
                })
            }).collect::<Vec<_>>();

            let legacy_resp = json!({
                "id": chat_resp.id,
                "object": "text_completion",
                "created": chat_resp.created,
                "model": chat_resp.model,
                "choices": choices,
                "usage": chat_resp.usage
            });
            if debug_logger::is_enabled(&debug_cfg) {
                let payload = json!({
                    "kind": "exchange_summary",
                    "protocol": "openai",
                    "trace_id": trace_id,
                    "request_path": uri.path(),
                    "original_codex_request": original_body.as_ref(),
                    "gemini_request": gemini_body_for_debug.as_ref(),
                    "gemini_raw_response": gemini_resp.clone(),
                    "converted_codex_response": legacy_resp.clone(),
                });
                debug_logger::write_exchange_payload(
                    &debug_cfg,
                    Some(&trace_id),
                    "exchange_summary",
                    &payload,
                )
                .await;
            }

            return Response::builder()
                .status(StatusCode::OK)
                .header("Content-Type", "application/json")
                .header("X-Account-Email", email.as_str())
                .header("X-Mapped-Model", mapped_model.as_str())
                .header("X-Upstream-Model", config.final_model.as_str())
                .header("X-Session-Id", session_scope.client_id.as_str())
                .header("X-Antigravity-Session-Id", session_scope.client_id.as_str())
                .timing_core(&TimingHeaders::new(
                    clean_ms,
                    norm_ms,
                    think_fill_ms,
                    ttft_ms,
                    headers_ms,
                    0.0,
                    "",
                ))
                .body(Body::from(
                    serde_json::to_string(&legacy_resp).unwrap_or_default(),
                ))
                .unwrap()
                .into_response();
        };

        failure_statuses.record(status);
        let status_code = status.as_u16();
        last_error = format!("HTTP {}: {}", status_code, error_text);
        let classification = crate::proxy::pipeline::UpstreamClassification::classify(
            status_code,
            &error_text,
            retry_after.as_deref(),
        );
        let scheduling = token_manager.get_sticky_config().await;

        // [Pipeline Auto-Heal] 503 过载 / 400 tool_choice 冲突 / 400 损坏图片：统一在同一账号原地修复重试
        if let Some(heal) = crate::proxy::pipeline::auto_heal::try_auto_heal_with_retry_policy(
            &classification,
            status_code,
            &error_text,
            retry_after.as_deref(),
            &mut gemini_body,
            &mut heal_flags,
            &mut retry_state,
            &account_id,
            &scheduling,
            config.request_type != "image_gen",
        ) {
            force_rotate = false;
            crate::proxy::stage_trace::StageTrace::alert_current(
                "self_heal",
                format!(
                    "protocol=openai_responses kind={} status={} account={}",
                    heal.kind, status_code, email
                ),
            );
            if !apply_retry_strategy(
                RetryStrategy::GraceRetry(std::time::Duration::from_millis(heal.delay_ms)),
                attempt,
                max_attempts,
                status_code,
                &trace_id,
            )
            .await
            {
                break;
            }
            if config.request_type != "image_gen" {
                retry_credentials = Some((
                    access_token.clone(),
                    project_id.clone(),
                    email.clone(),
                    account_id.clone(),
                    0,
                    session_lease.clone(),
                ));
            }
            continue;
        }

        // CacheFirst policy is shared across protocols and runs before account penalties.
        if let Some(strategy) = retry_state.cache_first_failure(
            &account_id,
            status_code,
            &error_text,
            retry_after.as_deref(),
            &token_manager.get_sticky_config().await,
            config.request_type != "image_gen",
        ) {
            if apply_retry_strategy(strategy, attempt, max_attempts, status_code, &trace_id).await {
                retry_credentials = Some((
                    access_token.clone(),
                    project_id.clone(),
                    email.clone(),
                    account_id.clone(),
                    0,
                    session_lease.clone(),
                ));
                continue;
            }
            break;
        }

        tracing::error!(
            "[Codex-Upstream] Error Response {}: {}",
            status_code,
            error_text
        );

        // 3. 统一流水线判定与标记限流状态(用于 UI 显示)

        // [FIX Signature] 400 思考签名无效：清库 + 流水线去签名，同账号原地重试
        if classification.is_thought_signature_error() {
            if !retried_without_thinking {
                retried_without_thinking = true;
                tracing::warn!(
                    "[{}] Pipeline: Thinking signature error detected on upstream (HTTP {}). Purging corrupted signatures and retrying on same account.",
                    trace_id, status_code
                );
                session_scope.purge_signatures(&mapped_model);
                force_rotate = false;
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                continue;
            } else {
                tracing::warn!(
                    "[{}] Pipeline: Thinking signature error persisted after retry without thinking. Terminating retry loop.",
                    trace_id
                );
            }
        }

        // [FIX Corrupted Image] 已由 `pipeline::auto_heal::try_auto_heal` 统一处理（见上）

        if classification.is_model_not_found() {
            // [Pipeline Policy] UA 门禁短路 + 单模型临时熔断 + 是否换号的决策已下沉（与 chat 路径同构）
            let custom_ua = state.upstream.get_user_agent_override().await;
            let can_rotate = crate::proxy::pipeline::policy::apply_model_not_found_with_ua(
                &mapped_model,
                status_code,
                &error_text,
                custom_ua.as_deref(),
                attempt,
                pool_size,
                || token_manager.mark_model_unsupported(&account_id, &mapped_model, Some(900)),
            );
            if can_rotate {
                tracing::warn!(
                    "[{}] 上游报错模型不可用 (HTTP {})，已标记账号 {} 对模型 [{}] 临时熔断，继续换号重试 ({}/{})...",
                    trace_id, status_code, email, mapped_model, attempt, pool_size
                );
                force_rotate = true;
                continue;
            }
            tracing::warn!(
                "[{}] Pipeline: Target model [{}] not found on upstream (HTTP {}). Pool exhausted without account-level lockout.",
                trace_id, mapped_model, status_code
            );
            let protocol = if is_responses_api {
                "responses"
            } else {
                "openai"
            };
            let dual_err = crate::proxy::handlers::common::build_dual_track_error_with_ua(
                protocol,
                status_code,
                &mapped_model,
                &error_text,
                custom_ua.as_deref(),
            );
            return Response::builder()
                .status(StatusCode::from_u16(status_code).unwrap_or(StatusCode::NOT_FOUND))
                .header("X-Account-Email", email.as_str())
                .header("X-Mapped-Model", mapped_model.as_str())
                .body(Body::from(
                    serde_json::to_string(&dual_err).unwrap_or_default(),
                ))
                .unwrap()
                .into_response();
        }

        if classification.should_lock_account_on_attempt(
            status_code,
            &error_text,
            retry_after.as_deref(),
            attempt,
        ) {
            let needs_quota_refresh = token_manager
                .mark_rate_limited_fast(
                    &email,
                    status_code,
                    retry_after.as_deref(),
                    &error_text,
                    Some(&mapped_model),
                )
                .await;
            if needs_quota_refresh {
                token_manager
                    .refresh_quota_lock_after_fast_mark(&email, Some(&mapped_model))
                    .await;
            }
        }

        if classification.abandons_sticky_account() {
            token_manager.abandon_session_lease(&mut session_lease);
        }

        // [Grace Policy] SchedulingMode 不参与此裁决：429 微间隙 GraceRetry 属于
        // “账号池保护”（Fixes #3329/#3409），单账号等待已由 RequestRetryState 限制为
        // 每账号一次，不存在无界等待。Claude 因历史原因仍按调度模式决定是否原地等待。

        let strategy = retry_state.determine_strategy_adaptive(
            &account_id,
            status_code,
            &error_text,
            retry_after.as_deref(),
            false,
            attempt,
            pool_size,
        );

        // 执行退备
        if apply_retry_strategy(
            strategy.clone(),
            attempt,
            max_attempts,
            status_code,
            &trace_id,
        )
        .await
        {
            if matches!(strategy, RetryStrategy::GraceRetry(_)) {
                retry_credentials = Some((
                    access_token.clone(),
                    project_id.clone(),
                    email.clone(),
                    account_id.clone(),
                    0,
                    session_lease.clone(),
                ));
            }
            force_rotate = should_rotate_account(status_code, Some(&strategy));
            if force_rotate {
                crate::proxy::pipeline::rotation::record_attempted_account(
                    &mut attempted_accounts,
                    &account_id,
                    pool_size,
                    &mut force_rotate,
                );
            }
            continue;
        } else {
            // 不可重试
            let protocol = if is_responses_api {
                "responses"
            } else {
                "openai"
            };
            let dual_err = crate::proxy::handlers::common::build_dual_track_error(
                protocol,
                status_code,
                &mapped_model,
                &error_text,
            );
            return (
                status,
                [
                    ("X-Account-Email", email.as_str()),
                    ("X-Mapped-Model", mapped_model.as_str()),
                ],
                axum::Json(dual_err),
            )
                .into_response();
        }
    }

    // 所有尝试均失败
    let final_status = failure_statuses.final_status();
    let headers = crate::proxy::handlers::common::build_token_error_headers(
        Some(mapped_model.as_str()),
        last_email.as_deref(),
        &last_error,
    );
    let protocol = if is_responses_api {
        "responses"
    } else {
        "openai"
    };
    let mut dual_err = crate::proxy::handlers::common::build_dual_track_error(
        protocol,
        final_status.as_u16(),
        &mapped_model,
        &last_error,
    );
    crate::proxy::pipeline::stream_error::attach_error_usage(
        &mut dual_err,
        protocol,
        terminal_stream_usage.as_ref(),
    );
    retry_state.finish_error_response((final_status, headers, axum::Json(dual_err)).into_response())
}

/// Helper to resolve token limits (context length and max output tokens) for any model ID.
pub fn resolve_model_token_limits(model_id: &str) -> (i64, i64) {
    if !model_id.is_empty() {
        if let Some(info) = crate::models::official_model::OfficialModelCatalog::get(model_id) {
            let max_tokens = info.max_tokens.unwrap_or(1_048_576);
            let max_output = info.max_output_tokens.unwrap_or(65_536);
            return (max_tokens, max_output);
        }
        let lower = model_id.to_lowercase();
        if lower.contains("claude") {
            if lower.contains("5-5") || lower.contains("5.5") {
                return (1_000_000, 128_000);
            }
            return (200_000, 64_000);
        }
        if lower.contains("gpt") || lower.contains("o1") || lower.contains("o3") {
            return (131_072, 32_768);
        }
    }
    (1_048_576, 65_536)
}

#[derive(Debug, serde::Deserialize, Default)]
pub struct PropsQuery {
    pub model: Option<String>,
}

/// llama.cpp compatible /props and /v1/props endpoint.
/// Hermes Agent and local LLM clients probe this to determine runtime n_ctx.
pub async fn handle_props(
    query: Result<axum::extract::Query<PropsQuery>, axum::extract::rejection::QueryRejection>,
) -> impl IntoResponse {
    let model_id = query.ok().and_then(|q| q.0.model).unwrap_or_default();
    let (n_ctx, n_predict) = resolve_model_token_limits(&model_id);
    let alias = if model_id.is_empty() {
        "default"
    } else {
        model_id.as_str()
    };

    let resp = json!({
        "default_generation_settings": {
            "n_ctx": n_ctx,
            "n_predict": n_predict,
        },
        "total_slots": 1,
        "model_alias": alias
    });

    (StatusCode::OK, Json(resp))
}

/// vLLM compatible /version and /v1/version endpoint.
/// Hermes Agent and vLLM clients probe this to detect server type.
pub async fn handle_version() -> impl IntoResponse {
    let resp = json!({
        "version": env!("CARGO_PKG_VERSION"),
        "app": "antigravity-tools",
        "name": "antigravity-tools"
    });

    (StatusCode::OK, Json(resp))
}

#[cfg(test)]
mod model_probe_compatibility_tests {
    use super::*;

    #[test]
    fn test_resolve_model_token_limits() {
        let (ctx, out) = resolve_model_token_limits("gemini-3.8-flash-high");
        assert!(ctx >= 131_072, "ctx={ctx}");
        assert_eq!(out, 65_536);

        // Aliased model inherits mapped official model limits
        let (ctx_aliased, out_aliased) = resolve_model_token_limits("gpt-4o");
        assert!(ctx_aliased >= 131_072, "ctx_aliased={ctx_aliased}");
        assert_eq!(out_aliased, 32_768);

        // Uncataloged fallbacks
        let (ctx_claude, out_claude) = resolve_model_token_limits("custom-claude-unknown");
        assert_eq!(ctx_claude, 200_000);
        assert_eq!(out_claude, 64_000);

        let (ctx_gpt, out_gpt) = resolve_model_token_limits("unknown-gpt-custom");
        assert_eq!(ctx_gpt, 131_072);
        assert_eq!(out_gpt, 32_768);

        let (ctx_def, out_def) = resolve_model_token_limits("");
        assert_eq!(ctx_def, 1_048_576);
        assert_eq!(out_def, 65_536);
    }

    #[tokio::test]
    async fn test_handle_version() {
        let resp = handle_version().await.into_response();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_handle_props() {
        let query = Ok(axum::extract::Query(PropsQuery {
            model: Some("gemini-3.8-flash-high".to_string()),
        }));
        let resp = handle_props(query).await.into_response();
        assert_eq!(resp.status(), StatusCode::OK);
    }
}

pub async fn handle_list_models(State(state): State<AppState>) -> impl IntoResponse {
    let entries = crate::proxy::handlers::common::collect_model_list_entries(&state).await;
    let data = crate::proxy::handlers::common::openai_style_model_objects(entries);

    Json(json!({
        "object": "list",
        "data": data
    }))
}

/// OpenAI Models API: GET /v1/models/{model}
/// 检索指定模型的详细元数据
pub async fn handle_retrieve_model(
    State(state): State<AppState>,
    axum::extract::Path(model): axum::extract::Path<String>,
) -> impl IntoResponse {
    use crate::proxy::common::model_mapping::{
        find_dynamic_model, get_all_dynamic_models_metadata,
    };

    // [MODEL GATE] GET /v1/models/:model 探测端点同样接入门禁（含已下线模型族 404）
    if let Err(rejection) = crate::proxy::handlers::common::enforce_model_gate(
        &state,
        Some(model.as_str()),
        "openai",
        None,
    )
    .await
    {
        return rejection;
    }

    let only_raw = *state.only_raw_quota_models.read().await;
    let resolved_id = if let Some(matched_id) = find_dynamic_model(
        &state.custom_mapping,
        Some(&state.token_manager),
        only_raw,
        &model,
    )
    .await
    {
        Some(matched_id)
    } else if crate::models::official_model::OfficialModelCatalog::get(&model).is_some() {
        Some(model.clone())
    } else {
        None
    };

    if let Some(matched_id) = resolved_id {
        let (fallback_ctx, fallback_out) = resolve_model_token_limits(&matched_id);
        let metadata_map = get_all_dynamic_models_metadata(
            &state.custom_mapping,
            Some(&state.token_manager),
            only_raw,
        )
        .await;

        let meta = metadata_map
            .get(&matched_id)
            .or_else(|| metadata_map.get(&matched_id.to_lowercase()));
        let ctx_len = meta
            .and_then(|m| m.max_tokens)
            .map(|v| v as i64)
            .unwrap_or(fallback_ctx);
        let out_len = meta
            .and_then(|m| m.max_output_tokens)
            .map(|v| v as i64)
            .unwrap_or(fallback_out);

        let mut obj = json!({
            "id": matched_id,
            "object": "model",
            "created": 1706745600,
            "owned_by": "antigravity",
            "context_length": ctx_len,
            "max_context_length": ctx_len,
            "max_input_tokens": ctx_len,
            "max_model_len": ctx_len,
            "max_tokens": ctx_len,
            "input_token_limit": ctx_len,
            "max_completion_tokens": out_len,
            "max_output_tokens": out_len,
            "output_token_limit": out_len,
        });
        if let Some(m) = meta {
            if let Ok(serde_json::Value::Object(map)) = serde_json::to_value(m) {
                if let Some(target) = obj.as_object_mut() {
                    for (k, v) in map {
                        target.insert(k, v);
                    }
                }
            }
        }

        (StatusCode::OK, Json(obj)).into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": {
                    "message": format!("The model '{}' does not exist", model),
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found"
                }
            })),
        )
            .into_response()
    }
}

/// OpenAI Images API: POST /v1/images/generations
/// 处理图像生成请求，转换为 Gemini API 格式
pub async fn handle_chat_redirection(
    State(state): State<AppState>,
    headers: HeaderMap,
    upstream_recorder: Option<
        axum::extract::Extension<crate::proxy::monitor::UpstreamRequestBodyHolder>,
    >,
    Json(body): Json<Value>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    handle_chat_completions(State(state), headers, upstream_recorder, None, Json(body)).await
}

async fn intercept_chat_to_image(
    state: AppState,
    headers: HeaderMap,
    tenant_id: Option<String>,
    body: Value,
    model_name: &str,
) -> Result<Response, (StatusCode, String)> {
    // 1. Extract prompt from messages
    let mut prompt = String::new();
    if let Some(messages) = body.get("messages").and_then(|v| v.as_array()) {
        for msg in messages {
            if msg.get("role").and_then(|v| v.as_str()) == Some("user") {
                if let Some(content) = msg.get("content") {
                    if let Some(s) = content.as_str() {
                        prompt = s.to_string();
                    } else if let Some(arr) = content.as_array() {
                        for part in arr {
                            if part.get("type").and_then(|v| v.as_str()) == Some("text") {
                                prompt.push_str(
                                    part.get("text").and_then(|v| v.as_str()).unwrap_or(""),
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    if prompt.is_empty() {
        prompt = "A beautiful painting".to_string(); // fallback
    }

    let is_stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // 2. Call internal image generator
    let img_req = json!({
        "prompt": prompt,
        "model": model_name,
        "n": 1,
        "response_format": "url"
    });

    match handle_images_generations_internal(state, headers, tenant_id, img_req).await {
        Ok((email, img_res)) => {
            // Extract URL
            let mut img_markdown = String::new();
            if let Some(data) = img_res.get("data").and_then(|v| v.as_array()) {
                for item in data {
                    if let Some(url) = item.get("url").and_then(|v| v.as_str()) {
                        img_markdown.push_str(&format!("![Generated Image]({})\n\n", url));
                    }
                }
            }

            if img_markdown.is_empty() {
                img_markdown = "Failed to extract image URL from generation result.".to_string();
            }

            // 3. Construct Chat Completion Response
            if is_stream {
                use axum::body::Body;

                let chunk = json!({
                    "id": format!("chatcmpl-img-{}", uuid::Uuid::new_v4()),
                    "object": "chat.completion.chunk",
                    "created": chrono::Utc::now().timestamp(),
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": img_markdown
                        },
                        "finish_reason": null
                    }]
                });

                let done_chunk = json!({
                    "id": format!("chatcmpl-img-{}", uuid::Uuid::new_v4()),
                    "object": "chat.completion.chunk",
                    "created": chrono::Utc::now().timestamp(),
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                });

                let sse_data = format!(
                    "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
                    chunk.to_string(),
                    done_chunk.to_string()
                );

                let body = Body::from(sse_data);
                Ok(Response::builder()
                    .header("Content-Type", "text/event-stream")
                    .header("Cache-Control", "no-cache")
                    .header("X-Account-Email", email)
                    .body(body)
                    .unwrap())
            } else {
                let resp = json!({
                    "id": format!("chatcmpl-img-{}", uuid::Uuid::new_v4()),
                    "object": "chat.completion",
                    "created": chrono::Utc::now().timestamp(),
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": img_markdown
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
                });

                Ok((
                    StatusCode::OK,
                    [("X-Account-Email", email.as_str())],
                    Json(resp),
                )
                    .into_response())
            }
        }
        Err((status, msg, _email)) => Err((status, msg)),
    }
}

pub async fn handle_images_generations(
    State(state): State<AppState>,
    headers: HeaderMap,
    user_identity: Option<
        axum::extract::Extension<crate::proxy::middleware::auth::UserTokenIdentity>,
    >,
    Json(body): Json<Value>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    match handle_images_generations_internal(
        state,
        headers,
        user_identity.map(|identity| identity.token_id.clone()),
        body,
    )
    .await
    {
        Ok((email_header, openai_response)) => Ok((
            StatusCode::OK,
            [("X-Account-Email", email_header.as_str())],
            Json(openai_response),
        )
            .into_response()),
        // Attach the attempted account to error responses too, so the traffic log shows
        // which account the failed (e.g. 502/503) image request used.
        Err((status, msg, email_opt)) => {
            let email = email_opt.unwrap_or_default();
            Ok((status, [("X-Account-Email", email)], msg).into_response())
        }
    }
}

pub async fn handle_images_generations_internal(
    state: AppState,
    headers: HeaderMap,
    tenant_id: Option<String>,
    body: Value,
) -> Result<(String, Value), (StatusCode, String, Option<String>)> {
    // 1. 解析请求参数
    let prompt = body.get("prompt").and_then(|v| v.as_str()).ok_or((
        StatusCode::BAD_REQUEST,
        "Missing 'prompt' field".to_string(),
        None,
    ))?;
    let affinity_key = crate::proxy::thinking_store::SessionScope::resolve(
        &headers,
        Some(&body),
        None,
        prompt,
        tenant_id.as_deref(),
    )
    .affinity_key;

    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("gemini-3.1-flash-image");

    // [MODEL GATE] 生图入口做存在性校验（不做整池配额检查，避免与文本配额桶误杀）：
    // 不存在 / 已下线 → 404 model not exist，属于网关自答，不进入选号与账号惩罚。
    if let Some((status, reason)) = crate::proxy::common::model_gate::gate_inbound_model_exists(
        Some(model),
        &state.custom_mapping,
        &state.token_manager,
    )
    .await
    .rejection()
    {
        return Err((status, reason.to_string(), None));
    }

    let n = body.get("n").and_then(|v| v.as_u64()).unwrap_or(1) as usize;

    let size = body.get("size").and_then(|v| v.as_str());

    let response_format = body
        .get("response_format")
        .and_then(|v| v.as_str())
        .unwrap_or("b64_json");

    let quality = body.get("quality").and_then(|v| v.as_str());

    let image_size = generation_image_size_param(&body)
        .map_err(|message| (StatusCode::BAD_REQUEST, message, None))?;

    let style = body
        .get("style")
        .and_then(|v| v.as_str())
        .unwrap_or("vivid");

    // Canvas compatibility extension: OpenAI's standard generations endpoint does not define
    // this top-level field. Accept only inline data:image URLs and never fetch remote URLs.
    let input_images = parse_generation_input_images(body.get("image"))
        .map_err(|message| (StatusCode::BAD_REQUEST, message, None))?;

    info!(
        model = model,
        image_count = input_images.len(),
        n = n,
        size = size.unwrap_or("auto"),
        quality = quality.unwrap_or("auto"),
        style = style,
        "[Images] Received generation request"
    );

    // 2. 使用 common_utils 解析图片配置（统一逻辑，支持动态计算宽高比和 quality 映射）
    let (image_config, clean_model_name) =
        crate::proxy::mappers::common_utils::try_parse_image_config_with_params(
            model, size, quality, image_size,
        )
        .map_err(|message| (StatusCode::BAD_REQUEST, message, None))?;

    // 3. Prompt Enhancement（保留原有逻辑）
    let mut final_prompt = prompt.to_string();
    if quality == Some("hd") {
        final_prompt.push_str(", (high quality, highly detailed, 4k resolution, hdr)");
    }
    match style {
        "vivid" => final_prompt.push_str(", (vivid colors, dramatic lighting, rich details)"),
        "natural" => final_prompt.push_str(", (natural lighting, realistic, photorealistic)"),
        _ => {}
    }
    let contents_parts = build_image_contents(final_prompt, &input_images, None);

    // 4. 并发发送请求
    // 注意：不再在外部获取 Token，而是移入 Task 内部并在重试时获取
    let upstream = state.upstream.clone();
    let token_manager = state.token_manager.clone();
    let image_scheduler = state.image_scheduler.clone();
    let request_timeout = state.request_timeout;
    let max_pool_size = token_manager.len();
    // [FIX #3485] 自适应多账号池与单账号退避最大重试次数 (单账号3次，多账号整池两轮)
    let max_attempts = crate::proxy::handlers::common::calculate_max_retry_attempts(max_pool_size);

    // [Pipeline Budget] JoinSet 子任务不会继承 task_local：在 spawn 前捕获请求上下文，
    // 并在子任务内恢复，否则图片链路既没有 270s 总预算，也不会被大 body 重放护栏计数。
    let request_budget = crate::proxy::pipeline::budget::current();
    let usage_holder = crate::proxy::monitor::UpstreamRequestBodyHolder::current();

    let mut tasks = JoinSet::new();

    // Track the last account actually attempted, so error responses (502/503) can be
    // attributed to an account in the traffic log instead of showing "(none)".
    let attempted_account = std::sync::Arc::new(std::sync::Mutex::new(None::<String>));

    for _ in 0..n {
        let upstream = upstream.clone();
        let token_manager = token_manager.clone();
        let request_budget = request_budget.as_ref().map(|ctx| ctx.fork());
        let usage_holder = usage_holder.clone();
        let contents_parts = contents_parts.clone();
        let image_config = image_config.clone(); // 使用解析后的完整配置
        let _response_format = response_format.to_string();

        let model_to_use = clean_model_name.clone();
        let attempted_account = attempted_account.clone();
        let image_scheduler = image_scheduler.clone();
        let affinity_key = affinity_key.clone();

        tasks.spawn(async move {
            let mut image_permit = None;
            let mut last_error = String::new();
            let mut force_rotate = false;
            let mut retry_state = RequestRetryState::default();
            let mut retry_credentials: Option<crate::proxy::token_manager::LeasedCredentials> = None;
            let mut failure_statuses = FailureStatusTracker::default();
            let mut used_attempts = 0;

            while let Some(attempt) = next_rotation_attempt(
                &mut used_attempts,
                max_attempts,
                retry_credentials.is_some(),
            ) {
                if crate::proxy::pipeline::terminal::gate_exhausted(
                    "openai_images",
                    attempt + 1,
                    &mut failure_statuses,
                    &mut last_error,
                ) {
                    break;
                }
                let (access_token, project_id, email, account_id, _wait_ms, mut session_lease) =
                    if let Some(credentials) = retry_credentials.take() {
                        credentials
                    } else {
                        drop(image_permit.take());
                        match token_manager
                            .get_image_token_with_lease(
                                force_rotate,
                                Some(&affinity_key),
                                &model_to_use,
                                &image_scheduler,
                                request_timeout,
                            )
                            .await
                        {
                            Ok((
                                access_token,
                                project_id,
                                email,
                                account_id,
                                wait_ms,
                                permit,
                                lease,
                            )) => {
                                image_permit = Some(permit);
                                (access_token, project_id, email, account_id, wait_ms, lease)
                            }
                            Err((status, e)) => {
                                last_error = format!("Token error: {}", e);
                                failure_statuses.record(status);
                                if status == StatusCode::TOO_MANY_REQUESTS {
                                    return Err((status, e));
                                }
                                if attempt < max_attempts - 1 {
                                    tokio::time::sleep(Duration::from_millis(500)).await;
                                    continue;
                                }
                                break;
                            }
                        }
                    };
                if let Ok(mut g) = attempted_account.lock() {
                    *g = Some(email.clone());
                }

                // [FIX] Resolve to the account-specific dynamic image model, exactly like the
                // chat (openai.rs:232) and gemini (gemini.rs:155) handlers do. Sending the static
                // alias (e.g. "gemini-3-pro-image") made upstream return 404 "Requested entity was
                // not found" because each account exposes its own concrete image model id.
                let resolved_model = token_manager
                    .resolve_dynamic_model_for_account(&account_id, &model_to_use)
                    .await;

                let gemini_body = json!({
                    "project": project_id,
                    "requestId": format!("agent-{}", uuid::Uuid::new_v4()),
                    "model": resolved_model,
                    "userAgent": "antigravity",
                    "requestType": "image_gen",
                    "request": {
                        "contents": [{
                            "role": "user",
                            "parts": contents_parts
                        }],
                        "generationConfig": {
                            "candidateCount": 1, // 强制单张
                            "imageConfig": image_config // ✅ 使用完整配置（包含 aspectRatio 和 imageSize）
                        },
                        "safetySettings": [
                            { "category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF" },
                            { "category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF" },
                            { "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF" },
                            { "category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF" },
                        ]
                    }
                });

                let upstream_call = upstream.call_v1_internal(
                    "generateContent",
                    &access_token,
                    gemini_body,
                    None,
                    Some(account_id.as_str()),
                );
                match crate::proxy::pipeline::budget::scope_optional(
                    request_budget.clone(),
                    upstream_call,
                )
                .await
                {
                    Ok(call_result) => {
                        let response = call_result.response;
                        let status = response.status();
                        if !status.is_success() {
                            failure_statuses.record(status);
                            let retry_after = response
                                .headers()
                                .get("Retry-After")
                                .and_then(|header| header.to_str().ok())
                                .map(str::to_string);
                            let err_text = response.text().await.unwrap_or_default();
                            let status_code = status.as_u16();
                            last_error = format!("Upstream error {}: {}", status, err_text);
                            let strategy = (status_code == 429).then(|| {
                                retry_state.determine_strategy(
                                    &account_id,
                                    status_code,
                                    &err_text,
                                    retry_after.as_deref(),
                                    false,
                                )
                            });
                            // 统一流水线限流裁决：500/503等服务异常绝不打入限流
                            let classification = crate::proxy::pipeline::UpstreamClassification::classify(
                                status_code,
                                &err_text,
                                retry_after.as_deref(),
                            );
                            if classification.abandons_sticky_account() {
                                token_manager.abandon_session_lease(&mut session_lease);
                            }
                            let should_mark_limited = classification.should_lock_account();
                            let needs_quota_refresh = if should_mark_limited {
                                tracing::warn!(
                                    "[Images] Account {} rate limited/error ({}), rotating...",
                                    email,
                                    status_code
                                );
                                token_manager
                                    .mark_rate_limited_fast(
                                        &email,
                                        status_code,
                                        retry_after.as_deref(),
                                        &err_text,
                                        Some(&resolved_model),
                                    )
                                    .await
                            } else {
                                false
                            };
                            if !matches!(strategy.as_ref(), Some(RetryStrategy::GraceRetry(_))) {
                                drop(image_permit.take());
                            }
                            if needs_quota_refresh {
                                token_manager
                                    .refresh_quota_lock_after_fast_mark(
                                        &email,
                                        Some(&resolved_model),
                                    )
                                    .await;
                            }

                            if let Some(strategy) = strategy {
                                if apply_retry_strategy(
                                    strategy.clone(),
                                    attempt,
                                    max_attempts,
                                    status_code,
                                    "image_generation",
                                )
                                .await
                                {
                                    if matches!(strategy, RetryStrategy::GraceRetry(_)) {
                                        retry_credentials = Some((
                                            access_token.clone(),
                                            project_id.clone(),
                                            email.clone(),
                                            account_id.clone(),
                                            0,
                                            session_lease.clone(),
                                        ));
                                    }
                                    force_rotate =
                                        should_rotate_account(status_code, Some(&strategy));
                                    continue;
                                }
                            }

                            if status_code == 503 || status_code == 500 {
                                force_rotate = true;
                                continue; // Retry loop
                            }

                            // [FIX] 403/404 usually mean THIS account lacks the image model or
                            // project access. Rotate to another account instead of failing the
                            // whole request, so an image-capable account can serve it.
                            if (status_code == 403 || status_code == 404)
                                && attempt < max_attempts - 1
                            {
                                // 与文本链路对齐：该账号缺模型/权限时写单模型 900s 熔断，
                                // 避免下一轮再选中同一不可用账号。
                                let _ = crate::proxy::pipeline::policy::apply_model_not_found(
                                    attempt,
                                    max_attempts,
                                    || {
                                        token_manager.mark_model_unsupported(
                                            &account_id,
                                            &resolved_model,
                                            Some(900),
                                        )
                                    },
                                );
                                tracing::warn!(
                                    "[Images] Account {} returned {} for image gen, rotating to another account",
                                    email,
                                    status_code
                                );
                                force_rotate = true;
                                continue;
                            }

                            // Other errors: return
                            return Err((failure_statuses.final_status(), last_error));
                        }
                        match crate::proxy::pipeline::usage_diagnostics::read_json_response(
                            response,
                            usage_holder.as_ref().and_then(|holder| {
                                holder.begin_usage_attempt(&resolved_model, &account_id, "image_generation")
                            }),
                        ).await {
                            Ok(json) => {
                                if response_has_inline_image_data(&json) {
                                    token_manager.commit_session_lease(&mut session_lease);
                                    token_manager.mark_account_success(&account_id);
                                    token_manager
                                        .clear_persisted_live_limit(
                                            &account_id,
                                            Some(&model_to_use),
                                        );
                                    return Ok((json, email));
                                }

                                // [Fixes #3756] 上游虽返回 200，但未包含合法图片数据（如安全策略拦截、拒答或空候选集）
                                let diag = extract_image_generation_diagnostic(&json);
                                tracing::warn!(
                                    account = %email,
                                    status = %diag.status,
                                    reason = %diag.message,
                                    "[Images] Upstream returned 200 OK without inline image data"
                                );
                                failure_statuses.record(diag.status);
                                last_error = diag.message.clone();

                                // 若被安全策略拦截（提示词违规或模型拒答），换账号重试相同 Prompt 毫无意义且会白白浪费额度，直接终态返回
                                if diag.is_safety_or_policy {
                                    return Err((diag.status, diag.message));
                                }

                                // 其它情况（如格式漂移或偶发空候选集）标记换号重试
                                force_rotate = true;
                                drop(image_permit.take());
                                continue;
                            }
                            Err(e) => {
                                return Err((
                                    StatusCode::BAD_GATEWAY,
                                    format!("Parse error: {}", e),
                                ))
                            }
                        }
                    }
                    Err(e) => {
                        // 终态（首字节 / 上传 / 全局预算 / 重放上限）都不是账号故障：
                        // 不轮换账号、不重复上传同一份 body（图片同样受预算与重放护栏约束）
                        if let Some(handling) = crate::proxy::pipeline::terminal::apply_terminal(
                            &e,
                            &mut failure_statuses,
                            "openai_images",
                            attempt + 1,
                            max_attempts,
                            0.0,
                            |err| {
                                token_manager.mark_account_transient_failure(&account_id, 30, err)
                            },
                        ) {
                            last_error = format!(
                                "Upstream terminal ({}): {}",
                                handling.terminal.as_str(),
                                crate::proxy::upstream::client::strip_internal_markers(&e)
                            );
                            if handling.disposition.stop_rotation {
                                break;
                            }
                            continue;
                        }
                        // [Gateway Self-Answer Guard] 模型不存在 / 无可用账号：不罚号、不轮换
                        if let Some((status, reason)) =
                            crate::proxy::common::model_gate::self_answer_disposition(&e)
                        {
                            failure_statuses.record(status);
                            last_error = reason.to_string();
                            break;
                        }
                        last_error = format!("Network error: {}", e);
                        failure_statuses.record(StatusCode::BAD_GATEWAY);
                        drop(image_permit.take());
                        continue;
                    }
                }
            }

            // All attempts failed
            Err((
                failure_statuses.final_status(),
                format!("Max retries exhausted. Last error: {}", last_error),
            ))
        });
    }

    // 5. 收集结果
    let mut images: Vec<Value> = Vec::new();
    let mut errors: Vec<String> = Vec::new();
    let mut used_email: Option<String> = None;
    let mut failure_statuses = FailureStatusTracker::default();

    while let Some(task) = tasks.join_next().await {
        match task {
            Ok(result) => match result {
                Ok((gemini_resp, email_used)) => {
                    // Capture the email from the first successful task for logging
                    if used_email.is_none() {
                        used_email = Some(email_used);
                    }
                    let raw = gemini_resp.get("response").unwrap_or(&gemini_resp);
                    if let Some(parts) = raw
                        .get("candidates")
                        .and_then(|c| c.get(0))
                        .and_then(|cand| cand.get("content"))
                        .and_then(|content| content.get("parts"))
                        .and_then(|p| p.as_array())
                    {
                        for part in parts {
                            if let Some(img) = part.get("inlineData") {
                                let data = img.get("data").and_then(|v| v.as_str()).unwrap_or("");
                                if !data.is_empty() {
                                    if response_format == "url" {
                                        let mime_type = img
                                            .get("mimeType")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("image/png");
                                        images.push(json!({
                                            "url": format!("data:{};base64,{}", mime_type, data)
                                        }));
                                    } else {
                                        images.push(json!({
                                            "b64_json": data
                                        }));
                                    }
                                    tracing::debug!("[Images] Task succeeded");
                                }
                            }
                        }
                    }
                }
                Err((status, e)) => {
                    tracing::error!("[Images] Task failed: {}", e);
                    failure_statuses.record(status);
                    errors.push(e);
                }
            },
            Err(e) => {
                let err_msg = format!("Task join error: {}", e);
                tracing::error!("[Images] Task join error: {}", e);
                failure_statuses.record(StatusCode::BAD_GATEWAY);
                errors.push(err_msg);
            }
        }
    }

    if images.is_empty() {
        let error_msg = if !errors.is_empty() {
            errors.join("; ")
        } else {
            "No images generated".to_string()
        };
        tracing::error!("[Images] All {} requests failed. Errors: {}", n, error_msg);

        let status = failure_statuses.final_status();

        let attempted = used_email
            .clone()
            .or_else(|| attempted_account.lock().ok().and_then(|g| g.clone()));
        return Err((status, error_msg, attempted));
    }

    // 部分成功时记录警告
    if !errors.is_empty() {
        tracing::warn!(
            "[Images] Partial success: {} out of {} requests succeeded. Errors: {}",
            images.len(),
            n,
            errors.join("; ")
        );
    }

    tracing::info!(
        "[Images] Successfully generated {} out of {} requested image(s)",
        images.len(),
        n
    );

    // 6. 构建 OpenAI 格式响应
    let openai_response = json!({
        "created": chrono::Utc::now().timestamp(),
        "data": images
    });

    // [FIX] 图像生成成功后触发配额刷新 (Issue #1995)
    tokio::spawn(async move {
        let _ = account::refresh_all_quotas_logic().await;
    });

    let email_header = used_email.unwrap_or_default();
    Ok((email_header, openai_response))
}

pub async fn handle_images_edits(
    State(state): State<AppState>,
    headers: HeaderMap,
    user_identity: Option<
        axum::extract::Extension<crate::proxy::middleware::auth::UserTokenIdentity>,
    >,
    mut multipart: axum::extract::Multipart,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    tracing::info!("[Images] Received edit request");

    let mut input_images: Vec<NormalizedInputImage> = Vec::new();
    let mut mask_data: Option<NormalizedInputImage> = None;
    let mut total_input_image_bytes = 0;
    let mut prompt = String::new();
    let mut n = 1;
    let mut size: Option<String> = None;
    let mut response_format = "b64_json".to_string();
    let mut model = "gemini-3.1-flash-image".to_string();
    let mut aspect_ratio: Option<String> = None;
    let mut image_size_param: Option<String> = None;
    let mut quality: Option<String> = None;
    let mut style: Option<String> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Multipart error: {}", e)))?
    {
        let name = field.name().unwrap_or("").to_string();

        if is_edit_image_field(&name) {
            let mime_type = field
                .content_type()
                .map(|content_type| content_type.to_string())
                .unwrap_or_else(|| "image/png".to_string());
            let data = field
                .bytes()
                .await
                .map_err(|e| (StatusCode::BAD_REQUEST, format!("Image read error: {}", e)))?;
            let image = normalized_image_from_bytes(
                &data,
                &mime_type,
                input_images.len() + 1,
                total_input_image_bytes,
            )
            .map_err(|message| (StatusCode::BAD_REQUEST, message))?;
            total_input_image_bytes = total_input_image_bytes.saturating_add(image.decoded_len);
            input_images.push(image);
        } else if name == "mask" {
            let mime_type = field
                .content_type()
                .map(|content_type| content_type.to_string())
                .unwrap_or_else(|| "image/png".to_string());
            let data = field
                .bytes()
                .await
                .map_err(|e| (StatusCode::BAD_REQUEST, format!("Mask read error: {}", e)))?;
            let mask = normalized_image_from_bytes(
                &data,
                &mime_type,
                input_images.len(),
                total_input_image_bytes,
            )
            .map_err(|message| (StatusCode::BAD_REQUEST, message))?;
            total_input_image_bytes = total_input_image_bytes.saturating_add(mask.decoded_len);
            mask_data = Some(mask);
        } else if name == "prompt" {
            prompt = field
                .text()
                .await
                .map_err(|e| (StatusCode::BAD_REQUEST, format!("Prompt read error: {}", e)))?;
        } else if name == "n" {
            if let Ok(val) = field.text().await {
                n = val.parse().unwrap_or(1);
            }
        } else if name == "size" {
            let val = field
                .text()
                .await
                .map_err(|e| (StatusCode::BAD_REQUEST, format!("Size read error: {}", e)))?;
            size = Some(val);
        } else if name == "image_size" || name == "imageSize" {
            if let Ok(val) = field.text().await {
                image_size_param = Some(val);
            }
        } else if name == "quality" {
            if let Ok(val) = field.text().await {
                quality = Some(val);
            }
        } else if name == "aspect_ratio" {
            if let Ok(val) = field.text().await {
                aspect_ratio = Some(val);
            }
        } else if name == "style" {
            if let Ok(val) = field.text().await {
                style = Some(val);
            }
        } else if name == "response_format" {
            if let Ok(val) = field.text().await {
                response_format = val;
            }
        } else if name == "model" {
            if let Ok(val) = field.text().await {
                if !val.is_empty() {
                    model = val;
                }
            }
        }
    }

    // Validation: Require either 'image' (standard edit) OR 'prompt' (generation)
    // If reference images are present, we treat it as generation with image context
    if prompt.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "Missing prompt".to_string()));
    }

    // [MODEL GATE] 生图编辑同样在选号前做存在性校验（网关自答，不罚号）
    if let Some((status, reason)) = crate::proxy::common::model_gate::gate_inbound_model_exists(
        Some(model.as_str()),
        &state.custom_mapping,
        &state.token_manager,
    )
    .await
    .rejection()
    {
        return Err((status, reason.to_string()));
    }
    let affinity_key = crate::proxy::thinking_store::SessionScope::resolve(
        &headers,
        None,
        None,
        prompt.clone(),
        user_identity
            .as_ref()
            .map(|identity| identity.token_id.as_str()),
    )
    .affinity_key;

    tracing::info!(
        model = model,
        n = n,
        size = size.as_deref().unwrap_or("auto"),
        aspect_ratio = aspect_ratio.as_deref().unwrap_or("auto"),
        image_size = image_size_param.as_deref().unwrap_or("auto"),
        quality = quality.as_deref().unwrap_or("auto"),
        style = style.as_deref().unwrap_or("auto"),
        image_count = input_images.len(),
        has_mask = mask_data.is_some(),
        "[Images] Received edit request metadata"
    );

    // 2. Prepare Config (Aspect Ratio / Size)
    // Priority: aspect_ratio param > size param
    // Priority: image_size param > quality param (derived from model suffix or default)

    // We reuse parse_image_config_with_params but need to adapt the inputs
    let size_input = edit_size_input(aspect_ratio.as_deref(), size.as_deref());

    let (image_config, clean_model_name) =
        crate::proxy::mappers::common_utils::try_parse_image_config_with_params(
            &model,
            size_input,
            quality.as_deref(),
            image_size_param.as_deref(),
        )
        .map_err(|message| (StatusCode::BAD_REQUEST, message))?;

    // 3. Construct Contents
    let mut final_prompt = prompt.clone();
    if let Some(s) = style {
        final_prompt.push_str(&format!(", style: {}", s));
    }
    let contents_parts = build_image_contents(final_prompt, &input_images, mask_data.as_ref());

    // 4. 并发发送请求
    // 注意：不再在外部获取 Token，而是移入 Task 内部
    let upstream = state.upstream.clone();
    let token_manager = state.token_manager.clone();
    let image_scheduler = state.image_scheduler.clone();
    let request_timeout = state.request_timeout;
    let max_pool_size = token_manager.len();
    // [FIX #3485] 自适应多账号池与单账号退避最大重试次数 (单账号3次，多账号整池两轮)
    let max_attempts = crate::proxy::handlers::common::calculate_max_retry_attempts(max_pool_size);

    let request_budget = crate::proxy::pipeline::budget::current();
    let usage_holder = crate::proxy::monitor::UpstreamRequestBodyHolder::current();

    let mut tasks = JoinSet::new();
    for _ in 0..n {
        let upstream = upstream.clone();
        let token_manager = token_manager.clone();
        let request_budget = request_budget.as_ref().map(|ctx| ctx.fork());
        let usage_holder = usage_holder.clone();
        let contents_parts = contents_parts.clone();
        let image_config = image_config.clone();
        let response_format = response_format.clone();
        let model_to_use = clean_model_name.clone();
        let image_scheduler = image_scheduler.clone();
        let affinity_key = affinity_key.clone();

        tasks.spawn(async move {
            let mut image_permit = None;
            let mut last_error = String::new();
            let mut force_rotate = false;
            let mut retry_state = RequestRetryState::default();
            let mut retry_credentials: Option<crate::proxy::token_manager::LeasedCredentials> = None;
            let mut failure_statuses = FailureStatusTracker::default();
            let mut used_attempts = 0;

            while let Some(attempt) = next_rotation_attempt(
                &mut used_attempts,
                max_attempts,
                retry_credentials.is_some(),
            ) {
                if crate::proxy::pipeline::terminal::gate_exhausted(
                    "openai_images",
                    attempt + 1,
                    &mut failure_statuses,
                    &mut last_error,
                ) {
                    break;
                }
                // 4.1 获取 Token
                let (access_token, project_id, email, account_id, _wait_ms, mut session_lease) =
                    if let Some(credentials) = retry_credentials.take() {
                        credentials
                    } else {
                        drop(image_permit.take());
                        match token_manager
                            .get_image_token_with_lease(
                                force_rotate,
                                Some(&affinity_key),
                                image_account_selection_target(&model_to_use),
                                &image_scheduler,
                                request_timeout,
                            )
                            .await
                        {
                            Ok((access_token, project_id, email, account_id, wait_ms, permit, lease)) => {
                                image_permit = Some(permit);
                                (access_token, project_id, email, account_id, wait_ms, lease)
                            }
                            Err((status, e)) => {
                                last_error = format!("Token error: {}", e);
                                failure_statuses.record(status);
                                if status == StatusCode::TOO_MANY_REQUESTS {
                                    return Err((status, e));
                                }
                                if attempt < max_attempts - 1 {
                                    tokio::time::sleep(Duration::from_millis(500)).await;
                                    continue;
                                }
                                break;
                            }
                        }
                    };

                let resolved_model = token_manager
                    .resolve_dynamic_model_for_account(&account_id, &model_to_use)
                    .await;

                // 4.2 Construct Request Body (Need project_id and account-resolved model)
                let gemini_body = build_image_edit_body(
                    project_id.clone(),
                    &resolved_model,
                    contents_parts.clone(),
                    image_config.clone(),
                );

                let upstream_call = upstream.call_v1_internal(
                    "generateContent",
                    &access_token,
                    gemini_body,
                    None,
                    Some(account_id.as_str()),
                );
                match crate::proxy::pipeline::budget::scope_optional(
                    request_budget.clone(),
                    upstream_call,
                )
                .await
                {
                    Ok(call_result) => {
                        let response = call_result.response;
                        let status = response.status();
                        if !status.is_success() {
                            failure_statuses.record(status);
                            let retry_after = response
                                .headers()
                                .get("Retry-After")
                                .and_then(|header| header.to_str().ok())
                                .map(str::to_string);
                            let err_text = response.text().await.unwrap_or_default();
                            let status_code = status.as_u16();
                            last_error = format!("Upstream error {}: {}", status, err_text);
                            let strategy = (status_code == 429).then(|| {
                                retry_state.determine_strategy(
                                    &account_id,
                                    status_code,
                                    &err_text,
                                    retry_after.as_deref(),
                                    false,
                                )
                            });
                            // 统一流水线限流裁决：500/503等服务异常绝不打入限流
                            let classification =
                                crate::proxy::pipeline::UpstreamClassification::classify(
                                    status_code,
                                    &err_text,
                                    retry_after.as_deref(),
                                );
                            if classification.abandons_sticky_account() {
                                token_manager.abandon_session_lease(&mut session_lease);
                            }
                            let should_mark_limited = classification.should_lock_account();
                            let needs_quota_refresh = if should_mark_limited {
                                tracing::warn!(
                                    "[Images] Account {} rate limited/error ({}), rotating...",
                                    email,
                                    status_code
                                );
                                token_manager
                                    .mark_rate_limited_fast(
                                        &email,
                                        status_code,
                                        retry_after.as_deref(),
                                        &err_text,
                                        Some(&resolved_model),
                                    )
                                    .await
                            } else {
                                false
                            };
                            if !matches!(strategy.as_ref(), Some(RetryStrategy::GraceRetry(_))) {
                                drop(image_permit.take());
                            }
                            if needs_quota_refresh {
                                token_manager
                                    .refresh_quota_lock_after_fast_mark(
                                        &email,
                                        Some(&resolved_model),
                                    )
                                    .await;
                            }

                            if let Some(strategy) = strategy {
                                if apply_retry_strategy(
                                    strategy.clone(),
                                    attempt,
                                    max_attempts,
                                    status_code,
                                    "image_edit",
                                )
                                .await
                                {
                                    if matches!(strategy, RetryStrategy::GraceRetry(_)) {
                                        retry_credentials = Some((
                                            access_token.clone(),
                                            project_id.clone(),
                                            email.clone(),
                                            account_id.clone(),
                                            0,
                                            session_lease.clone(),
                                        ));
                                    }
                                    force_rotate =
                                        should_rotate_account(status_code, Some(&strategy));
                                    continue;
                                }
                            }

                            if status_code == 503 || status_code == 500 {
                                continue; // Retry loop
                            }

                            // [FIX] 403/404 usually mean THIS account lacks the image model or
                            // project access. Rotate to another account instead of failing the
                            // whole request (same contract as image generations).
                            if (status_code == 403 || status_code == 404)
                                && attempt < max_attempts - 1
                            {
                                // 与文本链路对齐：该账号缺模型/权限时写单模型 900s 熔断。
                                let _ = crate::proxy::pipeline::policy::apply_model_not_found(
                                    attempt,
                                    max_attempts,
                                    || {
                                        token_manager.mark_model_unsupported(
                                            &account_id,
                                            &resolved_model,
                                            Some(900),
                                        )
                                    },
                                );
                                tracing::warn!(
                                    "[Images] Account {} returned {} for image edit, rotating to another account",
                                    email,
                                    status_code
                                );
                                force_rotate = true;
                                continue;
                            }
                            return Err((failure_statuses.final_status(), last_error));
                        }
                        match crate::proxy::pipeline::usage_diagnostics::read_json_response(
                            response,
                            usage_holder.as_ref().and_then(|holder| {
                                holder.begin_usage_attempt(&resolved_model, &account_id, "image_edit")
                            }),
                        ).await {
                            Ok(json) => {
                                if response_has_inline_image_data(&json) {
                                    token_manager.commit_session_lease(&mut session_lease);
                                    token_manager.mark_account_success(&account_id);
                                    token_manager.clear_persisted_live_limit(
                                        &account_id,
                                        Some(&model_to_use),
                                    );
                                    return Ok((json, response_format.clone(), email));
                                }

                                // [Fixes #3756] 上游编辑接口虽返回 200，但未包含合法图片数据
                                let diag = extract_image_generation_diagnostic(&json);
                                tracing::warn!(
                                    account = %email,
                                    status = %diag.status,
                                    reason = %diag.message,
                                    "[Images] Upstream edit returned 200 OK without inline image data"
                                );
                                failure_statuses.record(diag.status);
                                last_error = diag.message.clone();

                                if diag.is_safety_or_policy {
                                    return Err((diag.status, diag.message));
                                }

                                force_rotate = true;
                                drop(image_permit.take());
                                continue;
                            }
                            Err(e) => {
                                return Err((
                                    StatusCode::BAD_GATEWAY,
                                    format!("Parse error: {}", e),
                                ))
                            }
                        }
                    }
                    Err(e) => {
                        // 终态（首字节 / 上传 / 全局预算 / 重放上限）都不是账号故障：
                        // 不轮换账号、不重复上传同一份 body（图片同样受预算与重放护栏约束）
                        if let Some(handling) = crate::proxy::pipeline::terminal::apply_terminal(
                            &e,
                            &mut failure_statuses,
                            "openai_images",
                            attempt + 1,
                            max_attempts,
                            0.0,
                            |err| {
                                token_manager.mark_account_transient_failure(&account_id, 30, err)
                            },
                        ) {
                            last_error = format!(
                                "Upstream terminal ({}): {}",
                                handling.terminal.as_str(),
                                crate::proxy::upstream::client::strip_internal_markers(&e)
                            );
                            if handling.disposition.stop_rotation {
                                break;
                            }
                            continue;
                        }
                        // [Gateway Self-Answer Guard] 模型不存在 / 无可用账号：不罚号、不轮换
                        if let Some((status, reason)) =
                            crate::proxy::common::model_gate::self_answer_disposition(&e)
                        {
                            failure_statuses.record(status);
                            last_error = reason.to_string();
                            break;
                        }
                        last_error = format!("Network error: {}", e);
                        failure_statuses.record(StatusCode::BAD_GATEWAY);
                        drop(image_permit.take());
                        continue;
                    }
                }
            }
            Err((
                failure_statuses.final_status(),
                format!("Max retries exhausted. Last error: {}", last_error),
            ))
        });
    }

    // 5. Collect Results
    let mut images: Vec<Value> = Vec::new();
    let mut errors: Vec<String> = Vec::new();
    let mut used_email: Option<String> = None;
    let mut failure_statuses = FailureStatusTracker::default();

    while let Some(task) = tasks.join_next().await {
        match task {
            Ok(result) => match result {
                Ok((gemini_resp, response_format, email_used)) => {
                    if used_email.is_none() {
                        used_email = Some(email_used);
                    }
                    let raw = gemini_resp.get("response").unwrap_or(&gemini_resp);
                    if let Some(parts) = raw
                        .get("candidates")
                        .and_then(|c| c.get(0))
                        .and_then(|cand| cand.get("content"))
                        .and_then(|content| content.get("parts"))
                        .and_then(|p| p.as_array())
                    {
                        for part in parts {
                            if let Some(img) = part.get("inlineData") {
                                let data = img.get("data").and_then(|v| v.as_str()).unwrap_or("");
                                if !data.is_empty() {
                                    if response_format == "url" {
                                        let mime_type = img
                                            .get("mimeType")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("image/png");
                                        images.push(json!({
                                            "url": format!("data:{};base64,{}", mime_type, data)
                                        }));
                                    } else {
                                        images.push(json!({
                                            "b64_json": data
                                        }));
                                    }
                                    tracing::debug!("[Images] Task succeeded");
                                }
                            }
                        }
                    }
                }
                Err((status, e)) => {
                    tracing::error!("[Images] Task failed: {}", e);
                    failure_statuses.record(status);
                    errors.push(e);
                }
            },
            Err(e) => {
                let err_msg = format!("Task join error: {}", e);
                tracing::error!("[Images] Task join error: {}", e);
                failure_statuses.record(StatusCode::BAD_GATEWAY);
                errors.push(err_msg);
            }
        }
    }

    if images.is_empty() {
        let error_msg = if !errors.is_empty() {
            errors.join("; ")
        } else {
            "No images generated".to_string()
        };
        tracing::error!(
            "[Images] All {} edit requests failed. Errors: {}",
            n,
            error_msg
        );
        let status = failure_statuses.final_status();

        return Err((status, error_msg));
    }

    if !errors.is_empty() {
        tracing::warn!(
            "[Images] Partial success: {} out of {} requests succeeded. Errors: {}",
            images.len(),
            n,
            errors.join("; ")
        );
    }

    tracing::info!(
        "[Images] Successfully generated {} out of {} requested edited image(s)",
        images.len(),
        n
    );

    let openai_response = json!({
        "created": chrono::Utc::now().timestamp(),
        "data": images
    });

    tokio::spawn(async move {
        let _ = account::refresh_all_quotas_logic().await;
    });

    let email_header = used_email.unwrap_or_default();
    Ok((
        StatusCode::OK,
        [
            ("X-Mapped-Model", clean_model_name.as_str()),
            ("X-Account-Email", email_header.as_str()),
        ],
        Json(openai_response),
    )
        .into_response())
}

// ==========================================
// CODE INTEGRATION: Codex WebSocket Handler
// ==========================================

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use futures::StreamExt;
use uuid::Uuid;

// ==========================================

// CODE INTEGRATION: Global Tool Call Cache

// ==========================================

use std::sync::OnceLock;

use tokio::sync::RwLock as TokioRwLock;

use std::collections::HashMap;

static WEBSOCKET_TOOL_CALL_CACHE: OnceLock<TokioRwLock<HashMap<String, Value>>> = OnceLock::new();

pub fn get_cached_tool_call(call_id: &str) -> Option<Value> {
    if let Some(cache) = WEBSOCKET_TOOL_CALL_CACHE.get() {
        if let Ok(guard) = cache.try_read() {
            return guard.get(call_id).cloned();
        }
    }

    None
}

pub fn insert_cached_tool_call(call_id: String, item: Value) {
    if call_id.is_empty() {
        return;
    }

    let cache = WEBSOCKET_TOOL_CALL_CACHE.get_or_init(|| TokioRwLock::new(HashMap::new()));

    if let Ok(mut guard) = cache.try_write() {
        guard.insert(call_id, item);
    }
}

#[derive(Debug, Clone)]

struct WebsocketSessionState {
    last_request: Option<Value>,
    last_response_output: Value,
    last_response_id: String,
    tool_call_cache: std::collections::HashMap<String, Value>,
}

pub async fn handle_responses_websocket(
    ws: WebSocketUpgrade,
    headers: HeaderMap,
    State(state): State<AppState>,
    user_identity: Option<
        axum::extract::Extension<crate::proxy::middleware::auth::UserTokenIdentity>,
    >,
) -> Response {
    ws.on_upgrade(move |socket| handle_websocket_session(socket, headers, state, user_identity))
}

async fn handle_websocket_session(
    mut socket: WebSocket,
    headers: HeaderMap,
    state: AppState,
    user_identity: Option<
        axum::extract::Extension<crate::proxy::middleware::auth::UserTokenIdentity>,
    >,
) {
    tracing::info!("Codex responses websocket: client connected");
    let mut session_state = WebsocketSessionState {
        last_request: None,
        last_response_output: json!([]),
        last_response_id: String::new(),
        tool_call_cache: std::collections::HashMap::new(),
    };

    let connection_id = format!("ws-{}", Uuid::new_v4());
    while let Some(msg_result) = socket.recv().await {
        let msg = match msg_result {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!("responses websocket: read message failed: {:?}", e);
                break;
            }
        };

        let text = match msg {
            Message::Text(t) => t,
            Message::Binary(b) => match String::from_utf8(b) {
                Ok(s) => s,
                Err(_) => continue,
            },
            Message::Close(_) => {
                tracing::info!("responses websocket: client disconnected");
                break;
            }
            _ => continue,
        };

        let payload: Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(e) => {
                let error_ev = json!({
                    "type": "error",
                    "error": {
                        "message": format!("Invalid JSON: {}", e),
                        "type": "invalid_request_error"
                    }
                });
                let _ = socket.send(Message::Text(error_ev.to_string())).await;
                continue;
            }
        };
        drop(text);
        let ws_trace_id = format!("ws_{}", chrono::Utc::now().timestamp_subsec_millis());
        let debug_cfg = state.debug_logging.read().await.clone();
        if debug_logger::is_enabled(&debug_cfg) {
            let payload_log = json!({
                "kind": "codex_websocket_raw_request",
                "protocol": "codex_websocket",
                "trace_id": ws_trace_id,
                "payload": debug_value_without_inline_data(&payload),
            });
            debug_logger::write_exchange_payload(
                &debug_cfg,
                Some(&ws_trace_id),
                "codex_websocket_raw_request",
                &payload_log,
            )
            .await;
        }

        if should_handle_prewarm_locally(&payload, &session_state) {
            let mut pending = session_state.clone();
            let (created, completed) = match prepare_websocket_prewarm(
                &payload,
                &mut pending,
                &headers,
                &connection_id,
            )
            .await
            {
                Ok(events) => events,
                Err(message) => {
                    let event = json!({"type":"error","error":{"type":"invalid_request_error","message":message}});
                    if socket.send(Message::Text(event.to_string())).await.is_err() {
                        return;
                    }
                    continue;
                }
            };
            if socket
                .send(Message::Text(created.to_string()))
                .await
                .is_err()
                || socket
                    .send(Message::Text(completed.to_string()))
                    .await
                    .is_err()
            {
                return;
            }
            session_state = pending;
            if debug_logger::is_enabled(&debug_cfg) {
                let payload_log = json!({
                    "kind": "codex_websocket_local_response",
                    "protocol": "codex_websocket",
                    "trace_id": ws_trace_id,
                    "events": [created, completed],
                });
                debug_logger::write_exchange_payload(
                    &debug_cfg,
                    Some(&ws_trace_id),
                    "codex_websocket_local_response",
                    &payload_log,
                )
                .await;
            }
            continue;
        }

        // Stage the merge: an invalid request, failed generation, or disconnected
        // writer must not consume the last successfully delivered turn.
        let mut pending = session_state.clone();
        let mut normalized = match normalize_responses_websocket_request(payload, &mut pending) {
            Ok(body) => body,
            Err(message) => {
                let event = json!({"type":"error","error":{"type":"invalid_request_error","message":message}});
                if socket.send(Message::Text(event.to_string())).await.is_err() {
                    return;
                }
                continue;
            }
        };
        if let Err(message) =
            prepare_websocket_generation(&mut normalized, &mut pending, &headers, &connection_id)
                .await
        {
            let event = json!({"type":"error","error":{"type":"invalid_request_error","code":"previous_response_not_found","message":message}});
            if socket.send(Message::Text(event.to_string())).await.is_err() {
                return;
            }
            continue;
        }
        let mut usage = crate::proxy::middleware::monitor::WebsocketTurnUsage::begin(
            &headers,
            &normalized,
            user_identity.as_ref().map(|i| &i.0),
        );
        let response = handle_completions(
            axum::extract::OriginalUri(axum::http::Uri::from_static("/v1/responses")),
            State(state.clone()),
            headers.clone(),
            None,
            user_identity.clone(),
            Json(normalized),
        )
        .await;
        usage.response(response.status().as_u16(), response.headers());
        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
                .await
                .unwrap_or_default();
            let event = websocket_http_error_event(status, &body);
            usage.observe(&event);
            usage.record(
                state.monitor.clone(),
                user_identity.as_ref().map(|i| i.0.clone()),
            );
            if socket.send(Message::Text(event.to_string())).await.is_err() {
                return;
            }
            continue;
        }
        let events = websocket_response_events(response.into_body().into_data_stream());
        futures::pin_mut!(events);
        let mut logged_events = debug_logger::is_enabled(&debug_cfg).then(Vec::new);
        let mut disconnected = false;
        while let Some(event) = events.next().await {
            // Account for actual upstream observations even if the client leaves
            // while sending the terminal. Only delivered output is resumable.
            usage.observe(&event);
            if socket.send(Message::Text(event.to_string())).await.is_err() {
                disconnected = true;
                break;
            }
            if let Some(log) = logged_events.as_mut() {
                log.push(debug_value_without_inline_data(&event));
            }
            if commit_websocket_response(&mut pending, &event) {
                session_state = pending;
                break;
            }
        }
        usage.record(
            state.monitor.clone(),
            user_identity.as_ref().map(|i| i.0.clone()),
        );
        if disconnected {
            return;
        }
        if let Some(events) = logged_events {
            debug_logger::write_exchange_payload(
                &debug_cfg,
                Some(&ws_trace_id),
                "codex_websocket_converted_response",
                &json!({"kind":"codex_websocket_converted_response","events":events}),
            )
            .await;
        }
    }
}

// Resolve a first or historical branch cursor before dispatch. Current socket
// turns have already merged their own local snapshot, including store=false.
async fn prepare_websocket_generation(
    body: &mut Value,
    pending: &mut WebsocketSessionState,
    headers: &HeaderMap,
    connection_id: &str,
) -> Result<(), String> {
    let previous = body["previous_response_id"]
        .as_str()
        .filter(|id| !id.is_empty());
    let new_root = previous.is_none() && pending.last_response_id.is_empty();
    let mut explicit =
        crate::proxy::thinking_store::stable_session_winner(headers, Some(body), None);
    if let Some(id) = previous {
        // An explicit cursor not represented by the current socket snapshot must
        // resolve independently. Do not splice a sibling, failed or expired turn
        // onto the last successful turn. Keep the cursor for native HTTP storage.
        let (parent, stored) = crate::proxy::http_session_store::get_session_with_parent(id)
            .await
            .ok_or_else(|| format!("Previous response '{}' was not found", id))?;
        if explicit.is_none() {
            explicit = stored.scope_identity().map(str::to_owned);
        }
        let history = pending.last_request.as_mut().expect("normalized request");
        let delta = take_responses_input_items(history);
        let delta =
            crate::proxy::http_session_store::resolve_input_references(&parent.input_items, delta)?;
        let prepared = crate::proxy::http_session_store::prepare_session_input_with_storage(
            parent.input_items,
            delta,
            &pending.tool_call_cache,
            false,
        );
        history["input"] = if body.get("generate").and_then(Value::as_bool) == Some(false) {
            // The saved parent was already bounded; newly supplied input hasn't
            // been dispatched yet and must reach the model before abbreviation.
            Value::Array(prepared.merged)
        } else {
            history_without_inline_media(&Value::Array(prepared.merged))
        };
        if history.get("instructions").is_none() && !parent.instructions.is_empty() {
            history["instructions"] = json!(parent.instructions);
        }
        if history.get("model").is_none() && !parent.model.is_empty() {
            history["model"] = json!(parent.model);
            body["model"] = history["model"].clone();
        }
        // An imported anonymous parent uses its restored content anchor, as it
        // does over HTTP; never invent a new socket identity for it.
    }
    let new_identity = if new_root {
        explicit.or_else(|| Some(connection_id.to_owned()))
    } else {
        explicit
    };
    if let Some(id) = new_identity {
        body["session_id"] = json!(id);
        if let Some(history) = pending.last_request.as_mut() {
            history["session_id"] = json!(id);
        }
    }
    Ok(())
}

async fn prepare_websocket_prewarm(
    payload: &Value,
    pending: &mut WebsocketSessionState,
    headers: &HeaderMap,
    connection_id: &str,
) -> Result<(Value, Value), String> {
    validate_responses_continuation(payload).map_err(|(_, _, msg)| msg.to_owned())?;
    // The pending request lives on an idle socket until its first generation.
    // Bound the entire retained JSON (including audio and tools), allowing for
    // base64 expansion. A sizing writer does not allocate a second huge buffer.
    let max_bytes = crate::proxy::config::get_multimodal_config()
        .max_total_image_mb
        .max(1)
        .saturating_mul(1024 * 1024);
    if serialized_json_len(payload)
        > max_bytes
            .saturating_mul(4)
            .div_ceil(3)
            .saturating_add(64 * 1024)
    {
        return Err(
            "WebSocket prewarm payload exceeds the configured media retention limit".to_owned(),
        );
    }
    validate_responses_input_image_limits_with_mode(payload.get("input"), false)?;
    let mut validated = payload.clone();
    normalize_responses_plaintext(&mut validated).map_err(|(_, _, msg)| msg.to_owned())?;
    if payload["previous_response_id"]
        .as_str()
        .is_none_or(str::is_empty)
    {
        let input = take_responses_input_items(&mut validated);
        validated["input"] = json!(crate::proxy::http_session_store::resolve_input_references(
            &[],
            input
        )?);
    }
    let events = handle_prewarm_locally(&validated, pending);
    // Only metadata is needed for parent lookup/session resolution here;
    // the original, validated input is already owned by pending.last_request.
    let mut body = json!({"generate":false});
    if let Some(id) = payload.get("previous_response_id") {
        body["previous_response_id"] = id.clone();
    }
    if let Some(id) =
        crate::proxy::thinking_store::stable_session_winner(headers, Some(payload), None)
    {
        body["session_id"] = json!(id);
    }
    if body
        .get("previous_response_id")
        .and_then(Value::as_str)
        .is_none_or(str::is_empty)
        && crate::proxy::thinking_store::stable_session_winner(headers, Some(&body), None).is_none()
    {
        body["session_id"] = json!(connection_id);
    }
    prepare_websocket_generation(&mut body, pending, headers, connection_id).await?;
    Ok(events)
}

fn should_handle_prewarm_locally(payload: &Value, state: &WebsocketSessionState) -> bool {
    if state.last_request.is_some() {
        return false;
    }
    let event_type = payload.get("type").and_then(|v| v.as_str()).unwrap_or("");
    if event_type != "response.create" {
        return false;
    }
    if let Some(generate) = payload.get("generate").and_then(|v| v.as_bool()) {
        if !generate {
            return true;
        }
    }
    false
}

fn handle_prewarm_locally(payload: &Value, state: &mut WebsocketSessionState) -> (Value, Value) {
    let response_id = format!("resp_prewarm_{}", Uuid::new_v4());
    let created_at = chrono::Utc::now().timestamp();
    let model = payload
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    let created_ev = json!({
        "type": "response.created",
        "sequence_number": 0,
        "response": {
            "id": &response_id,
            "object": "response",
            "created_at": created_at,
            "status": "in_progress",
            "background": false,
            "error": null,
            "output": [],
            "model": model,
        }
    });

    let completed_ev = json!({
        "type": "response.completed",
        "sequence_number": 1,
        "response": {
            "id": &response_id,
            "object": "response",
            "created_at": created_at,
            "status": "completed",
            "background": false,
            "error": null,
            "output": [],
            "usage": {
                "input_tokens": 0,
                "input_tokens_details": {
                    "cached_tokens": 0
                },
                "output_tokens": 0,
                "output_tokens_details": {
                    "reasoning_tokens": 0
                },
                "total_tokens": 0
            },
            "model": model,
        }
    });

    // There is no model call on prewarm. Keep newly supplied media until the
    // first real generation; normal commit retains only bounded history.
    let mut normalized = payload.clone();
    if let Some(obj) = normalized.as_object_mut() {
        obj.remove("type");
        obj.remove("generate");
    }
    state.last_request = Some(normalized);
    state.last_response_output = json!([]);
    state.last_response_id = response_id;

    (created_ev, completed_ev)
}

fn normalize_responses_websocket_request(
    mut payload: Value,
    state: &mut WebsocketSessionState,
) -> Result<Value, String> {
    normalize_responses_plaintext(&mut payload).map_err(|(_, _, msg)| msg.to_owned())?;
    let event_type = payload
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    match event_type.as_str() {
        "response.create" => {
            if state.last_request.is_none() {
                if let Some(obj) = payload.as_object_mut() {
                    obj.remove("type");
                    obj.insert("stream".to_string(), Value::Bool(true));
                    if !obj.contains_key("input") {
                        obj.insert("input".to_string(), json!([]));
                    }
                }
                let model_name = payload.get("model").and_then(|v| v.as_str()).unwrap_or("");
                if model_name.is_empty()
                    && payload["previous_response_id"]
                        .as_str()
                        .is_none_or(str::is_empty)
                {
                    return Err("missing model in response.create request".to_string());
                }
                validate_responses_input_image_limits(payload.get("input"))?;
                if payload["previous_response_id"]
                    .as_str()
                    .is_none_or(str::is_empty)
                {
                    let input = take_responses_input_items(&mut payload);
                    payload["input"] = json!(
                        crate::proxy::http_session_store::resolve_input_references(&[], input)?
                    );
                }
                state.last_request = Some(history_without_inline_media(&payload));
                Ok(payload)
            } else {
                normalize_response_subsequent_request(payload, state)
            }
        }
        "response.append" => normalize_response_subsequent_request(payload, state),
        _ => Err(format!(
            "unsupported websocket request type: {}",
            event_type
        )),
    }
}

fn normalize_response_subsequent_request(
    mut payload: Value,
    state: &mut WebsocketSessionState,
) -> Result<Value, String> {
    if state.last_request.is_none() {
        return Err("websocket request received before response.create".to_string());
    }
    validate_responses_input_image_limits(payload.get("input"))?;

    let previous = payload["previous_response_id"]
        .as_str()
        .filter(|id| !id.is_empty());
    let external_parent = previous.is_some_and(|id| id != state.last_response_id);
    let has_identity =
        crate::proxy::thinking_store::extract_body_session_id(Some(&payload)).is_some();
    if !external_parent {
        let resets_parent = should_replace_websocket_transcript(&payload)
            || payload["input"].as_array().is_some_and(|items| {
                items.iter().any(|item| {
                    matches!(
                        item["type"].as_str(),
                        Some("compaction" | "compaction_summary")
                    )
                })
            });
        let mut history = if resets_parent {
            Vec::new()
        } else {
            state
                .last_request
                .as_ref()
                .and_then(|v| v["input"].as_array())
                .cloned()
                .unwrap_or_default()
        };
        if !resets_parent {
            if let Some(output) = state.last_response_output.as_array() {
                history.extend(output.iter().cloned());
            }
        }
        let incoming = take_responses_input_items(&mut payload);
        payload["input"] = json!(crate::proxy::http_session_store::resolve_input_references(
            &history, incoming
        )?);
    }
    let last = state.last_request.as_mut().expect("checked above");
    if let Some(obj) = payload.as_object_mut() {
        // The initial generate:false event prepared these options for this
        // specific generation. After it completes, ordinary turns inherit only
        // the existing connection defaults, never a sibling's settings.
        let from_prewarm = state.last_response_id.starts_with("resp_prewarm_");
        let token_fields = [
            "max_tokens",
            "max_completion_tokens",
            "maxCompletionTokens",
            "maxTokens",
            "max_output_tokens",
            "maxOutputTokens",
        ];
        let overrides_tokens = token_fields.iter().any(|field| obj.contains_key(*field));
        let prewarm_options = [
            "max_tokens",
            "max_completion_tokens",
            "maxCompletionTokens",
            "maxTokens",
            "maxOutputTokens",
            "store",
            "reasoning",
            "text",
            "max_output_tokens",
            "max_tool_calls",
            "parallel_tool_calls",
            "temperature",
            "top_p",
            "top_logprobs",
            "truncation",
            "include",
            "stream_options",
            "service_tier",
            "background",
            "metadata",
            "safety_identifier",
            "user",
            "prompt",
            "prompt_cache_key",
            "prompt_cache_retention",
        ];
        for field in [
            "model",
            "instructions",
            "tools",
            "tool_choice",
            "session_id",
        ]
        .into_iter()
        .chain(prewarm_options.into_iter().filter(|_| from_prewarm))
        {
            // A historical branch gets configuration from its own parent,
            // not from the unrelated latest turn on this connection.
            if !external_parent
                && !obj.contains_key(field)
                && (field != "session_id" || !has_identity)
                && (!overrides_tokens || !token_fields.contains(&field))
            {
                if let Some(value) = last.get_mut(field).map(Value::take) {
                    obj.insert(field.to_owned(), value);
                }
            }
        }
    }
    if external_parent {
        state.tool_call_cache.clear();
        state.last_response_output = json!([]);
        if let Some(obj) = payload.as_object_mut() {
            obj.remove("type");
            obj.insert("stream".into(), Value::Bool(true));
        }
        state.last_request = Some(history_without_inline_media(&payload));
        return Ok(payload);
    }
    let replace = should_replace_websocket_transcript(&payload);
    let incoming = take_responses_input_items(&mut payload);
    let compacted = incoming.iter().any(|item| {
        matches!(
            item["type"].as_str(),
            Some("compaction" | "compaction_summary")
        )
    });
    if replace || compacted {
        state.tool_call_cache.clear();
    }
    let parent = if replace {
        Vec::new()
    } else {
        let mut last_request = state.last_request.take().expect("checked above");
        let mut history = take_responses_input_items(&mut last_request);
        if let Value::Array(output) = std::mem::take(&mut state.last_response_output) {
            history.extend(output);
        }
        history
    };
    let prepared = crate::proxy::http_session_store::prepare_session_input_with_storage(
        parent,
        incoming,
        &state.tool_call_cache,
        false,
    );
    let deduped = dedupe_function_calls_by_call_id(prepared.merged);
    if let Some(obj) = payload.as_object_mut() {
        obj.remove("type");
        obj.remove("previous_response_id");
        obj.insert("input".to_string(), Value::Array(deduped));
        obj.insert("stream".to_string(), Value::Bool(true));
    }
    state.last_request = Some(history_without_inline_media(&payload));
    Ok(payload)
}

fn should_replace_websocket_transcript(payload: &Value) -> bool {
    // An explicit append can legally contain new assistant/tool history.
    // Shared replay detection handles a true full prefix and compaction.
    if payload["type"] == "response.append" {
        return false;
    }

    let previous_response_id = payload
        .get("previous_response_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if !previous_response_id.is_empty() {
        return false;
    }
    if let Some(input_array) = payload.get("input").and_then(|v| v.as_array()) {
        for item in input_array {
            let item_type = responses_input_item_type(&item).to_string();
            if item_type == "function_call" || item_type == "custom_tool_call" {
                return true;
            }
            if item_type == "message" {
                let role = item.get("role").and_then(|v| v.as_str()).unwrap_or("");
                if role == "assistant" {
                    return true;
                }
            }
        }
    }
    false
}

fn dedupe_function_calls_by_call_id(items: Vec<Value>) -> Vec<Value> {
    use std::collections::HashSet;
    let mut seen_call_ids = HashSet::new();
    let mut filtered = Vec::new();
    for item in items {
        let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if item_type == "function_call" || item_type == "custom_tool_call" {
            if let Some(call_id) = item.get("call_id").and_then(|v| v.as_str()) {
                if !call_id.is_empty() {
                    if seen_call_ids.contains(call_id) {
                        continue;
                    }
                    seen_call_ids.insert(call_id.to_string());
                }
            }
        }
        filtered.push(item);
    }
    filtered
}

fn convert_codex_to_openai_request(mut body: Value) -> Value {
    let instructions = body
        .get("instructions")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    let (interaction_ledger, mut step_markers) = codex_ledger_from_body(&body);
    let input_items = body
        .as_object_mut()
        .and_then(|obj| obj.remove("input"))
        .map(|value| match value {
            Value::Array(items) => items,
            Value::String(text) => vec![json!({
                "type": "message",
                "role": "user",
                "content": text
            })],
            other => vec![json!({
                "type": "message",
                "role": "user",
                "content": other
            })],
        })
        .unwrap_or_default();

    let mut messages = Vec::new();
    if !instructions.is_empty() {
        messages.push(json!({ "role": "system", "content": instructions }));
    }

    let mut call_id_to_name = std::collections::HashMap::new();
    let mut skipped_incomplete_custom_call_ids = std::collections::HashSet::new();

    {
        for item in &input_items {
            let item_type = responses_input_item_type(item);
            if item_type == "custom_tool_call"
                && item.get("status").and_then(|v| v.as_str()) == Some("incomplete")
            {
                if let Some(call_id) = item
                    .get("call_id")
                    .and_then(|v| v.as_str())
                    .or_else(|| item.get("id").and_then(|v| v.as_str()))
                {
                    skipped_incomplete_custom_call_ids.insert(call_id.to_string());
                }
                continue;
            }
            match item_type {
                "function_call" | "custom_tool_call" | "local_shell_call" | "web_search_call" => {
                    let call_id = item
                        .get("call_id")
                        .and_then(|v| v.as_str())
                        .or_else(|| item.get("id").and_then(|v| v.as_str()))
                        .unwrap_or("unknown");
                    let name = if item_type == "local_shell_call" {
                        "local_shell_call".to_string()
                    } else if item_type == "web_search_call" {
                        "web_search_call".to_string()
                    } else {
                        responses_item_tool_name(&item).unwrap_or_else(|| "unknown".to_string())
                    };
                    call_id_to_name.insert(call_id.to_string(), name);
                }
                _ => {}
            }
        }
    }

    {
        for mut item in input_items {
            let item_type = responses_input_item_type(&item).to_string();
            let step_marker = step_markers.pop_front();
            if item_type == "custom_tool_call"
                && item.get("status").and_then(|v| v.as_str()) == Some("incomplete")
            {
                continue;
            }
            match item_type.as_str() {
                "message" => {
                    let role = item
                        .get("role")
                        .and_then(Value::as_str)
                        .unwrap_or("user")
                        .to_string();
                    let (content, joined_text, has_media) =
                        responses_message_content(&mut item, step_marker);
                    if is_codex_transcript_only_assistant_message(&item, &joined_text) {
                        continue;
                    }

                    if role == "assistant" && joined_text.trim().is_empty() && !has_media {
                        continue;
                    }

                    messages.push(json!({"role":role,"content":content}));
                }
                "reasoning" => {
                    messages.push(responses_reasoning_message(&item));
                }
                "function_call" | "custom_tool_call" | "local_shell_call" | "web_search_call" => {
                    let mut name =
                        responses_item_tool_name(&item).unwrap_or_else(|| "unknown".to_string());
                    let mut args_str = item
                        .get("arguments")
                        .and_then(|v| v.as_str())
                        .unwrap_or("{}")
                        .to_string();
                    let call_id = item
                        .get("call_id")
                        .and_then(|v| v.as_str())
                        .or_else(|| item.get("id").and_then(|v| v.as_str()))
                        .unwrap_or("unknown");

                    if item_type == "custom_tool_call" {
                        if let Some(input) = item.get("input").and_then(|v| v.as_str()) {
                            args_str = serde_json::to_string(&json!({ "input": input }))
                                .unwrap_or_else(|_| "{}".to_string());
                        }
                    } else if item_type == "local_shell_call" || name == "local_shell_call" {
                        name = "shell".to_string();
                        if let Some(action) = item.get("action") {
                            if let Some(exec) = action.get("exec") {
                                let mut args_obj = serde_json::Map::new();
                                if let Some(cmd) = exec.get("command") {
                                    let cmd_val = if cmd.is_string() {
                                        json!([cmd])
                                    } else {
                                        cmd.clone()
                                    };
                                    args_obj.insert("command".to_string(), cmd_val);
                                }
                                if let Some(wd) =
                                    exec.get("working_directory").or(exec.get("workdir"))
                                {
                                    args_obj.insert("workdir".to_string(), wd.clone());
                                }
                                args_str = serde_json::to_string(&args_obj)
                                    .unwrap_or_else(|_| "{}".to_string());
                            }
                        }
                    } else if item_type == "web_search_call" || name == "web_search_call" {
                        name = "google_search".to_string();
                        if let Some(action) = item.get("action") {
                            let mut args_obj = serde_json::Map::new();
                            if let Some(q) = action.get("query") {
                                args_obj.insert("query".to_string(), q.clone());
                            }
                            args_str = serde_json::to_string(&args_obj)
                                .unwrap_or_else(|_| "{}".to_string());
                        }
                    }

                    messages.push(json!({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": call_id,
                            "type": "function",
                            "function": { "name": name, "arguments": args_str }
                        }]
                    }));
                }
                "function_call_output" | "custom_tool_call_output" => {
                    let call_id = item
                        .get("call_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();
                    if item_type == "custom_tool_call_output"
                        && skipped_incomplete_custom_call_ids.contains(&call_id)
                    {
                        tracing::warn!(
                            "Skipping output for incomplete custom tool call {}",
                            call_id
                        );
                        continue;
                    }
                    let (mut output_str, output_media) = responses_tool_output_parts(&mut item);

                    let name = match call_id_to_name.get(&call_id).cloned().or_else(|| {
                        get_cached_tool_call(&call_id).and_then(|v| responses_item_tool_name(&v))
                    }) {
                        Some(name) => name,
                        None if item_type == "custom_tool_call_output" => {
                            tracing::warn!(
                                "Skipping orphan custom_tool_call_output for unknown call_id {}",
                                call_id
                            );
                            continue;
                        }
                        None => "shell".to_string(),
                    };

                    output_str = prefix_with_step_marker(step_marker, output_str);
                    let output_content =
                        build_responses_tool_output_content(output_str, output_media);

                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": output_content
                    }));
                }
                _ => {}
            }
        }
    }

    let dropped = drop_leading_orphan_tool_history(&mut messages);
    if dropped > 0 {
        tracing::warn!(
            dropped_messages = dropped,
            "[Responses Compat] Dropped leading orphan websocket tool history"
        );
    }
    if rewrite_terminal_assistant_prefill(&mut messages) {
        tracing::debug!(
            "[Responses Compat] Rewrote websocket terminal assistant text prefill as user input"
        );
    }

    if let Some(obj) = body.as_object_mut() {
        obj.insert("messages".to_string(), Value::Array(messages));
        if let Some(ledger) = interaction_ledger {
            obj.insert("_interaction_ledger".to_string(), json!(ledger));
        }
        obj.remove("input");
        obj.remove("instructions");
    }
    body
}

fn websocket_http_error_event(status: u16, body: &[u8]) -> Value {
    let envelope = serde_json::from_slice::<Value>(body).unwrap_or(Value::Null);
    let error = envelope
        .get("error")
        .filter(|value| value.is_object())
        .cloned()
        .unwrap_or_else(
            || json!({"type":"server_error","code":status,"message":String::from_utf8_lossy(body)}),
        );
    let mut event = json!({"type":"error","status":status,"error":error});
    for field in ["usage", "usageMetadata"] {
        if let Some(value) = envelope.get(field) {
            event[field] = value.clone();
        }
    }
    event
}

// Forward native Responses events unchanged. The same mapper now owns HTTP and
// WebSocket terminal semantics, reasoning items, tool output and sparse usage.
fn websocket_response_events<S, E>(source: S) -> impl futures::Stream<Item = Value>
where
    S: futures::Stream<Item = Result<Bytes, E>>,
    E: std::fmt::Display,
{
    async_stream::stream! {
        futures::pin_mut!(source);
        let mut buffer = bytes::BytesMut::new();
        let mut response_id = Value::Null;
        let mut model = Value::Null;
        let mut next_sequence = 0;
        let mut failure = "Responses stream ended before a terminal event".to_string();
        'read: while let Some(chunk) = source.next().await {
            match chunk {
                Ok(chunk) => buffer.extend_from_slice(&chunk),
                Err(err) => { failure = format!("Responses body read failed: {err}"); break; }
            }
            while let Some(end) = buffer.iter().position(|b| *b == b'\n') {
                let line = buffer.split_to(end + 1);
                let Some(data) = line.strip_prefix(b"data:") else { continue };
                let event: Value = match serde_json::from_slice(data) {
                    Ok(event) => event,
                    Err(_) => { failure = "Invalid event from Responses stream".into(); break 'read; }
                };
                if let Some(id) = event.pointer("/response/id") { response_id = id.clone(); }
                if let Some(value) = event.pointer("/response/model") { model = value.clone(); }
                if let Some(seq) = event["sequence_number"].as_u64() { next_sequence = seq.saturating_add(1); }
                let terminal = matches!(event["type"].as_str(), Some("response.completed" | "response.incomplete" | "response.failed"));
                yield event;
                if terminal { return; }
            }
        }
        // A body error/EOF can occur after response.created and visible deltas.
        // An error envelope is terminal here; never fabricate response.completed.
        let error = json!({"code":"stream_error","message":failure});
        yield json!({"type":"response.failed","sequence_number":next_sequence,"response":{"id":response_id,"model":model,"object":"response","status":"failed","error":error,"output":[]}});
    }
}

fn commit_websocket_response(state: &mut WebsocketSessionState, event: &Value) -> bool {
    let response = &event["response"];
    let resumable = match event["type"].as_str() {
        Some("response.completed") => response["status"] == "completed",
        Some("response.incomplete") => {
            response["incomplete_details"]["reason"] == "max_output_tokens"
                && response["error"].is_null()
        }
        _ => false,
    };
    let Some(id) = response["id"].as_str().filter(|id| !id.is_empty()) else {
        return false;
    };
    let Some(output) = response["output"].as_array() else {
        return false;
    };
    if !resumable
        || !output.iter().any(|item| match item["type"].as_str() {
            Some("message") => item["content"].as_array().is_some_and(|parts| {
                parts.iter().any(|part| {
                    part["type"] == "output_text"
                        && part["text"].as_str().is_some_and(|text| !text.is_empty())
                })
            }),
            Some("function_call" | "custom_tool_call") => {
                item["call_id"].as_str().is_some_and(|id| !id.is_empty())
            }
            _ => false,
        })
    {
        return false;
    }
    state.last_response_output = history_without_inline_media(&response["output"]);
    state.last_response_id = id.to_owned();
    for call in output.iter().filter(|item| {
        matches!(
            item["type"].as_str(),
            Some("function_call" | "custom_tool_call")
        )
    }) {
        if let Some(call_id) = call["call_id"].as_str() {
            state
                .tool_call_cache
                .insert(call_id.to_string(), call.clone());
        }
    }
    true
}

const CONTEXT_SUMMARY_PROMPT: &str = r#"You are a context compression specialist. Your task is to create a structured XML snapshot of the conversation history.

This snapshot will become the Agent's ONLY memory of the past. All key details, plans, errors, and user instructions MUST be preserved.

First, think through the entire history in a private <scratchpad>. Review the user's overall goal, the agent's actions, tool outputs, file modifications, and any unresolved issues. Identify every piece of information critical for future actions.

After reasoning, generate the final <state_snapshot> XML object. Information must be extremely dense. Omit any irrelevant conversational filler.

The structure MUST be as follows:

<state_snapshot>
  <overall_goal>
    <!-- Describe the user's high-level goal in one concise sentence -->
  </overall_goal>

  <technical_context>
    <!-- Tech stack: frameworks, languages, toolchain, dependency versions -->
  </technical_context>

  <file_system_state>
    <!-- List files that were created, read, modified, or deleted. Note their status -->
  </file_system_state>

  <code_changes>
    <!-- Key code snippets (preserve function signatures and important logic) -->
  </code_changes>

  <debugging_history>
    <!-- List all errors encountered, with stack traces, and how they were fixed -->
  </debugging_history>

  <current_plan>
    <!-- Step-by-step plan. Mark completed steps -->
  </current_plan>

  <user_preferences>
    <!-- User's work preferences for this project (test commands, code style, etc.) -->
  </user_preferences>

  <key_decisions>
    <!-- Critical architectural decisions and design choices -->
  </key_decisions>

  <latest_thinking_signature>
    <!-- [CRITICAL] Preserve the last valid thinking signature -->
    <!-- Format: base64-encoded signature string -->
    <!-- This MUST be copied exactly as-is, no modifications -->
  </latest_thinking_signature>
</state_snapshot>

**IMPORTANT**:
1. Code snippets must be complete, including function signatures and key logic
2. Error messages must be preserved verbatim, including line numbers and stacks
3. File paths must use absolute paths
4. The thinking signature must be copied exactly, no modifications
"#;

async fn call_openai_gemini_sync(
    model: &str,
    request: &OpenAIRequest,
    token_manager: &std::sync::Arc<crate::proxy::TokenManager>,
    upstream: &std::sync::Arc<crate::proxy::upstream::client::UpstreamClient>,
    trace_id: &str,
) -> Result<String, String> {
    let (access_token, project_id, _, account_id, _wait_ms) = token_manager
        .get_token("gemini", false, None, model)
        .await
        .map_err(|e| format!("Failed to get account: {}", e))?;

    let token_obj = token_manager.get_token_by_id(&account_id);
    let session_id = format!("bg_sid_{}", chrono::Utc::now().timestamp_subsec_millis());
    let (gemini_body, _, _, _) =
        transform_openai_request(request, &project_id, &session_id, token_obj.as_ref())?;

    debug!(
        "[{}] [OpenAI-BG] Calling {} via cloudcode v1internal for summary",
        trace_id, model
    );

    // 走共享的辅助调用通道。
    //
    // 历史实现把 `transform_openai_request` 产出的**已包装 cloudcode 信封**
    // （内含 project / request / model / userAgent / requestId / requestType）
    // POST 到 `https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`
    // —— host 与形状双双不符。实测该 host 对 Antigravity 账号恒返回
    // 403 `ACCESS_TOKEN_SCOPE_INSUFFICIENT`（token 不具备公共 Generative Language API 权限），
    // 因此该后台摘要从未成功过。
    let gemini_response = upstream
        .call_v1_internal_auxiliary(
            "generateContent",
            &access_token,
            gemini_body,
            Some(account_id.as_str()),
            SUMMARY_REQUEST_TIMEOUT_SECS,
        )
        .await?;

    // 上游返回 `{"response": {...}}` 包装体，必须先解包。
    let unwrapped = crate::proxy::mappers::gemini::unwrap_response(&gemini_response);

    // 只拼接「非思考 part」的文本：前面可能存在 `{"thought": true, "text": ""}` 的空块，
    // 直接取 `parts[0].text` 会拿到空串并把空摘要当成功。
    unwrapped
        .get("candidates")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("content"))
        .and_then(|c| c.get("parts"))
        .and_then(|parts| parts.as_array())
        .map(|parts| {
            parts
                .iter()
                .filter(|part| !crate::proxy::thinking_store::is_thought_part(part))
                .filter_map(|part| part.get("text").and_then(|t| t.as_str()))
                .collect::<String>()
        })
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| "Failed to extract text from response".to_string())
}

async fn try_compress_openai_with_summary(
    original_request: &OpenAIRequest,
    trace_id: &str,
    token_manager: &std::sync::Arc<crate::proxy::TokenManager>,
    upstream: &std::sync::Arc<crate::proxy::upstream::client::UpstreamClient>,
    session_id_str: &str,
) -> Result<OpenAIRequest, String> {
    info!(
        "[{}] [Layer-3] [OpenAI] Starting context compression with XML summary",
        trace_id
    );

    let last_signature =
        crate::proxy::mappers::context_manager::ContextManager::extract_last_openai_valid_signature(
            session_id_str,
        );

    if let Some(ref sig) = last_signature {
        debug!(
            "[{}] [Layer-3] [OpenAI] Extracted signature (len: {})",
            trace_id,
            sig.len()
        );
    }

    let mut summary_messages = original_request.messages.clone();

    let signature_instruction = if let Some(ref sig) = last_signature {
        format!("\n\n**CRITICAL**: The last thinking signature is:\n```\n{}\n```\nYou MUST include this EXACTLY in the <latest_thinking_signature> section.", sig)
    } else {
        "\n\n**Note**: No thinking signature found in history. Leave <latest_thinking_signature> empty.".to_string()
    };

    summary_messages.push(OpenAIMessage {
        role: "user".to_string(),
        content: Some(
            crate::proxy::mappers::openai::models::OpenAIContent::String(format!(
                "{}{}",
                CONTEXT_SUMMARY_PROMPT, signature_instruction
            )),
        ),
        refusal: None,
        reasoning_content: None,
        signature: None,
        tool_calls: None,
        tool_call_id: None,
        name: None,
    });

    // 兜底简单任务模型完全由当前号池动态推导（优先最新 flash-low，其次无后缀 flash），绝不硬编码
    let background_model =
        crate::proxy::common::model_gate::pick_background_model_from_pool(token_manager.as_ref())
            .ok_or_else(|| "No flash model available in the current quota pool".to_string())?;

    let mut summary_request = original_request.clone();
    summary_request.messages = summary_messages;
    summary_request.model = background_model.to_string();
    summary_request.stream = false;
    summary_request.max_tokens = Some(8000);
    summary_request.temperature = Some(0.3);

    debug!(
        "[{}] [Layer-3] [OpenAI] Calling {} for summary generation",
        trace_id, background_model
    );

    let xml_summary = call_openai_gemini_sync(
        &background_model,
        &summary_request,
        token_manager,
        upstream,
        trace_id,
    )
    .await?;

    info!(
        "[{}] [Layer-3] [OpenAI] Generated XML summary (len: {} chars)",
        trace_id,
        xml_summary.len()
    );

    let mut forked_messages = vec![
        OpenAIMessage {
            role: "user".to_string(),
            content: Some(crate::proxy::mappers::openai::models::OpenAIContent::String(format!(
                "Context has been compressed. Here is the structured summary of our conversation history:\n\n{}",
                xml_summary
            ))),
            refusal: None,
            reasoning_content: None,
            signature: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
        OpenAIMessage {
            role: "assistant".to_string(),
            content: Some(crate::proxy::mappers::openai::models::OpenAIContent::String(
                "I have reviewed the compressed context summary. I understand the current state and will continue from here.".to_string()
            )),
            refusal: None,
            reasoning_content: None,
            signature: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
    ];

    if let Some(last_msg) = original_request.messages.last() {
        if last_msg.role == "user" {
            if !matches!(&last_msg.content, Some(crate::proxy::mappers::openai::models::OpenAIContent::String(s)) if s.contains(CONTEXT_SUMMARY_PROMPT))
            {
                forked_messages.push(last_msg.clone());
            }
        }
    }

    let mut forked_request = original_request.clone();
    forked_request.messages = forked_messages;
    Ok(forked_request)
}

#[cfg(test)]
mod nonstream_peek_timeout_tests {
    use super::*;
    use crate::proxy::pipeline::stream_error::{
        attach_error_usage, observe_response_errors, UpstreamStreamErrorObserver,
    };
    use crate::proxy::pipeline::stream_peek::{
        is_heartbeat, peek_first_data, PeekOutcome, StreamDialect,
    };
    use crate::proxy::sticky_config::SchedulingMode;
    use futures::StreamExt;

    #[tokio::test(start_paused = true)]
    async fn timeout_keeps_buffered_prefill_usage_and_cache_affinity() {
        for protocol in ["openai", "openai_responses"] {
            for mode in [SchedulingMode::CacheFirst, SchedulingMode::Balance] {
                for measured in [true, false] {
                    let observer = UpstreamStreamErrorObserver::default();
                    let count = if measured { 4096 } else { 0 };
                    let raw = json!({"usageMetadata":{"promptTokenCount":count,"cachedContentTokenCount":count}});
                    let source = futures::stream::iter(vec![Ok::<_, String>(Bytes::from(
                        format!("data: {raw}\n\n"),
                    ))])
                    .chain(futures::stream::pending());
                    let upstream = observe_response_errors(source, observer.clone(), true);
                    let mut mapped = crate::proxy::mappers::openai::streaming::create_openai_sse_stream_with_context(Box::pin(upstream), "gemini-flash".into(), "timeout-regression".into(), 1, None, true, None, None);
                    let outcome = peek_first_data(
                        &mut mapped,
                        1,
                        |b| is_heartbeat(b, StreamDialect::OpenAi),
                        |_| false,
                    )
                    .await;
                    let PeekOutcome::Timeout { budget_secs, .. } = outcome else {
                        panic!("expected a timeout, got {outcome:?}");
                    };
                    let mut failures = FailureStatusTracker::default();
                    let mut retry_state = RequestRetryState::default();
                    let usage = record_nonstream_peek_timeout(
                        protocol,
                        budget_secs,
                        &observer,
                        &mut failures,
                        &mut retry_state,
                        mode,
                        true,
                    );
                    assert_eq!(failures.final_status(), StatusCode::GATEWAY_TIMEOUT);
                    assert_eq!(usage.is_some(), measured);
                    let mut error = json!({"error":{"code":504,"message":"first data timeout"}});
                    attach_error_usage(
                        &mut error,
                        if protocol == "openai_responses" {
                            "responses"
                        } else {
                            "openai"
                        },
                        usage.as_ref(),
                    );
                    if measured {
                        assert_eq!(error["usageMetadata"]["promptTokenCount"], 4096);
                        let field = if protocol == "openai_responses" {
                            "input_tokens_details"
                        } else {
                            "prompt_tokens_details"
                        };
                        assert_eq!(error["usage"][field]["cached_tokens"], 4096);
                    } else {
                        assert!(error.get("usage").is_none());
                    }
                    let response = retry_state.finish_error_response(
                        (failures.final_status(), Json(error)).into_response(),
                    );
                    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
                    assert_eq!(
                        response.headers().get("x-am-cache-affinity").is_some(),
                        mode == SchedulingMode::CacheFirst
                    );
                    drop(mapped); // The timeout returns this terminal error; no replay path is entered.
                }
            }
        }
    }
}

#[cfg(test)]
mod responses_http_history_tests {
    use super::{save_session_unless_response_cancelled, take_responses_input_items};
    use crate::proxy::http_session_store::{
        get_session, get_session_with_parent, prepare_session_input_with_storage,
        save_session_delta,
    };
    use crate::proxy::mappers::openai::streaming::create_codex_sse_stream;
    use bytes::Bytes;
    use futures::{stream, StreamExt};
    use serde_json::{json, Value};
    use std::collections::HashMap;

    #[tokio::test]
    async fn http_string_input_preserves_first_turn_and_previous_response_delta() {
        let first = "First question: 中文\n\"quoted\"";
        let followup = "Continue from the previous answer.";
        for store in [true, false] {
            let mut string_body = json!({"model":"gemini-pro", "input":first});
            let initial = take_responses_input_items(&mut string_body);
            assert_eq!(initial, vec![json!({"role":"user","content":first})]);
            assert!(string_body.get("input").is_none());
            assert_eq!(string_body["model"], "gemini-pro");
            let parent_id = format!("resp-string-{}", uuid::Uuid::new_v4());
            save_session_delta(
                parent_id.clone(), None, initial.clone(),
                vec![json!({"type":"message","role":"assistant","content":[{"type":"output_text","text":"First answer"}]})],
                "Keep context".into(), "gemini-pro".into(), "routing-string".into(),
            ).await;
            let (session, parent) = get_session_with_parent(&parent_id).await.unwrap();
            let mut body = json!({"previous_response_id":parent_id,"input":followup});
            let next = take_responses_input_items(&mut body);
            let prepared = prepare_session_input_with_storage(
                session.input_items.clone(),
                next,
                &HashMap::new(),
                store,
            );
            assert_eq!(prepared.merged.len(), 3);
            assert_eq!(prepared.merged[0], initial[0]);
            assert_eq!(
                prepared.merged[2],
                json!({"role":"user","content":followup})
            );
            assert_eq!(parent.routing_session_id(), "routing-string");
            assert!(!prepared.reset_parent);
            assert_eq!(prepared.delta.len(), usize::from(store));
            let mut array_body = json!({"input":[{"role":"user","content":followup}]});
            let array = prepare_session_input_with_storage(
                session.input_items,
                take_responses_input_items(&mut array_body),
                &HashMap::new(),
                store,
            );
            assert_eq!(array.merged, prepared.merged);
            assert_eq!(array.delta, prepared.delta);
            if store {
                let child_id = format!("resp-string-child-{}", uuid::Uuid::new_v4());
                save_session_delta(
                    child_id.clone(),
                    Some(parent),
                    prepared.delta,
                    Vec::new(),
                    session.instructions,
                    session.model,
                    "routing-string".into(),
                )
                .await;
                assert_eq!(
                    get_session(&child_id).await.unwrap().input_items,
                    prepared.merged
                );
            }
        }
    }

    fn sse(frame: Value) -> Result<Bytes, String> {
        Ok(Bytes::from(format!("data: {frame}\n\n")))
    }

    async fn stream_with_history(
        frames: Vec<Result<Bytes, String>>,
        store: bool,
    ) -> (String, Value) {
        let id = format!("resp-history-{}", uuid::Uuid::new_v4());
        let parent_id = format!("{id}-parent");
        save_session_delta(parent_id.clone(), None,
            vec![json!({"role":"user","content":"First question"})],
            vec![json!({"type":"message","role":"assistant","content":[{"type":"output_text","text":"First answer"}]})],
            String::new(), "gemini-pro".into(), "routing-history".into()).await;
        let (_, parent) = get_session_with_parent(&parent_id).await.unwrap();
        let (tx, rx) =
            tokio::sync::oneshot::channel::<(Vec<Value>, tokio::sync::oneshot::Sender<()>)>();
        let save_id = id.clone();
        let save_task = tokio::spawn(async move {
            if let Ok((outputs, ack_tx)) = rx.await {
                save_session_unless_response_cancelled(
                    ack_tx,
                    save_session_delta(
                        save_id,
                        Some(parent),
                        vec![json!({"role":"user","content":"Next question"})],
                        outputs,
                        String::new(),
                        "gemini-pro".into(),
                        "routing-history".into(),
                    ),
                )
                .await;
            }
        });
        let callback = if store {
            Some(tx)
        } else {
            drop(tx);
            None
        };
        let mut mapped = create_codex_sse_stream(
            Box::pin(stream::iter(frames)),
            "gemini-pro".into(),
            id.clone(),
            3,
            1,
            id.clone(),
            callback,
            store,
        );
        let mut terminal = Value::Null;
        while let Some(chunk) = mapped.next().await {
            for line in String::from_utf8_lossy(&chunk.unwrap()).lines() {
                if let Some(data) = line.strip_prefix("data: ") {
                    if let Ok(event) = serde_json::from_str::<Value>(data) {
                        if matches!(
                            event["type"].as_str(),
                            Some("response.completed" | "response.incomplete" | "response.failed")
                        ) {
                            terminal = event["response"].clone();
                            if store && matches!(terminal["status"].as_str(), Some("completed")) {
                                assert!(
                                    get_session(&id).await.is_some(),
                                    "save barrier precedes terminal event"
                                );
                            }
                            if store
                                && terminal["incomplete_details"]["reason"] == "max_output_tokens"
                                && terminal["error"].is_null()
                            {
                                assert!(
                                    get_session(&id).await.is_some(),
                                    "truncated history precedes terminal event"
                                );
                            }
                        }
                    }
                }
            }
        }
        drop(mapped);
        save_task.await.unwrap();
        (id, terminal)
    }

    #[tokio::test]
    async fn max_tokens_stream_persists_text_and_tool_continuations() {
        for (part, continuation) in [
            (
                json!({"text":"Partial answer"}),
                json!({"role":"user","content":"Continue"}),
            ),
            (
                json!({"functionCall":{"id":"tool-limit","name":"lookup","args":{"city":"Paris"}}}),
                json!({"type":"function_call_output","call_id":"tool-limit","output":"sunny"}),
            ),
        ] {
            for store in [true, false] {
                let frames = vec![
                    sse(
                        json!({"candidates":[{"content":{"parts":[part]}}],"usageMetadata":{"promptTokenCount":100,"cachedContentTokenCount":80}}),
                    ),
                    sse(
                        json!({"candidates":[{"finishReason":"MAX_TOKENS"}],"usageMetadata":{"candidatesTokenCount":5}}),
                    ),
                    sse(json!({"usageMetadata":{"cachedContentTokenCount":0}})),
                ];
                let (id, response) = stream_with_history(frames, store).await;
                assert_eq!(response["status"], "incomplete");
                assert_eq!(
                    response["incomplete_details"]["reason"],
                    "max_output_tokens"
                );
                assert!(response["error"].is_null());
                assert_eq!(response["usage"]["input_tokens"], 100);
                assert_eq!(
                    response["usage"]["input_tokens_details"]["cached_tokens"],
                    0
                );
                assert_eq!(response["usage"]["output_tokens"], 5);
                let restored = get_session_with_parent(&id).await;
                assert_eq!(restored.is_some(), store);
                if let Some((saved, parent)) = restored {
                    assert_eq!(parent.routing_session_id(), "routing-history");
                    assert_eq!(
                        &saved.input_items[3..],
                        response["output"].as_array().unwrap()
                    );
                    let mut next = continuation.clone();
                    if next["type"] == "function_call_output" {
                        next["call_id"] = response["output"][0]["call_id"].clone();
                    }
                    let prepared = prepare_session_input_with_storage(
                        saved.input_items.clone(),
                        vec![next.clone()],
                        &HashMap::new(),
                        true,
                    );
                    assert!(prepared.merged.starts_with(&saved.input_items));
                    assert_eq!(prepared.merged.last(), Some(&next));
                }
            }
        }
    }

    #[tokio::test]
    async fn failed_empty_and_refused_streams_do_not_save_history() {
        let partial = json!({"content":{"parts":[{"text":"partial"}]}});
        for frames in [
            vec![sse(json!({"candidates":[partial.clone()]}))],
            vec![sse(json!({"candidates":[{"finishReason":"MAX_TOKENS"}]}))],
            vec![sse(
                json!({"candidates":[{"content":{"parts":[{"thought":true,"text":"thinking"}]},"finishReason":"MAX_TOKENS"}]}),
            )],
            vec![sse(
                json!({"candidates":[{"content":{"parts":[{"text":"partial"}]},"finishReason":"SAFETY"}]}),
            )],
            vec![
                sse(
                    json!({"candidates":[{"content":{"parts":[{"text":"partial"}]},"finishReason":"MAX_TOKENS"}]}),
                ),
                Err("connection reset before stream EOF".into()),
            ],
        ] {
            let (id, response) = stream_with_history(frames, true).await;
            assert_ne!(response["status"], "completed");
            assert!(
                get_session(&id).await.is_none(),
                "non-resumable response: {response}"
            );
        }
    }

    #[tokio::test]
    async fn cancelled_truncated_response_drops_pending_history_save() {
        let id = format!("resp-truncated-cancel-{}", uuid::Uuid::new_v4());
        let (tx, rx) =
            tokio::sync::oneshot::channel::<(Vec<Value>, tokio::sync::oneshot::Sender<()>)>();
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let save_id = id.clone();
        let save_task = tokio::spawn(async move {
            let (outputs, ack_tx) = rx.await.unwrap();
            save_session_unless_response_cancelled(ack_tx, async move {
                entered_tx.send(()).unwrap();
                std::future::pending::<()>().await;
                save_session_delta(
                    save_id,
                    None,
                    Vec::new(),
                    outputs,
                    String::new(),
                    "gemini-pro".into(),
                    "routing-cancel".into(),
                )
                .await;
            })
            .await;
        });
        let mapped = create_codex_sse_stream(
            Box::pin(stream::iter(vec![sse(
                json!({"candidates":[{"content":{"parts":[{"text":"partial"}]},"finishReason":"MAX_TOKENS"}]}),
            )])),
            "gemini-pro".into(),
            id.clone(),
            1,
            0,
            id.clone(),
            Some(tx),
            true,
        );
        let reader = tokio::spawn(async move {
            let _ = mapped.collect::<Vec<_>>().await;
        });
        tokio::time::timeout(std::time::Duration::from_secs(1), entered_rx)
            .await
            .unwrap()
            .unwrap();
        reader.abort();
        let _ = reader.await;
        tokio::time::timeout(std::time::Duration::from_secs(1), save_task)
            .await
            .unwrap()
            .unwrap();
        assert!(get_session(&id).await.is_none());
    }
}

#[cfg(test)]
mod responses_websocket_history_tests {
    use super::*;

    fn state() -> WebsocketSessionState {
        WebsocketSessionState {
            last_request: None,
            last_response_output: json!([]),
            last_response_id: String::new(),
            tool_call_cache: Default::default(),
        }
    }

    fn input(text: &str, shorthand: bool) -> Value {
        if shorthand {
            json!(text)
        } else {
            json!([{"role":"user","content":text}])
        }
    }

    #[test]
    fn websocket_create_and_append_preserve_string_turns_and_assistant_output() {
        for event in ["response.create", "response.append"] {
            for first_shorthand in [true, false] {
                for next_shorthand in [true, false] {
                    for store in [true, false] {
                        let mut state = state();
                        let first=normalize_responses_websocket_request(json!({"type":"response.create","model":"gemini-pro","instructions":"Keep context","store":store,"input": input("Question one", first_shorthand)}), &mut state).unwrap();
                        assert_eq!(
                            convert_codex_to_openai_request(first)["messages"],
                            json!([{"role":"system","content":"Keep context"},{"role":"user","content":"Question one"}])
                        );
                        state.last_response_id = "resp-first".into();
                        state.last_response_output = json!([{"id":"assistant-1","type":"message","role":"assistant","content":[{"type":"output_text","text":"Answer one"}]}]);
                        let next=normalize_responses_websocket_request(json!({"type":event,"previous_response_id":"resp-first","store":store,"input":input("Question two", next_shorthand)}),&mut state).unwrap();
                        assert_eq!(next["model"], "gemini-pro");
                        assert_eq!(next["store"], store);
                        assert_eq!(
                            convert_codex_to_openai_request(next)["messages"],
                            json!([{"role":"system","content":"Keep context"},{"role":"user","content":"Question one"},{"role":"assistant","content":"Answer one"},{"role":"user","content":"Question two"}])
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn websocket_prewarm_preserves_string_input_for_first_generation() {
        for shorthand in [true, false] {
            let mut state = state();
            let prewarm = json!({"type":"response.create","model":"gemini-pro","generate":false,"input":input("Warm context",shorthand)});
            assert!(should_handle_prewarm_locally(&prewarm, &state));
            let (_, completed) = handle_prewarm_locally(&prewarm, &mut state);
            assert_eq!(completed["response"]["usage"]["total_tokens"], 0);
            let next = json!({"type":"response.create","previous_response_id":completed["response"]["id"],"input":"First question"});
            assert!(!should_handle_prewarm_locally(&next, &state));
            let normalized = normalize_responses_websocket_request(next, &mut state).unwrap();
            assert_eq!(
                convert_codex_to_openai_request(normalized)["messages"],
                json!([{"role":"user","content":"Warm context"},{"role":"user","content":"First question"}])
            );
        }
    }
}

#[cfg(test)]
mod websocket_native_responses_tests {
    use super::*;
    use crate::proxy::mappers::openai::streaming::create_codex_sse_stream;
    use futures::{stream, StreamExt};
    fn state() -> WebsocketSessionState {
        WebsocketSessionState {
            last_request: None,
            last_response_output: json!([]),
            last_response_id: String::new(),
            tool_call_cache: Default::default(),
        }
    }
    fn frame(value: Value) -> Result<Bytes, String> {
        Ok(Bytes::from(format!("data: {value}\n\n")))
    }
    #[tokio::test]
    async fn native_terminal_usage_and_history_are_preserved_on_the_socket() {
        for (name, reason, terminal, detail, save) in [
            ("complete", "STOP", "response.completed", None, true),
            (
                "truncated",
                "MAX_TOKENS",
                "response.incomplete",
                Some("max_output_tokens"),
                true,
            ),
            (
                "filtered",
                "SAFETY",
                "response.incomplete",
                Some("content_filter"),
                false,
            ),
            ("503", "", "response.failed", None, false),
            ("transport", "", "response.failed", None, false),
        ] {
            let response_id = format!("resp-ws-test-{}", Uuid::new_v4());
            let mut committed = state();
            let mut pending = committed.clone();
            let mut body = normalize_responses_websocket_request(
                json!({"type":"response.create","model":"gemini-pro","input":"Task","store":false}),
                &mut pending,
            )
            .unwrap();
            prepare_websocket_generation(&mut body, &mut pending, &HeaderMap::new(), "ws-stable")
                .await
                .unwrap();
            assert_eq!(body["session_id"], "ws-stable");
            assert_eq!(body["stream"], true);
            assert_eq!(body["store"], false);
            let mut frames = vec![frame(
                json!({"candidates":[{"content":{"parts":[{"thought":true,"text":"Thinking only","thoughtSignature":"c2lnbmF0dXJl"},{"text":"Visible answer"}]}}],"usageMetadata":{"promptTokenCount":100,"cachedContentTokenCount":80}}),
            )];
            if name == "503" {
                frames.push(frame(
                    json!({"error":{"code":503,"message":"generation failed"}}),
                ));
            } else if name == "transport" {
                frames.push(Err("connection closed before message completed".into()));
            } else {
                frames.push(frame(json!({"candidates":[{"finishReason":reason}],"usageMetadata":{"candidatesTokenCount":5}})));
            }
            let source = create_codex_sse_stream(
                Box::pin(stream::iter(frames)),
                "gemini-pro".into(),
                response_id.clone(),
                1,
                0,
                response_id.clone(),
                None,
                false,
            );
            let events: Vec<_> = websocket_response_events(source).collect().await;
            let last = events.last().unwrap();
            assert_eq!(last["type"], terminal, "{name}");
            assert_eq!(
                last["response"]["incomplete_details"]["reason"].as_str(),
                detail,
                "{name}"
            );
            assert_eq!(
                last["response"]["usage"]["input_tokens_details"]["cached_tokens"], 80,
                "{name}"
            );
            if name == "503" {
                assert_eq!(last["response"]["error"]["upstream_status"], 503);
            }
            assert_eq!(
                events
                    .iter()
                    .filter(|e| matches!(
                        e["type"].as_str(),
                        Some("response.completed" | "response.incomplete" | "response.failed")
                    ))
                    .count(),
                1
            );
            assert_eq!(commit_websocket_response(&mut pending, last), save);
            if save {
                committed = pending;
                assert_eq!(committed.last_response_id, response_id);
                let output = committed.last_response_output.as_array().unwrap();
                assert!(output.iter().any(|i| i["type"] == "reasoning"));
                let message = output.iter().find(|i| i["type"] == "message").unwrap();
                assert_eq!(message["content"][0]["text"], "Visible answer");
                let mut next=normalize_responses_websocket_request(json!({"type":"response.append","previous_response_id":response_id,"input":"Continue","store":false}),&mut committed).unwrap();
                prepare_websocket_generation(
                    &mut next,
                    &mut committed,
                    &HeaderMap::new(),
                    "ws-stable",
                )
                .await
                .unwrap();
                assert!(next.get("previous_response_id").is_none());
                assert_eq!(next["session_id"], "ws-stable");
                let visible: Vec<_> = next["input"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|i| i["role"] == "assistant" && i["type"] == "message")
                    .collect();
                assert_eq!(visible.len(), 1);
                assert_eq!(visible[0]["content"][0]["text"], "Visible answer");
            } else {
                assert!(committed.last_request.is_none());
            }
        }
    }
    #[tokio::test]
    async fn relay_preserves_tools_and_sparse_zero_without_global_storage() {
        let id = Uuid::new_v4().to_string();
        let frames = vec![
            frame(
                json!({"candidates":[{"content":{"parts":[{"functionCall":{"id":id,"name":"scope__scopeLookup","args":{"city":"Paris"}}}]}}],"usageMetadata":{"promptTokenCount":100,"cachedContentTokenCount":80}}),
            ),
            frame(
                json!({"candidates":[{"finishReason":"STOP"}],"usageMetadata":{"candidatesTokenCount":5}}),
            ),
            frame(json!({"usageMetadata":{"cachedContentTokenCount":0}})),
        ];
        let source = create_codex_sse_stream(
            Box::pin(stream::iter(frames)),
            "gemini-pro".into(),
            id.clone(),
            1,
            0,
            id.clone(),
            None,
            false,
        );
        let events: Vec<_> = websocket_response_events(source).collect().await;
        let term = events.last().unwrap();
        assert_eq!(term["type"], "response.completed");
        assert_eq!(term["response"]["usage"]["input_tokens"], 100);
        assert_eq!(
            term["response"]["usage"]["input_tokens_details"]["cached_tokens"],
            0
        );
        let mut st = state();
        assert!(commit_websocket_response(&mut st, term));
        let call = &st.tool_call_cache[&id];
        assert_eq!(call["name"], "scopeLookup");
        assert_eq!(call["namespace"], "scope");
        let qualified = "scope__scopeLookup";
        assert_eq!(responses_item_tool_name(call).as_deref(), Some(qualified));
        let tools = Some(vec![
            json!({"type":"namespace","name":"scope","tools":[{"type":"function","name":"scopeLookup"}]}),
        ]);
        assert!(
            crate::proxy::mappers::openai::request::extract_client_tool_names(&tools)
                .contains(qualified)
        );
        let req = convert_codex_to_openai_request(
            json!({"model":"gemini-pro","input":[{"role":"user","content":"Look it up"},call,{"type":"function_call_output","call_id":id,"output":"Paris"}]}),
        );
        let messages = req["messages"].as_array().unwrap();
        assert_eq!(messages[1]["tool_calls"][0]["function"]["name"], qualified);
        assert_eq!(messages[2]["name"], qualified);
        assert_eq!(
            serde_json::from_str::<Value>(call["arguments"].as_str().unwrap()).unwrap(),
            json!({"city":"Paris"})
        );
        assert!(get_cached_tool_call(&id).is_none());
    }
    #[tokio::test]
    async fn full_transcript_replacement_and_explicit_aliases_keep_the_expected_identity() {
        for (field, expected) in [
            (None, "ws-connection"),
            (Some("conversation_id"), "new-session"),
        ] {
            let mut st = state();
            let mut root = normalize_responses_websocket_request(json!({"type":"response.create","model":"gemini-pro","tools":[],"instructions":"Continue","input":"First"}), &mut st).unwrap();
            prepare_websocket_generation(&mut root, &mut st, &HeaderMap::new(), "ws-connection")
                .await
                .unwrap();
            assert_eq!(root["session_id"], "ws-connection");
            let event = json!({"type":"response.completed","response":{"id":"response-first","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Answer"}]}]}});
            assert!(commit_websocket_response(&mut st, &event));
            let mut followup = json!({"type":"response.create","input":[{"role":"user","content":"First"},{"role":"assistant","content":"Answer"},{"role":"user","content":"Next"}]});
            if let Some(field) = field {
                followup[field] = json!(expected);
            }
            let mut normalized = normalize_responses_websocket_request(followup, &mut st).unwrap();
            prepare_websocket_generation(
                &mut normalized,
                &mut st,
                &HeaderMap::new(),
                "ws-connection",
            )
            .await
            .unwrap();
            assert_eq!(normalized["session_id"], expected);
            assert_eq!(normalized["input"].as_array().unwrap().len(), 3);
            assert_eq!(normalized["model"], "gemini-pro");
            assert_eq!(normalized["tools"], json!([]));
            assert_eq!(normalized["instructions"], "Continue");
        }
    }
    #[test]
    fn upfront_http_errors_keep_observed_cache_usage_and_status() {
        let original = json!({"error":{"code":"unavailable","message":"retry"},"usage":{"input_tokens":100,"input_tokens_details":{"cached_tokens":0}},"usageMetadata":{"promptTokenCount":100}});
        let event = websocket_http_error_event(503, original.to_string().as_bytes());
        assert_eq!(event["status"], 503);
        assert_eq!(event["error"], original["error"]);
        assert_eq!(event["usage"], original["usage"]);
        assert_eq!(event["usageMetadata"], original["usageMetadata"]);
        assert!(websocket_http_error_event(400, b"invalid body")
            .get("usage")
            .is_none());
    }
    #[tokio::test]
    async fn broken_responses_body_never_emits_success_or_commits() {
        let created = "event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"r\",\"model\":\"model\"}}\n\n";
        for broken in [
            None,
            Some(Ok(Bytes::from_static(b"data: {bad}\n\n"))),
            Some(Err("body transport error".to_string())),
        ] {
            let mut frames: Vec<Result<Bytes, String>> = created
                .as_bytes()
                .chunks(7)
                .map(|s| Ok(Bytes::copy_from_slice(s)))
                .collect();
            frames.extend(broken);
            let events: Vec<_> = websocket_response_events(stream::iter(frames))
                .collect()
                .await;
            assert_eq!(events.len(), 2);
            assert_eq!(events[1]["type"], "response.failed");
            assert_eq!(events[1]["response"]["id"], "r");
            assert!(!commit_websocket_response(&mut state(), &events[1]));
        }
    }
    #[tokio::test]
    async fn first_websocket_cursor_restores_parent_once_and_keeps_local_history() {
        use crate::proxy::http_session_store::save_session_delta;
        let id = Uuid::new_v4().to_string();
        save_session_delta(
            id.clone(),
            None,
            vec![json!({"role":"user","content":"old"})],
            vec![json!({"role":"assistant","type":"message","content":"prior answer"})],
            "Keep context".into(),
            "gemini-pro".into(),
            "routing-root".into(),
        )
        .await;
        let mut st = state();
        let mut body=normalize_responses_websocket_request(json!({"type":"response.create","model":"gemini-pro","previous_response_id":id,"input":"new","store":false}),&mut st).unwrap();
        prepare_websocket_generation(&mut body, &mut st, &HeaderMap::new(), "connection")
            .await
            .unwrap();
        assert_eq!(body["input"], "new");
        assert_eq!(body["previous_response_id"], id);
        assert!(body.get("session_id").is_none());
        assert_eq!(
            st.last_request.as_ref().unwrap()["input"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
        assert_eq!(
            st.last_request.as_ref().unwrap()["instructions"],
            "Keep context"
        );
    }
}

#[cfg(test)]
mod websocket_cursor_regressions {
    use super::*;

    fn state() -> WebsocketSessionState {
        WebsocketSessionState {
            last_request: None,
            last_response_output: json!([]),
            last_response_id: String::new(),
            tool_call_cache: Default::default(),
        }
    }
    fn output(id: &str, answer: &str) -> Value {
        json!({"type":"response.completed","response":{"id":id,"status":"completed","output":[{"id":format!("message-{id}"),"type":"message","role":"assistant","content":[{"type":"output_text","text":answer}]}]}})
    }
    async fn wire(
        s: &WebsocketSessionState,
        mut req: Value,
    ) -> Result<(Value, WebsocketSessionState), String> {
        req["store"] = json!(false);
        let mut pending = s.clone();
        let mut body = normalize_responses_websocket_request(req, &mut pending)?;
        prepare_websocket_generation(&mut body, &mut pending, &HeaderMap::new(), "ws-consistent")
            .await?;
        Ok((body, pending))
    }

    #[tokio::test]
    async fn returning_to_a_stored_parent_restores_its_real_identity_only() {
        use crate::proxy::http_session_store::save_session_delta_with_identity;
        for owner in [
            Some("ws-consistent"),
            Some("resp-looking-but-explicit"),
            None,
        ] {
            let root_id = format!("resp-{}", Uuid::new_v4());
            let mut first = json!({"type":"response.create","model":"g","input":"first"});
            if let Some(explicit) = owner {
                first["session_id"] = json!(explicit);
            }
            let (_, mut s) = wire(&state(), first).await.unwrap();
            let root = output(&root_id, "first answer");
            assert!(commit_websocket_response(&mut s, &root));
            // HTTP's anonymous case deliberately has a resp-* chain identifier
            // but no scope identity. A genuine client ID may itself start resp-.
            save_session_delta_with_identity(
                root_id.clone(),
                None,
                vec![json!({"role":"user","content":"first"})],
                root["response"]["output"].as_array().unwrap().clone(),
                String::new(),
                "g".into(),
                owner.unwrap_or(&root_id).to_owned(),
                owner.map(str::to_owned),
            )
            .await;
            let (second,mut child) = wire(&s,json!({"type":"response.create","previous_response_id":root_id,"session_id":"another-session","input":"second"})).await.unwrap();
            assert_eq!(second["session_id"], "another-session");
            assert!(commit_websocket_response(
                &mut child,
                &output("latest", "second answer")
            ));
            let request =
                json!({"type":"response.create","previous_response_id":root_id,"input":"branch"});
            let (branch, _) = wire(&child, request.clone()).await.unwrap();
            assert_eq!(
                branch["session_id"].as_str(),
                owner,
                "select the identity actually used for the named parent"
            );
            let mut overriding = request;
            overriding["conversation_id"] = json!("requested-alias");
            let (branch, _) = wire(&child, overriding).await.unwrap();
            assert_eq!(branch["session_id"], "requested-alias");
        }
    }
    #[tokio::test]
    async fn explicit_old_or_failed_cursor_is_not_silently_replaced_by_latest() {
        use crate::proxy::http_session_store::{get_session_with_parent, save_session_delta};
        let root_id = Uuid::new_v4().to_string();
        let second_id = Uuid::new_v4().to_string();
        let failed_id = Uuid::new_v4().to_string();
        let (_, mut s) = wire(
            &state(),
            json!({"type":"response.create","model":"g","input":"root turn"}),
        )
        .await
        .unwrap();
        let root = output(&root_id, "root answer");
        assert!(commit_websocket_response(&mut s, &root));
        save_session_delta(
            root_id.clone(),
            None,
            vec![json!({"role":"user","content":"root turn"})],
            root["response"]["output"].as_array().unwrap().clone(),
            String::new(),
            "g".into(),
            "routing".into(),
        )
        .await;
        let (_,mut s2)=wire(&s,json!({"type":"response.append","previous_response_id":root_id,"input":"separate child"})).await.unwrap();
        assert!(commit_websocket_response(
            &mut s2,
            &output(&second_id, "child answer")
        ));
        let (_,mut failed)=wire(&s2,json!({"type":"response.append","previous_response_id":second_id,"input":"failed generation"})).await.unwrap();
        assert!(!commit_websocket_response(
            &mut failed,
            &json!({"type":"response.failed","response":{"id":failed_id,"status":"failed","output":[]}})
        ));
        assert!(get_session_with_parent(&failed_id).await.is_none());
        let delta = |id: &str| json!({"type":"response.append","previous_response_id":id,"input":"correct branch"});
        let (control, _) = wire(&s2, delta(&second_id)).await.unwrap();
        assert!(control["input"].to_string().contains("child answer"));
        let (branch, _) = wire(&s2, delta(&root_id)).await.unwrap();
        assert_eq!(branch["previous_response_id"], root_id);
        let (_, branch_state) = wire(&s2, delta(&root_id)).await.unwrap();
        let history = &branch_state.last_request.as_ref().unwrap()["input"];
        assert!(history.to_string().contains("root answer"));
        assert!(!history.to_string().contains("child answer"));
        let bad = wire(&s2, delta(&failed_id)).await;
        assert!(
            bad.is_err(),
            "a failed cursor must be rejected or resolved separately"
        );
    }

    #[tokio::test]
    async fn explicit_compaction_replaces_parent_on_both_transports() {
        use crate::proxy::http_session_store::prepare_session_input_with_storage;
        let (_, mut st) = wire(
            &state(),
            json!({"type":"response.create","model":"g","input":"superseded-old-transcript"}),
        )
        .await
        .unwrap();
        let first = output("done", "superseded-old-answer");
        assert!(commit_websocket_response(&mut st, &first));
        let new = vec![
            json!({"type":"compaction_summary"}),
            json!({"role":"user","content":"condensed-continuation"}),
        ];
        let req = json!({"type":"response.create","previous_response_id":"done","input":new});
        let (out, _) = wire(&st, req).await.unwrap();
        let mut parent = vec![json!({"role":"user","content":"superseded-old-transcript"})];
        parent.extend(
            first["response"]["output"]
                .as_array()
                .unwrap()
                .iter()
                .cloned(),
        );
        let http = prepare_session_input_with_storage(parent, new, &Default::default(), false);
        assert_eq!(
            out["input"],
            json!(http.merged),
            "same explicit compaction must not resend superseded parent"
        );
    }

    #[tokio::test]
    async fn complete_replay_with_correct_cursor_does_not_duplicate_unlabelled_user_turn() {
        use crate::proxy::http_session_store::prepare_session_input_with_storage;
        let (_, mut st) = wire(
            &state(),
            json!({"type":"response.create","model":"g","input":"opening"}),
        )
        .await
        .unwrap();
        let first = output("done", "answer");
        assert!(commit_websocket_response(&mut st, &first));
        let mut parent = vec![json!({"role":"user","content":"opening"})];
        parent.extend(
            first["response"]["output"]
                .as_array()
                .unwrap()
                .iter()
                .cloned(),
        );
        let mut replay = parent.clone();
        replay.push(json!({"role":"user","content":"followup"}));
        let req = json!({"type":"response.create","previous_response_id":"done","input":replay});
        let (out, _) = wire(&st, req).await.unwrap();
        let http = prepare_session_input_with_storage(parent, replay, &Default::default(), false);
        assert_eq!(
            out["input"],
            json!(http.merged),
            "same complete replay with a correct cursor must not repeat the first user turn"
        );
    }
    #[tokio::test]
    async fn prewarming_from_an_http_parent_does_not_forget_the_parent() {
        use crate::proxy::http_session_store::save_session_delta;
        let root_id = Uuid::new_v4().to_string();
        save_session_delta(
            root_id.clone(),
            None,
            vec![json!({"role":"user","content":"persisted-root-question"})],
            vec![json!({"role":"assistant","content":"persisted-root-answer"})],
            "Previous instructions".to_string(),
            "g".into(),
            "saved-root".into(),
        )
        .await;
        let empty = || WebsocketSessionState {
            last_request: None,
            last_response_output: json!([]),
            last_response_id: String::new(),
            tool_call_cache: Default::default(),
        };
        let mut s1 = empty();
        let mut first=normalize_responses_websocket_request(json!({"type":"response.create","model":"g","previous_response_id":root_id,"input":"onward","store":false}),&mut s1).unwrap();
        prepare_websocket_generation(&mut first, &mut s1, &HeaderMap::new(), "ws-one")
            .await
            .unwrap();
        assert!(s1.last_request.as_ref().unwrap()["input"]
            .to_string()
            .contains("persisted-root-answer"));
        let mut s2 = empty();
        let prewarm = json!({"type":"response.create","generate":false,"model":"g","previous_response_id":root_id,"input":[],"store":false});
        assert!(should_handle_prewarm_locally(&prewarm, &s2));
        let (_, prewarmed) =
            prepare_websocket_prewarm(&prewarm, &mut s2, &HeaderMap::new(), "ws-one")
                .await
                .unwrap();
        let mut next=normalize_responses_websocket_request(json!({"type":"response.create","model":"g","previous_response_id":prewarmed["response"]["id"],"input":"onward","store":false}),&mut s2).unwrap();
        prepare_websocket_generation(&mut next, &mut s2, &HeaderMap::new(), "ws-one")
            .await
            .unwrap();
        assert!(
            next["input"].to_string().contains("persisted-root-answer")
                || next["previous_response_id"] == root_id,
            "first generated turn after prewarm lacks the actual parent"
        );
    }
}
#[cfg(test)]
mod responses_continuity_tests {
    use super::*;
    use crate::proxy::http_session_store::save_session_delta;
    const PNG:&str="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC";
    fn new_state() -> WebsocketSessionState {
        WebsocketSessionState {
            last_request: None,
            last_response_output: json!([]),
            last_response_id: String::new(),
            tool_call_cache: Default::default(),
        }
    }
    fn terminal(id: &str, answer: &str) -> Value {
        json!({"type":"response.completed","response":{"id":id,"status":"completed","output":[{"id":format!("msg_{id}"),"type":"message","role":"assistant","content":[{"type":"output_text","text":answer}]}]}})
    }

    #[tokio::test]
    async fn strict_parent_selection_distinguishes_missing_stored_and_new_conversations() {
        let id = Uuid::new_v4().to_string();
        assert!(load_responses_parent(None, true).await.unwrap().is_none());
        let err = load_responses_parent(Some(&id), true).await.unwrap_err();
        assert_eq!(err["error"]["code"], "previous_response_not_found");
        assert_eq!(err["error"]["param"], "previous_response_id");
        assert!(load_responses_parent(Some(&id), false)
            .await
            .unwrap()
            .is_none());
        save_session_delta(
            id.clone(),
            None,
            vec![json!({"role":"user","content":"root"})],
            vec![],
            String::new(),
            "g".into(),
            "a".into(),
        )
        .await;
        assert_eq!(
            load_responses_parent(Some(&id), true)
                .await
                .unwrap()
                .unwrap()
                .0
                .input_items[0]["content"],
            "root"
        );
    }

    #[tokio::test]
    async fn explicit_append_without_a_cursor_keeps_new_assistant_and_tool_deltas_and_deduplicates_replay(
    ) {
        let mut st = new_state();
        normalize_responses_websocket_request(
            json!({"type":"response.create","model":"g","input":"first"}),
            &mut st,
        )
        .unwrap();
        let output = terminal("first", "answer");
        assert!(commit_websocket_response(&mut st, &output));
        let parent = vec![
            json!({"role":"user","content":"first"}),
            output["response"]["output"][0].clone(),
        ];
        let deltas = vec![
            json!([{"role":"assistant","content":"inserted"},{"role":"user","content":"next"}]),
            json!([{"type":"function_call","id":"tool","call_id":"c","name":"lookup","arguments":"{}"},{"type":"function_call_output","call_id":"c","output":"found"}]),
        ];
        for delta in deltas {
            let expected = [parent.clone(), delta.as_array().unwrap().clone()].concat();
            for supplied in [delta, json!(expected)] {
                for id in [None, Some("first")] {
                    let mut req = json!({"type":"response.append","input":supplied});
                    if let Some(id) = id {
                        req["previous_response_id"] = json!(id);
                    }
                    let merged =
                        normalize_responses_websocket_request(req, &mut st.clone()).unwrap();
                    assert_eq!(merged["input"], json!(expected));
                }
            }
        }
    }

    #[tokio::test]
    async fn prewarm_pending_media_reaches_first_generation_then_only_bounded_history_is_kept() {
        let id = Uuid::new_v4().to_string();
        save_session_delta(
            id.clone(),
            None,
            vec![json!({"role":"user","content":"old"})],
            vec![],
            String::new(),
            "g".into(),
            "a".into(),
        )
        .await;
        for parent in [None, Some(id.as_str())] {
            let mut st = new_state();
            let mut pre = json!({"type":"response.create","generate":false,"model":"g","input":[{"role":"user","content":[{"type":"input_image","image_url":PNG}]}]});
            if let Some(parent) = parent {
                pre["previous_response_id"] = json!(parent);
            }
            let (_, done) = prepare_websocket_prewarm(&pre, &mut st, &HeaderMap::new(), "ws")
                .await
                .unwrap();
            assert_eq!(done["response"]["usage"]["total_tokens"], 0);
            let mut req=normalize_responses_websocket_request(json!({"type":"response.create","previous_response_id":done["response"]["id"],"input":"Describe it"}),&mut st).unwrap();
            prepare_websocket_generation(&mut req, &mut st, &HeaderMap::new(), "ws")
                .await
                .unwrap();
            let incoming = req["input"].as_array().unwrap();
            assert_eq!(incoming.len(), if parent.is_some() { 3 } else { 2 });
            assert_eq!(incoming[incoming.len() - 2]["content"][0]["image_url"], PNG);
            assert!(!st.last_request.as_ref().unwrap().to_string().contains(PNG));
            assert!(commit_websocket_response(
                &mut st,
                &terminal("generated", "red")
            ));
            let next=normalize_responses_websocket_request(json!({"type":"response.append","previous_response_id":"generated","input":"again"}),&mut st).unwrap();
            assert!(!next["input"].to_string().contains(PNG));
            assert!(next["input"]
                .to_string()
                .contains("[historical image omitted]"));
        }
    }

    #[tokio::test]
    async fn prewarm_inherits_first_generation_options_and_allows_overrides() {
        let pre = json!({"type":"response.create","model":"g","generate":false,"store":false,
            "max_output_tokens":51,"reasoning":{"effort":"low"},"text":{"format":{"type":"json_object"}}});
        for override_store in [false, true] {
            let mut st = new_state();
            let (_, done) = prepare_websocket_prewarm(&pre, &mut st, &HeaderMap::new(), "ws")
                .await
                .unwrap();
            let mut dispatch = json!({"type":"response.append","input":"hello","previous_response_id":done["response"]["id"]});
            if override_store {
                dispatch["store"] = json!(true);
                dispatch["max_output_tokens"] = json!(13);
            }
            let body = normalize_responses_websocket_request(dispatch, &mut st).unwrap();
            assert_eq!(responses_store_enabled(&body), override_store);
            assert_eq!(
                body["max_output_tokens"],
                if override_store { 13 } else { 51 }
            );
            assert_eq!(body["reasoning"], pre["reasoning"]);
            assert_eq!(body["text"], pre["text"]);
        }
    }

    #[tokio::test]
    async fn invalid_cursor_and_unsupported_compaction_do_not_consume_socket_history() {
        let prior = json!({"type":"response.create","model":"g","input":"important"});
        let mut st = new_state();
        normalize_responses_websocket_request(prior, &mut st).unwrap();
        assert!(commit_websocket_response(
            &mut st,
            &terminal("parent", "answer")
        ));
        for (key, bad) in [
            ("previous_response_id", json!(5)),
            (
                "input",
                json!([{"type":"compaction","encrypted_content":"opaque"},{"role":"user","content":"continue"}]),
            ),
        ] {
            let mut payload =
                json!({"type":"response.append","input":"next","previous_response_id":"parent"});
            payload[key] = bad;
            assert!(validate_responses_continuation(&payload).is_err());
            assert!(normalize_responses_websocket_request(payload, &mut st).is_err());
            assert_eq!(st.last_response_id, "parent");
            assert!(st.last_request.as_ref().unwrap()["input"]
                .to_string()
                .contains("important"));
        }
    }

    #[test]
    fn refusal_and_interleaved_media_keep_their_order_on_replay() {
        let request = json!({"input":[{"role":"user","content":[{"type":"input_text","text":"before"},
            {"type":"input_image","image_url":PNG},{"type":"input_text","text":"after"}]},
            {"role":"assistant","content":[{"type":"refusal","refusal":"declined"}]},{"role":"user","content":"next"}]});
        let result = convert_codex_to_openai_request(request);
        let msgs = result["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);
        let blocks = msgs[0]["content"].as_array().unwrap();
        assert_eq!(blocks[0]["text"], "before");
        assert_eq!(blocks[1]["image_url"]["url"], PNG);
        assert_eq!(blocks[2]["text"], "after");
        assert_eq!(msgs[1]["content"], "declined");
    }

    #[test]
    fn custom_data_url_argument_survives_borrowed_and_owned_storage() {
        let call = json!({"type":"custom_tool_call","name":"inspect","call_id":"a","input":PNG});
        assert_eq!(history_without_inline_media(&call), call);
        assert_eq!(into_history_without_inline_media(call.clone()), call);
    }

    #[tokio::test]
    async fn responses_stream_and_json_take_wire_types_from_qualified_declarations() {
        use crate::proxy::mappers::openai::{
            models::OpenAIResponse, request::ResponsesToolKinds,
            streaming::create_codex_sse_stream_with_context,
        };
        let raw = "{\"still\":\"raw text\"}";
        let tools = json!([{"type":"custom","name":"mytool"},{"type":"custom","name":"flat__lookup"},{"type":"function","name":"shell","parameters":{"type":"object","properties":{"input":{"type":"string"}}}},{"type":"namespace","name":"scope","tools":[{"type":"custom","name":"custom"},{"type":"function","name":"apply_patch","parameters":{"type":"object","properties":{"input":{"type":"string"}}}}]}]);
        let kinds = ResponsesToolKinds::from_tools(tools.as_array().map(|v| v.as_slice()));
        for (name, args, is_custom) in [
            ("mytool", json!({"input":raw}), true),
            ("flat__lookup", json!({"input":raw}), true),
            ("shell", json!({"input":raw}), false),
            ("scope__custom", json!({"input":raw}), true),
            ("scope__apply_patch", json!({"input":raw}), false),
            ("mytool", json!({"input":raw,"another":[1,2]}), false),
            ("scope__custom", json!({"input":7}), false),
            ("mytool", json!({"other":"untouched"}), false),
        ] {
            let id = Uuid::new_v4().to_string();
            let frame = json!({"candidates":[{"content":{"parts":[{"functionCall":{"id":id,"name":name,"args":args}}]},"finishReason":"STOP"}]});
            let source = create_codex_sse_stream_with_context(
                Box::pin(futures::stream::iter(vec![Ok::<Bytes, String>(
                    Bytes::from(format!("data: {frame}\n\n")),
                )])),
                "g".into(),
                id.clone(),
                1,
                0,
                id.clone(),
                None,
                false,
                None,
                kinds.clone(),
            );
            let events: Vec<_> = websocket_response_events(source).collect().await;
            let terminal = &events.last().unwrap()["response"];
            assert_eq!(terminal["status"], "completed");
            let stream_item = &terminal["output"][0];
            let chat:OpenAIResponse=serde_json::from_value(json!({"id":"chat","object":"chat.completion","created":1,"model":"g","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","tool_calls":[{"id":id,"type":"function","function":{"name":name,"arguments":args.to_string()}}]}}]})).unwrap();
            let nonstream = convert_chat_response_to_responses_with_tools(&chat, false, &kinds);
            let item = &nonstream["output"][0];
            assert_eq!(nonstream["status"], "completed");
            for k in ["type", "name", "namespace", "call_id", "input", "arguments"] {
                assert_eq!(stream_item[k], item[k], "{name}/{k}");
            }
            assert_eq!(responses_item_tool_name(item).as_deref(), Some(name));
            if name == "flat__lookup" {
                assert!(item.get("namespace").is_none());
                assert_eq!(item["name"], "flat__lookup");
            }
            if is_custom {
                assert_eq!(item["type"], "custom_tool_call");
                assert_eq!(item["input"], raw);
                assert_eq!(
                    events
                        .iter()
                        .find(|e| e["type"] == "response.custom_tool_call_input.delta")
                        .unwrap()["delta"],
                    raw
                );
                assert_eq!(
                    events
                        .iter()
                        .find(|e| e["type"] == "response.custom_tool_call_input.done")
                        .unwrap()["input"],
                    raw
                );
            } else {
                assert_eq!(item["type"], "function_call");
                assert_eq!(item["arguments"], args.to_string());
            }
            let result = if is_custom {
                "custom_tool_call_output"
            } else {
                "function_call_output"
            };
            let replay = convert_codex_to_openai_request(
                json!({"input":[{"role":"user","content":"use tool"},item,{"type":result,"call_id":id,"output":"done"}]}),
            );
            let f = &replay["messages"][1]["tool_calls"][0]["function"];
            assert_eq!(f["name"], name);
            assert_eq!(
                serde_json::from_str::<Value>(f["arguments"].as_str().unwrap()).unwrap(),
                args
            );
        }
    }
    #[tokio::test]
    async fn responses_custom_tool_json_and_stream_use_the_same_raw_input() {
        use crate::proxy::mappers::openai::{
            models::OpenAIResponse, streaming::create_codex_sse_stream,
        };
        let raw = "{\"nested\":true}\nDo not reparse freeform input.";
        for (name, args, custom) in [
            ("scope__apply_patch", json!({"input":raw}), true),
            ("apply_patch_v2", json!({"input":""}), true),
            ("shell", json!({"input":raw}), true),
            ("shell", json!({"command":["pwd"]}), false),
            ("apply_patch", json!({"input":raw,"option":true}), false),
            ("lookup", json!({"input":raw}), false),
        ] {
            let id = Uuid::new_v4().to_string();
            let f = json!({"candidates":[{"content":{"parts":[{"functionCall":{"name":name,"id":id,"args":args}}]},"finishReason":"STOP"}]});
            let source = create_codex_sse_stream(
                Box::pin(futures::stream::iter(vec![Ok::<Bytes, String>(
                    Bytes::from(format!("data: {f}\n\n")),
                )])),
                "g".into(),
                id.clone(),
                1,
                0,
                id.clone(),
                None,
                false,
            );
            let events: Vec<_> = websocket_response_events(source).collect().await;
            let item = &events.last().unwrap()["response"]["output"][0];
            let resp:OpenAIResponse=serde_json::from_value(json!({"id":"chat","object":"chat.completion","created":1,"model":"g","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","tool_calls":[{"id":id,"type":"function","function":{"name":name,"arguments":args.to_string()}}]}}]})).unwrap();
            let json_resp = convert_chat_response_to_responses(&resp, false);
            assert_eq!(json_resp["status"], "completed");
            let json_item = &json_resp["output"][0];
            for key in ["type", "call_id", "name", "namespace", "arguments", "input"] {
                assert_eq!(item[key], json_item[key], "{name}/{key}");
            }
            let restored = if custom {
                let text = args["input"].as_str().unwrap();
                assert_eq!(item["type"], "custom_tool_call");
                assert_eq!(item["input"], text);
                assert_eq!(
                    events
                        .iter()
                        .find(|e| e["type"] == "response.custom_tool_call_input.delta")
                        .unwrap()["delta"],
                    text
                );
                assert_eq!(
                    events
                        .iter()
                        .find(|e| e["type"] == "response.custom_tool_call_input.done")
                        .unwrap()["input"],
                    text
                );
                json!({"input":item["input"]})
            } else {
                assert_eq!(item["type"], "function_call");
                serde_json::from_str::<Value>(item["arguments"].as_str().unwrap()).unwrap()
            };
            assert_eq!(restored, args);
        }
    }
}

#[cfg(test)]
mod r19_responses_regressions {
    use super::*;

    fn state() -> WebsocketSessionState {
        WebsocketSessionState {
            last_request: None,
            last_response_output: json!([]),
            last_response_id: String::new(),
            tool_call_cache: Default::default(),
        }
    }

    #[test]
    fn lossy_tool_schema_is_accepted_but_resource_limits_remain_enforced() {
        let schema = json!({"oneOf":[{"type":"string"},{"type":"string","minLength":2}]});
        for tools in [
            json!([{"type":"function","function":{"name":"check","parameters":schema}}]),
            json!([{"type":"namespace","name":"repo","tools":[{"type":"function","name":"check","parameters":schema}]}]),
        ] {
            assert!(validate_openai_tool_schemas(tools.as_array().map(Vec::as_slice)).is_ok());
            let body = json!({"type":"response.create","model":"g","input":"hi","tools":tools});
            assert!(validate_responses_continuation(&body).is_ok());
            assert!(normalize_responses_websocket_request(body, &mut state()).is_ok());
        }
        let schema = json!({"type":"object", "description":"x".repeat(crate::proxy::common::json_schema::MAX_SCHEMA_BYTES + 1)});
        for tools in [
            json!([{"type":"function","function":{"name":"check","parameters":schema}}]),
            json!([{"type":"namespace","name":"repo","tools":[{"type":"function","name":"check","parameters":schema}]}]),
        ] {
            assert!(validate_openai_tool_schemas(tools.as_array().map(Vec::as_slice)).is_err());
            let body = json!({"type":"response.create","model":"g","input":"hi","tools":tools});
            assert_eq!(
                validate_responses_continuation(&body).unwrap_err().0,
                "unsupported_tool_schema"
            );
            assert!(normalize_responses_websocket_request(body, &mut state()).is_err());
        }
    }

    #[test]
    fn ambiguous_namespace_tools_are_rejected_in_both_declaration_orders() {
        let flat = json!({"type":"function","name":"repo__lookup","parameters":{"type":"object"}});
        let ns =
            json!({"type":"namespace","name":"repo","tools":[{"type":"custom","name":"lookup"}]});
        for tools in [json!([flat, ns]), json!([ns, flat])] {
            let body = json!({"type":"response.create","model":"g","input":"hi","tools":tools});
            assert_eq!(
                validate_responses_continuation(&body).unwrap_err().0,
                "ambiguous_tool_name"
            );
            assert!(normalize_responses_websocket_request(body, &mut state()).is_err());
        }
        assert!(validate_responses_continuation(&json!({"tools":[{"type":"function","name":"plain"},{"type":"namespace","name":"repo","tools":[{"type":"custom","name":"lookup"}]}]})).is_ok());
    }

    #[test]
    fn unresolved_files_and_reasoning_without_plaintext_fail_before_history_is_consumed() {
        for part in [
            json!({"type":"input_file","file_data":"data:application/pdf;base64,not-base64"}),
            json!({"type":"input_file","file_id":"file-one"}),
            json!({"type":"input_file","file_url":"file:///private/file.pdf"}),
            json!({"type":"input_image","file_id":"image-one"}),
        ] {
            for item in [
                json!({"role":"user","content":[part,{"type":"input_text","text":"what?"}]}),
                json!({"type":"function_call_output","call_id":"call-one","output":[part]}),
            ] {
                let body = json!({"type":"response.create","model":"g","input":[item]});
                assert_eq!(
                    validate_responses_continuation(&body).unwrap_err().0,
                    "unsupported_file_input"
                );
                assert!(normalize_responses_websocket_request(body, &mut state()).is_err());
            }
        }
        for encrypted in ["opaque-A", "opaque-B"] {
            assert_eq!(
                validate_responses_continuation(
                    &json!({"input":[{"type":"reasoning","encrypted_content":encrypted}]})
                )
                .unwrap_err()
                .0,
                "unsupported_reasoning"
            );
        }
        assert!(validate_responses_continuation(&json!({"input":[{"type":"reasoning","summary":[{"type":"summary_text","text":"plain"}]},{"type":"function_call","name":"business","arguments":{"type":"input_file","encrypted_content":"business data"}}]})).is_ok());
    }

    #[tokio::test]
    async fn token_alias_overrides_prewarmed_limit_without_duplicate_fields() {
        let aliases = [
            "max_tokens",
            "max_completion_tokens",
            "maxCompletionTokens",
            "maxTokens",
            "max_output_tokens",
            "maxOutputTokens",
        ];
        for previous in aliases {
            for next in aliases {
                let mut s = state();
                let mut pre = json!({"type":"response.create","model":"g","generate":false});
                pre[previous] = json!(100);
                let (_, done) =
                    prepare_websocket_prewarm(&pre, &mut s, &HeaderMap::new(), "socket")
                        .await
                        .unwrap();
                let mut event = json!({"type":"response.create","previous_response_id":done["response"]["id"],"input":"hi"});
                event[next] = json!(5);
                let normalized = normalize_responses_websocket_request(event, &mut s).unwrap();
                let typed: crate::proxy::mappers::openai::models::OpenAIRequest =
                    serde_json::from_value(normalized).unwrap();
                assert_eq!(typed.max_tokens, Some(5), "{previous} -> {next}");
            }
        }
    }

    #[test]
    fn representable_documents_reach_gemini_with_their_source_and_mime() {
        for file in [
            json!({"type":"input_file","file_data":"data:application/pdf;base64,JVBERi0xLjQKJSVFT0Y="}),
            json!({"type":"input_file","filename":"report.pdf","file_data":"JVBERi0xLjQKJSVFT0Y="}),
            json!({"type":"input_file","file_url":"https://example.invalid/report.pdf"}),
        ] {
            let body = json!({"model":"gemini-3.7-flash-high","input":[{"role":"user","content":[file,{"type":"input_text","text":"summarize"}]}]});
            assert!(validate_responses_continuation(&body).is_ok());
            let converted = convert_codex_to_openai_request(body);
            let typed: OpenAIRequest = serde_json::from_value(converted).unwrap();
            validate_openai_file_inputs(&typed).unwrap();
            let (mapped, _, _, _) =
                crate::proxy::mappers::openai::request::transform_openai_request_with_session(
                    &typed,
                    "project",
                    "gemini-3.7-flash-high",
                    None,
                    "session",
                    Some("session"),
                    true,
                    false,
                )
                .unwrap();
            let part = &mapped["request"]["contents"][0]["parts"][0];
            if let Some(uri) = file["file_url"].as_str() {
                assert_eq!(
                    *part,
                    json!({"fileData":{"fileUri":uri,"mimeType":"application/pdf"}})
                );
            } else {
                assert_eq!(part["inlineData"]["mimeType"], "application/pdf");
                assert_eq!(part["inlineData"]["data"], "JVBERi0xLjQKJSVFT0Y=");
            }
        }
    }

    #[test]
    fn websocket_item_references_preserve_parent_answers() {
        let answer = json!({"type":"message","id":"answer","role":"assistant","content":"seven"});
        let mut s = state();
        s.last_request = Some(
            json!({"model":"g","input":[{"role":"user","content":"first"},answer,{"role":"user","content":"second"}]}),
        );
        s.last_response_output =
            json!([{"type":"message","id":"last","role":"assistant","content":"done"}]);
        s.last_response_id = "parent".into();
        let normalized=normalize_responses_websocket_request(json!({"type":"response.append","previous_response_id":"parent","input":[{"type":"item_reference","id":"answer"},{"role":"user","content":"explain"}]}),&mut s).unwrap();
        assert!(normalized["input"].as_array().unwrap().contains(&answer));
        assert!(!normalized["input"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["type"] == "item_reference"));
        let before = s.last_request.clone();
        assert!(normalize_responses_websocket_request(
            json!({"type":"response.append","input":[{"type":"item_reference","id":"missing"}]}),
            &mut s
        )
        .is_err());
        assert_eq!(s.last_request, before);
    }
}

#[cfg(test)]
mod responses_plaintext_compatibility_tests {
    use super::*;
    use crate::proxy::http_session_store::{
        prepare_session_input_with_storage, save_session_delta,
    };

    fn state() -> WebsocketSessionState {
        WebsocketSessionState {
            last_request: None,
            last_response_output: json!([]),
            last_response_id: String::new(),
            tool_call_cache: Default::default(),
        }
    }

    fn assert_nonthinking_model_keeps_summary(body: Value) {
        let request: OpenAIRequest = serde_json::from_value(body).unwrap();
        let (mapped, _, _, _) =
            crate::proxy::mappers::openai::request::transform_openai_request_with_session(
                &request,
                "project",
                "gemini-2.5-flash",
                None,
                "plaintext-test",
                None,
                true,
                false,
            )
            .unwrap();
        let matches: Vec<_> = mapped["request"]["contents"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|turn| turn["parts"].as_array().unwrap())
            .filter(|part| {
                part["text"]
                    .as_str()
                    .is_some_and(|text| text.contains("Readable context"))
            })
            .collect();
        assert_eq!(matches.len(), 1, "summary must be sent once: {mapped}");
        assert_eq!(
            mapped["request"]["contents"]
                .to_string()
                .matches("Readable context")
                .count(),
            1
        );
        assert_ne!(matches[0]["thought"], true);
        assert!(matches[0].get("thoughtSignature").is_none());
    }

    #[test]
    fn plaintext_reasoning_survives_foreign_encryption_and_chat_conversion() {
        for (key, value) in [
            (
                "summary",
                json!([{"type":"summary_text","text":"Readable context"}]),
            ),
            ("text", json!("Readable context")),
            (
                "content",
                json!([{"type":"reasoning_text","text":"Readable context"}]),
            ),
        ] {
            let mut reasoning =
                json!({"type":"reasoning","id":"rs-one","encrypted_content":"foreign-ciphertext"});
            reasoning[key] = value;
            let mut request = json!({"model":"g","input":[{"role":"user","content":"First question"},reasoning,{"role":"assistant","content":"First answer"},{"role":"user","content":"Followup"}]});
            normalize_responses_plaintext(&mut request).unwrap();
            assert!(request["input"][1].get("encrypted_content").is_none());
            // HTTP and Chat's Responses compatibility adapter share this mapping.
            let expected = responses_reasoning_message(&request["input"][1]);
            let mapped = convert_codex_to_openai_request(request);
            assert_eq!(mapped["messages"][1], expected);
            assert_eq!(expected["content"], "Readable context");
            assert!(expected.get("reasoning_content").is_none());
            assert_eq!(mapped["messages"][3]["content"], "Followup");
            assert_nonthinking_model_keeps_summary(mapped);
        }
    }

    #[test]
    fn plaintext_summary_precedes_text_and_business_fields_are_untouched() {
        let mut body = json!({"input":[{"type":"reasoning","summary":[{"type":"summary_text","text":"preferred"}],"text":"fallback","encrypted_content":"opaque"},{"type":"function_call","name":"business","arguments":{"type":"compaction_summary","text":"business","encrypted_content":"business cipher"}}]});
        let business = body["input"][1].clone();
        normalize_responses_plaintext(&mut body).unwrap();
        assert_eq!(
            responses_reasoning_message(&body["input"][0])["content"],
            "preferred"
        );
        assert_eq!(body["input"][1], business);
    }

    #[tokio::test]
    async fn plaintext_reasoning_survives_websocket_prewarm_and_generation() {
        let input = json!([
            {"role":"user","content":"First question"},
            {"type":"reasoning","summary":[{"type":"summary_text","text":"Readable context"}],"encrypted_content":"foreign-ciphertext"},
            {"role":"assistant","content":"First answer"}
        ]);
        let mut socket = state();
        let initial = normalize_responses_websocket_request(
            json!({"type":"response.create","model":"g","input":input}),
            &mut socket,
        )
        .unwrap();
        assert!(initial["input"][1].get("encrypted_content").is_none());

        let mut socket = state();
        let (_, completed) = prepare_websocket_prewarm(
            &json!({"type":"response.create","model":"g","generate":false,"store":false,"input":input}),
            &mut socket, &HeaderMap::new(), "connection",
        ).await.unwrap();
        assert!(!socket
            .last_request
            .as_ref()
            .unwrap()
            .to_string()
            .contains("foreign-ciphertext"));
        let next = normalize_responses_websocket_request(
            json!({"type":"response.create","previous_response_id":completed["response"]["id"],"input":"Followup"}),
            &mut socket,
        ).unwrap();
        let converted = convert_codex_to_openai_request(next);
        assert_eq!(converted["messages"][1]["content"], "Readable context");
        assert_eq!(converted["messages"][3]["content"], "Followup");
        assert_nonthinking_model_keeps_summary(converted);
    }

    #[test]
    fn native_signed_reasoning_keeps_its_existing_thought_path() {
        use base64::Engine;
        let signature = base64::engine::general_purpose::STANDARD.encode([0x12; 48]);
        let item = json!({"type":"reasoning","summary":[{"type":"summary_text","text":"Native thought"}],"thoughtSignature":signature});
        let message = responses_reasoning_message(&item);
        assert_eq!(message["reasoning_content"], "Native thought");
        assert_eq!(message["thoughtSignature"], signature);
        assert_eq!(message["content"], "");
        for signature in [
            "external-token",
            crate::proxy::thinking_store::SENTINEL_SIGNATURE,
        ] {
            let mut foreign = item.clone();
            foreign["thoughtSignature"] = json!(signature);
            let message = responses_reasoning_message(&foreign);
            assert_eq!(message["content"], "Native thought");
            assert!(message.get("reasoning_content").is_none());
            assert!(message.get("thoughtSignature").is_none());
        }
    }

    #[tokio::test]
    async fn plaintext_compaction_resets_http_history_and_reaches_model_input() {
        for summary in [
            json!({"text":"Compacted context"}),
            json!({"summary":[{"type":"summary_text","text":"Compacted context"}]}),
            json!({"content":[{"type":"input_text","text":"Compacted context"}],"encrypted_content":"optional-opaque-backup"}),
        ] {
            let id = Uuid::new_v4().to_string();
            save_session_delta(
                id.clone(),
                None,
                vec![json!({"role":"user","content":"stale history"})],
                vec![],
                String::new(),
                "g".into(),
                "route".into(),
            )
            .await;
            let mut marker = summary;
            marker["type"] = json!("compaction_summary");
            marker["status"] = json!("completed");
            let mut body = json!({"model":"gemini-3.7-flash-high","previous_response_id":id,"input":[marker,{"role":"user","content":"Next question"}]});
            normalize_responses_plaintext(&mut body).unwrap();
            let once = body.clone();
            normalize_responses_plaintext(&mut body).unwrap();
            assert_eq!(
                body, once,
                "HTTP must accept already normalized WS input without repeating the summary"
            );
            let (parent, _) = load_responses_parent(Some(&id), true)
                .await
                .unwrap()
                .unwrap();
            let prepared = prepare_session_input_with_storage(
                parent.input_items,
                take_responses_input_items(&mut body),
                &Default::default(),
                true,
            );
            assert!(prepared.reset_parent);
            assert_eq!(
                prepared.merged,
                vec![
                    json!({"role":"user","content":"Compacted context"}),
                    json!({"role":"user","content":"Next question"})
                ]
            );
            body["input"] = json!(prepared.merged);
            let request: OpenAIRequest =
                serde_json::from_value(convert_codex_to_openai_request(body)).unwrap();
            let (mapped, _, _, _) =
                crate::proxy::mappers::openai::request::transform_openai_request_with_session(
                    &request,
                    "project",
                    "gemini-3.7-flash-high",
                    None,
                    "session",
                    Some("session"),
                    true,
                    false,
                )
                .unwrap();
            let content = mapped["request"]["contents"].to_string();
            assert!(content.contains("Compacted context"));
            assert!(content.contains("Next question"));
            assert!(!content.contains("stale history"));
            assert!(!content.contains("optional-opaque-backup"));
        }
    }

    #[tokio::test]
    async fn plaintext_compaction_replaces_socket_history_and_survives_prewarm() {
        let input = json!([{"type":"compaction_summary","status":"completed","text":"Compacted context"},{"role":"user","content":"Current question"}]);
        let mut socket = state();
        normalize_responses_websocket_request(
            json!({"type":"response.create","model":"g","input":"stale history"}),
            &mut socket,
        )
        .unwrap();
        socket.last_response_id = "local-parent".into();
        socket.last_response_output = json!([{"role":"assistant","content":"stale answer"}]);
        let body = normalize_responses_websocket_request(
            json!({"type":"response.append","previous_response_id":"local-parent","input":input}),
            &mut socket,
        )
        .unwrap();
        assert_eq!(
            body["input"],
            json!([{"role":"user","content":"Compacted context"},{"role":"user","content":"Current question"}])
        );

        let parent = Uuid::new_v4().to_string();
        save_session_delta(
            parent.clone(),
            None,
            vec![json!({"role":"user","content":"stale HTTP history"})],
            vec![],
            String::new(),
            "g".into(),
            "route".into(),
        )
        .await;
        let mut socket = state();
        let prewarm = json!({"type":"response.create","model":"g","generate":false,"store":false,"previous_response_id":parent,"input":input});
        let (_, completed) =
            prepare_websocket_prewarm(&prewarm, &mut socket, &HeaderMap::new(), "connection")
                .await
                .unwrap();
        assert_eq!(
            socket.last_request.as_ref().unwrap()["input"],
            body["input"]
        );
        let next = normalize_responses_websocket_request(json!({"type":"response.create","previous_response_id":completed["response"]["id"],"input":"Followup"}), &mut socket).unwrap();
        assert_eq!(next["input"].as_array().unwrap().len(), 3);
        assert_eq!(next["input"][0]["content"], "Compacted context");
        assert!(!next.to_string().contains("stale HTTP history"));
    }

    #[tokio::test]
    async fn opaque_context_rejection_preserves_pending_socket_history() {
        for item in [
            json!({"type":"reasoning","summary":[],"encrypted_content":"opaque"}),
            json!({"type":"reasoning","summary":[{"type":"summary_text","text":"  "}],"encrypted_content":"opaque"}),
            json!({"type":"compaction_summary","encrypted_content":"opaque","status":"completed"}),
            json!({"type":"compaction_summary","content":{"encrypted_content":"opaque"}}),
            json!({"type":"compaction","encrypted_content":"opaque"}),
        ] {
            let body = json!({"type":"response.create","model":"g","input":[item,{"role":"user","content":"Continue"}]});
            let mut normalized = body.clone();
            assert!(normalize_responses_plaintext(&mut normalized).is_err());
            assert_eq!(normalized, body);
            let mut socket = state();
            assert!(
                prepare_websocket_prewarm(&body, &mut socket, &HeaderMap::new(), "connection")
                    .await
                    .is_err()
            );
            assert!(socket.last_request.is_none());
            assert!(normalize_responses_websocket_request(body, &mut socket).is_err());
            assert!(socket.last_request.is_none());
        }
        let mut marker = json!({"input":[{"type":"compaction_summary","status":"completed","metadata":{"tokens":100}},{"role":"user","content":"Complete replacement transcript"}]});
        normalize_responses_plaintext(&mut marker).unwrap();
        assert_eq!(marker["input"][0], json!({"type":"compaction_summary"}));
    }
}
