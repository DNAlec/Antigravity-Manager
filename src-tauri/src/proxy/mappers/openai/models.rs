// OpenAI 数据模型

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{json, Value};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(remote = "Self")]
pub struct OpenAIRequest {
    pub model: String,
    #[serde(default)]
    pub messages: Vec<OpenAIMessage>,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub n: Option<u32>, // [NEW] 支持多候选结果数量
    #[serde(
        default,
        rename = "max_tokens",
        alias = "max_completion_tokens",
        alias = "maxCompletionTokens",
        alias = "maxTokens",
        alias = "max_output_tokens",
        alias = "maxOutputTokens"
    )]
    pub max_tokens: Option<u32>,
    pub temperature: Option<f64>,
    #[serde(rename = "top_p")]
    pub top_p: Option<f64>,
    #[serde(rename = "presence_penalty")]
    pub presence_penalty: Option<f64>,
    #[serde(rename = "frequency_penalty")]
    pub frequency_penalty: Option<f64>,
    pub seed: Option<i64>,
    pub stop: Option<Value>,
    pub response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(rename = "tool_choice")]
    pub tool_choice: Option<Value>,
    #[serde(rename = "parallel_tool_calls")]
    pub parallel_tool_calls: Option<bool>,
    // Codex proprietary fields
    pub instructions: Option<String>,
    pub input: Option<Value>,
    // [NEW] Image generation parameters (for Chat API compatibility)
    #[serde(default)]
    pub size: Option<String>,
    #[serde(default)]
    pub quality: Option<String>,
    #[serde(default, rename = "personGeneration")]
    pub person_generation: Option<String>,
    // [NEW] Thinking/Extended Thinking 支持 (兼容 Anthropic/Claude 协议)
    #[serde(default)]
    pub thinking: Option<ThinkingConfig>,
    // Codex Responses API reasoning controls. Applied only to tiered Flash models.
    #[serde(default)]
    pub reasoning: Option<ReasoningConfig>,
    // [NEW] OpenAI o1/o3/o4 reasoning_effort (and aliases) support
    #[serde(
        default,
        rename = "reasoning_effort",
        alias = "reasoningEffort",
        alias = "thinkingLevel",
        alias = "thinking_level"
    )]
    pub reasoning_effort: Option<String>,
    // [NEW] Direct imageSize support (for Gemini native parameter)
    #[serde(default, rename = "imageSize")]
    pub image_size: Option<String>,
    /// Client/session id used to store and restore full thinking blocks.
    #[serde(default, rename = "session_id")]
    pub session_id: Option<String>,
    // [NEW] OpenAI stream_options support (include_usage gate)
    #[serde(default, rename = "stream_options")]
    pub stream_options: Option<StreamOptions>,
    // Responses API Structured Outputs / text config (e.g. text.format)
    #[serde(default)]
    pub text: Option<Value>,
}

impl<'de> Deserialize<'de> for OpenAIRequest {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut value = Value::deserialize(deserializer)?;
        normalize_chat_request(&mut value).map_err(serde::de::Error::custom)?;
        // The remote derive retains the ordinary typed validation after normalization.
        Self::deserialize(value).map_err(serde::de::Error::custom)
    }
}

impl Serialize for OpenAIRequest {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        Self::serialize(self, serializer)
    }
}

fn normalize_chat_request(value: &mut Value) -> Result<(), String> {
    let Some(body) = value.as_object_mut() else {
        return Err("Chat request must be an object".into());
    };
    if body.get("stream").is_some_and(Value::is_null) {
        body.remove("stream");
    }
    let mut token_limit = None;
    for key in [
        "max_tokens",
        "max_completion_tokens",
        "maxCompletionTokens",
        "maxTokens",
        "max_output_tokens",
        "maxOutputTokens",
    ] {
        if let Some(value) = body.remove(key).filter(|value| !value.is_null()) {
            if token_limit
                .as_ref()
                .is_some_and(|previous| previous != &value)
            {
                return Err("Conflicting non-null maximum token limits".into());
            }
            token_limit = Some(value);
        }
    }
    if let Some(limit) = token_limit {
        body.insert("max_tokens".into(), limit);
    }

    if let Some(functions) = body.remove("functions").filter(|v| !v.is_null()) {
        let functions = functions.as_array().ok_or("functions must be an array")?;
        let mut tools = Vec::with_capacity(functions.len());
        for function in functions {
            required_function_name(function)?;
            if let Some(schema) = function.get("parameters") {
                crate::proxy::common::json_schema::validate_tool_schema(schema)?;
            }
            tools.push(json!({"type":"function", "function":function}));
        }
        if body
            .get("tools")
            .is_some_and(|value| !value.is_null() && value != &json!([]))
        {
            return Err("Specify either functions or tools, not both".into());
        }
        body.insert("tools".into(), Value::Array(tools));
    }
    if let Some(choice) = body.remove("function_call").filter(|v| !v.is_null()) {
        let choice = match choice.as_str() {
            Some("auto" | "none") => choice,
            Some(_) => {
                return Err("function_call must be auto, none, or an object with a name".into())
            }
            None => {
                json!({"type":"function", "function":{"name":required_function_name(&choice)?}})
            }
        };
        if body
            .get("tool_choice")
            .is_some_and(|v| !v.is_null() && v != &choice)
        {
            return Err("Conflicting function_call and tool_choice".into());
        }
        body.insert("tool_choice".into(), choice);
    }

    let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) else {
        return Ok(());
    };
    let mut used_ids: std::collections::HashSet<String> = messages
        .iter()
        .filter_map(|message| message.get("tool_calls").and_then(Value::as_array))
        .flatten()
        .filter_map(|call| call.get("id").and_then(Value::as_str))
        .map(str::to_owned)
        .collect();
    let mut pending: Vec<(String, String)> = Vec::new();
    for (index, message) in messages.iter_mut().enumerate() {
        let Some(message) = message.as_object_mut() else {
            continue;
        };
        if let Some(function) = message.remove("function_call").filter(|v| !v.is_null()) {
            if message.get("role").and_then(Value::as_str) != Some("assistant") {
                return Err("Historical function_call requires an assistant message".into());
            }
            let parsed: ToolFunction =
                serde_json::from_value(function.clone()).map_err(|e| e.to_string())?;
            required_function_name(&function)?;
            let arguments: Value = serde_json::from_str(&parsed.arguments)
                .map_err(|e| format!("Invalid function_call arguments: {e}"))?;
            if !arguments.is_object() {
                return Err("function_call arguments must encode an object".into());
            }
            if message
                .get("tool_calls")
                .is_some_and(|v| !v.is_null() && v != &json!([]))
            {
                return Err(
                    "Assistant message cannot contain both function_call and tool_calls".into(),
                );
            }
            let mut id = format!("call_legacy_{index}");
            while !used_ids.insert(id.clone()) {
                id.push('_');
            }
            pending.push((parsed.name, id.clone()));
            message.insert(
                "tool_calls".into(),
                json!([{"id":id, "type":"function", "function":function}]),
            );
        }
        if message.get("role").and_then(Value::as_str) == Some("function") {
            let name = message
                .get("name")
                .and_then(Value::as_str)
                .ok_or("Function result is missing its name")?;
            let Some(position) = pending
                .iter()
                .rposition(|(pending_name, _)| pending_name == name)
            else {
                // A trimmed history can start with a result. Preserve the existing
                // orphan-result handling instead of inventing a call association.
                continue;
            };
            let (_, id) = pending.remove(position);
            if message
                .get("tool_call_id")
                .is_some_and(|value| !value.is_null() && value.as_str() != Some(id.as_str()))
            {
                return Err("Function result has a conflicting tool_call_id".into());
            }
            message.insert("role".into(), json!("tool"));
            message.insert("tool_call_id".into(), json!(id));
        }
    }
    Ok(())
}

fn required_function_name(function: &Value) -> Result<&str, String> {
    function
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.trim().is_empty())
        .ok_or_else(|| "Function is missing a non-empty name".into())
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

/// Thinking 配置 (兼容 Anthropic 和 OpenAI 扩展协议)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ThinkingConfig {
    #[serde(rename = "type")]
    pub thinking_type: Option<String>, // "enabled", "disabled", or "adaptive"
    #[serde(
        default,
        rename = "budget_tokens",
        alias = "budgetTokens",
        alias = "max_tokens",
        alias = "maxTokens"
    )]
    pub budget_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>, // "low", "high", or "max"
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReasoningConfig {
    pub effort: Option<String>,
    #[serde(
        default,
        rename = "max_tokens",
        alias = "maxTokens",
        alias = "budget_tokens",
        alias = "budgetTokens"
    )]
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseFormatJSONSchema {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseFormat {
    pub r#type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<ResponseFormatJSONSchema>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum OpenAIContent {
    String(String),
    Array(Vec<OpenAIContentBlock>),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum OpenAIContentBlock {
    #[serde(rename = "text", alias = "input_text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: OpenAIImageUrl },
    #[serde(rename = "file")]
    File { file: OpenAIInputFile },
    #[serde(rename = "audio_url")]
    AudioUrl { audio_url: AudioUrlContent },
    // [NEW] OpenAI 官方多模态音频入参: {"type":"input_audio","input_audio":{"data":"<base64>","format":"wav"}}
    #[serde(rename = "input_audio", alias = "audio")]
    InputAudio { input_audio: OpenAIInputAudio },
    // [NEW] 视频多模态输入: {"type":"video_url","video_url":{"url":"..."}}
    #[serde(rename = "video_url")]
    VideoUrl { video_url: OpenAIVideoUrl },
}

/// A representable Responses document, independent of an external Files API.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OpenAIInputFile {
    pub file_data: Option<String>,
    pub file_url: Option<String>,
    pub filename: Option<String>,
    #[serde(default, alias = "mimeType")]
    pub mime_type: Option<String>,
}

impl OpenAIInputFile {
    pub fn to_gemini_part(&self) -> Option<Value> {
        let inferred = || {
            let source = self
                .filename
                .as_deref()
                .or(self.file_url.as_deref())
                .unwrap_or_default();
            let path = source
                .split(['?', '#'])
                .next()
                .unwrap_or(source)
                .to_ascii_lowercase();
            match path.rsplit('.').next().unwrap_or_default() {
                "pdf" => "application/pdf",
                "txt" | "md" => "text/plain",
                "csv" => "text/csv",
                "json" => "application/json",
                "html" | "htm" => "text/html",
                "png" => "image/png",
                "jpg" | "jpeg" => "image/jpeg",
                _ => "application/octet-stream",
            }
        };
        match (self.file_data.as_deref(), self.file_url.as_deref()) {
            (Some(source), None) if !source.is_empty() => {
                let (mime, data) = if let Some(source) = source.strip_prefix("data:") {
                    let (header, data) = source.split_once(',')?;
                    let mime = header.strip_suffix(";base64")?;
                    if mime.is_empty() {
                        return None;
                    }
                    (mime, data)
                } else {
                    (self.mime_type.as_deref().unwrap_or_else(inferred), source)
                };
                if data.is_empty() {
                    return None;
                }
                Some(json!({"inlineData":{"mimeType":mime,"data":data}}))
            }
            (None, Some(uri)) => {
                let parsed = url::Url::parse(uri).ok()?;
                if !matches!(parsed.scheme(), "http" | "https" | "gs")
                    || parsed.host_str().is_none()
                {
                    return None;
                }
                Some(
                    json!({"fileData":{"fileUri":uri,"mimeType":self.mime_type.as_deref().unwrap_or_else(inferred)}}),
                )
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OpenAIVideoUrl {
    pub url: String,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "mimeType",
        alias = "mime_type",
        alias = "format"
    )]
    pub mime_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OpenAIImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AudioUrlContent {
    pub url: String,
    /// 可选的显式 MIME/格式 (如 "audio/wav" 或 "wav")，缺省时从 data URL / 扩展名推断
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "mimeType",
        alias = "mime_type",
        alias = "format"
    )]
    pub mime_type: Option<String>,
}

/// OpenAI `input_audio` 内容块: base64 音频 + 格式标识
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OpenAIInputAudio {
    /// base64 编码的音频数据 (也兼容传入 data: URL)
    pub data: String,
    /// "wav" | "mp3" | "m4a" | "ogg" | "flac" | "aiff" ... 亦接受完整 MIME
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "mimeType",
        alias = "mime_type"
    )]
    pub format: Option<String>,
}

impl OpenAIInputAudio {
    /// 归一化后的 Gemini MIME 类型
    pub fn mime_type(&self) -> String {
        crate::proxy::audio::normalize_audio_mime(self.format.as_deref().unwrap_or("mp3"))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenAIMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<OpenAIContent>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "thought")]
    pub reasoning_content: Option<String>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        alias = "thoughtSignature",
        alias = "thought_signature"
    )]
    pub signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<ToolFunction>,

    #[serde(
        skip_serializing_if = "Option::is_none",
        alias = "thoughtSignature",
        alias = "thought_signature"
    )]
    pub signature: Option<String>,

    // [NEW] Fields for apply_patch_call
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation: Option<ApplyPatchOperation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplyPatchOperation {
    pub r#type: String,
    pub diff: String,
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<OpenAIUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: OpenAIMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    #[serde(skip)]
    pub input_tokens_by_modality: Option<Value>,
    #[serde(skip)]
    pub raw_output_tokens: Option<u32>,
    #[serde(skip)]
    pub total_thought_tokens: Option<u32>,
    #[serde(skip)]
    pub total_tool_use_tokens: Option<u32>,
    #[serde(skip)]
    pub gemini_total_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionTokensDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
}

impl OpenAIUsage {
    pub fn to_responses_usage_value(&self) -> Value {
        let cached_tokens = self
            .prompt_tokens_details
            .as_ref()
            .and_then(|details| details.cached_tokens)
            .unwrap_or(0);
        let reasoning_tokens = self
            .completion_tokens_details
            .as_ref()
            .and_then(|details| details.reasoning_tokens)
            .unwrap_or(0);

        json!({
            "input_tokens": self.prompt_tokens,
            "input_tokens_details": {
                "cached_tokens": cached_tokens
            },
            "output_tokens": self.completion_tokens,
            "output_tokens_details": {
                "reasoning_tokens": reasoning_tokens
            },
            "total_tokens": self.total_tokens
        })
    }
}

impl From<&crate::proxy::pipeline::CanonicalUsage> for OpenAIUsage {
    fn from(c: &crate::proxy::pipeline::CanonicalUsage) -> Self {
        Self {
            prompt_tokens: c.total_input_tokens,
            completion_tokens: c.output_tokens,
            total_tokens: c.total_tokens,
            prompt_tokens_details: if c.cached_tokens > 0 {
                Some(PromptTokensDetails {
                    cached_tokens: Some(c.cached_tokens),
                })
            } else {
                None
            },
            completion_tokens_details: if c.reasoning_tokens > 0 {
                Some(CompletionTokensDetails {
                    reasoning_tokens: Some(c.reasoning_tokens),
                })
            } else {
                None
            },
            input_tokens_by_modality: None,
            raw_output_tokens: Some(c.output_tokens),
            total_thought_tokens: if c.reasoning_tokens > 0 {
                Some(c.reasoning_tokens)
            } else {
                None
            },
            total_tool_use_tokens: None,
            gemini_total_tokens: Some(c.total_tokens),
        }
    }
}

#[cfg(test)]
mod r26_request_boundary_tests {
    use super::*;

    fn request(extra: Value) -> Result<OpenAIRequest, serde_json::Error> {
        let mut body =
            json!({"model":"gemini-2.5-flash","messages":[{"role":"user","content":"Hello"}]});
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        serde_json::from_value(body)
    }

    #[test]
    fn r26_nullable_stream_preserves_boolean_semantics_and_rejects_bad_types() {
        for (extra, expected) in [
            (json!({}), false),
            (json!({"stream":null}), false),
            (json!({"stream":false}), false),
            (json!({"stream":true}), true),
        ] {
            let parsed = request(extra).unwrap();
            assert_eq!(parsed.stream, expected);
            let roundtrip: OpenAIRequest =
                serde_json::from_value(serde_json::to_value(parsed).unwrap()).unwrap();
            assert_eq!(roundtrip.stream, expected);
        }
        for invalid in [json!("false"), json!(0), json!({}), json!([])] {
            assert!(request(json!({"stream":invalid})).is_err());
        }
    }

    #[test]
    fn r26_token_aliases_accept_null_with_one_effective_limit_and_reject_conflicts() {
        let aliases = [
            "max_tokens",
            "max_completion_tokens",
            "maxCompletionTokens",
            "maxTokens",
            "max_output_tokens",
            "maxOutputTokens",
        ];
        for active in aliases {
            let mut controls = serde_json::Map::new();
            for alias in aliases {
                controls.insert(alias.into(), Value::Null);
            }
            controls.insert(active.into(), json!(16));
            assert_eq!(
                request(Value::Object(controls)).unwrap().max_tokens,
                Some(16)
            );
        }
        assert_eq!(
            request(json!({"max_tokens":16,"max_completion_tokens":16}))
                .unwrap()
                .max_tokens,
            Some(16)
        );
        assert_eq!(
            request(json!({"max_tokens":null,"max_completion_tokens":null}))
                .unwrap()
                .max_tokens,
            None
        );
        for invalid in [
            json!({"max_tokens":16,"max_completion_tokens":32}),
            json!({"max_tokens":null,"max_completion_tokens":"16"}),
            json!({"max_completion_tokens":-1}),
        ] {
            assert!(request(invalid).is_err());
        }
    }

    #[test]
    fn r26_legacy_normalization_does_not_accept_incomplete_calls_or_conflicting_envelopes() {
        for invalid in [
            json!({"functions":"lookup"}),
            json!({"functions":[{"parameters":{}}]}),
            json!({"function_call":{"wrong_name":"lookup"}}),
            json!({"function_call":"required"}),
            json!({"functions":[{"name":"lookup"}],"tools":[{"type":"function","function":{"name":"other"}}]}),
            json!({"function_call":"none","tool_choice":"auto"}),
            json!({"messages":[{"role":"assistant","function_call":{"name":"lookup"}}]}),
            json!({"messages":[{"role":"assistant","function_call":{"name":"lookup","arguments":"broken"}}]}),
            json!({"messages":[{"role":"assistant","function_call":{"name":"lookup","arguments":"[]"}}]}),
            json!({"messages":[{"role":"user","function_call":{"name":"lookup","arguments":"{}"}}]}),
            json!({"messages":[{"role":"assistant","tool_calls":[{"id":"call_local","type":"function","function":{"name":"lookup"}}]}]}),
        ] {
            assert!(
                request(invalid.clone()).is_err(),
                "accepted invalid request: {invalid}"
            );
        }
    }

    #[test]
    fn r26_legacy_ids_are_stable_and_do_not_collide_with_modern_ids() {
        let input = json!({"messages":[
            {"role":"user","content":"Run both"},
            {"role":"assistant","function_call":{"name":"lookup","arguments":"{}"}},
            {"role":"function","name":"lookup","content":"first"},
            {"role":"assistant","tool_calls":[{"id":"call_legacy_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},
            {"role":"tool","tool_call_id":"call_legacy_1","content":"second"}
        ]});
        let parsed = request(input.clone()).unwrap();
        let repeated = request(input).unwrap();
        let legacy_id = &parsed.messages[1].tool_calls.as_ref().unwrap()[0].id;
        assert_ne!(legacy_id, "call_legacy_1");
        assert_eq!(
            parsed.messages[2].tool_call_id.as_deref(),
            Some(legacy_id.as_str())
        );
        assert_eq!(
            &repeated.messages[1].tool_calls.as_ref().unwrap()[0].id,
            legacy_id
        );
        assert_eq!(
            parsed.messages[4].tool_call_id.as_deref(),
            Some("call_legacy_1")
        );
    }
}
