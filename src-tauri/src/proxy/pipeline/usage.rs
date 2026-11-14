use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

// Aliases describe the same cumulative observation, not independent counters.
const INPUT_PATHS: &[&str] = &[
    "/promptTokenCount",
    "/prompt_tokens",
    "/input_tokens",
    "/total_input_tokens",
];
const CACHE_PATHS: &[&str] = &[
    "/cache_read_input_tokens",
    "/cachedContentTokenCount",
    "/cached_content_token_count",
    "/total_cached_tokens",
    "/cached_tokens",
    "/prompt_tokens_details/cached_tokens",
    "/input_tokens_details/cached_tokens",
];
const OUTPUT_PATHS: &[&str] = &[
    "/total_output_tokens",
    "/candidatesTokenCount",
    "/candidates_token_count",
    "/completion_tokens",
    "/output_tokens",
];
const REASONING_PATHS: &[&str] = &[
    "/total_thought_tokens",
    "/totalThoughtTokens",
    "/thoughtsTokenCount",
    "/completion_tokens_details/reasoning_tokens",
    "/output_tokens_details/reasoning_tokens",
];
const TOTAL_PATHS: &[&str] = &["/total_tokens", "/totalTokenCount"];

fn counter(usage: &Value, paths: &[&str]) -> Option<u32> {
    paths.iter().find_map(|path| {
        usage
            .pointer(path)
            .and_then(Value::as_u64)
            .and_then(|v| u32::try_from(v).ok())
    })
}

fn detailed_reasoning(usage: &Value) -> Option<u32> {
    usage
        .get("candidatesTokensDetails")
        .and_then(Value::as_array)
        .and_then(|details| {
            details
                .iter()
                .filter(|item| is_reasoning_detail(item))
                .filter_map(|item| counter(item, &["/tokenCount"]))
                .reduce(u32::saturating_add)
        })
}

fn is_reasoning_detail(item: &Value) -> bool {
    item.get("modality")
        .and_then(Value::as_str)
        .is_some_and(|name| {
            name.eq_ignore_ascii_case("THINKING") || name.eq_ignore_ascii_case("REASONING")
        })
}

fn remove_path(value: &mut Value, path: &str) {
    if let Some((parent, key)) = path.rsplit_once('/') {
        if let Some(object) = value.pointer_mut(parent).and_then(Value::as_object_mut) {
            object.remove(key);
        }
    }
}

/// Merge cumulative upstream usage by field presence, never by addition or maximum.
/// Sparse frames leave earlier counters intact; explicit values (including zero)
/// replace them. Null is not a numeric observation and cannot erase an earlier
/// counter. Keep other unknown metadata so native Gemini responses remain lossless.
pub fn merge_cumulative_usage(current: &mut Option<Value>, update: &Value) {
    if !update.is_object() {
        return;
    }

    fn merge_value(current: &mut Value, update: &Value) {
        if update.is_null() && current.is_number() {
            return;
        }
        if let (Some(current), Some(update)) = (current.as_object_mut(), update.as_object()) {
            for (key, value) in update {
                match current.get_mut(key) {
                    Some(existing) => merge_value(existing, value),
                    None => {
                        current.insert(key.clone(), value.clone());
                    }
                }
            }
        } else {
            *current = update.clone();
        }
    }

    let mut update = update.clone();
    let current = current.get_or_insert_with(|| json!({}));
    // A null/invalid details container must not erase numeric child counters.
    for parent in [
        "prompt_tokens_details",
        "input_tokens_details",
        "completion_tokens_details",
        "output_tokens_details",
    ] {
        if update.get(parent).is_some_and(|value| !value.is_object()) {
            update.as_object_mut().unwrap().remove(parent);
        }
    }
    let anthropic_cache_update =
        counter(current, &["/cache_read_input_tokens"]).and_then(|_| counter(&update, CACHE_PATHS));
    for paths in [
        INPUT_PATHS,
        CACHE_PATHS,
        OUTPUT_PATHS,
        REASONING_PATHS,
        TOTAL_PATHS,
        &["/cache_creation_input_tokens"],
        &["/total_tool_use_tokens"],
    ] {
        // Invalid numeric values (including null) are not observations. In
        // particular, one malformed field must not freeze all later Claude usage.
        for path in paths {
            if update.pointer(path).is_some() && counter(&update, &[*path]).is_none() {
                remove_path(&mut update, path);
            }
        }
        if counter(&update, paths).is_some() {
            for path in paths {
                remove_path(current, path);
            }
        }
    }
    if counter(&update, REASONING_PATHS).is_some() || detailed_reasoning(&update).is_some() {
        for path in REASONING_PATHS {
            remove_path(current, path);
        }
        if let Some(details) = current
            .get_mut("candidatesTokensDetails")
            .and_then(Value::as_array_mut)
        {
            details.retain(|item| !is_reasoning_detail(item));
        }
    }
    // TEXT-only details are not an observation of the previously known thinking
    // count. Keep that count while retaining the newer non-thinking metadata.
    if update.get("candidatesTokensDetails").is_some()
        && detailed_reasoning(&update).is_none()
        && counter(&update, REASONING_PATHS).is_none()
    {
        if let Some(reasoning) = detailed_reasoning(current) {
            if !current
                .get("output_tokens_details")
                .is_some_and(Value::is_object)
            {
                current["output_tokens_details"] = json!({});
            }
            current["output_tokens_details"]["reasoning_tokens"] = json!(reasoning);
        }
    }
    merge_value(current, &update);
    // Preserve Anthropic's uncached-input convention when a later frame uses a
    // generic cache alias. The retained canonical alias must carry the new value.
    if let Some(cached) = anthropic_cache_update {
        current["cache_read_input_tokens"] = json!(cached);
    }
}

/// Presence-aware observations of cumulative usage across protocol events.
/// Counters omitted by a later event survive; explicit zero replaces old values.
/// Keep Anthropic's uncached input separate until cache counters are available.
#[derive(Debug, Default)]
enum OutputKind {
    #[default]
    Combined,
    GeminiCandidates,
    Separate,
}

#[derive(Debug, Default)]
pub struct ObservedUsage {
    input: Option<u32>,
    output: Option<u32>,
    cached: Option<u32>,
    creation: Option<u32>,
    reasoning: Option<u32>,
    tool_use: Option<u32>,
    anthropic_input: bool,
    output_kind: OutputKind,
    reasoning_is_separate: bool,
    total: Option<u32>,
}

impl ObservedUsage {
    pub fn merge_event(&mut self, event: &Value) {
        if let Some(usage) = [
            "/usage",
            "/usageMetadata",
            "/response/usage",
            "/response/usageMetadata",
            "/message/usage",
            "/message/usageMetadata",
            "/delta/usage",
        ]
        .iter()
        .find_map(|path| event.pointer(path).filter(|value| value.is_object()))
        {
            self.merge_usage(usage);
        }
    }

    pub fn merge_usage(&mut self, usage: &Value) {
        fn replace(dst: &mut Option<u32>, value: Option<u32>) {
            if value.is_some() {
                *dst = value;
            }
        }
        replace(
            &mut self.input,
            counter(
                usage,
                &[
                    "/promptTokenCount",
                    "/prompt_tokens",
                    "/input_tokens",
                    "/total_input_tokens",
                ],
            ),
        );
        let cache_read = counter(usage, &["/cache_read_input_tokens"]);
        let creation = counter(usage, &["/cache_creation_input_tokens"]);
        self.anthropic_input |= cache_read.is_some() || creation.is_some();
        replace(&mut self.creation, creation);
        replace(
            &mut self.cached,
            cache_read.or_else(|| {
                counter(
                    usage,
                    &[
                        "/cachedContentTokenCount",
                        "/cached_content_token_count",
                        "/total_cached_tokens",
                        "/cached_tokens",
                        "/prompt_tokens_details/cached_tokens",
                        "/input_tokens_details/cached_tokens",
                    ],
                )
            }),
        );

        let details = detailed_reasoning(usage);
        let separate = counter(usage, &REASONING_PATHS[..3]);
        let reasoning = separate
            .or_else(|| details.filter(|v| *v > 0))
            .or_else(|| counter(usage, REASONING_PATHS))
            .or(details);
        if let Some(reasoning) = reasoning {
            self.reasoning = Some(reasoning);
            self.reasoning_is_separate = separate.is_some();
        }
        replace(&mut self.total, counter(usage, TOTAL_PATHS));
        replace(
            &mut self.tool_use,
            counter(usage, &["/total_tool_use_tokens"]),
        );
        if let Some(output) = counter(usage, &["/total_output_tokens"]) {
            self.output = Some(output);
            self.output_kind = OutputKind::Separate;
        } else if let Some(output) =
            counter(usage, &["/candidatesTokenCount", "/candidates_token_count"])
        {
            self.output = Some(output);
            self.output_kind = OutputKind::GeminiCandidates;
        } else if let Some(output) = counter(usage, &["/completion_tokens", "/output_tokens"]) {
            self.output = Some(output);
            self.output_kind = OutputKind::Combined;
        }
    }

    pub fn input_tokens(&self) -> Option<u32> {
        self.input.map(|input| {
            if self.anthropic_input {
                input
                    .saturating_add(self.cached.unwrap_or(0))
                    .saturating_add(self.creation.unwrap_or(0))
            } else if self.cached.is_some_and(|cached| cached > input) {
                // Preserve CanonicalUsage's compatibility with older Anthropic
                // gateways that report uncached input with a cached_tokens alias.
                input.saturating_add(self.cached.unwrap_or(0))
            } else {
                input
            }
        })
    }
    pub fn output_tokens(&self) -> Option<u32> {
        // A reasoning-only frame still reports measured output. Once a combined
        // output total arrives it is authoritative, including an explicit zero.
        // Input-only frames retain unknown output rather than inventing a zero.
        self.output
            .map(|output| match self.output_kind {
                OutputKind::Separate => output
                    .saturating_add(self.reasoning.unwrap_or(0))
                    .saturating_add(self.tool_use.unwrap_or(0)),
                OutputKind::GeminiCandidates if self.reasoning_is_separate => {
                    output.saturating_add(self.reasoning.unwrap_or(0))
                }
                _ => output,
            })
            .or(self.reasoning)
    }
    pub fn total_tokens(&self) -> Option<u32> {
        self.total.or_else(|| {
            (self.input.is_some() || self.output_tokens().is_some()).then(|| {
                self.input_tokens()
                    .unwrap_or(0)
                    .saturating_add(self.output_tokens().unwrap_or(0))
            })
        })
    }
    pub fn cached_tokens(&self) -> Option<u32> {
        self.cached
    }
    pub fn reasoning_tokens(&self) -> Option<u32> {
        self.reasoning
    }
}

/// 统一用量与缓存计算引擎（CanonicalUsage）
/// 权威收拢 Google Gemini 原生 usageMetadata，并提供向四大协议专属格式的标准散开方法
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalUsage {
    /// 全量总上下文输入 (Total Prompt Context = uncached + cached)
    pub total_input_tokens: u32,
    /// 真正未命中的增量输入 (Uncached Input = total_input_tokens - cached_tokens)
    pub uncached_input_tokens: u32,
    /// 命中的缓存 Token (Cached Tokens)
    pub cached_tokens: u32,
    /// 模型输出总量 (Output Tokens, 含思维链)
    pub output_tokens: u32,
    /// 思考思维链 Token 消耗 (Reasoning Tokens)
    pub reasoning_tokens: u32,
    /// 总消耗 Token = total_input_tokens + output_tokens
    pub total_tokens: u32,
}

impl CanonicalUsage {
    /// 直接由离散字段构造标准用量对象
    pub fn new(
        total_input_tokens: u32,
        cached_tokens: u32,
        output_tokens: u32,
        reasoning_tokens: u32,
    ) -> Self {
        let uncached_input_tokens = total_input_tokens.saturating_sub(cached_tokens);
        let total_tokens = total_input_tokens.saturating_add(output_tokens);
        Self {
            total_input_tokens,
            cached_tokens,
            uncached_input_tokens,
            output_tokens,
            reasoning_tokens,
            total_tokens,
        }
    }

    /// 从 Google upstream 原生 usageMetadata 或包含 usageMetadata 的任意 JSON 中权威抽取收拢
    pub fn from_gemini(raw: &Value) -> Self {
        let usage = raw.get("usageMetadata").unwrap_or(raw);

        let mut observed = ObservedUsage::default();
        observed.merge_usage(usage);
        let total_input_tokens = observed.input_tokens().unwrap_or(0);
        let cached_tokens = observed.cached_tokens().unwrap_or(0);
        Self {
            total_input_tokens,
            uncached_input_tokens: if observed.anthropic_input {
                observed.input.unwrap_or(0)
            } else {
                total_input_tokens.saturating_sub(cached_tokens)
            },
            cached_tokens,
            output_tokens: observed.output_tokens().unwrap_or(0),
            reasoning_tokens: observed.reasoning_tokens().unwrap_or(0),
            total_tokens: observed.total_tokens().unwrap_or(0),
        }
    }

    /// 缩放算法（对齐 Claude Code 上下文阈值）：超过 30k 且开启 scaling 时按阶梯缩放
    pub fn scale_claude_tokens(total_raw: u32, context_limit: u32) -> u32 {
        if total_raw <= 30_000 {
            return total_raw;
        }
        const TARGET_MAX: f64 = 195_000.0;
        let ratio = total_raw as f64 / context_limit.max(1) as f64;
        if ratio <= 0.5 {
            let display_ratio = ratio * 0.6;
            (display_ratio * TARGET_MAX) as u32
        } else if ratio <= 0.7 {
            let progress = (ratio - 0.5) / 0.2;
            let display_ratio = 0.3 + progress * 0.2;
            (display_ratio * TARGET_MAX) as u32
        } else if ratio <= 0.85 {
            let progress = (ratio - 0.7) / 0.15;
            let display_ratio = 0.5 + progress * 0.2;
            (display_ratio * TARGET_MAX) as u32
        } else {
            let progress = (ratio - 0.85) / 0.15;
            let display_ratio = 0.7 + progress * 0.27;
            (display_ratio.min(0.97) * TARGET_MAX) as u32
        }
    }

    /// 散开为 Anthropic Claude 格式：
    /// Claude 规范要求 input_tokens 仅表达未缓存增量，缓存单列在 cache_read_input_tokens
    pub fn to_claude_usage(&self, scaling_enabled: bool, context_limit: u32) -> Value {
        let (reported_input, reported_cache) = if scaling_enabled
            && self.total_input_tokens > 30_000
        {
            let scaled_total = Self::scale_claude_tokens(self.total_input_tokens, context_limit);
            if self.total_input_tokens > 0 {
                let cache_ratio = (self.cached_tokens as f64) / (self.total_input_tokens as f64);
                let sc_cache = (scaled_total as f64 * cache_ratio) as u32;
                (scaled_total.saturating_sub(sc_cache), sc_cache)
            } else {
                (scaled_total, 0)
            }
        } else {
            (self.uncached_input_tokens, self.cached_tokens)
        };

        // This is a cumulative snapshot, not a sparse delta. An explicit zero
        // must survive serialization so outer gateways can clear older cache counts.
        json!({
            "input_tokens": reported_input,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": reported_cache,
            "cache_creation_input_tokens": 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claude_usage_snapshot_serializes_zero_cache_as_a_number() {
        for (input, scaling) in [(100, false), (100, true), (100_000, true)] {
            let snapshot = CanonicalUsage::new(input, 0, 5, 0).to_claude_usage(scaling, 1_000_000);
            assert_eq!(
                snapshot
                    .get("cache_read_input_tokens")
                    .and_then(Value::as_u64),
                Some(0)
            );
        }
    }

    #[test]
    fn reasoning_only_usage_counts_output_until_an_explicit_total_arrives() {
        for initial in [
            json!({"thoughtsTokenCount":5}),
            json!({"total_thought_tokens":5}),
            json!({"output_tokens_details":{"reasoning_tokens":5}}),
        ] {
            let mut observed = ObservedUsage::default();
            observed.merge_usage(&initial);
            assert_eq!(observed.output_tokens(), Some(5));
            assert_eq!(observed.total_tokens(), Some(5));
            assert_eq!(CanonicalUsage::from_gemini(&initial).output_tokens, 5);
            observed.merge_usage(&json!({"output_tokens":0}));
            assert_eq!(observed.output_tokens(), Some(0));
            assert_eq!(observed.reasoning_tokens(), Some(5));
            observed.merge_usage(&json!({"output_tokens":8}));
            assert_eq!(observed.output_tokens(), Some(8));
        }
        let mut input_only = ObservedUsage::default();
        input_only.merge_usage(&json!({"promptTokenCount":100}));
        assert_eq!(input_only.output_tokens(), None);
        assert_eq!(input_only.total_tokens(), Some(100));
    }

    #[test]
    fn observed_usage_preserves_sparse_counters_and_accepts_zero() {
        let mut observed = ObservedUsage::default();
        observed.merge_event(&json!({"usageMetadata":{"promptTokenCount":100,"cachedContentTokenCount":40,"candidatesTokenCount":1}}));
        observed.merge_event(&json!({"usageMetadata":{"cachedContentTokenCount":80}}));
        observed.merge_event(&json!({"usageMetadata":{"candidatesTokenCount":5}}));
        assert_eq!(
            (
                observed.input_tokens(),
                observed.output_tokens(),
                observed.cached_tokens()
            ),
            (Some(100), Some(5), Some(80))
        );
        observed.merge_event(
            &json!({"usageMetadata":{"cached_content_token_count":0,"candidatesTokenCount":0}}),
        );
        assert_eq!(
            (
                observed.input_tokens(),
                observed.output_tokens(),
                observed.cached_tokens()
            ),
            (Some(100), Some(0), Some(0))
        );
    }

    #[test]
    fn observed_usage_tracks_anthropic_message_start_and_sparse_delta() {
        let mut observed = ObservedUsage::default();
        observed.merge_event(&json!({"type":"message_start","usage":null,"message":{"usage":{"input_tokens":20,"cache_read_input_tokens":80,"cache_creation_input_tokens":3,"output_tokens":0}}}));
        observed.merge_event(&json!({"type":"message_delta","usage":{"output_tokens":5}}));
        assert_eq!(
            (
                observed.input_tokens(),
                observed.output_tokens(),
                observed.cached_tokens()
            ),
            (Some(103), Some(5), Some(80))
        );
        observed.merge_event(&json!({"usage":{"cache_read_input_tokens":0}}));
        assert_eq!(
            (observed.input_tokens(), observed.cached_tokens()),
            (Some(23), Some(0))
        );
    }

    #[test]
    fn observed_usage_keeps_reasoning_when_details_omit_it() {
        let mut observed = ObservedUsage::default();
        observed.merge_usage(&json!({"total_output_tokens":5,"total_thought_tokens":10}));
        observed
            .merge_usage(&json!({"candidatesTokensDetails":[{"modality":"TEXT","tokenCount":5}]}));
        assert_eq!(
            (observed.reasoning_tokens(), observed.output_tokens()),
            (Some(10), Some(15))
        );
        observed.merge_usage(
            &json!({"candidatesTokensDetails":[{"modality":"THINKING","tokenCount":0}]}),
        );
        assert_eq!(
            (observed.reasoning_tokens(), observed.output_tokens()),
            (Some(0), Some(5))
        );
        observed.merge_usage(&json!({"output_tokens_details":{"reasoning_tokens":7}}));
        assert_eq!(observed.reasoning_tokens(), Some(7));
    }

    #[test]
    fn observed_usage_ignores_invalid_counters_without_erasing_good_values() {
        let mut observed = ObservedUsage::default();
        observed.merge_usage(&json!({"prompt_tokens":100,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":80}}));
        observed.merge_usage(&json!({"prompt_tokens":null,"completion_tokens":-1,"prompt_tokens_details":{"cached_tokens":4294967296u64}}));
        assert_eq!(
            (
                observed.input_tokens(),
                observed.output_tokens(),
                observed.cached_tokens()
            ),
            (Some(100), Some(5), Some(80))
        );
    }

    #[test]
    fn cumulative_usage_null_preserves_numeric_counters_but_zero_replaces_them() {
        let mut usage =
            Some(json!({"cachedContentTokenCount":80,"details":{"counter":3,"label":"old"}}));
        merge_cumulative_usage(
            &mut usage,
            &json!({"cachedContentTokenCount":null,"details":{"counter":null,"label":null},"unknown":null}),
        );
        assert_eq!(usage.as_ref().unwrap()["cachedContentTokenCount"], 80);
        assert_eq!(usage.as_ref().unwrap()["details"]["counter"], 3);
        assert_eq!(usage.as_ref().unwrap()["details"]["label"], Value::Null);
        assert_eq!(usage.as_ref().unwrap().get("unknown"), Some(&Value::Null));
        merge_cumulative_usage(
            &mut usage,
            &json!({"cachedContentTokenCount":0,"details":{"counter":0}}),
        );
        assert_eq!(
            usage.as_ref().unwrap()["cachedContentTokenCount"].as_u64(),
            Some(0)
        );
        assert_eq!(
            usage.as_ref().unwrap()["details"]["counter"].as_u64(),
            Some(0)
        );
    }

    #[test]
    fn cumulative_usage_preserves_sparse_fields_and_replaces_explicit_values() {
        let mut usage = None;
        merge_cumulative_usage(
            &mut usage,
            &json!({
                "promptTokenCount": 100,
                "cachedContentTokenCount": 80,
                "candidatesTokenCount": 3,
                "futureDetails": {"cached": 80, "other": 2},
                "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 100}]
            }),
        );
        let sparse = json!({"candidatesTokenCount": 5, "futureDetails": {"other": 1}});
        merge_cumulative_usage(&mut usage, &sparse);
        merge_cumulative_usage(&mut usage, &sparse);
        let merged = usage.as_ref().unwrap();
        assert_eq!(merged["promptTokenCount"], 100);
        assert_eq!(merged["cachedContentTokenCount"], 80);
        assert_eq!(merged["candidatesTokenCount"], 5);
        assert_eq!(merged["futureDetails"], json!({"cached": 80, "other": 1}));
        assert_eq!(merged["promptTokensDetails"][0]["tokenCount"], 100);

        merge_cumulative_usage(
            &mut usage,
            &json!({
                "cachedContentTokenCount": 0,
                "candidatesTokenCount": 2,
                "futureDetails": {"cached": 0},
                "promptTokensDetails": []
            }),
        );
        let merged = usage.unwrap();
        assert_eq!(merged["cachedContentTokenCount"], 0);
        assert_eq!(merged["candidatesTokenCount"], 2);
        assert_eq!(merged["futureDetails"], json!({"cached": 0, "other": 1}));
        assert_eq!(merged["promptTokensDetails"], json!([]));
    }

    #[test]
    fn cumulative_usage_ignores_non_object_metadata() {
        let mut usage = Some(json!({"cachedContentTokenCount": 80}));
        for update in [Value::Null, json!(0), json!([])] {
            merge_cumulative_usage(&mut usage, &update);
        }
        assert_eq!(usage, Some(json!({"cachedContentTokenCount": 80})));
    }

    #[test]
    fn test_canonical_usage_from_gemini_and_diffusion() {
        let gemini_raw = json!({
            "usageMetadata": {
                "promptTokenCount": 99000,
                "cachedContentTokenCount": 94000,
                "candidatesTokenCount": 4100,
                "candidatesTokensDetails": [
                    { "modality": "TEXT", "tokenCount": 3100 },
                    { "modality": "THINKING", "tokenCount": 1000 }
                ]
            }
        });

        let usage = CanonicalUsage::from_gemini(&gemini_raw);
        assert_eq!(usage.total_input_tokens, 99000);
        assert_eq!(usage.cached_tokens, 94000);
        assert_eq!(usage.uncached_input_tokens, 5000);
        assert_eq!(usage.output_tokens, 4100);
        assert_eq!(usage.reasoning_tokens, 1000);
        assert_eq!(usage.total_tokens, 103100);

        // 验证散开到 Claude
        let claude_val = usage.to_claude_usage(false, 200000);
        assert_eq!(claude_val["input_tokens"], 5000);
        assert_eq!(claude_val["output_tokens"], 4100);
        assert_eq!(claude_val["cache_read_input_tokens"], 94000);
    }

    #[test]
    fn test_canonical_usage_healing_anthropic_input() {
        // 模拟 Anthropic 历史格式：input_tokens 存的是未缓存部分 (5000)，cached 存的是 (94000)
        let raw = json!({
            "input_tokens": 5000,
            "cached_tokens": 94000,
            "output_tokens": 110
        });
        let usage = CanonicalUsage::from_gemini(&raw);
        assert_eq!(usage.total_input_tokens, 99000);
        assert_eq!(usage.cached_tokens, 94000);
        assert_eq!(usage.uncached_input_tokens, 5000);
        assert_eq!(usage.output_tokens, 110);
    }
}

#[cfg(test)]
mod normalization_regressions {
    use super::*;
    #[test]
    fn separate_thoughts_are_output_but_combined_protocol_usage_is_not_added_twice() {
        let native = json!({"promptTokenCount":7339,"candidatesTokenCount":3,"thoughtsTokenCount":5,"totalTokenCount":7347});
        let usage = CanonicalUsage::from_gemini(&native);
        assert_eq!(
            (
                usage.output_tokens,
                usage.reasoning_tokens,
                usage.total_tokens
            ),
            (8, 5, 7347)
        );
        let combined = json!({"prompt_tokens":7339,"completion_tokens":8,"completion_tokens_details":{"reasoning_tokens":5},"total_tokens":7347});
        assert_eq!(CanonicalUsage::from_gemini(&combined).output_tokens, 8);
        let mut observed = ObservedUsage::default();
        observed.merge_usage(&json!({"candidatesTokenCount":3}));
        observed.merge_usage(&json!({"thoughtsTokenCount":5}));
        assert_eq!(observed.output_tokens(), Some(8));
        observed.merge_usage(&json!({"thoughtsTokenCount":0}));
        assert_eq!(observed.output_tokens(), Some(3));
    }
    #[test]
    fn cache_alias_switch_preserves_uncached_anthropic_input() {
        let mut usage =
            Some(json!({"input_tokens":100,"cache_read_input_tokens":80,"output_tokens":5}));
        merge_cumulative_usage(&mut usage, &json!({"cached_tokens":50}));
        let actual = CanonicalUsage::from_gemini(usage.as_ref().unwrap());
        assert_eq!(
            (
                actual.total_input_tokens,
                actual.uncached_input_tokens,
                actual.cached_tokens
            ),
            (150, 100, 50)
        );
        merge_cumulative_usage(&mut usage, &json!({"cachedContentTokenCount":0}));
        let actual = CanonicalUsage::from_gemini(usage.as_ref().unwrap());
        assert_eq!((actual.total_input_tokens, actual.cached_tokens), (100, 0));
    }

    #[test]
    fn null_details_containers_keep_observed_children() {
        let mut usage = Some(
            json!({"prompt_tokens":100,"completion_tokens":8,"prompt_tokens_details":{"cached_tokens":80},"output_tokens_details":{"reasoning_tokens":5}}),
        );
        merge_cumulative_usage(
            &mut usage,
            &json!({"prompt_tokens_details":null,"output_tokens_details":null}),
        );
        let actual = CanonicalUsage::from_gemini(usage.as_ref().unwrap());
        assert_eq!((actual.cached_tokens, actual.reasoning_tokens), (80, 5));
        merge_cumulative_usage(
            &mut usage,
            &json!({"input_tokens_details":{"cached_tokens":0},"completion_tokens_details":{"reasoning_tokens":0}}),
        );
        let actual = CanonicalUsage::from_gemini(usage.as_ref().unwrap());
        assert_eq!((actual.cached_tokens, actual.reasoning_tokens), (0, 0));
    }

    #[test]
    fn aliases_replace_previous_observations_without_invalid_field_poisoning() {
        let mut usage = None;
        merge_cumulative_usage(
            &mut usage,
            &json!({"promptTokenCount":100,"cachedContentTokenCount":80,"candidatesTokenCount":3,"thoughtsTokenCount":5}),
        );
        merge_cumulative_usage(
            &mut usage,
            &json!({"promptTokenCount":"invalid","cached_content_token_count":0,"total_thought_tokens":0}),
        );
        merge_cumulative_usage(&mut usage, &json!({"candidatesTokenCount":5}));
        let actual = CanonicalUsage::from_gemini(usage.as_ref().unwrap());
        assert_eq!(
            (
                actual.total_input_tokens,
                actual.cached_tokens,
                actual.output_tokens,
                actual.reasoning_tokens
            ),
            (100, 0, 5, 0)
        );
        merge_cumulative_usage(
            &mut usage,
            &json!({"cache_read_input_tokens":null,"cachedContentTokenCount":40}),
        );
        assert_eq!(
            CanonicalUsage::from_gemini(usage.as_ref().unwrap()).cached_tokens,
            40
        );
    }
}
