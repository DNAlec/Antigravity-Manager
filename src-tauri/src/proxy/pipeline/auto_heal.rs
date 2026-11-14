//! 纯思考空回复流式自愈管道（Thinking Dropout Auto-Healing Pipeline）
//!
//! 核心防御场景：
//! 超长上下文（如 40 万+ tokens / 700+ 轮）下，大模型（尤其是 Gemini Flash 系列）在完成思考后，
//! 概率性因注意力坍缩直接输出结束符 `<end_of_turn>` / `STOP`，导致 `candidatesTokenCount == 0`，
//! 即“只返回了思考块，正文与工具调用皆为空”。
//! 客户端（如 JeikCode / Codex / Claude Code 等）收到仅有思考的回复后，会因状态机无法推进而直接异常断开。
//!
//! 自愈拦截策略（Tail Interception & Auto-Piping）：
//! 1. 思考块正常实时透传给下游客户端，确保首字延迟（TTFT）与实时思考动画丝滑展示；
//! 2. 拦截流末尾的过早终止符（`finishReason: "STOP"` 与 `[DONE]`），不向客户端发射；
//! 3. 向客户端发射 SSE 心跳注释（`: auto-healing empty thinking\n\n`）保持连接存活；
//! 4. 自动构造带原上下文的自愈续跑请求（根据语言自适应追加 `"继续"` 或 `"Continue."`），在相同账号上并发起上游调用；
//! 5. 将上游续跑流无缝缝合至当前下游客户端连接；
//! 6. 严格实施单次自愈上限（`max_auto_heals = 1`）与失败兜底熔断，彻底消除死循环风险与客户端断开。

use bytes::{Bytes, BytesMut};
use futures::{Stream, StreamExt};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use crate::proxy::pipeline::inbound::InboundThinkingPipeline;
use crate::proxy::upstream::client::UpstreamClient;

/// 思考空回复自愈上下文
pub struct ThinkingAutoHealContext {
    pub upstream: Arc<UpstreamClient>,
    pub method: &'static str,
    pub access_token: String,
    pub original_body: Value,
    pub query_string: Option<&'static str>,
    pub extra_headers: HashMap<String, String>,
    pub account_id: Option<String>,
    pub trace_id: String,
    /// 请求级预算上下文（构造时捕获）
    ///
    /// 续跑发生在 `Body::from_stream` 被 poll 时，此时入口中间件的
    /// `task_local` 已经失效；必须显式恢复，否则续跑不受全局预算与
    /// 大 body 重放护栏约束。
    pub budget: Option<Arc<crate::proxy::pipeline::budget::RequestContext>>,
}

/// 根据历史上下文的用户语言偏好，解析最自然的续跑提示词
pub fn resolve_continuation_prompt(body: &Value) -> &'static str {
    let contents = body
        .get("request")
        .and_then(|r| r.get("contents"))
        .or_else(|| body.get("contents"))
        .and_then(|c| c.as_array());

    if let Some(contents) = contents {
        for content in contents.iter().rev() {
            if content.get("role").and_then(|r| r.as_str()) == Some("user") {
                if let Some(parts) = content.get("parts").and_then(|p| p.as_array()) {
                    for p in parts.iter().rev() {
                        if let Some(text) = p.get("text").and_then(|t| t.as_str()) {
                            if text.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c)) {
                                return "继续";
                            } else {
                                return "Continue.";
                            }
                        }
                    }
                }
            }
        }
    }
    "Continue."
}

/// 向 Gemini contents 中安全注入或融合续跑提示词，严格保持角色交替
pub fn append_or_merge_continuation(contents: &mut Vec<Value>, continuation_prompt: &str) {
    let last_role = contents
        .last()
        .and_then(|c| c.get("role"))
        .and_then(|r| r.as_str())
        .unwrap_or("");

    if last_role == "user" {
        // [FIX #3694] Gemini API 严格要求 multi-turn 对话中角色严格交替 (user <-> model)。
        // 若原上下文最后一轮已为 user 轮次，追加新的 user 轮次将产生连续 user [user, user]，
        // 触发 Gemini HTTP 400 拒收报错: "Please ensure that multiturn requests alternate between user and model roles."
        // 因此直接将续跑提示词合入该 user 轮次的 parts 末尾。
        if let Some(last_turn) = contents.last_mut() {
            if let Some(parts) = last_turn.get_mut("parts").and_then(|p| p.as_array_mut()) {
                if let Some(last_part) = parts.last_mut().and_then(|p| p.as_object_mut()) {
                    if let Some(Value::String(text)) = last_part.get_mut("text") {
                        if !text.is_empty() {
                            text.push_str("\n\n");
                        }
                        text.push_str(continuation_prompt);
                    } else {
                        parts.push(json!({ "text": continuation_prompt }));
                    }
                } else {
                    parts.push(json!({ "text": continuation_prompt }));
                }
            } else {
                last_turn["parts"] = json!([{ "text": continuation_prompt }]);
            }
        }
    } else {
        // 最后一轮为 model/assistant 或内容为空，安全追加独立的 user 轮次
        contents.push(json!({
            "role": "user",
            "parts": [{ "text": continuation_prompt }]
        }));
    }
}

/// 构造用于自愈续跑的上游请求 Body
pub fn create_auto_heal_continuation_body(
    original_body: &Value,
    continuation_prompt: &str,
) -> Value {
    let mut new_body = original_body.clone();

    // 1. 为 requestId 追加自愈标记，避免上游缓存或去重干扰
    if let Some(req_id) = new_body.get_mut("requestId").and_then(|v| v.as_str()) {
        new_body["requestId"] = json!(format!("{}_heal1", req_id));
    }
    if let Some(req) = new_body.get_mut("request").and_then(|r| r.as_object_mut()) {
        if let Some(req_id) = req.get_mut("requestId").and_then(|v| v.as_str()) {
            req["requestId"] = json!(format!("{}_heal1", req_id));
        }
    }

    // 2. 严格交替注入/融合 User 续跑提示轮次
    if let Some(contents) = new_body
        .get_mut("request")
        .and_then(|r| r.get_mut("contents"))
        .and_then(|c| c.as_array_mut())
    {
        append_or_merge_continuation(contents, continuation_prompt);
    } else if let Some(contents) = new_body.get_mut("contents").and_then(|c| c.as_array_mut()) {
        append_or_merge_continuation(contents, continuation_prompt);
    }

    new_body
}

/// 统一的「原地自愈」请求状态（每个请求一份）
#[derive(Debug, Default, Clone)]
pub struct AutoHealFlags {
    /// 已因 tool_choice/thinking 冲突剥离过 thinkingConfig
    pub retried_tool_choice_conflict: bool,
    /// 已因 503/过载降级过思考预算
    pub retried_with_budget_degraded: bool,
    /// 已因 503 容量枯竭 (MODEL_CAPACITY_EXHAUSTED) 原地升档为 High
    pub retried_with_capacity_promoted: bool,
    /// 已因 400 图像解码失败剔除过损坏图片
    pub retried_with_image_omitted: bool,
    // Record only authorized repairs, in order: promotion and degradation do not commute.
    repairs: Vec<PayloadRepair>,
}

#[derive(Debug, Clone, Copy)]
enum PayloadRepair {
    PromoteCapacity,
    DegradeBudget,
    StripThinking,
    OmitImages,
}

impl PayloadRepair {
    fn apply(self, body: &mut Value) {
        match self {
            Self::PromoteCapacity => {
                InboundThinkingPipeline::try_promote_to_high_capacity(body);
            }
            Self::DegradeBudget => {
                InboundThinkingPipeline::try_degrade_thinking_budget(body);
            }
            Self::StripThinking => {
                fn strip(request: &mut Value) {
                    for key in ["generationConfig", "generation_config"] {
                        if let Some(config) = request.get_mut(key).and_then(Value::as_object_mut) {
                            config.remove("thinkingConfig");
                            config.remove("thinking_config");
                        }
                    }
                }
                strip(body);
                if let Some(request) = body.get_mut("request") {
                    strip(request);
                }
            }
            Self::OmitImages => {
                crate::proxy::mappers::common_utils::omit_images_from_gemini_payload(body, false);
            }
        }
    }
}

impl AutoHealFlags {
    /// Replay approved payload transforms on a fresh account/project/request wrapper.
    /// This does not authorize another retry or restore stale credentials/identifiers.
    pub fn reapply(&self, body: &mut Value) {
        for repair in &self.repairs {
            repair.apply(body);
        }
    }
}

/// 一次自愈的执行结果
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutoHealOutcome {
    /// 诊断用短标签（写入日志/阶段时间线）
    pub kind: &'static str,
    /// 同账号重试前的等待时长
    pub delay_ms: u64,
}

/// Prepare a repair without changing the live request until its replay is allowed.
/// This runs before ordinary CacheFirst failure handling and before account penalties.
#[allow(clippy::too_many_arguments)]
pub fn try_auto_heal_with_retry_policy(
    classification: &crate::proxy::pipeline::UpstreamClassification,
    status_code: u16,
    error_text: &str,
    retry_after: Option<&str>,
    body: &mut Value,
    flags: &mut AutoHealFlags,
    retry_state: &mut crate::proxy::pipeline::rotation::RequestRetryState,
    account_id: &str,
    scheduling: &crate::proxy::sticky_config::StickySessionConfig,
    cache_bound: bool,
) -> Option<AutoHealOutcome> {
    if crate::proxy::upstream::request_policy::request_policy().single_attempt
        || classification.is_model_not_found()
        || classification.is_internal_gateway_message()
    {
        return None;
    }
    let mut repaired_body = body.clone();
    let mut repaired_flags = flags.clone();
    let mut heal = try_auto_heal(
        classification,
        status_code,
        error_text,
        &mut repaired_body,
        &mut repaired_flags,
    )?;
    heal.delay_ms = retry_state.authorize_payload_repair(
        account_id,
        heal.delay_ms,
        crate::proxy::upstream::retry::parse_retry_delay(error_text, retry_after),
        scheduling,
        crate::proxy::pipeline::rotation::cache_first_preserves_failure(
            status_code,
            error_text,
            retry_after,
            scheduling,
            cache_bound,
        ),
    )?;
    if repaired_flags.retried_with_image_omitted && !flags.retried_with_image_omitted {
        // Publishing the single-image blacklist is a side effect, so defer it
        // until the policy has authorized replay, just like the body and flags.
        crate::proxy::mappers::common_utils::purge_corrupted_images_from_gemini_payload(body);
    }
    *body = repaired_body;
    *flags = repaired_flags;
    Some(heal)
}

/// 通用原地自愈执行器（协议无关，操作的是已转换的 Gemini Canonical IR）
///
/// 四个适配器过去各写一份同样的判断与改写：
/// 1. 503/529 过载 → 降级思考预算（保住 prefix cache）；
/// 2. 400 tool_choice/thinking 冲突 → 剥离 thinkingConfig；
/// 3. 400 图像解码失败 → 剔除损坏图片并降级为文本占位。
///
/// 三者的共同点：**都是对已转换完成的 Gemini 报文做原地修复，且必须在同账号重试**
/// （换号只会让同一份坏载荷再失败一次，并浪费轮换预算）。
///
/// 返回 `Some` 表示已在同一账号上完成修复，调用方应 `continue` 重试；
/// `None` 表示本次不需要 / 无法自愈，继续走既有错误策略。
pub fn try_auto_heal(
    classification: &crate::proxy::pipeline::UpstreamClassification,
    status_code: u16,
    error_text: &str,
    body: &mut Value,
    flags: &mut AutoHealFlags,
) -> Option<AutoHealOutcome> {
    // 0. 上游容量枯竭 (MODEL_CAPACITY_EXHAUSTED / 503)：原地将 medium/low/tiered 升档为 high 算力通道
    if crate::proxy::upstream::client::UpstreamClient::is_model_capacity_exhausted(
        rquest::StatusCode::from_u16(status_code)
            .unwrap_or(rquest::StatusCode::SERVICE_UNAVAILABLE),
        error_text,
    ) && !flags.retried_with_capacity_promoted
    {
        if InboundThinkingPipeline::try_promote_to_high_capacity(body) {
            flags.retried_with_capacity_promoted = true;
            flags.repairs.push(PayloadRepair::PromoteCapacity);
            tracing::warn!(
                "[AutoHeal] Prepared HIGH tier repair for upstream MODEL_CAPACITY_EXHAUSTED"
            );
            return Some(AutoHealOutcome {
                kind: "promote_tier_to_high_capacity",
                delay_ms: 50,
            });
        }
    }

    // 1. 上游过载：降级思考预算，避免被调度到高算力集群继续排队
    if crate::proxy::pipeline::UpstreamClassification::is_server_overloaded(status_code, error_text)
        && !flags.retried_with_budget_degraded
        && InboundThinkingPipeline::try_degrade_thinking_budget(body)
    {
        flags.retried_with_budget_degraded = true;
        flags.repairs.push(PayloadRepair::DegradeBudget);
        tracing::warn!(
            "[AutoHeal] Prepared thinking budget reduction for upstream {}/overloaded (preserving prefix cache) err={}",
            status_code,
            crate::proxy::upstream::client::strip_internal_markers(error_text)
        );
        return Some(AutoHealOutcome {
            kind: "degrade_thinking_budget",
            delay_ms: 150,
        });
    }

    // 2. 强制工具调用与思考冲突：剥离 thinkingConfig 后重试
    if crate::proxy::pipeline::UpstreamClassification::is_thinking_tool_choice_conflict(
        status_code,
        error_text,
    ) && !flags.retried_tool_choice_conflict
    {
        flags.retried_tool_choice_conflict = true;
        PayloadRepair::StripThinking.apply(body);
        flags.repairs.push(PayloadRepair::StripThinking);
        tracing::warn!(
            "[AutoHeal] Prepared thinkingConfig removal for upstream tool_choice/thinking conflict"
        );
        return Some(AutoHealOutcome {
            kind: "strip_thinking_on_tool_choice_conflict",
            delay_ms: 50,
        });
    }

    // 3. 图像解码/处理失败：剔除损坏图片（降级为文本占位）后重试
    if classification.is_image_processing_error() && !flags.retried_with_image_omitted {
        let purged =
            crate::proxy::mappers::common_utils::omit_images_from_gemini_payload(body, false);
        if purged == 0 {
            tracing::warn!(
                "[AutoHeal] upstream 400 unable to process input image, but no inline image was purged; skipping retry to prevent duplicate 400"
            );
            return None;
        }
        flags.retried_with_image_omitted = true;
        flags.repairs.push(PayloadRepair::OmitImages);
        tracing::warn!(
            "[AutoHeal] Prepared replacement of {} corrupted image(s) with text placeholders",
            purged
        );
        return Some(AutoHealOutcome {
            kind: "purge_corrupted_images",
            delay_ms: 50,
        });
    }

    None
}

#[cfg(test)]
mod retry_policy_tests {
    use super::*;
    use crate::proxy::pipeline::rotation::{RequestRetryState, RetryStrategy};
    use crate::proxy::sticky_config::{SchedulingMode, StickySessionConfig};

    fn body() -> Value {
        json!({"model": "gemini-pro-agent-medium", "request": {
            "generationConfig": {"thinkingConfig": {"thinkingLevel": "MEDIUM", "thinkingBudget": 10000}},
            "contents": [{"role": "user", "parts": [{"text": "Keep this prefix"}]}]
        }})
    }

    fn repair(
        error: &str,
        retry_after: Option<&str>,
        body: &mut Value,
        flags: &mut AutoHealFlags,
        state: &mut RequestRetryState,
        scheduling: &StickySessionConfig,
    ) -> Option<AutoHealOutcome> {
        let classification =
            crate::proxy::pipeline::UpstreamClassification::classify(503, error, retry_after);
        try_auto_heal_with_retry_policy(
            &classification,
            503,
            error,
            retry_after,
            body,
            flags,
            state,
            "warm",
            scheduling,
            true,
        )
    }

    #[test]
    fn payload_repair_survives_rebuild_and_later_transport_retry() {
        let scheduling = StickySessionConfig {
            mode: SchedulingMode::CacheFirst,
            ..Default::default()
        };
        for (status, error) in [
            (
                400,
                "Thinking may not be enabled when tool_choice forces tool use",
            ),
            (400, "unable to process input image"),
            (503, "MODEL_CAPACITY_EXHAUSTED"),
            (503, "model overloaded"),
        ] {
            let mut original = body();
            original["project"] = json!("old-project");
            original["requestId"] = json!("old-request");
            original["request"]["contents"][0]["parts"]
                .as_array_mut()
                .unwrap()
                .push(json!({"inlineData":{"mimeType":"image/png", "data":"!!!!not-base64!!!!"}}));
            let mut repaired = original.clone();
            let mut flags = AutoHealFlags::default();
            let mut state = RequestRetryState::default();
            let classification =
                crate::proxy::pipeline::UpstreamClassification::classify(status, error, None);
            assert!(try_auto_heal_with_retry_policy(
                &classification,
                status,
                error,
                None,
                &mut repaired,
                &mut flags,
                &mut state,
                "warm",
                &scheduling,
                true,
            )
            .is_some());
            // A payload-specific 400 repair leaves one transient replay available.
            if status == 400 {
                assert!(matches!(
                    state.cache_first_failure("warm", 502, "bad gateway", None, &scheduling, true,),
                    Some(RetryStrategy::GraceRetry(_))
                ));
            }
            for attempt in 1..=3 {
                let mut rebuilt = original.clone();
                rebuilt["project"] = json!("current-project");
                rebuilt["requestId"] = json!(format!("new-request-{attempt}"));
                flags.reapply(&mut rebuilt);
                assert_eq!(rebuilt["request"], repaired["request"], "{error}");
                assert_eq!(rebuilt["model"], repaired["model"], "{error}");
                assert_eq!(rebuilt["project"], "current-project");
                assert_eq!(rebuilt["requestId"], format!("new-request-{attempt}"));
                let before = rebuilt.clone();
                flags.reapply(&mut rebuilt);
                assert_eq!(rebuilt, before, "reapplying repairs must be idempotent");
            }
        }
    }

    #[tokio::test]
    async fn payload_repair_denied_image_does_not_publish_blacklist() {
        let image = "payload_repair_denied_image_unique_fixture";
        let mut body = json!({"contents":[{"role":"user", "parts":[{
            "inlineData":{"mimeType":"image/png", "data":image}
        }]}]});
        let original = body.clone();
        let mut flags = AutoHealFlags::default();
        let mut state = RequestRetryState::default();
        let scheduling = StickySessionConfig {
            mode: SchedulingMode::CacheFirst,
            max_wait_seconds: 1,
            ..Default::default()
        };
        let error = "unable to process input image";
        let classification =
            crate::proxy::pipeline::UpstreamClassification::classify(400, error, None);
        crate::proxy::pipeline::budget::scope(
            crate::proxy::pipeline::budget::RequestContext::new(std::time::Instant::now()),
            async {
                assert!(try_auto_heal_with_retry_policy(
                    &classification,
                    400,
                    error,
                    None,
                    &mut body,
                    &mut flags,
                    &mut state,
                    "warm",
                    &scheduling,
                    true,
                )
                .is_none());
            },
        )
        .await;
        assert_eq!(body, original);
        assert!(flags.repairs.is_empty());
        assert!(!crate::proxy::mappers::common_utils::is_corrupted_image(
            image
        ));
        assert!(try_auto_heal_with_retry_policy(
            &classification,
            400,
            error,
            None,
            &mut body,
            &mut flags,
            &mut state,
            "warm",
            &scheduling,
            true,
        )
        .is_some());
        assert!(crate::proxy::mappers::common_utils::is_corrupted_image(
            image
        ));
    }

    #[test]
    fn payload_repair_replays_capacity_and_budget_in_approved_order() {
        // Balance mode can approve both transformations; their order matters.
        for errors in [
            ["model overloaded", "MODEL_CAPACITY_EXHAUSTED"],
            ["MODEL_CAPACITY_EXHAUSTED", "model overloaded"],
        ] {
            let original = body();
            let mut repaired = original.clone();
            let mut flags = AutoHealFlags::default();
            let mut state = RequestRetryState::default();
            for error in errors {
                assert!(repair(
                    error,
                    None,
                    &mut repaired,
                    &mut flags,
                    &mut state,
                    &Default::default()
                )
                .is_some());
            }
            let mut rebuilt = original;
            flags.reapply(&mut rebuilt);
            assert_eq!(rebuilt, repaired);
        }
    }

    #[test]
    fn payload_repair_runs_before_cache_first_stop_and_preserves_prefix() {
        let scheduling = StickySessionConfig {
            mode: SchedulingMode::CacheFirst,
            ..Default::default()
        };
        for (error, kind) in [
            ("MODEL_CAPACITY_EXHAUSTED", "promote_tier_to_high_capacity"),
            ("model overloaded", "degrade_thinking_budget"),
        ] {
            let mut body = body();
            let prefix = body["request"]["contents"].clone();
            let mut flags = AutoHealFlags::default();
            let mut state = RequestRetryState::default();
            let heal = repair(error, None, &mut body, &mut flags, &mut state, &scheduling).unwrap();
            assert_eq!(heal.kind, kind);
            assert_eq!(body["request"]["contents"], prefix);
            assert_eq!(flags.repairs.len(), 1);
            if kind == "promote_tier_to_high_capacity" {
                assert_eq!(
                    body["request"]["generationConfig"]["thinkingConfig"]["thinkingLevel"],
                    "HIGH"
                );
            } else {
                assert_eq!(
                    body["request"]["generationConfig"]["thinkingConfig"]["thinkingBudget"],
                    4000
                );
            }
            assert_eq!(
                state.cache_first_failure("warm", 503, error, None, &scheduling, true),
                Some(RetryStrategy::NoRetry)
            );
        }
    }

    #[test]
    fn payload_repair_denied_by_replay_or_delay_does_not_change_request() {
        let scheduling = StickySessionConfig {
            mode: SchedulingMode::CacheFirst,
            max_wait_seconds: 1,
            ..Default::default()
        };
        for already_retried in [false, true] {
            let mut state = RequestRetryState::default();
            if already_retried {
                assert_eq!(
                    state.authorize_payload_repair("warm", 50, None, &scheduling, true),
                    Some(50)
                );
            }
            let mut body = body();
            let original = body.clone();
            let mut flags = AutoHealFlags::default();
            let retry_after = if already_retried { None } else { Some("2") };
            assert!(repair(
                "MODEL_CAPACITY_EXHAUSTED",
                retry_after,
                &mut body,
                &mut flags,
                &mut state,
                &scheduling
            )
            .is_none());
            assert_eq!(body, original);
            assert!(flags.repairs.is_empty());
            assert!(!flags.retried_with_capacity_promoted);
        }
    }

    #[test]
    fn payload_repair_request_fix_does_not_consume_transient_replay() {
        let scheduling = StickySessionConfig {
            mode: SchedulingMode::CacheFirst,
            ..Default::default()
        };
        let mut body = body();
        let mut flags = AutoHealFlags::default();
        let mut state = RequestRetryState::default();
        let error = "Thinking may not be enabled when tool_choice forces tool use";
        let classification =
            crate::proxy::pipeline::UpstreamClassification::classify(400, error, None);
        assert!(try_auto_heal_with_retry_policy(
            &classification,
            400,
            error,
            None,
            &mut body,
            &mut flags,
            &mut state,
            "warm",
            &scheduling,
            true
        )
        .is_some());
        assert!(repair(
            "MODEL_CAPACITY_EXHAUSTED",
            None,
            &mut body,
            &mut flags,
            &mut state,
            &scheduling
        )
        .is_some());
    }

    #[test]
    fn payload_repair_does_not_intercept_model_or_quota_recovery() {
        for (status, error) in [
            (503, "model not found"),
            (503, "no quota accounts"),
            (
                429,
                r#"{"error":{"details":[{"@type":"type.googleapis.com/google.rpc.QuotaFailure"}]}}"#,
            ),
        ] {
            let mut body = body();
            let original = body.clone();
            let mut flags = AutoHealFlags::default();
            let mut state = RequestRetryState::default();
            let classification =
                crate::proxy::pipeline::UpstreamClassification::classify(status, error, None);
            assert!(try_auto_heal_with_retry_policy(
                &classification,
                status,
                error,
                None,
                &mut body,
                &mut flags,
                &mut state,
                "warm",
                &Default::default(),
                true
            )
            .is_none());
            assert_eq!(body, original);
            assert!(flags.repairs.is_empty());
        }
    }

    #[tokio::test]
    async fn payload_repair_exhausted_budget_does_not_change_request() {
        use crate::proxy::pipeline::budget;
        budget::scope(
            budget::RequestContext::new(std::time::Instant::now()),
            async {
                let mut body = body();
                let original = body.clone();
                let mut flags = AutoHealFlags::default();
                let mut state = RequestRetryState::default();
                assert!(repair(
                    "MODEL_CAPACITY_EXHAUSTED",
                    None,
                    &mut body,
                    &mut flags,
                    &mut state,
                    &Default::default()
                )
                .is_none());
                assert_eq!(body, original);
                assert!(flags.repairs.is_empty());
                assert!(!flags.retried_with_capacity_promoted);
            },
        )
        .await;
    }

    #[test]
    fn payload_repair_single_attempt_process_guard() {
        // Also run this test alone in a fresh process with
        // ANTIGRAVITY_UPSTREAM_SINGLE_ATTEMPT=1 (the setting is cached per process).
        if !crate::proxy::upstream::request_policy::request_policy().single_attempt {
            return;
        }
        let mut body = body();
        let original = body.clone();
        let mut flags = AutoHealFlags::default();
        let mut state = RequestRetryState::default();
        assert!(repair(
            "MODEL_CAPACITY_EXHAUSTED",
            None,
            &mut body,
            &mut flags,
            &mut state,
            &Default::default()
        )
        .is_none());
        assert_eq!(body, original);
        assert!(flags.repairs.is_empty());
        assert!(!flags.retried_with_capacity_promoted);
    }
}

/// 检查 Gemini candidate 中的部件类型分布
fn inspect_gemini_candidate_parts(
    candidate: &Value,
    saw_thought: &mut bool,
    saw_content: &mut bool,
    saw_tool_call: &mut bool,
) {
    if let Some(parts) = candidate
        .get("content")
        .and_then(|c| c.get("parts"))
        .and_then(|p| p.as_array())
    {
        for part in parts {
            let is_thought = part
                .get("thought")
                .and_then(|t| t.as_bool())
                .unwrap_or(false);
            if is_thought {
                *saw_thought = true;
            }
            if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                if !is_thought && !text.trim().is_empty() {
                    *saw_content = true;
                }
            }
            if crate::proxy::mappers::gemini::has_media(part)
                || super::stream_error::has_code_execution_output(part)
            {
                *saw_content = true;
            }
            if part.get("functionCall").is_some() {
                *saw_tool_call = true;
            }
        }
    }
}

/// Keep SSE framing intact until all data lines of an event are available.
/// Only the empty-thinking completion is rewritten; ordinary events keep their
/// original bytes, including comments, CRLF and multiline JSON.
#[derive(Default)]
struct AutoHealEventDecoder {
    pending: BytesMut,
    event: BytesMut,
}

struct AutoHealEvent {
    raw: Bytes,
    data: Vec<u8>,
    terminated: bool,
}

impl AutoHealEventDecoder {
    fn push(&mut self, bytes: &[u8]) -> Vec<AutoHealEvent> {
        self.pending.extend_from_slice(bytes);
        let mut events = Vec::new();
        while let Some(pos) = self.pending.iter().position(|&b| b == b'\n') {
            let line = self.pending.split_to(pos + 1);
            let blank = line.as_ref() == b"\n" || line.as_ref() == b"\r\n";
            self.event.extend_from_slice(&line);
            if blank {
                if let Some(event) = self.take_event(true) {
                    events.push(event);
                }
            }
        }
        events
    }

    fn finish(&mut self) -> Option<AutoHealEvent> {
        self.event.extend_from_slice(&self.pending.split());
        self.take_event(false)
    }

    fn take_event(&mut self, terminated: bool) -> Option<AutoHealEvent> {
        if self.event.is_empty() {
            return None;
        }
        let raw = self.event.split().freeze();
        let mut data = Vec::new();
        let mut has_data = false;
        for line in raw.split(|&b| b == b'\n') {
            let line = line.strip_suffix(b"\r").unwrap_or(line);
            if let Some(value) = line.strip_prefix(b"data:") {
                if has_data {
                    data.push(b'\n');
                }
                data.extend_from_slice(value.strip_prefix(b" ").unwrap_or(value));
                has_data = true;
            }
        }
        Some(AutoHealEvent {
            raw,
            data,
            terminated,
        })
    }
}

impl AutoHealEvent {
    fn into_bytes(self) -> Bytes {
        if self.terminated {
            return self.raw;
        }
        // An EOF-flushed event may be followed by a continuation or heartbeat.
        // Complete its separator so the next event cannot become part of its data.
        let mut raw = self.raw.to_vec();
        if !raw.ends_with(b"\n") {
            raw.push(b'\n');
        }
        raw.push(b'\n');
        Bytes::from(raw)
    }

    fn with_json(self, value: &Value) -> Bytes {
        let mut raw = Vec::new();
        // Preserve event/id/retry/comment fields when replacing the data field.
        for line in self.raw.split(|&b| b == b'\n') {
            let line = line.strip_suffix(b"\r").unwrap_or(line);
            if !line.is_empty() && !line.starts_with(b"data:") {
                raw.extend_from_slice(line);
                raw.push(b'\n');
            }
        }
        raw.extend_from_slice(b"data: ");
        raw.extend_from_slice(&serde_json::to_vec(value).expect("JSON Value serializes"));
        raw.extend_from_slice(b"\n\n");
        Bytes::from(raw)
    }
}

fn auto_heal_events<S, E>(mut stream: Pin<Box<S>>) -> impl Stream<Item = Result<AutoHealEvent, E>>
where
    S: Stream<Item = Result<Bytes, E>> + Send + ?Sized + 'static,
    E: Send + 'static,
{
    async_stream::stream! {
        let mut decoder = AutoHealEventDecoder::default();
        while let Some(item) = stream.next().await {
            match item {
                Ok(bytes) => {
                    for event in decoder.push(&bytes) { yield Ok(event); }
                }
                Err(error) => {
                    // A transport close can follow a complete data line before
                    // its blank separator. Preserve already measured output,
                    // but never turn a truncated JSON fragment into an event.
                    if let Some(event) = decoder.finish() {
                        if event.data == b"[DONE]" || serde_json::from_slice::<Value>(&event.data).is_ok() {
                            yield Ok(event);
                        }
                    }
                    yield Err(error);
                    return;
                }
            }
        }
        if let Some(event) = decoder.finish() { yield Ok(event); }
    }
}

#[derive(Default)]
struct EmptyThinkingState {
    saw_thought: bool,
    saw_content: bool,
    saw_tool_call: bool,
    finish_reason: Option<String>,
    rejected: bool,
    usage: Option<Value>,
}

impl EmptyThinkingState {
    fn needs_continuation(&self) -> bool {
        self.saw_thought
            && !self.saw_content
            && !self.saw_tool_call
            && self.finish_reason.is_some()
            && !self.rejected
    }

    fn continuation_is_complete(&self) -> bool {
        self.rejected || ((self.saw_content || self.saw_tool_call) && self.finish_reason.is_some())
    }

    fn observe(&mut self, inner: &Value) -> (bool, bool) {
        if let Some(usage) = inner.get("usageMetadata") {
            crate::proxy::pipeline::usage::merge_cumulative_usage(&mut self.usage, usage);
        }
        self.rejected |= crate::proxy::pipeline::stream_error::error_envelope(inner).is_some()
            || crate::proxy::pipeline::stream_error::response_is_refusal(inner);
        let mut chunk_has_finish = false;
        let mut chunk_has_parts = false;
        if let Some(candidates) = inner.get("candidates").and_then(Value::as_array) {
            for candidate in candidates {
                inspect_gemini_candidate_parts(
                    candidate,
                    &mut self.saw_thought,
                    &mut self.saw_content,
                    &mut self.saw_tool_call,
                );
                chunk_has_parts |= candidate
                    .pointer("/content/parts")
                    .and_then(Value::as_array)
                    .is_some_and(|parts| !parts.is_empty());
                if let Some(reason) = candidate
                    .get("finishReason")
                    .or_else(|| candidate.get("finish_reason"))
                    .and_then(Value::as_str)
                    .filter(|reason| !reason.is_empty() && *reason != "FINISH_REASON_UNSPECIFIED")
                {
                    chunk_has_finish = true;
                    self.finish_reason = Some(reason.to_owned());
                }
            }
        }
        (chunk_has_finish, chunk_has_parts)
    }

    fn process(&mut self, event: AutoHealEvent) -> Option<Bytes> {
        if event.data == b"[DONE]" {
            return (self.saw_content || self.saw_tool_call || self.rejected)
                .then(|| event.into_bytes());
        }
        let Ok(mut json) = serde_json::from_slice::<Value>(&event.data) else {
            return Some(event.into_bytes());
        };
        let inner = if json.get("response").is_some_and(Value::is_object) {
            json.get_mut("response").unwrap()
        } else {
            &mut json
        };
        let (chunk_has_finish, chunk_has_parts) = self.observe(inner);
        if self.rejected
            || self.saw_content
            || self.saw_tool_call
            || !self.saw_thought
            || !chunk_has_finish
        {
            return Some(event.into_bytes());
        }
        if let Some(candidates) = inner.get_mut("candidates").and_then(Value::as_array_mut) {
            for candidate in candidates {
                if let Some(object) = candidate.as_object_mut() {
                    object.remove("finishReason");
                    object.remove("finish_reason");
                }
            }
        }
        if !chunk_has_parts {
            if let Some(object) = inner.as_object_mut() {
                object.remove("candidates");
            }
        }
        Some(event.with_json(&json))
    }
}

fn add_usage_counter(left: Option<u32>, right: Option<u32>) -> Option<u32> {
    (left.is_some() || right.is_some())
        .then(|| left.unwrap_or(0).saturating_add(right.unwrap_or(0)))
}

fn usage_counter(usage: &Value, key: &str) -> Option<u32> {
    usage
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|count| u32::try_from(count).ok())
}

/// Each upstream attempt reports cumulative snapshots. Sum the two final/current
/// snapshots, never the successive frames within either attempt. Normalize known
/// aliases so counters omitted by attempt two cannot inherit attempt one's cache.
fn combined_auto_heal_usage(first: Option<&Value>, second: Option<&Value>) -> Option<Value> {
    let (Some(first), Some(second)) = (first, second) else {
        return first.or(second).cloned();
    };
    use crate::proxy::pipeline::usage::{merge_cumulative_usage, CanonicalUsage, ObservedUsage};
    let mut a = ObservedUsage::default();
    a.merge_usage(first);
    let mut b = ObservedUsage::default();
    b.merge_usage(second);
    let mut merged = Some(first.clone());
    merge_cumulative_usage(&mut merged, second);
    let mut result = merged.unwrap();
    // Keep unknown metadata, but replace every recognized alias consistently.
    for key in [
        "promptTokenCount",
        "prompt_tokens",
        "input_tokens",
        "total_input_tokens",
        "cache_read_input_tokens",
        "cachedContentTokenCount",
        "cached_content_token_count",
        "total_cached_tokens",
        "cached_tokens",
        "total_output_tokens",
        "candidatesTokenCount",
        "candidates_token_count",
        "completion_tokens",
        "output_tokens",
        "total_thought_tokens",
        "totalThoughtTokens",
        "thoughtsTokenCount",
        "total_tokens",
        "totalTokenCount",
        "cache_creation_input_tokens",
        "total_tool_use_tokens",
    ] {
        result.as_object_mut().unwrap().remove(key);
    }
    for (parent, child) in [
        ("prompt_tokens_details", "cached_tokens"),
        ("input_tokens_details", "cached_tokens"),
        ("completion_tokens_details", "reasoning_tokens"),
        ("output_tokens_details", "reasoning_tokens"),
    ] {
        if let Some(object) = result.get_mut(parent).and_then(Value::as_object_mut) {
            object.remove(child);
        }
    }
    // A cache-only snapshot reports known cached input even when the uncached
    // count is unavailable. Include that known component per attempt, otherwise
    // combining it with an earlier prompt can produce cache > prompt and trigger
    // the legacy uncached-input heuristic in downstream adapters.
    let known_input = |usage: &Value, observed: &ObservedUsage| {
        observed.input_tokens().or_else(|| {
            add_usage_counter(
                observed.cached_tokens(),
                usage_counter(usage, "cache_creation_input_tokens"),
            )
        })
    };
    let first_input = known_input(first, &a);
    let second_input = known_input(second, &b);
    let input = add_usage_counter(first_input, second_input);
    let cached = add_usage_counter(a.cached_tokens(), b.cached_tokens());
    let output = add_usage_counter(a.output_tokens(), b.output_tokens());
    let reasoning = add_usage_counter(a.reasoning_tokens(), b.reasoning_tokens());
    if let Some(count) = input {
        result["promptTokenCount"] = json!(count);
    }
    if let Some(count) = cached {
        result["cachedContentTokenCount"] = json!(count);
    }
    let known_total = |usage: &Value, input, output| {
        usage_counter(usage, "total_tokens")
            .or_else(|| usage_counter(usage, "totalTokenCount"))
            .or_else(|| add_usage_counter(input, output))
    };
    if let Some(count) = add_usage_counter(
        known_total(first, first_input, a.output_tokens()),
        known_total(second, second_input, b.output_tokens()),
    ) {
        result["totalTokenCount"] = json!(count);
    }
    if let Some(output) = output {
        if reasoning.is_some_and(|reasoning| reasoning > output) {
            // An authoritative combined output zero can coexist with an older
            // reasoning detail. Preserve that convention rather than double bill it.
            result["output_tokens"] = json!(output);
            result["output_tokens_details"]["reasoning_tokens"] = json!(reasoning.unwrap());
        } else {
            result["candidatesTokenCount"] = json!(output.saturating_sub(reasoning.unwrap_or(0)));
            if let Some(reasoning) = reasoning {
                result["thoughtsTokenCount"] = json!(reasoning);
            }
        }
    }
    if let Some(created) = add_usage_counter(
        usage_counter(first, "cache_creation_input_tokens"),
        usage_counter(second, "cache_creation_input_tokens"),
    ) {
        // Anthropic input excludes both cache reads and writes. Do not attach its
        // cache aliases to Gemini's inclusive promptTokenCount.
        result.as_object_mut().unwrap().remove("promptTokenCount");
        result
            .as_object_mut()
            .unwrap()
            .remove("cachedContentTokenCount");
        if input.is_some() {
            result["input_tokens"] = json!(CanonicalUsage::from_gemini(first)
                .uncached_input_tokens
                .saturating_add(CanonicalUsage::from_gemini(second).uncached_input_tokens));
        }
        if let Some(cached) = cached {
            result["cache_read_input_tokens"] = json!(cached);
        }
        result["cache_creation_input_tokens"] = json!(created);
    }
    if let Some(count) = add_usage_counter(
        usage_counter(first, "total_tool_use_tokens"),
        usage_counter(second, "total_tool_use_tokens"),
    ) {
        result["total_tool_use_tokens"] = json!(count);
    }
    for key in [
        "promptTokensDetails",
        "candidatesTokensDetails",
        "cacheTokensDetails",
        "toolUsePromptTokensDetails",
    ] {
        let mut details = std::collections::BTreeMap::<String, Value>::new();
        for source in [first, second] {
            for item in source
                .get(key)
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let Some(modality) = item.get("modality").and_then(Value::as_str) else {
                    continue;
                };
                let count = add_usage_counter(
                    details
                        .get(modality)
                        .and_then(|item| usage_counter(item, "tokenCount")),
                    usage_counter(item, "tokenCount"),
                );
                let mut merged = item.clone();
                if let Some(count) = count {
                    merged["tokenCount"] = json!(count);
                }
                details.insert(modality.to_owned(), merged);
            }
        }
        if !details.is_empty() {
            result[key] = json!(details.into_values().collect::<Vec<_>>());
        }
    }
    Some(result)
}

fn continuation_error_event(
    status: u16,
    message: &str,
    body: Option<&Value>,
    usage: Option<&Value>,
) -> Bytes {
    let mut error = body
        .and_then(crate::proxy::pipeline::stream_error::error_envelope)
        .cloned()
        .unwrap_or_else(|| json!({"message":message}));
    error["code"] = json!(status);
    if error.get("message").is_none() {
        error["message"] = json!(message);
    }
    let mut response = json!({"error":error});
    crate::proxy::pipeline::stream_error::attach_usage_metadata(&mut response, usage);
    Bytes::from(format!("data: {response}\n\n"))
}

fn continuation_event(
    event: AutoHealEvent,
    first_usage: Option<&Value>,
    state: &mut EmptyThinkingState,
) -> Bytes {
    let Ok(mut json) = serde_json::from_slice::<Value>(&event.data) else {
        return event.into_bytes();
    };
    let inner = if json.get("response").is_some_and(Value::is_object) {
        json.get_mut("response").unwrap()
    } else {
        &mut json
    };
    let has_usage = inner.get("usageMetadata").is_some_and(Value::is_object);
    state.observe(inner);
    // Do not emit a successful finish before the continuation has produced an
    // answer/tool result. EOF/error will report the incomplete attempt instead.
    let mut stripped_finish = false;
    if !state.continuation_is_complete() {
        if let Some(candidates) = inner.get_mut("candidates").and_then(Value::as_array_mut) {
            for candidate in candidates {
                if let Some(object) = candidate.as_object_mut() {
                    stripped_finish |= object.remove("finishReason").is_some();
                    stripped_finish |= object.remove("finish_reason").is_some();
                }
            }
        }
    }

    if has_usage || stripped_finish || state.finish_reason.is_some() || state.rejected {
        if let Some(usage) = combined_auto_heal_usage(first_usage, state.usage.as_ref()) {
            inner["usageMetadata"] = usage;
            return event.with_json(&json);
        }
    }
    if stripped_finish {
        return event.with_json(&json);
    }
    event.into_bytes()
}

#[cfg(test)]
mod continuation_accounting_tests {
    use super::*;
    use crate::proxy::pipeline::usage::{merge_cumulative_usage, CanonicalUsage};
    fn event(value: Value) -> AutoHealEvent {
        let mut decoder = AutoHealEventDecoder::default();
        decoder
            .push(format!("data: {value}\n\n").as_bytes())
            .pop()
            .unwrap()
    }
    fn parsed(bytes: Bytes) -> Value {
        serde_json::from_str(
            String::from_utf8_lossy(&bytes)
                .trim()
                .strip_prefix("data: ")
                .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn r26_code_output_finishes_without_auto_heal_but_thinking_alone_still_continues() {
        for code in [
            json!({"executableCode":{"language":"PYTHON","code":"print(42)"}}),
            json!({"codeExecutionResult":{"outcome":"OUTCOME_OK","output":"42"}}),
            json!({"executable_code":{"code":"print(42)"}}),
            json!({"code_execution_result":{"output":"42"}}),
        ] {
            for thinking in [false, true] {
                let mut parts = vec![code.clone()];
                if thinking {
                    parts.insert(0, json!({"thought":true,"text":"computing"}));
                }
                let mut state = EmptyThinkingState::default();
                let input = json!({"response":{"candidates":[{"content":{"parts":parts},"finishReason":"STOP"}]}});
                let output = parsed(state.process(event(input.clone())).unwrap());
                assert_eq!(output, input, "code output must keep its terminal frame");
                assert!(!state.needs_continuation());
                assert!(state.continuation_is_complete());
            }
        }
        let mut state = EmptyThinkingState::default();
        let output = parsed(
            state
                .process(event(json!({"candidates":[{"content":{"parts":[
            {"thought":true,"text":"computing"},{"codeExecutionResult":{"output":""}}
        ]},"finishReason":"STOP"}]})))
                .unwrap(),
        );
        assert!(state.needs_continuation());
        assert!(!state.continuation_is_complete());
        assert!(output["candidates"][0].get("finishReason").is_none());
    }

    #[test]
    fn r26_code_output_without_finish_does_not_complete_continuation() {
        let mut state = EmptyThinkingState::default();
        state.observe(&json!({"candidates":[{"content":{"parts":[{"codeExecutionResult":{"output":"42"}}]}}]}));
        assert!(!state.continuation_is_complete());
        assert!(!state.needs_continuation());
    }
    #[test]
    fn continuation_empty_or_unspecified_finish_is_not_successful_wire() {
        for reason in ["STOP", "", "FINISH_REASON_UNSPECIFIED"] {
            for thinking in [false, true] {
                let mut state = EmptyThinkingState::default();
                let value = json!({"candidates":[{"content":{"parts":if thinking {vec![json!({"text":"thinking","thought":true})]}else{vec![]}},"finishReason":reason}]});
                let out = parsed(continuation_event(
                    event(value),
                    Some(&json!({"promptTokenCount":100,"cachedContentTokenCount":80})),
                    &mut state,
                ));
                assert!(out["candidates"][0].get("finishReason").is_none());
                assert_eq!(out["usageMetadata"]["cachedContentTokenCount"], 80);
                assert!(!state.continuation_is_complete());
            }
        }
    }
    #[test]
    fn continuation_first_finish_and_trailing_usage_survive() {
        let mut state = EmptyThinkingState::default();
        let value = json!({"candidates":[{"content":{"parts":[{"text":"thinking","thought":true,"thoughtSignature":"signed"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":100,"thoughtsTokenCount":5}});
        let out = parsed(state.process(event(value)).unwrap());
        assert!(out["candidates"][0].get("finishReason").is_none());
        assert_eq!(
            out["candidates"][0]["content"]["parts"][0]["thoughtSignature"],
            "signed"
        );
        assert_eq!(out["usageMetadata"]["promptTokenCount"], 100);
        state.process(event(
            json!({"usageMetadata":{"cachedContentTokenCount":80}}),
        ));
        assert!(state.needs_continuation());
        assert_eq!(state.usage.unwrap()["cachedContentTokenCount"], 80);
    }
    #[test]
    fn continuation_sums_attempt_snapshots_once_and_honors_zero() {
        let first =
            json!({"promptTokenCount":100,"cachedContentTokenCount":80,"thoughtsTokenCount":5});
        let mut second = None;
        for update in [
            json!({"promptTokenCount":200,"candidatesTokenCount":7}),
            json!({"promptTokenCount":200,"candidatesTokenCount":7}),
            json!({"cachedContentTokenCount":30}),
            json!({"cachedContentTokenCount":0}),
        ] {
            merge_cumulative_usage(&mut second, &update);
        }
        let combined = combined_auto_heal_usage(Some(&first), second.as_ref()).unwrap();
        let usage = CanonicalUsage::from_gemini(&combined);
        assert_eq!(
            (
                usage.total_input_tokens,
                usage.cached_tokens,
                usage.output_tokens,
                usage.reasoning_tokens
            ),
            (300, 80, 12, 5)
        );
        assert_eq!(usage.total_tokens, 312);
        let sparse =
            combined_auto_heal_usage(Some(&first), Some(&json!({"promptTokenCount":200}))).unwrap();
        assert_eq!(CanonicalUsage::from_gemini(&sparse).cached_tokens, 80);
    }
    #[test]
    fn continuation_cache_only_attempt_does_not_invent_uncached_input() {
        for second in [
            json!({"cachedContentTokenCount":200}),
            json!({"cache_read_input_tokens":200}),
            json!({"cachedContentTokenCount":200,"thoughtsTokenCount":3}),
        ] {
            let merged = combined_auto_heal_usage(
                Some(&json!({"promptTokenCount":100,"cachedContentTokenCount":80})),
                Some(&second),
            )
            .unwrap();
            let usage = CanonicalUsage::from_gemini(&merged);
            assert_eq!(
                (
                    usage.total_input_tokens,
                    usage.cached_tokens,
                    usage.uncached_input_tokens
                ),
                (300, 280, 20)
            );
            assert_eq!(usage.total_tokens, 300 + usage.output_tokens);
        }
    }
    #[test]
    fn continuation_aliases_reasoning_and_cache_creation_keep_meaning() {
        let first = json!({"input_tokens":20,"cache_read_input_tokens":70,"cache_creation_input_tokens":10,"output_tokens":5,"output_tokens_details":{"reasoning_tokens":3}});
        let second = json!({"promptTokenCount":200,"cachedContentTokenCount":100,"candidatesTokenCount":7,"thoughtsTokenCount":2});
        let merged = combined_auto_heal_usage(Some(&first), Some(&second)).unwrap();
        let usage = CanonicalUsage::from_gemini(&merged);
        assert_eq!(
            (
                usage.total_input_tokens,
                usage.cached_tokens,
                usage.uncached_input_tokens,
                usage.output_tokens,
                usage.reasoning_tokens
            ),
            (300, 170, 120, 14, 5)
        );
        assert_eq!(merged["cache_creation_input_tokens"], 10);
        let zero = combined_auto_heal_usage(
            Some(&json!({"output_tokens":0,"output_tokens_details":{"reasoning_tokens":5}})),
            Some(&json!({"output_tokens":0})),
        )
        .unwrap();
        assert_eq!(CanonicalUsage::from_gemini(&zero).output_tokens, 0);
    }
    #[test]
    fn continuation_failures_preserve_structured_error_and_usage_without_text() {
        let usage =
            json!({"promptTokenCount":100,"cachedContentTokenCount":80,"thoughtsTokenCount":5});
        for status in [429, 503, 502] {
            let value = parsed(continuation_error_event(
                status,
                "continuation failed",
                Some(
                    &json!({"error":{"message":"quota","details":[{"@type":"google.rpc.QuotaFailure"}]}}),
                ),
                Some(&usage),
            ));
            assert_eq!(value["error"]["code"], status);
            assert_eq!(value["error"]["message"], "quota");
            assert_eq!(
                value["error"]["details"][0]["@type"],
                "google.rpc.QuotaFailure"
            );
            assert_eq!(value["usageMetadata"], usage);
            assert!(value.get("candidates").is_none());
        }
    }
    #[test]
    fn continuation_requires_real_completion_and_never_retries_refusal() {
        for reason in ["", "FINISH_REASON_UNSPECIFIED"] {
            let mut state = EmptyThinkingState::default();
            state.observe(&json!({"candidates":[{"content":{"parts":[{"text":"partial"}]},"finishReason":reason}]}));
            assert!(!state.continuation_is_complete());
        }
        for reason in [
            "SAFETY",
            "RECITATION",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
            "SPII",
            "IMAGE_SAFETY",
            "IMAGE_PROHIBITED_CONTENT",
            "MODEL_ARMOR",
        ] {
            for thinking in [false, true] {
                let mut state = EmptyThinkingState::default();
                state.observe(&json!({"candidates":[{"content":{"parts":if thinking {vec![json!({"text":"thinking","thought":true})]} else {vec![]}},"finishReason":reason}]}));
                assert!(!state.needs_continuation());
                assert!(state.continuation_is_complete());
            }
        }
    }
}

#[cfg(test)]
mod event_boundary_tests {
    use super::*;

    #[test]
    fn complete_events_preserve_multiline_crlf_utf8_and_eof() {
        for ending in ["\r\n\r\n", "\r\n", ""] {
            let wire = format!("event: message\r\ndata: {{\"response\":{{\"candidates\":[{{\"content\":{{\"parts\":[{{\"text\":\"你好\"}}]}},\r\ndata:\"finishReason\":\"STOP\"}}],\"usageMetadata\":{{\"cachedContentTokenCount\":80}}}}}}{ending}");
            for size in [1, 7, wire.len()] {
                let mut decoder = AutoHealEventDecoder::default();
                let mut state = EmptyThinkingState::default();
                let mut output = Vec::new();
                for chunk in wire.as_bytes().chunks(size) {
                    for event in decoder.push(chunk) {
                        if let Some(bytes) = state.process(event) {
                            output.extend_from_slice(&bytes);
                        }
                    }
                }
                if let Some(event) = decoder.finish() {
                    if let Some(bytes) = state.process(event) {
                        output.extend_from_slice(&bytes);
                    }
                }
                let normalized = if ending.is_empty() {
                    format!("{wire}\n\n")
                } else if ending == "\r\n" {
                    format!("{wire}\n")
                } else {
                    wire.clone()
                };
                assert_eq!(output, normalized.as_bytes());
                assert!(state.saw_content);
                assert_eq!(state.finish_reason.as_deref(), Some("STOP"));
                assert!(!state.needs_continuation());
            }
        }
    }

    #[tokio::test]
    async fn transport_error_preserves_complete_pending_event_but_not_partial_json() {
        for complete in [false, true] {
            let bytes = if complete {
                "data: {\"usageMetadata\":{\"cachedContentTokenCount\":80}}\n"
            } else {
                "data: {\"usageMetadata\":{\"cachedContentTokenCount\":80"
            };
            let source = futures::stream::iter(vec![
                Ok(Bytes::from(bytes)),
                Err("connection reset".to_string()),
            ]);
            let events = auto_heal_events(Box::pin(source)).collect::<Vec<_>>().await;
            assert_eq!(events.len(), if complete { 2 } else { 1 });
            assert!(matches!(events.last(),Some(Err(error)) if error == "connection reset"));
            if complete {
                assert_eq!(
                    serde_json::from_slice::<Value>(&events[0].as_ref().unwrap().data).unwrap()
                        ["usageMetadata"]["cachedContentTokenCount"],
                    80
                );
            }
        }
    }

    #[test]
    fn continuation_refusal_and_errors_do_not_get_fallback_text() {
        for payload in [
            json!({"promptFeedback":{"blockReason":"SAFETY"}}),
            json!({"error":{"code":429}}),
            json!({"type":"error","message":"failed"}),
        ] {
            let mut state = EmptyThinkingState::default();
            state.observe(&payload);
            assert!(state.continuation_is_complete());
        }
        let mut state = EmptyThinkingState::default();
        state.observe(&json!({"candidates":[{"content":{"parts":[{"thought":true,"text":"reason"}]},"finishReason":"STOP"}]}));
        assert!(!state.continuation_is_complete());
    }

    #[test]
    fn eof_frame_stays_separate_from_continuation_heartbeat_and_data() {
        for ending in ["", "\n", "\r\n"] {
            let wire =
                format!("data: {{\"usageMetadata\":{{\"cachedContentTokenCount\":80}}}}{ending}");
            let mut decoder = AutoHealEventDecoder::default();
            assert!(decoder.push(wire.as_bytes()).is_empty());
            let mut bytes = decoder.finish().unwrap().into_bytes().to_vec();
            bytes.extend_from_slice(b": auto-healing\n\ndata: {\"candidates\":[]}\n\n");
            let mut decoder = AutoHealEventDecoder::default();
            let frames = decoder.push(&bytes);
            assert_eq!(frames.len(), 3);
            assert_eq!(
                serde_json::from_slice::<Value>(&frames[0].data).unwrap()["usageMetadata"]
                    ["cachedContentTokenCount"],
                80
            );
            assert!(frames[1].data.is_empty());
            assert_eq!(
                serde_json::from_slice::<Value>(&frames[2].data).unwrap(),
                json!({"candidates":[]})
            );
        }
    }

    #[test]
    fn whole_request_continuation_is_disabled_for_parallel_generations() {
        assert!(requests_multiple_candidates(
            &json!({"request":{"generationConfig":{"candidateCount":2}}})
        ));
        assert!(requests_multiple_candidates(
            &json!({"generation_config":{"candidate_count":2}})
        ));
        assert!(!requests_multiple_candidates(
            &json!({"request":{"generationConfig":{"candidateCount":1}}})
        ));
        assert!(!requests_multiple_candidates(&json!({})));
    }

    #[test]
    fn empty_thinking_intercepts_only_complete_terminal_event() {
        let wire = "event: message\ndata: {\"response\":{\"candidates\":[{\"content\":{\"parts\":[{\"thought\":true,\"text\":\"reason\",\"thoughtSignature\":\"signature\"}]},\ndata: \"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":100}}}\n\n";
        let mut decoder = AutoHealEventDecoder::default();
        let mut state = EmptyThinkingState::default();
        let mut frames = decoder.push(wire.as_bytes());
        let output = state.process(frames.remove(0)).unwrap();
        assert!(state.needs_continuation());
        let text = std::str::from_utf8(&output).unwrap();
        assert!(text.starts_with("event: message\n"));
        assert!(text.contains("signature"));
        assert!(!text.contains("finishReason"));
        assert!(text.contains("usageMetadata"));
        for payload in [
            json!({"error":{"code":429}}),
            json!({"promptFeedback":{"blockReason":"SAFETY"}}),
        ] {
            let wire = format!("data: {payload}\n\n");
            let event = decoder.push(wire.as_bytes()).remove(0);
            assert_eq!(state.process(event).unwrap().as_ref(), wire.as_bytes());
            assert!(!state.needs_continuation());
        }
    }
}

fn requests_multiple_candidates(body: &Value) -> bool {
    let request = body.get("request").unwrap_or(body);
    let generation = request
        .get("generationConfig")
        .or_else(|| request.get("generation_config"));
    generation
        .and_then(|cfg| {
            cfg.get("candidateCount")
                .or_else(|| cfg.get("candidate_count"))
        })
        .and_then(Value::as_u64)
        .is_some_and(|n| n > 1)
}

/// 统一纯思考空回复自愈流式包装器（Pipeline First）
///
/// 对上游原始 Gemini SSE 流进行透明包装：
/// - 思考内容实时透传；
/// - 若正常产出正文或工具调用，全流程无任何额外开销；
/// - 若发现仅产出思考后立即返回终止符，截留终止信号并在同一连接中自动续跑；
/// - 续跑上限严格为 1 次，具有完备的熔断与兜底保障。
pub fn wrap_stream_with_empty_thinking_auto_heal<S, E>(
    stream: Pin<Box<S>>,
    ctx: ThinkingAutoHealContext,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, E>> + Send>>
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    E: std::fmt::Display + Send + 'static,
{
    // Preserve the original stream during single-attempt experiments: a hidden
    // continuation would generate again and change the request under comparison.
    // A continuation is one new generation; it cannot continue an individual
    // branch of a candidate set. Preserve each branch's own terminal frame.
    if requests_multiple_candidates(&ctx.original_body)
        || crate::proxy::upstream::request_policy::request_policy().single_attempt
    {
        return stream;
    }
    let usage_holder = crate::proxy::monitor::UpstreamRequestBodyHolder::current();
    let stream = async_stream::stream! {
        let mut state = EmptyThinkingState::default();
        let events = auto_heal_events(stream);
        futures::pin_mut!(events);
        while let Some(event) = events.next().await {
            match event {
                Ok(event) => { if let Some(bytes) = state.process(event) { yield Ok(bytes); } }
                Err(error) => { yield Err(error); return; }
            }
        }

        // 核心自愈判断条件：
        // 1. 存在思考块 (saw_thought)
        // 2. 无任何正文 (saw_content == false)
        // 3. 无任何工具调用 (saw_tool_call == false)
        // 4. 上游正常返回了结束符 (saw_finish_reason == true)
        if state.needs_continuation() {
            tracing::warn!(
                "[{}] [Stream-AutoHeal] 🚨 Detected empty thinking completion (thought present, 0 content, 0 tool_calls, finishReason={:?}). Triggering auto-heal continuation 1/1...",
                ctx.trace_id, state.finish_reason
            );

            // 发射 SSE 心跳注释保持客户端下游连接存活
            yield Ok(Bytes::from(": auto-healing empty thinking\n\n"));

            let prompt = resolve_continuation_prompt(&ctx.original_body);
            let heal_body = create_auto_heal_continuation_body(&ctx.original_body, prompt);

            let call_res = crate::proxy::pipeline::budget::scope_optional(
                ctx.budget.clone(),
                ctx.upstream.call_v1_internal_with_headers(
                    ctx.method,
                    &ctx.access_token,
                    heal_body,
                    ctx.query_string,
                    ctx.extra_headers.clone(),
                    ctx.account_id.as_deref(),
                ),
            )
            .await;

            match call_res {
                Ok(call_success) if call_success.response.status().is_success() => {
                    tracing::info!(
                        "[{}] [Stream-AutoHeal] ✓ Continuation request succeeded (HTTP 200), piping healed stream directly into client connection...",
                        ctx.trace_id
                    );
                    let usage_attempt = usage_holder.as_ref().and_then(|holder| {
                        let model = ctx.original_body.get("model")
                            .and_then(Value::as_str).unwrap_or("");
                        holder.begin_usage_attempt(model, ctx.account_id.as_deref().unwrap_or(""), "auto_heal")
                    });
                    let stream2 = crate::proxy::pipeline::usage_diagnostics::observe_stream(
                        call_success.response.bytes_stream(), usage_attempt,
                    );
                    let events = auto_heal_events(Box::pin(stream2));
                    futures::pin_mut!(events);
                    let mut stream2_state = EmptyThinkingState::default();
                    let mut saw_done = false;
                    while let Some(event) = events.next().await {
                        match event {
                            Ok(event) => {
                                if event.data == b"[DONE]" { saw_done=true; continue; }
                                yield Ok(continuation_event(event,state.usage.as_ref(),&mut stream2_state));
                            }
                            Err(error) => {
                                if !stream2_state.continuation_is_complete() {
                                    let usage=combined_auto_heal_usage(state.usage.as_ref(),stream2_state.usage.as_ref());
                                    yield Ok(continuation_error_event(502,&format!("Auto-heal continuation stream failed: {error}"),None,usage.as_ref()));
                                    return;
                                }
                                break;
                            }
                        }
                    }
                    if !stream2_state.continuation_is_complete() {
                        let usage=combined_auto_heal_usage(state.usage.as_ref(),stream2_state.usage.as_ref());
                        yield Ok(continuation_error_event(502,"Auto-heal continuation ended without a complete response",None,usage.as_ref()));
                    } else if saw_done {
                        yield Ok(Bytes::from("data: [DONE]\n\n"));
                    }
                }
                Ok(call_fail) => {
                    let status=call_fail.response.status().as_u16();
                    let body=call_fail.response.json::<Value>().await.ok();
                    let mut failed=EmptyThinkingState::default();
                    if let Some(body)=body.as_ref() {
                        failed.observe(body.get("response").filter(|value|value.is_object()).unwrap_or(body));
                    }
                    let usage=combined_auto_heal_usage(state.usage.as_ref(),failed.usage.as_ref());
                    yield Ok(continuation_error_event(status,"Auto-heal continuation request failed",body.as_ref(),usage.as_ref()));
                }
                Err(error) => {
                    yield Ok(continuation_error_event(502,&format!("Auto-heal continuation request failed: {error}"),None,state.usage.as_ref()));
                }
            }
        }
    };
    Box::pin(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_continuation_prompt_chinese() {
        let body = json!({
            "request": {
                "contents": [
                    { "role": "user", "parts": [{ "text": "帮我看看这个报错怎么解决" }] }
                ]
            }
        });
        assert_eq!(resolve_continuation_prompt(&body), "继续");
    }

    #[test]
    fn test_resolve_continuation_prompt_english() {
        let body = json!({
            "request": {
                "contents": [
                    { "role": "user", "parts": [{ "text": "Please fix this bug in the repo" }] }
                ]
            }
        });
        assert_eq!(resolve_continuation_prompt(&body), "Continue.");
    }

    #[test]
    fn test_create_auto_heal_continuation_body() {
        let original = json!({
            "requestId": "agent/1234/1",
            "request": {
                "requestId": "agent/1234/1",
                "contents": [
                    { "role": "user", "parts": [{ "text": "Hello" }] }
                ]
            }
        });
        let healed = create_auto_heal_continuation_body(&original, "Continue.");
        assert_eq!(healed["requestId"], "agent/1234/1_heal1");
        assert_eq!(healed["request"]["requestId"], "agent/1234/1_heal1");
        let contents = healed["request"]["contents"].as_array().unwrap();
        // [FIX #3694] 最后一轮已是 user 时合入该轮次，严格保持单轮或交替规则，避免 [user, user] 导致 400
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[0]["parts"][0]["text"], "Hello\n\nContinue.");
    }

    #[test]
    fn test_create_auto_heal_continuation_body_when_last_turn_is_model() {
        let original = json!({
            "request": {
                "contents": [
                    { "role": "user", "parts": [{ "text": "What is 2+2?" }] },
                    { "role": "model", "parts": [{ "text": "Let me calculate." }] }
                ]
            }
        });
        let healed = create_auto_heal_continuation_body(&original, "Continue.");
        let contents = healed["request"]["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 3);
        assert_eq!(contents[1]["role"], "model");
        assert_eq!(contents[2]["role"], "user");
        assert_eq!(contents[2]["parts"][0]["text"], "Continue.");
    }

    #[test]
    fn test_inspect_candidate_parts_pure_thought() {
        let cand = json!({
            "content": {
                "parts": [
                    { "thought": true, "text": "I should run git pull" }
                ]
            }
        });
        let mut saw_thought = false;
        let mut saw_content = false;
        let mut saw_tool_call = false;
        inspect_gemini_candidate_parts(
            &cand,
            &mut saw_thought,
            &mut saw_content,
            &mut saw_tool_call,
        );
        assert!(saw_thought);
        assert!(!saw_content);
        assert!(!saw_tool_call);
    }

    #[test]
    fn test_inspect_candidate_parts_with_tool_call() {
        let cand = json!({
            "content": {
                "parts": [
                    { "thought": true, "text": "I will call bash" },
                    { "functionCall": { "name": "run_command", "args": {} } }
                ]
            }
        });
        let mut saw_thought = false;
        let mut saw_content = false;
        let mut saw_tool_call = false;
        inspect_gemini_candidate_parts(
            &cand,
            &mut saw_thought,
            &mut saw_content,
            &mut saw_tool_call,
        );
        assert!(saw_thought);
        assert!(!saw_content);
        assert!(saw_tool_call);
    }

    #[test]
    fn test_inspect_candidate_parts_with_content() {
        let cand = json!({
            "content": {
                "parts": [
                    { "thought": true, "text": "thinking..." },
                    { "text": "Here is the response" }
                ]
            }
        });
        let mut saw_thought = false;
        let mut saw_content = false;
        let mut saw_tool_call = false;
        inspect_gemini_candidate_parts(
            &cand,
            &mut saw_thought,
            &mut saw_content,
            &mut saw_tool_call,
        );
        assert!(saw_thought);
        assert!(saw_content);
        assert!(!saw_tool_call);
    }

    #[test]
    fn test_inspect_candidate_parts_inline_data() {
        let cand = json!({
            "content": {
                "parts": [
                    { "thought": true, "text": "generating image..." },
                    { "inlineData": { "mimeType": "image/png", "data": "base64..." } }
                ]
            }
        });
        let mut saw_thought = false;
        let mut saw_content = false;
        let mut saw_tool_call = false;
        inspect_gemini_candidate_parts(
            &cand,
            &mut saw_thought,
            &mut saw_content,
            &mut saw_tool_call,
        );
        assert!(saw_thought);
        assert!(saw_content);
        assert!(!saw_tool_call);
    }

    #[test]
    fn test_create_auto_heal_continuation_body_root_contents() {
        let original = json!({
            "requestId": "req_root_123",
            "contents": [
                { "role": "user", "parts": [{ "text": "Do task" }] }
            ]
        });
        let healed = create_auto_heal_continuation_body(&original, "继续");
        assert_eq!(healed["requestId"], "req_root_123_heal1");
        let contents = healed["contents"].as_array().unwrap();
        // [FIX #3694] 最后一轮已是 user 时合入该轮次
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[0]["parts"][0]["text"], "Do task\n\n继续");
    }

    #[test]
    fn test_resolve_continuation_prompt_fallback() {
        let body = json!({
            "request": {
                "contents": []
            }
        });
        assert_eq!(resolve_continuation_prompt(&body), "Continue.");
    }

    #[test]
    fn test_auto_heal_strips_thinking_on_tool_choice_conflict() {
        let classification = crate::proxy::pipeline::UpstreamClassification::classify(
            400,
            "thinking may not be enabled when tool_choice forces tool use",
            None,
        );
        let mut body = json!({
            "request": {
                "generationConfig": { "thinkingConfig": { "includeThoughts": true } }
            }
        });
        let mut flags = AutoHealFlags::default();
        let out = try_auto_heal(
            &classification,
            400,
            "thinking may not be enabled when tool_choice forces tool use",
            &mut body,
            &mut flags,
        );
        assert_eq!(
            out.map(|o| o.kind),
            Some("strip_thinking_on_tool_choice_conflict")
        );
        assert!(flags.retried_tool_choice_conflict);
        assert!(body["request"]["generationConfig"]
            .get("thinkingConfig")
            .is_none());
        assert_eq!(flags.repairs.len(), 1);

        // 第二次不再自愈（单次上限）
        let again = try_auto_heal(
            &classification,
            400,
            "thinking may not be enabled when tool_choice forces tool use",
            &mut body,
            &mut flags,
        );
        assert!(again.is_none());
    }

    #[test]
    fn test_auto_heal_purges_corrupted_images_once() {
        let classification = crate::proxy::pipeline::UpstreamClassification::classify(
            400,
            "unable to process input image",
            None,
        );
        let mut body = json!({
            "contents": [{
                "role": "user",
                "parts": [{
                    "inlineData": { "mimeType": "image/png", "data": "!!!!not-base64!!!!" }
                }]
            }]
        });
        let mut flags = AutoHealFlags::default();
        let out = try_auto_heal(
            &classification,
            400,
            "unable to process input image",
            &mut body,
            &mut flags,
        );
        // 损坏图片被剔除（或占位）；关键是不再重试第二次
        assert!(out.is_some());
        assert!(flags.retried_with_image_omitted);
        let again = try_auto_heal(
            &classification,
            400,
            "unable to process input image",
            &mut body,
            &mut flags,
        );
        assert!(again.is_none());
    }
}
