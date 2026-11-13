// Gemini 原生协议的发散适配器
//
// 只负责“v1internal SSE 行 → 客户端 Gemini SSE 行”的协议差异：
// 签名缓存、工具 ID 注入、图片统计、v1internal 解包、`__cloudCodeMeta` 首帧。
// 心跳、首字/状态切换/滑动超时、优雅收尾与首包错误契约全部由
// `pipeline::stream_lifecycle::run_stream_lifecycle` 统一驱动。

use bytes::Bytes;
use serde_json::Value;
use std::sync::Arc;

use crate::proxy::mappers::error_classifier::StreamErrorReport;
use crate::proxy::pipeline::stream_lifecycle::ProtocolStreamHandler;
use crate::proxy::thinking_store::TurnAccumulator;

pub struct GeminiStreamHandler {
    cloud_code_trace_id: Option<String>,
    session_id: String,
    model_name: String,
    message_count: usize,
    track_image_success: bool,
    image_success_manager: Option<Arc<crate::proxy::TokenManager>>,
    image_success_account: String,
    image_success_model: String,
    capture_template: TurnAccumulator,
    candidates: std::collections::BTreeMap<u64, CandidateState>,
    expected_candidates: usize,
    marked_multi_candidate: bool,
    buffered: bool,
    invalid_candidates: bool,
    saw_image_data: bool,
    saw_finish_reason: bool,
    blocked_prompt: bool,
    whole_prompt_blocked: bool,
    saw_function_call: bool,
    has_content: bool,
    has_thinking: bool,
    is_thinking_active: bool,
}

struct CandidateState {
    thinking: Option<TurnAccumulator>,
    finish: bool,
    active_thought: bool,
}

impl GeminiStreamHandler {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cloud_code_trace_id: Option<String>,
        session_id: String,
        model_name: String,
        message_count: usize,
        causal_anchor: String,
        track_image_success: bool,
        image_success_manager: Option<Arc<crate::proxy::TokenManager>>,
        image_success_account: String,
        image_success_model: String,
    ) -> Self {
        Self::new_with_context(
            cloud_code_trace_id,
            session_id,
            model_name,
            message_count,
            causal_anchor,
            track_image_success,
            image_success_manager,
            image_success_account,
            image_success_model,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_context(
        cloud_code_trace_id: Option<String>,
        session_id: String,
        model_name: String,
        message_count: usize,
        causal_anchor: String,
        track_image_success: bool,
        image_success_manager: Option<Arc<crate::proxy::TokenManager>>,
        image_success_account: String,
        image_success_model: String,
        text_context_anchor: Option<String>,
    ) -> Self {
        let mut thinking_acc = TurnAccumulator::with_anchor(causal_anchor);
        if let Some(anchor) = text_context_anchor {
            thinking_acc = thinking_acc.with_text_context_anchor(anchor);
        }
        Self {
            cloud_code_trace_id,
            session_id,
            model_name,
            message_count,
            track_image_success,
            image_success_manager,
            image_success_account,
            image_success_model,
            capture_template: thinking_acc,
            candidates: Default::default(),
            expected_candidates: 1,
            marked_multi_candidate: false,
            buffered: false,
            invalid_candidates: false,
            saw_image_data: false,
            saw_finish_reason: false,
            blocked_prompt: false,
            whole_prompt_blocked: false,
            saw_function_call: false,
            has_content: false,
            has_thinking: false,
            is_thinking_active: false,
        }
    }
    /// Buffered responses are recorded by the collector once the complete body succeeds.
    pub fn with_candidate_request(mut self, count: usize, buffered: bool) -> Self {
        self.expected_candidates = count.max(1);
        self.buffered = buffered;
        self
    }

    fn incomplete_candidates(&self) -> bool {
        !self.whole_prompt_blocked
            && (self.candidates.len() < self.expected_candidates
                || (self.expected_candidates > 1
                    && (0..self.expected_candidates as u64)
                        .any(|index| !self.candidates.contains_key(&index)))
                || self.candidates.values().any(|c| !c.finish))
    }
}

impl ProtocolStreamHandler for GeminiStreamHandler {
    fn observe_upstream_usage(&mut self, usage: &Value) -> Vec<Bytes> {
        vec![Bytes::from(format!(
            "data: {}\n\n",
            serde_json::json!({"usageMetadata":usage})
        ))]
    }

    fn initial_frames(&mut self) -> Vec<Bytes> {
        // 官方 Worker 会将 TraceID 作为 SSE 流的第 0 个数据包下发
        match &self.cloud_code_trace_id {
            Some(tid) => {
                let meta_pkg = serde_json::json!({ "__cloudCodeMeta": { "traceId": tid } });
                vec![Bytes::from(format!(
                    "data: {}\n\n",
                    serde_json::to_string(&meta_pkg).unwrap_or_default()
                ))]
            }
            None => Vec::new(),
        }
    }

    fn process_line(&mut self, line: &str) -> Vec<Bytes> {
        if !line.starts_with("data: ") {
            // 注释行 / 非数据行原样透传
            return vec![Bytes::from(format!("{}\n\n", line))];
        }

        let json_part = line.trim_start_matches("data: ").trim();
        if json_part == "[DONE]" {
            return vec![Bytes::from("data: [DONE]\n\n")];
        }

        let mut json: Value = match serde_json::from_str(json_part) {
            Ok(v) => v,
            Err(e) => {
                let error_json = serde_json::json!({
                    "error": {
                        "code": 502,
                        "message": format!("Gemini stream json parse error: {e}"),
                        "status": "INTERNAL"
                    }
                });
                return vec![Bytes::from(format!(
                    "data: {}\n\n",
                    serde_json::to_string(&error_json).unwrap_or_default()
                ))];
            }
        };

        if self.track_image_success && crate::proxy::pipeline::response_has_inline_image_data(&json)
        {
            self.saw_image_data = true;
        }

        self.whole_prompt_blocked |=
            crate::proxy::pipeline::stream_error::response_has_prompt_refusal(&json);
        self.blocked_prompt |= crate::proxy::pipeline::stream_error::response_is_refusal(&json);

        // v1internal 响应可能包一层 "response"
        let parts_snapshot = json
            .get("response")
            .unwrap_or(&json)
            .get("candidates")
            .and_then(|c| c.as_array())
            .cloned()
            .unwrap_or_default();

        if self.invalid_candidates {
            return Vec::new();
        }
        let Some(indices) = super::candidate_indices(&parts_snapshot) else {
            self.invalid_candidates = true;
            return vec![Bytes::from_static(b"data: {\"error\":{\"code\":502,\"status\":\"INTERNAL\",\"message\":\"Invalid or duplicate Gemini candidate index\"}}\n\n")];
        };
        let multiple = self.expected_candidates > 1
            || indices.iter().any(|i| *i != 0)
            || self.candidates.keys().any(|i| *i != 0);
        for (index, cand) in indices.into_iter().zip(&parts_snapshot) {
            let state = self
                .candidates
                .entry(index)
                .or_insert_with(|| CandidateState {
                    thinking: (!self.buffered).then(|| self.capture_template.clone()),
                    finish: false,
                    active_thought: false,
                });
            if super::has_candidate_finish(cand) {
                self.saw_finish_reason = true;
                state.finish = true;
            }
            if let Some(parts) = cand
                .get("content")
                .and_then(|c| c.get("parts"))
                .and_then(|p| p.as_array())
            {
                for part in parts {
                    let is_thought = part.get("thought").and_then(|v| v.as_bool()) == Some(true);
                    state.active_thought = is_thought;
                    if is_thought {
                        self.has_thinking = true;
                        self.is_thinking_active = true;
                    } else {
                        self.is_thinking_active = false;
                    }
                    if part.get("functionCall").is_some() {
                        self.saw_function_call = true;
                        self.has_content = true;
                        // 工具调用是明确的行动输出，必然意味着思考阶段已结束。
                        self.is_thinking_active = false;
                    } else if (!is_thought && part.get("text").is_some())
                        || part.get("inlineData").is_some()
                        || part.get("inline_data").is_some()
                    {
                        self.has_content = true;
                    }
                    if let Some(thinking) = &mut state.thinking {
                        thinking.ingest_part(part);
                    }
                }
            }
            // Publish before yielding the terminal event: the client can start the
            // next request immediately, without reading a usage trailer or EOF.
            // Taking the accumulator also makes a repeated terminal idempotent.
            if state.finish {
                if let Some(thinking) = state.thinking.take() {
                    if multiple && !self.marked_multi_candidate {
                        crate::proxy::SignatureCache::global()
                            .mark_multi_candidate_turn(&self.session_id, self.message_count);
                        self.marked_multi_candidate = true;
                    } else if !multiple {
                        if let Some(sig) = thinking.observed_signature() {
                            crate::proxy::SignatureCache::global().cache_session_signature(
                                &self.session_id,
                                sig.to_owned(),
                                self.message_count,
                            );
                        }
                    }
                    thinking.commit(&self.session_id);
                }
            }
        }

        self.is_thinking_active = self
            .candidates
            .values()
            .any(|c| c.active_thought && !c.finish);
        // 注入工具调用 ID
        crate::proxy::mappers::gemini::wrapper::inject_ids_to_response(&mut json, &self.model_name);

        // 解包 v1internal 信封
        if let Some(inner) = json.get_mut("response").map(|v| v.take()) {
            vec![Bytes::from(format!(
                "data: {}\n\n",
                serde_json::to_string(&inner).unwrap_or_default()
            ))]
        } else {
            vec![Bytes::from(format!(
                "data: {}\n\n",
                serde_json::to_string(&json).unwrap_or_default()
            ))]
        }
    }

    fn finalize_without_content(&self) -> bool {
        self.blocked_prompt || self.invalid_candidates
    }

    fn completion_tail_timeout(&self) -> Option<std::time::Duration> {
        // Only prompt refusal completes every candidate in the response.
        (self.whole_prompt_blocked || self.invalid_candidates)
            .then(|| std::time::Duration::from_millis(250))
    }

    fn has_content(&self) -> bool {
        self.has_content
    }

    fn has_thinking(&self) -> bool {
        self.has_thinking
    }

    fn is_thinking_active(&self) -> bool {
        self.is_thinking_active
    }

    fn emit_finalize(&mut self) -> Vec<Bytes> {
        if self.invalid_candidates {
            return Vec::new();
        }
        if (self.expected_candidates > 1 || self.candidates.len() > 1)
            && self.incomplete_candidates()
        {
            return vec![Bytes::from_static(b"data: {\"error\":{\"code\":502,\"status\":\"INTERNAL\",\"message\":\"Stream ended before all Gemini candidates completed\"}}\n\n")];
        }
        // 图片生成成功统计仅在优雅结束时提交（错误路径不会走到这里）
        if self.track_image_success && self.saw_image_data {
            if let Some(mgr) = &self.image_success_manager {
                mgr.mark_account_success(&self.image_success_account);
                mgr.clear_persisted_live_limit(
                    &self.image_success_account,
                    Some(&self.image_success_model),
                );
            }
        }

        if self.expected_candidates > 1 || self.candidates.len() > 1 {
            return Vec::new();
        }
        // [Fixes #3813] 情形 B 异常收尾：上游输出了正文但未下发任何 finishReason 即遭遇未预期 EOF
        // 必须提供明确的合成终态与 finishMessage 诊断，标明流在正常完成前被提前截断，杜绝下游误判为正常结束
        if self.has_content && !self.saw_finish_reason && !self.blocked_prompt {
            self.saw_finish_reason = true;
            let finish_reason = if self.saw_function_call {
                "STOP"
            } else {
                "MAX_TOKENS"
            };

            let recovery_json = serde_json::json!({
                "candidates": [{
                    "finishReason": finish_reason,
                    "finishMessage": "Stream closed prematurely (EOF without finishReason) by upstream. Preceding content preserved by gateway.",
                    "gatewayInterrupted": true,
                    "index": 0
                }]
            });

            return vec![Bytes::from(format!(
                "data: {}\n\n",
                serde_json::to_string(&recovery_json).unwrap_or_default()
            ))];
        }

        // 正常流（已含 finishReason）：Gemini 正常 EOF 不额外下发 [DONE]，保持既有行为
        Vec::new()
    }

    fn emit_abnormal_finalize(&mut self, error_report: &StreamErrorReport) -> Vec<Bytes> {
        let mut chunks = Vec::new();

        // 终态与内容防护：若上游已下发 finishReason 或从未产生任何正文/工具调用，绝不可伪造收尾
        if (self.expected_candidates > 1 || self.candidates.len() > 1)
            && self.incomplete_candidates()
        {
            return self.emit_initial_error(error_report);
        }
        if !self.has_content || self.saw_finish_reason {
            return chunks;
        }
        self.saw_finish_reason = true;

        // 工具调用契约保护：若本轮已有工具调用输出，终结原因保持 STOP，否则标记 MAX_TOKENS
        let finish_reason = if self.saw_function_call {
            "STOP"
        } else {
            "MAX_TOKENS"
        };

        // [Fixes #3813] 诊断信息如实传达：通过官方标准 finishMessage 及流内诊断字段透传真实中断原因，
        // 既保全已生成的成千上万 token 与客户端兼容性，又彻底避免下游将连接中断/超时误诊为模型自然输出上限
        let error_msg = error_report.client_message();
        let finish_message = format!(
            "Stream interrupted before completion ({}): {}. Preceding content preserved by gateway.",
            error_report.classified.error_type,
            error_report.raw
        );

        let recovery_json = serde_json::json!({
            "candidates": [{
                "finishReason": finish_reason,
                "finishMessage": finish_message,
                "gatewayInterrupted": true,
                "gatewayError": {
                    "type": error_report.classified.error_type,
                    "message": error_msg,
                    "raw": error_report.raw
                },
                "index": 0
            }]
        });

        chunks.push(Bytes::from(format!(
            "data: {}\n\n",
            serde_json::to_string(&recovery_json).unwrap_or_default()
        )));
        chunks
    }

    fn emit_initial_error(&mut self, error_report: &StreamErrorReport) -> Vec<Bytes> {
        let error_json = serde_json::json!({
            "error": {
                "code": error_report.upstream_status.unwrap_or(503),
                "message": error_report.client_message(),
                "status": match error_report.upstream_status.unwrap_or(503) {
                    400 => "INVALID_ARGUMENT", 401 => "UNAUTHENTICATED",
                    403 => "PERMISSION_DENIED", 404 => "NOT_FOUND",
                    429 => "RESOURCE_EXHAUSTED", 503 => "UNAVAILABLE",
                    504 => "DEADLINE_EXCEEDED", _ => "INTERNAL",
                },
                "function": error_report.function,
                "call_site": error_report.call_site(),
                "params": error_report.params
            }
        });
        vec![
            Bytes::from(format!(
                "data: {}\n\n",
                serde_json::to_string(&error_json).unwrap_or_default()
            )),
            Bytes::from("data: [DONE]\n\n"),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    fn test_gemini_handler() -> GeminiStreamHandler {
        GeminiStreamHandler::new(
            None,
            "session_gemini_phase".to_string(),
            "gemini-2.5-pro".to_string(),
            0,
            "anchor".to_string(),
            false,
            None,
            "acc@example.com".to_string(),
            "gemini-2.5-pro".to_string(),
        )
    }

    #[tokio::test]
    async fn blocking_prompt_feedback_finishes_cleanly_and_retains_usage_trailers() {
        for reason in [
            "SAFETY",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
            "MODEL_ARMOR",
            "FUTURE_BLOCK_REASON",
        ] {
            for ending in ["\n\n", ""] {
                let blocked = serde_json::json!({"response": {
                    "promptFeedback":{"blockReason":reason},
                    "usageMetadata":{"promptTokenCount":100,"cachedContentTokenCount":80}
                }});
                let trailer = serde_json::json!({"usageMetadata":{"candidatesTokenCount":0}});
                let wire = format!("data: {blocked}\n\ndata: {trailer}{ending}");
                let source = futures::stream::iter(
                    wire.as_bytes()
                        .chunks(3)
                        .map(|chunk| Ok::<_, String>(Bytes::copy_from_slice(chunk)))
                        .collect::<Vec<_>>(),
                );
                let stream = crate::proxy::pipeline::run_stream_lifecycle(
                    Box::pin(source),
                    test_gemini_handler(),
                    crate::proxy::pipeline::StreamLifecycleConfig::new(
                        "gemini",
                        "blocked_prompt",
                        "test".into(),
                    ),
                );
                let chunks = stream.collect::<Vec<_>>().await;
                assert!(chunks.iter().all(Result::is_ok), "{reason}: {chunks:?}");
                let output = chunks
                    .into_iter()
                    .flat_map(|chunk| chunk.unwrap().to_vec())
                    .collect::<Vec<_>>();
                let output = String::from_utf8(output).unwrap();
                assert!(output.contains(reason));
                assert!(output.contains("cachedContentTokenCount"));
                assert!(output.contains("candidatesTokenCount"));
                assert!(!output.contains("gatewayInterrupted"));
            }
        }
    }

    #[test]
    fn default_prompt_feedback_does_not_complete_an_empty_response() {
        for feedback in [
            serde_json::json!({}),
            serde_json::json!({"blockReason":null}),
            serde_json::json!({"blockReason":""}),
            serde_json::json!({"blockReason":"BLOCK_REASON_UNSPECIFIED"}),
        ] {
            let mut handler = test_gemini_handler();
            handler.process_line(&format!(
                "data: {}",
                serde_json::json!({"promptFeedback":feedback})
            ));
            assert!(!handler.finalize_without_content());
            assert!(handler.completion_tail_timeout().is_none());
        }
    }

    #[tokio::test]
    async fn blocked_prompt_transport_close_does_not_revoke_refusal() {
        let source = futures::stream::iter(vec![
            Ok(Bytes::from(
                "data: {\"promptFeedback\":{\"blockReason\":\"SAFETY\"}}\n\n",
            )),
            Err("connection reset after refusal".to_string()),
        ]);
        let stream = crate::proxy::pipeline::run_stream_lifecycle(
            Box::pin(source),
            test_gemini_handler(),
            crate::proxy::pipeline::StreamLifecycleConfig::new(
                "gemini",
                "blocked_prompt",
                "test".into(),
            ),
        );
        assert!(stream.collect::<Vec<_>>().await.iter().all(Result::is_ok));
    }

    #[test]
    fn test_gemini_handler_tracks_thinking_phase_lifecycle() {
        let mut handler = test_gemini_handler();

        handler.process_line(
            r#"data: {"candidates":[{"content":{"parts":[{"thought":true,"text":"Pondering."}]}}]}"#,
        );
        assert!(handler.has_thinking());
        assert!(handler.is_thinking_active());
        assert!(!handler.has_content());

        handler
            .process_line(r#"data: {"candidates":[{"content":{"parts":[{"text":"Answer."}]}}]}"#);
        assert!(handler.has_content());
        assert!(!handler.is_thinking_active());

        // finishReason-only chunk must also close the thinking phase.
        let mut finish_handler = test_gemini_handler();
        finish_handler.process_line(
            r#"data: {"candidates":[{"content":{"parts":[{"thought":true,"text":"Pondering."}]}}]}"#,
        );
        assert!(finish_handler.is_thinking_active());
        finish_handler.process_line(r#"data: {"candidates":[{"finishReason":"STOP"}]}"#);
        assert!(!finish_handler.is_thinking_active());
    }

    #[tokio::test]
    async fn test_gemini_stream_abnormal_interruption_preserves_content_without_tampering() {
        let chunk1_json = serde_json::json!({
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{ "text": "Here is code:\n```rust\nfn main() {" }]
                    }
                }]
            }
        });

        let items: Vec<Result<Bytes, String>> = vec![
            Ok(Bytes::from(format!(
                "data: {}\n\n",
                serde_json::to_string(&chunk1_json).unwrap()
            ))),
            Err("error reading a body from connection: connection reset by peer".to_string()),
        ];

        let handler = GeminiStreamHandler::new(
            Some("trace_gemini_test".to_string()),
            "session_gemini_test".to_string(),
            "gemini-2.5-pro".to_string(),
            1,
            "anchor".to_string(),
            false,
            None,
            "acc@example.com".to_string(),
            "gemini-2.5-pro".to_string(),
        );

        let lifecycle_cfg = crate::proxy::pipeline::StreamLifecycleConfig::new(
            "gemini",
            "test_gemini_stream_abnormal_interruption",
            "test".to_string(),
        );

        let mut stream = crate::proxy::pipeline::run_stream_lifecycle(
            Box::pin(futures::stream::iter(items)),
            handler,
            lifecycle_cfg,
        );

        let mut all_chunks = Vec::new();
        while let Some(result) = stream.next().await {
            if let Ok(bytes) = result {
                all_chunks.push(String::from_utf8(bytes.to_vec()).unwrap());
            }
        }
        let output = all_chunks.join("");

        assert!(output.contains("__cloudCodeMeta"));
        assert!(output.contains("```rust\\nfn main() {"));
        // 绝不注入虚假网关告警或篡改出站正文
        assert!(!output.contains("网关提示"));
        assert!(!output.contains("Gateway Notice"));
        assert!(output.contains("MAX_TOKENS"));
        // [Fixes #3813] 断流异常必须包含真实错误类型与诊断 finishMessage
        assert!(output.contains("finishMessage"));
        assert!(output.contains("gatewayInterrupted"));
        assert!(output.contains("connection reset by peer"));
    }

    #[tokio::test]
    async fn test_gemini_stream_eof_without_finish_reason_emits_premature_diagnostic() {
        let chunk_without_finish = serde_json::json!({
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{ "text": "Incomplete answer before EOF" }]
                    },
                    "index": 0
                }]
            }
        });

        // 模拟情形 B：产生了正文，但上游未经下发 finishReason 直接静默 EOF 退出
        let items: Vec<Result<Bytes, String>> = vec![Ok(Bytes::from(format!(
            "data: {}\n\n",
            serde_json::to_string(&chunk_without_finish).unwrap()
        )))];

        let handler = GeminiStreamHandler::new(
            Some("trace_gemini_premature_eof".to_string()),
            "session_gemini_premature_eof".to_string(),
            "gemini-2.5-pro".to_string(),
            1,
            "anchor".to_string(),
            false,
            None,
            "acc@example.com".to_string(),
            "gemini-2.5-pro".to_string(),
        );

        let lifecycle_cfg = crate::proxy::pipeline::StreamLifecycleConfig::new(
            "gemini",
            "test_gemini_premature_eof",
            "test".to_string(),
        );

        let mut stream = crate::proxy::pipeline::run_stream_lifecycle(
            Box::pin(futures::stream::iter(items)),
            handler,
            lifecycle_cfg,
        );

        let mut all_chunks = Vec::new();
        while let Some(result) = stream.next().await {
            if let Ok(bytes) = result {
                all_chunks.push(String::from_utf8(bytes.to_vec()).unwrap());
            }
        }
        let output = all_chunks.join("");

        assert!(output.contains("Incomplete answer before EOF"));
        assert!(output.contains("MAX_TOKENS"));
        assert!(output.contains("gatewayInterrupted"));
        assert!(output.contains("Stream closed prematurely"));
    }

    #[tokio::test]
    async fn test_gemini_stream_pure_thinking_does_not_set_has_content_and_propagates_err() {
        let thinking_json = serde_json::json!({
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{ "thought": true, "text": "Thinking deeply..." }]
                    }
                }]
            }
        });

        let items: Vec<Result<Bytes, String>> = vec![
            Ok(Bytes::from(format!(
                "data: {}\n\n",
                serde_json::to_string(&thinking_json).unwrap()
            ))),
            Err("error reading a body from connection: connection reset by peer".to_string()),
        ];

        let handler = GeminiStreamHandler::new(
            Some("trace_gemini_thinking".to_string()),
            "session_gemini_thinking".to_string(),
            "gemini-2.5-pro".to_string(),
            1,
            "anchor".to_string(),
            false,
            None,
            "acc@example.com".to_string(),
            "gemini-2.5-pro".to_string(),
        );

        let lifecycle_cfg = crate::proxy::pipeline::StreamLifecycleConfig::new(
            "gemini",
            "test_gemini_thinking",
            "test".to_string(),
        );

        let mut stream = crate::proxy::pipeline::run_stream_lifecycle(
            Box::pin(futures::stream::iter(items)),
            handler,
            lifecycle_cfg,
        );

        let mut all_chunks = Vec::new();
        let mut got_err = false;
        while let Some(result) = stream.next().await {
            match result {
                Ok(bytes) => all_chunks.push(String::from_utf8(bytes.to_vec()).unwrap()),
                Err(_) => {
                    got_err = true;
                }
            }
        }
        let output = all_chunks.join("");

        // 思考阶段如实透传错误，严禁下发假正文与 MAX_TOKENS 伪造收尾
        assert!(got_err, "Pure thinking interruption must propagate error");
        assert!(!output.contains("MAX_TOKENS"));
    }

    #[tokio::test]
    async fn test_gemini_stream_completed_stream_does_not_double_finalize() {
        let completed_json = serde_json::json!({
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{ "text": "Hello world" }]
                    },
                    "finishReason": "STOP",
                    "index": 0
                }]
            }
        });

        let items: Vec<Result<Bytes, String>> = vec![
            Ok(Bytes::from(format!(
                "data: {}\n\n",
                serde_json::to_string(&completed_json).unwrap()
            ))),
            // 上游在下发带 STOP 数据包后断开
            Err("connection reset by peer".to_string()),
        ];

        let handler = GeminiStreamHandler::new(
            Some("trace_gemini_done".to_string()),
            "session_gemini_done".to_string(),
            "gemini-2.5-pro".to_string(),
            1,
            "anchor".to_string(),
            false,
            None,
            "acc@example.com".to_string(),
            "gemini-2.5-pro".to_string(),
        );

        let lifecycle_cfg = crate::proxy::pipeline::StreamLifecycleConfig::new(
            "gemini",
            "test_gemini_done",
            "test".to_string(),
        );

        let mut stream = crate::proxy::pipeline::run_stream_lifecycle(
            Box::pin(futures::stream::iter(items)),
            handler,
            lifecycle_cfg,
        );

        let mut all_chunks = Vec::new();
        while let Some(result) = stream.next().await {
            if let Ok(bytes) = result {
                all_chunks.push(String::from_utf8(bytes.to_vec()).unwrap());
            }
        }
        let output = all_chunks.join("");

        // 已正常下发 STOP 的流，严禁再次触发 emit_abnormal_finalize 附加 MAX_TOKENS
        assert!(output.contains(r#""finishReason":"STOP""#));
        assert!(!output.contains("MAX_TOKENS"));
    }

    #[tokio::test]
    async fn test_gemini_stream_normal_eof_reaches_emit_finalize() {
        let completed_json = serde_json::json!({
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{ "text": "Normal completion" }]
                    },
                    "finishReason": "STOP",
                    "index": 0
                }]
            }
        });

        // 正常流：上游正常以 EOF (None) 终结，而非 Err 中断
        let items: Vec<Result<Bytes, String>> = vec![Ok(Bytes::from(format!(
            "data: {}\n\n",
            serde_json::to_string(&completed_json).unwrap()
        )))];

        let handler = GeminiStreamHandler::new(
            Some("trace_gemini_eof".to_string()),
            "session_gemini_eof".to_string(),
            "gemini-2.5-pro".to_string(),
            1,
            "anchor".to_string(),
            false,
            None,
            "acc@example.com".to_string(),
            "gemini-2.5-pro".to_string(),
        );

        let lifecycle_cfg = crate::proxy::pipeline::StreamLifecycleConfig::new(
            "gemini",
            "test_gemini_eof",
            "test".to_string(),
        );

        let mut stream = crate::proxy::pipeline::run_stream_lifecycle(
            Box::pin(futures::stream::iter(items)),
            handler,
            lifecycle_cfg,
        );

        let mut all_chunks = Vec::new();
        while let Some(result) = stream.next().await {
            if let Ok(bytes) = result {
                all_chunks.push(String::from_utf8(bytes.to_vec()).unwrap());
            }
        }
        let output = all_chunks.join("");

        assert!(output.contains("Normal completion"));
        assert!(output.contains(r#""finishReason":"STOP""#));
        assert!(!output.contains("MAX_TOKENS"));
    }
}
