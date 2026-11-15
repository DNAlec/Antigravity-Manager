# 协议可用性第十轮 AM 修复与复核

本轮修复 r37 审查确认的三项 AM 请求兼容性问题，基于 `codex/protocol-usability-round3`、HEAD `3535d6f99d75fe4faabbc1eb79370e47c53277ab` 的真实工作树，保留第七至九轮所有未提交改动。没有切换分支、提交、推送、合并或发布。本轮四个修改文件的原始脏版本在 `/tmp/am-r38-before`，增量 diff 在 `/tmp/am-r38-change.diff`。

## 修复结果

| 问题 | 最终行为 | 实现位置 |
| --- | --- | --- |
| Claude compaction 可读摘要或 null block 使整条请求解析失败 | 明确识别 compaction，把可读 content 作为普通文本送入现有映射；null、缺失和空摘要作为 no-op，周边正文不受影响；非字符串内容仍按已知类型错误拒绝 | [Claude 类型](../../src-tauri/src/proxy/mappers/claude/models.rs)、[请求内容映射](../../src-tauri/src/proxy/mappers/claude/request.rs) |
| foreign opaque reasoning 阻塞完整可读 Responses 对话 | 输入已含可读 user→assistant 历史时，可跳过没有可读摘要的外国加密 reasoning；已有可读摘要仍保留。HTTP、WebSocket 和预热入口共用同一归一化器，正文及媒体正常继续映射 | [Responses 入站验证与归一化](../../src-tauri/src/proxy/handlers/openai.rs) |
| Claude web_search 历史删除实际错误和可读结果 | 保留错误类型/错误码，以及成功结果的页面标题和 URL；字段白名单阻止密文或未知结果元数据进入模型正文。服务器工具不重新执行，也不伪造客户端工具调用 | [服务器工具结果投影](../../src-tauri/src/proxy/mappers/claude/request.rs) |

这些数据在适配器解析或转换前被拒绝/删除，通用 Gemini pipeline 无法恢复。按 Pipeline First，在已有协议归一化层识别协议差异，将可读内容送入现有公共文本和服务器结果映射，再交给原有 pipeline；未添加模型特例、工具执行逻辑或新的设置。

## 兼容性边界与相邻复核

Responses 的 opaque 降级只适用于非空字符串密文、无 `thoughtSignature`/`thought_signature`/`signature` 状态字段，而且请求自身已有可读 user→assistant 消息。可读内容包括普通字符串、text/input_text/output_text 块，以及 assistant 的官方 refusal 文本；assistant 必须无状态或已 completed。非 message 类型、未知内容块、空白内容、未完成回答和 `msg_thought_` 本地仅展示思考消息不能作为足够历史的依据。

仍拒绝 opaque-only、opaque 加一句“Continue”、只有 assistant 没有任务、非法密文类型、带签名状态的 opaque 项及纯密文 Responses compaction；验证在任何归一化修改和历史消耗之前完成。已有负向测试未删除，WebSocket/预热新增验证还检查失败时保留原来的 pending history。AM 不生成 `encrypted_content` 私有加密 envelope，本地 Gemini 状态保存在签名和会话存储中；本轮未新增解密器，也没有修改本地状态/游标校验。

交叉复核补齐两处同类边界：成功 web_search 结果的 title/url 能直接回答后续的“列出页面名称和链接”，应作为可读历史保留；合法 refusal 是已有适配器支持的可读回答，也应满足明文历史条件。两者均加入完整生产转换回归。搜索结果和错误只投影白名单字段，不转储 provider encrypted_content。

compaction 的空字符串虽然不承载摘要，本轮按用户的可用性要求与 null 一样跳过，避免空元数据拒绝整条可用对话。非字符串错误仍由 `Option<String>` 的类型校验处理。文本摘要中的字面 `<think>` 内容、普通文本顺序，以及可选外国签名/密文旁边的正文均有对照。

r37 响应侧“同一候选 STOP 后反转为 OTHER/NO_IMAGE”的合成序列仍只计观测：缺少合法协议契约或真实供应商序列证据，不作为本轮生产修复项。本轮没有修改响应流的终态处理。

## 既有测试期望校正

较广的定向测试发现旧 `failed_empty_and_refused_streams_do_not_save_history` 中一个子例过时：可读 partial 和明确 MAX_TOKENS 终态已经到达，随后发生 transport reset，现有生命周期会保留可续接的 incomplete 回答，而旧子例仍要求不保存历史。

在完全未含 r38 修改的 `/tmp/am-r37-review` 快照中单跑此测试，同样失败；证据是 `/tmp/am-r38-baseline-history.log`。本轮没有改变生产行为，只将这一子例迁移成正向回归，明确断言 `incomplete/max_output_tokens`、partial 正文、已保存历史及下一条用户消息可接续。其余四个负向子例（缺少模型终态、空 MAX_TOKENS、仅思考 MAX_TOKENS、SAFETY）原样保留；最终定向测试没有排除该测试。

## 永久回归与验证

新增 12 个 `r38_` 永久测试：11 个请求/入口测试，以及上述迁移的 1 个终态历史测试。请求矩阵覆盖 compaction 明文/可选元数据/null/缺失/空摘要、非法类型、六种合法搜索错误、多页面标题/URL、密文不转储、foreign opaque 的正常和负向边界、已存在摘要、合法拒答、图片保留、归一化幂等及 HTTP/WebSocket/预热一致性。

- [请求回归](../../src-tauri/src/proxy/handlers/openai_request_history_tests.rs) 使用真实 ClaudeRequest 解析和完整 Claude mapper，或真实 Responses normalizer→converter→OpenAIRequest→完整 Gemini mapper。
- [会话与终态回归](../../src-tauri/src/proxy/handlers/openai.rs) 使用生产 SSE mapper、保存回调、会话存储和历史合并路径。
- 正文与媒体断言读取最终 `request.contents`，不以保存在 typed 原始对象或审计 ledger 中代替真正送给模型的内容。

唯一 Cargo runner 使用隔离运行目录、真实 target 缓存，最终命令为：

```sh
cd /home/alec/workspace/Antigravity-Manager/src-tauri
ABV_DATA_DIR=/tmp/am-r38-runtime \
CARGO_TARGET_DIR=/home/alec/workspace/Antigravity-Manager/src-tauri/target \
cargo test --locked --lib -- \
  r38_ request_history_tests responses_plaintext_compatibility_tests \
  responses_continuity_tests responses_http_history_tests \
  responses_websocket_history_tests websocket_native_responses_tests \
  mappers::claude::request:: mappers::claude::models:: \
  mappers::claude::thinking_utils:: \
  --skip server_authoritative_ignores_client_reasoning_content_and_budget \
  --skip test_openai_responses_api_vs_chat_api_thinking_and_signature \
  --test-threads=1 > /tmp/am-r38-final-targeted.log 2>&1
```

最终结果：**114 passed / 0 failed / 0 ignored / 1953 filtered out**，编译 20.61s，测试 0.31s；12 个 `r38_` 永久回归全部通过，原有 opaque/history 负向回归也通过。最终通过依据为 `/tmp/am-r38-final-targeted.log`。既有两项 baseline 排除仍显式保留，未声称本轮修复。格式化仅涉及本轮修改的 Rust 文件；`git diff --check` 通过。没有运行 full suite、clippy、前端构建、release build 或发布预检。

所有验证均为本地离线生产转换与本地流/存储测试，未调用真实供应商或生产账号。完整性和报告见 `/tmp/am-r38-fix-review.md`。独立交叉复审 `/tmp/am-r38-request-cross-review.md` 已检查最终四个源文件哈希，结论无未解决 P1/P2；复审提出的 refusal 漏项已修复并通过最终测试。

`/tmp/am-r38-integrity.json` 对比 r37 的 587 文件基线：仅本轮四个文件变化，另新增本篇报告；其余既有脏文件内容保持不变，HEAD 不变。最终增量 diff 位于 `/tmp/am-r38-change.diff`。
