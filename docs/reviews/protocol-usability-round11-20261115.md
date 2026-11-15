# 协议可用性第十一轮 AM 修复与复核

本轮修复 r39 确认的两组 AM 请求兼容性问题（R6、R7），基于 `codex/protocol-usability-round3`、HEAD `3535d6f99d75fe4faabbc1eb79370e47c53277ab` 的真实工作树，保留前轮所有未提交改动。没有切换分支、提交、推送、合并或发布。

## 修复结果

| 问题 | 最终行为 | 实现位置 |
| --- | --- | --- |
| Claude 合法 `fallback` / `advisor_tool_result` 历史使整条请求 400 | 明确识别 fallback 但不把路由标记当作对话正文；单独标记消息不生成空上游消息。Advisor 可读结果原样保留，错误码以可读错误保留，加密结果明确说明正文不可读 | [Claude 类型](../../src-tauri/src/proxy/mappers/claude/models.rs)、[共享请求内容映射](../../src-tauri/src/proxy/mappers/claude/request.rs) |
| Responses `program_output.result` / `mcp_list_tools.error` 未进入模型历史 | `program` 代码与 `program_output` 结果、调用标识和 completed/incomplete 状态保留；MCP 列表保留服务器、工具名、可读描述和真实错误，包括失败及空列表 | [共享内置工具历史投影](../../src-tauri/src/proxy/handlers/openai_builtin_history.rs) |

按 Pipeline First，Claude 协议差异在现有 typed decoder 和共享内容映射处理；Responses 新项进入现有 `lower`/`normalize` 投影入口，HTTP、WebSocket、预热及直接转换共用同一处理逻辑。未复制 handler 分支、增加模型特例或变更响应流生命周期。

## 行为边界

- `fallback` 的 from/to/trigger 和可选元数据只表示原供应商路由边界，全部丢弃；trigger 为 null 或对象均不影响相邻正文。独立 fallback-only assistant 消息通过已有空 parts 处理移除。
- `advisor_result.text` 原样保留，包括字面 `<think>` 和看似工具调用的文本；空文本可跳过。`advisor_tool_result_error.error_code` 保留为可读错误；`advisor_redacted_result` 产生诚实的不可读说明，不传递或尝试解密密文。既有 server_tool_use 不重放 advisor，也不伪造客户端工具结果。
- Advisor 已知可读正文和错误码缺失或类型错误、未知 result variant 仍由明确类型校验拒绝；不采用全局未知 block 静默丢弃。
- Program 与 MCP 项只作为历史事实投影，不执行工具。program 的 fingerprint、MCP 的 schema/annotations 和未知可选元数据不进入模型正文。MCP 可选 description 的非字符串扩展可降级，实际 `result`、`error`、`tools` 和工具名仍要求正确类型。
- 原子验证沿用既有实现：所有新项成功投影后才替换 input，错误时不改写已有请求或 WebSocket/预热 pending history。连续归一化结果保持幂等。

本轮保证可读工具历史进入最终 Gemini `request.contents`，不声称保留外国工具的执行能力、密文状态或完整 schema 语义。

## 永久回归

[请求历史测试](../../src-tauri/src/proxy/handlers/openai_request_history_tests.rs) 新增 10 个 `r40_` 永久回归，使用真实 ClaudeRequest 解码与完整 Claude mapper，或 Responses normalizer→converter→OpenAIRequest→完整 Gemini mapper。断言读取最终 `request.contents`，不以 typed 中间对象替代模型可读内容。

覆盖 fallback 混合/独立 no-op 和可选元数据；advisor 字面正文、空文本、全部七种官方错误码、加密结果诚实说明、无工具重放、已知非法字段拒绝；program code/result/completed/incomplete；MCP 成功工具名/描述、失败和空列表；可选 opaque 元数据降级；直接与归一化路径一致、幂等、错误原子性；WebSocket 与预热的完整映射及失败时历史保持。

既有请求、会话连续性及 Claude collector/response 测试随受影响模块一起运行。未改写已有测试期望。

## 验证

初始 10 个新回归：**10 passed / 0 failed / 2067 filtered out**，日志 `/tmp/am-r40-initial.log`。

唯一 Cargo runner 使用以下最终定向命令：

```sh
cd /home/alec/workspace/Antigravity-Manager/src-tauri
ABV_DATA_DIR=/tmp/am-r40-runtime \
CARGO_TARGET_DIR=/home/alec/workspace/Antigravity-Manager/src-tauri/target \
cargo test --locked --lib -- \
  r40_ request_history_tests responses_plaintext_compatibility_tests \
  responses_continuity_tests responses_http_history_tests \
  responses_websocket_history_tests websocket_native_responses_tests \
  mappers::claude::request:: mappers::claude::models:: \
  mappers::claude::thinking_utils:: mappers::claude::collector:: \
  mappers::claude::response:: \
  --skip server_authoritative_ignores_client_reasoning_content_and_budget \
  --skip test_openai_responses_api_vs_chat_api_thinking_and_signature \
  --test-threads=1 > /tmp/am-r40-final-targeted.log 2>&1
```

最终结果：**131 passed / 0 failed / 0 ignored / 1946 filtered out**；编译缓存检查 0.68s、测试 0.35s，退出码 0。10 个新增回归及选中既有测试均通过。两个既有 baseline 排除如命令所示，本轮没有修复或隐藏其状态。编译器仍报告既有 59 条 warnings。

`git diff --check` 通过；仅对本轮四个 Rust 文件运行独立 rustfmt。没有运行 full suite、clippy、前端构建、release build 或发布预检，没有调用真实供应商或生产账号。HTTP 结论来自共享入口源码可达性和完整生产 mapper；WebSocket/预热另有生产归一化入口的定向测试，不声称远端端到端验证。

独立复核逐项检查四文件增量、fallback-only 空消息处理、advisor 三类结果、program/MCP 共享投影、原子性/幂等及 WebSocket/预热覆盖，结论无阻塞发现。复核者未另跑 Cargo；证据见 `/tmp/am-r40-request-cross-review.md`。

## 完整性

修改前四个文件的脏版本保存在 `/tmp/am-r40-before`，增量 diff 位于 `/tmp/am-r40-change.diff`。`/tmp/am-r40-integrity.json` 对比 r39 的 588 文件基线：仅上述四个源文件变化，另新增本篇报告；其余既有文件（含前轮未提交改动）保持不变，HEAD 与分支保持不变。
