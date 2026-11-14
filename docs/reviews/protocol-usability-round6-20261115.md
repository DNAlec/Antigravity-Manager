# 协议可用性第六轮修复与复审

基线：`62a2770aa8109003fe69caac000091f09009330a`。本轮延续 PR #11；开发分支 `codex/protocol-usability-round3`，未变更版本、发布或部署。

## 修复后的行为

- Claude / Responses 单答案流在首个有效帧固定候选身份。重排、稀疏帧中的其它候选不能混入正文、工具或媒体，也不能提前结束选中答案。渲染器与错误观察器共用身份；全局错误、选中候选失败与真实中断仍失败。普通 Chat 和原生 Gemini 保留多候选。
- 工具结果后的新指令按原文保留，空格、tab 差异和明确重复请求不再被“任务回显”去重删除。
- 普通助手正文以 `**Thinking**` 开头仍保留。Responses 入口及共享 inbound 管线均取消纯文本标题推断；明确 `msg_thought_` 私有展示项和显式 thought 块仍按既有规则处理。
- 官方 Chat assistant `refusal` 内容块保留为按序可读文本；token 估算与会话摘要同步兼容。
- 官方 Claude `tool_search_tool_result` 支持成功引用与错误记录，保留工具名、调用 ID 和错误正文；跨协议降为文本，不伪造本地工具调用。

## 验证与复查

唯一 Cargo 进程运行涉及的请求 mapper、Responses/Chat 输出、Claude usage tail、共享 inbound、observer、会话和上下文模块，以及全部 `r30_`、`r28_` 回归：**303 passed / 0 failed / 1712 filtered**。新增 r30 覆盖 6 个请求回归和 5 个响应回归，其中候选矩阵覆盖 6 种输入 × 5 条路径，另有非零候选的工具/媒体和 270KB 大帧。

单独复跑历史两项 reasoning 断言：**0 passed / 2 failed**。它们仍要求客户端历史 reasoning 不出现在出站报文；同样失败已在旧精确基线 `cf4cd0fbad6adda148f32b8e294fbf1e52596dfe` 复现，本轮未改该业务语义：

- `server_authoritative_ignores_client_reasoning_content_and_budget`
- `test_openai_responses_api_vs_chat_api_thinking_and_signature`

请求与响应实现者互审后，主代理再次审阅 selector、观察器、三条 handler 接线、历史过滤、typed 内容块及测试。首轮测试揭示共享管线的第二处标题误删，已修复；互审发现大帧测试误用无限制观察模式，已补 bounded/full 双模式及独立对照，最终通过。当前未确认新增阻断问题。

改动文件定向 rustfmt 与 `git diff --check` 通过。遵照 AGENTS.md，本次没有运行发布专用全仓 fmt/clippy 或前端构建。

测试为离线真实 mapper、collector、生命周期和 observer；不涉及生产账号、付费推理或认证/调度 HTTP 端到端。不是全仓测试通过声明。上游版本检查继续保留，应用更新动作继续禁用。

本地证据：`/tmp/am-r30-tests.log`、`/tmp/am-r30-known-baseline.log`、`/tmp/am-r30-request-fix-report.md`、`/tmp/am-r30-response-fix-report.md`。
