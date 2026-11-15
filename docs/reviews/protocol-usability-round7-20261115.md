# 协议可用性第七轮修复与复审

基线 `3535d6f99d75fe4faabbc1eb79370e47c53277ab`，延续开发分支 `codex/protocol-usability-round3` / PR #11。本轮修复请求侧4类、响应侧2类问题，未变更版本、发布或部署。

## 请求行为

- 共享inbound管线保留普通正文中的 `<think>`，包括围栏、行内、开头及未闭合XML。思考缓存只删除与明确保存思考内容一致的完整旧包装，UTF-8偏移处理正确。
- 已有Chat音频转写降级为助手文本；原正文顺序保留、完全相同文本不重复，归一化幂等。外部audio.id无法自行解析时不虚构内容。
- Responses已知内置工具历史统一投影为可读助手上下文，保留code interpreter代码/logs/图片、MCP参数/结果/错误、shell命令/stdout/stderr/outcome、图片生成的内联图像或SVG。实际内容有损保留即可，无需重新执行原工具。
- 官方local_shell的flat action与旧嵌套exec包装均保留命令、环境和工作目录；包括旧结构的字符串command。结果关联保留在可读历史中，不再生成空参数的可执行调用。
- 两条转换入口共用同一builtin归一化逻辑，先校验再替换，损坏的已知内容不会被成功伪造。

## 响应行为

- 缺少上游finishReason时，Chat/Claude的纯文本EOF或reset保留部分答案与usage，返回length/max_tokens；Responses非流式通过实际中间转换返回incomplete。Responses专用SSE原有中断判断保留。
- 未完成工具回合返回明确upstream_interrupted，不能获tool_calls/tool_use执行授权；非流式错误保留partial_response和usage。
- 多候选分别处理终态，已完成工具候选保持可用，未完成文本候选截断。STOP完成后的尾部usage或reset不撤销答案。
- `[DONE]`明确结束传输，但不伪造模型完成，连接仍保持pending时也会及时收尾。最后STOP帧缺少换行且随后reset时，先冲刷尾部再判断完成。
- IMAGE_RECITATION正确归类为拒绝；IMAGE_OTHER、工具生成失败和结构化上游错误仍保留错误语义。

## 独立复审

请求与响应实现者互审，另由原生Go修复者独立复查响应路径，主代理复核集成差异。已关闭复审发现的旧local_shell字符串兼容、audio重复归一化、尾部STOP+reset、DONE+pending及include_usage=false截断终帧usage位置问题。当前修改范围无剩余已确认问题。

同时跨仓复审sub2api时发现server tool_search的call_id可省略；已由主代理改为可读历史降级，并用相同探针复测通过，不编造工具调用关联。

## 验证

唯一Cargo进程编译测试目标并执行 **19项新增回归，全部通过**；涉及模块回归 **364 passed / 0 failed**。两组有重叠，不相加为独立测试总数。包含生产mapper、collector、lifecycle、observer及Responses私有JSON转换，使用离线流与本地夹具，无真实模型调用。

既有两项reasoning断言已在前轮旧精确基线复现，本轮目标回归按既定范围排除，未修改业务语义：

- `server_authoritative_ignores_client_reasoning_content_and_budget`
- `test_openai_responses_api_vs_chat_api_thinking_and_signature`

修改文件定向rustfmt及git diff --check通过。遵循AGENTS，本轮没有执行发布专用全仓fmt/clippy或前端构建，不宣称AM全仓测试全绿。上游版本检查继续保留，应用更新动作继续禁用。

证据：`/tmp/am-r32-final-tests.log`、`/tmp/am-r32-final-targeted.log`、`/tmp/am-r32-request-fix-report.md`、`/tmp/am-r32-response-fix-report.md`、`/tmp/am-r32-request-independent-review.md`、`/tmp/am-r32-response-cross-review.md`。
