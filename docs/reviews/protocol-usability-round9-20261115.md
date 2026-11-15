# 协议可用性第九轮 AM 修复与复核

本轮修复第九轮审查中的四类 AM 问题（总报告 #1、#5、#6、#10），以可读正文、已有附件和可靠的工具/完成状态为目标。基线为 `codex/protocol-usability-round3`、HEAD `3535d6f99d75fe4faabbc1eb79370e47c53277ab` 的实际工作树，保留第七、八轮未提交修复；本轮开始快照在 `/tmp/protocol-r36-before/am`。没有切换分支、提交、推送、合并或发布。

## 修复结果

| 问题 | 最终行为 | 实现位置 |
| --- | --- | --- |
| Claude 文本在工具名处分片，错误生成空参数调用 | `call:default_api:*` 始终作为正文保留，分片位置不再影响文本或产生执行请求；真正的结构化 `functionCall` 仍能生成带完整参数的 `tool_use` | [Claude 流式文本处理](../../src-tauri/src/proxy/mappers/claude/streaming.rs)；移除已无用途的恢复状态及启发式解析 |
| Responses 已完成工具历史丢失正文/截图 | `file_search` 保留检索正文；`apply_patch` 保留操作、diff、成功/失败说明；`computer` 保留动作和已有截图。转换成普通可读/媒体消息，不重新执行工具 | [内置工具历史规范化](../../src-tauri/src/proxy/handlers/openai_builtin_history.rs) |
| 有效 text/* 附件被当作坏图片删除 | 共享 inline 处理把 UTF-8 文本解码为普通文本块；短文本、Unicode、URL-safe base64、MIME 参数和字段别名可用。图片/PDF 等二进制校验保持原行为 | [公共附件规范化](../../src-tauri/src/proxy/mappers/common_utils.rs) |
| legacy completions 多候选丢失或串接 | legacy 复用 Chat 的逐候选状态，只做输出字段投影；保留候选索引、正文、终态和 usage。读取请求的候选数后再进入完成后的尾部等待，缺少候选时保留已有回答并报告中断 | [共享候选处理及 legacy 投影](../../src-tauri/src/proxy/mappers/openai/streaming.rs)、[请求候选数传递](../../src-tauri/src/proxy/handlers/openai.rs) |

遵循 Pipeline First：通用文本附件修复放在全部适配器共用的处理函数中；legacy 使用既有共享状态及流生命周期。Responses 的原生 item 类型需要在协议适配阶段识别，沿用已有历史降级器供两个 Responses 转换入口复用。没有在各入口复制工具执行或附件校验逻辑。

工具历史校验先完成再修改输入，错误的实际结果类型、补丁内容或截图 URL 会使本次规范化原子失败；可选 filename/score、安全检查等不要求无损还原。只有 provider 文件 ID 的截图不能凭空取得图像，保留明确的不可用说明，不尝试外部取回。文字附件内容按原文保留，包括 Markdown/HTML 和 `<think>` 字样，不把文档里的字面内容当成控制标记。

## 相邻审查闭环

请求侧与响应侧互审，主审再次复核共享状态改动和实际运行结果。过程中发现并关闭两处相邻问题：

- 完成后冻结候选输出时仍须接收明确的模型失败。主审发现早退可能吞掉后到的 `OTHER`/`NO_IMAGE`，现先识别错误再忽略重复正文。永久矩阵覆盖候选 0/1、Chat/legacy，并验证只结束一次。
- legacy 会把思考文本作为普通可见答案。复用 Chat 后原有 `thinking_only_legacy_success_retains_visible_reasoning_history` 回归暴露其历史没有保存；现只在 legacy 投影中把该候选的思考文本计入可见内容，恢复原有成功历史。Chat/Claude/native 的仅思考输出完成语义未被放宽。

还核对并覆盖了旧版单候选 STOP/MAX_TOKENS/SAFETY 的终帧 usage、晚到 usage、重复完成帧、不同候选尾片、候选延迟到达、遗漏候选、工具形态文本后真实结构化调用等行为。没有通过删除失败的旧回归来消除问题。

## 永久回归与最终验证

新增 **14 个 `r36_` 永久测试**：请求侧 6 个、响应侧 8 个，包含多个表驱动子场景。删除的 9 个旧 Claude 启发式恢复实现单测由生产 stream/collector 路径矩阵替代；真实结构化空参数工具的既有测试保留。

- [请求历史与完整请求 mapper 测试](../../src-tauri/src/proxy/handlers/openai_request_history_tests.rs)：新工具结果、截图、原子错误、幂等转换、Chat/Responses 及 tool result 文本附件。
- [共享附件测试](../../src-tauri/src/proxy/mappers/common_utils.rs)：短文本、别名、包装/非包装 payload、systemInstruction、元数据、UTF-8 与 URL-safe base64，以及正常 PNG/JSON/audio 和坏图片/PDF/文本对照。
- [流式及 collector 回归](../../src-tauri/src/proxy/mappers/openai/response_completion_tests.rs)：工具正文分片、结构化调用、多候选、usage、完成后模型错误和单一终态。

唯一 Cargo runner 使用真实 target 缓存、隔离运行数据。最终命令：

```sh
cd /home/alec/workspace/Antigravity-Manager/src-tauri
ABV_DATA_DIR=/tmp/am-r36-runtime \
CARGO_TARGET_DIR=/home/alec/workspace/Antigravity-Manager/src-tauri/target \
cargo test --locked --lib -- \
  r36_ request_history_tests mappers::common_utils \
  mappers::openai::streaming mappers::claude::streaming \
  handlers::openai::response_completion_tests \
  pipeline::stream_error pipeline::stream_lifecycle pipeline::candidate_selection \
  --skip server_authoritative_ignores_client_reasoning_content_and_budget \
  --skip test_openai_responses_api_vs_chat_api_thinking_and_signature \
  --test-threads=1 > /tmp/am-r36-final-targeted.log 2>&1
```

最终结果：**188 passed / 0 failed / 0 ignored / 1867 filtered out**；编译 31.70s，测试 7.85s。14 个新永久回归和先前失败的 legacy 思考历史回归全部通过。最终绿色依据仅为 `/tmp/am-r36-final-targeted.log`；中间夹具修正及相邻失败记录不作为最终通过证据。

上述两项已知 baseline 测试仍明确排除，未声称在本轮修复。仅格式化本轮修改文件；`git diff --check` 通过。没有运行 full suite、clippy、前端构建或发布预检。

全部验证离线，使用生产类型/规范化器/mapper/stream/collector 及本地字节流；未调用真实供应商或生产账号。native 文本附件验证的是实际共享 sanitizer 及调用点可达性，未运行完整 HTTP/native handler。裸 PDF 未提供 filename 时的 MIME 推断、供应商未观察到的 CR-only/BOM SSE 边界仍按第九轮审查记录，不在本轮四项修复中扩大范围。
