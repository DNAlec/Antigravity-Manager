# AM / sub2api 缓存连续性开发交接（2026-11-14）

**开发分支：`codex/cache-continuity-fixes`。暂缓上线。** R18 的 18 项原始用例已验收；R19 在相邻场景又确认两仓合计 6 项 P1、17 项 P2，以及 1 项只在异常上游输入时触发的 P3。此次保存的是含已知缺口的开发快照。没有用生产流量验证缓存命中率。

审查以 AM `fdcbaac0`、sub2api `9a4edb333` **加当时两边的工作树改动**为准。已将实际受审代码固定为 AM `23ca0990` 和 sub2api `6a8db55d0`；仅检出上面两个旧提交不能得到本次审查的版本。换机请在两个 fork 都检出上述同名开发分支；审查时 AM `origin/main` 为 `cdec9c34`，sub2api 为 `c9d2a0903`。

## 已验证的边界（R18）

- 原生 Gemini：仅思考候选先结束不会吞掉其他候选的正常答案；遇到明确 503 的部分回答不提交签名，成功的 STOP 可在用量尾帧之前供下一轮恢复。并行工具按各自调用身份回填签名。
- Responses HTTP / WebSocket：已覆盖普通用户消息的带图完整回放、有效游标、拒答及图文顺序。预热选项可在首个生成请求生效，`store:false` 不可读回问答；不支持的外部 opaque compaction 或非法游标会明确报错。
- 经 Messages 的工具回执：在原窗口保留两张图时，同名工具的两份回执能区分图片归属；Gemini 3 将媒体留在对应 `functionResponse.parts`。窗口收缩和 OpenAI Chat 入口见 R19，不要把这个结果外推到所有入口。

## R19：接手后优先修什么

编号沿用 [sub2api 中的两仓审查报告](https://github.com/DNAlec/sub2api/blob/codex/cache-continuity-fixes/docs/cache-continuity/r19-review.md)，报告内有各项输入、对照和所属代码。

| 优先级 / 编号 | 触发条件及下一步验收 |
| --- | --- |
| P1 #2（与 sub2api 联验） | 两候选各产一图，上游累计帧重发图片并 STOP；AM 的非流收集器将 2 张变 4 张。修后单帧、累计帧和独立 STOP 应向客户端交付相同两图，且 sub2api 最终计费都为 2 张（单价 $0.50 时合计 $1）。 |
| P1 #3–4（AM 流生命周期） | 明确 503 后成功重试，Chat、Legacy、直接 Responses mapper 恢复了失败的签名 A，Messages 未恢复任何签名。与原生 Gemini 对照，验收下一轮实际恢复成功签名 B，SQLite 不留失败回答。仅 `fileData`/`file_data` 加 STOP 的原生回答也应能完成流式和非流式交付。 |
| P1 #5–6（AM Responses） | 父游标下完整回放带图的 `function_call_output.output`，有独立 item ID 时丢图，仅有 call ID 时 5 条变 7 条。HTTP/WS 均应保留正确的 5 条及图片。引用父历史中非末尾助手项时，最终模型输入应仍包含已知答案实体。 |
| P2 #11–12 | Claude 工具调用未提供 ID 时，分帧生成相同 ID，`response` 封装内又可能漏补。Gemini 非流收集器还会合并两个各自签名的独立思考块。按最终调用 ID、块和签名逐一核对。 |
| P2 #13–17 | Responses 的平面与命名空间工具重名、文件输入静默消失、外部加密 reasoning 被吞、预热跨别名覆盖报重复字段、超时会话被读取后续期。覆盖 HTTP/WS、能表示的输入、需拒绝的输入及空闲 TTL。 |
| P2 #18–21（跨网关） | AM 清理了名为 `cache_control` 的业务属性和参数；二次转换缩窄 Schema 的 union 并丢掉 `minItems`；旧媒体窗口裁剪后位置编号失真；Chat 入口移动图片到另一份工具回执，模型却看见相同输入。验收用两仓串联后的最终 Schema、实参和 `contents`。 |

其余编号在 sub2api：P1 #1 是客户端收到空流却按一图计费；P2 #7–10 涉及热实例和流交付，#22–23 涉及工具图片和文档缓存断点；P3 只在同帧候选索引非法重复时触发。两边交付字节、最终计费与模型所见历史都要对齐，再判断生产切换。

## 换机复验

1. 将 [AM](https://github.com/DNAlec/Antigravity-Manager) 和 [sub2api](https://github.com/DNAlec/sub2api) 两个 fork 并列检出到 `Antigravity-Manager/`、`sub2api/`，两边都选择 `codex/cache-continuity-fixes`。从 [sub2api 交接索引](https://github.com/DNAlec/sub2api/blob/codex/cache-continuity-fixes/docs/cache-continuity/README.md) 找到 R19 报告和[已保存的复验材料](https://github.com/DNAlec/sub2api/tree/codex/cache-continuity-fixes/tools/cache-continuity/r19)。
2. 已安装 Rust、GTK/WebKitGTK 等 Tauri 开发依赖的 Linux 环境，可从 AM 仓库根运行以下仓内原有回归；它们分别保护原生签名提交、普通消息的媒体回放与 WS 非法游标，无需真实上游。这三个过滤器不覆盖 R19 全部场景。

   ```sh
   cd src-tauri
   cargo test --locked --no-default-features --lib only_an_upstream_terminal_can_publish_a_signed_stream_candidate
   cargo test --locked --no-default-features --lib bounded_media_matches_and_restores_only_the_verified_full_replay
   cargo test --locked --no-default-features --lib invalid_cursor_and_unsupported_compaction_do_not_consume_socket_history
   ```

3. R19 使用当时的生产代码段、SQLite 及受控上游。部分探针的断言是“观察到缺陷”，运行通过不表示已经修复；修复时需改为正确行为的断言，并保留原来的正常对照。未运行完整 Tauri、远程 CI 或供应商实流量。发布前应对将要发布的提交执行[项目发布检查](../RELEASE_GUIDE.md)，流量验证按[原始用量口径](cache-affinity-phase1.md)区分缺失计数与显式零。
