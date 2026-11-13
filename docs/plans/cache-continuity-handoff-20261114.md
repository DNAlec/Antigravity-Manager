# AM / sub2api 缓存连续性开发交接（2026-11-14）

两个 fork 的开发分支均为 `codex/cache-continuity-fixes`。**R19 修复及后续交叉复审已完成，本地定向回归通过；发布仍以对应提交的远端 CI 和项目发布流程为准。** 尚未合并生产分支、打标签或替换运行实例。

当前入口是 [sub2api 的修复与复审记录](https://github.com/DNAlec/sub2api/blob/codex/cache-continuity-fixes/docs/cache-continuity/r19-fixes.md)，它逐项对应 [R19 原始清单](https://github.com/DNAlec/sub2api/blob/codex/cache-continuity-fixes/docs/cache-continuity/r19-review.md) 的 #1–23 和异常候选 P3。R18 原始 18 项使用另一组编号；原始报告及观测日志保留为修前证据。

## 固定提交

| 仓库 | 本轮提交 | 开发 PR | 发布/回滚基线 |
| --- | --- | --- | --- |
| AM | `d30a6618`、`59064d83`、`91c466b9`；格式 `985f9832` | [#5](https://github.com/DNAlec/Antigravity-Manager/pull/5) | `cdec9c34`（v5.0.4） |
| sub2api | `21ef324c4` | [#6](https://github.com/DNAlec/sub2api/pull/6) | `c9d2a0903`（v0.2.20-nekocloud.4） |

AM 修复失败重试签名、累计媒体去重、文件输出、工具身份、Responses HTTP/WS 回放与输入验证、Schema 约束和工具媒体归属。交叉复审还补修了 Schema 父层约束及 nullable required、深层组合的重复遍历，以及仅思考 STOP / MAX_TOKENS 的失败历史提交。Gemini 按候选独立检查可交付内容；有效的其他候选不能替仅思考候选授权。Legacy 将思考作为正文交付，保留其正常成功语义。

## 已完成验证

- 永久流式回归 17/17：503、连接中断、EOF、拒答和成功对照；成功签名在下游终态可见之前发布。仅思考失败不得留下 SQLite 历史或会话签名。
- 仅思考补修后受影响的 Claude/Gemini/Chat mapper 与生命周期回归 184/184；此前相关 mapper/pipeline/历史组 308 项通过。正常文字、工具、内联图片、文件 URI 的组合/分离终态另有 16 组成功对照。
- OpenAI handler 72 项、会话存储 14 项、受控 HTTP/WS 探针 20 项；Schema 专项 49 项及实际 crate `r19_` 32 项通过。各组有重叠，不把数量相加。
- 两仓最终模型输入 wire 断言通过；真实 collector 的单帧、累计帧及独立 STOP 输出交给 sub2api 后均为 2 张、1.00（单价 0.50）。
- `cargo fmt -- --check` 通过。跨平台 Clippy、完整 Tauri 和 Android 构建由 [PR CI](https://github.com/DNAlec/Antigravity-Manager/pull/5/checks) 验证；以 PR 对应提交的实际结果为准。

## 复跑入口

将两仓并列检出到 `Antigravity-Manager/`、`sub2api/`。Go/Rust 版本以 manifest 为准，本次使用 Go 1.27.4、Rust 1.96.0，Linux 已具备 GTK/WebKitGTK 等 Tauri 依赖。AM 根目录可执行：

```sh
cd src-tauri
cargo test --locked --no-default-features -j2 --lib stream_completion_tests:: -- --test-threads=1
cargo test --locked --no-default-features -j2 --lib r19_ -- --test-threads=1
cargo test --locked --no-default-features -j2 --lib proxy::http_session_store:: -- --test-threads=1
```

测试使用独立临时 profile；如需自行指定，可设置 `ABV_DATA_DIR` 与 `ABV_DATA_DIR_POINTER_FILE` 到专用临时路径，避免混用个人资料目录。

两仓联验入口在 [sub2api 工具 README](https://github.com/DNAlec/sub2api/blob/codex/cache-continuity-fixes/tools/cache-continuity/r19/README.md)。`stream` 运行本仓永久回归后把当次 collector JSON 交给 Go 计费断言；默认 `billing` 使用旧 fixture，累计帧保留修前的 4 张结果，不能用于证明当前修复有效。

## 发布条件和限制

本轮独立复审在审查范围内没有剩余可复现缺陷；这不是生产实流量或供应商验签的证据。没有测量真实供应商缓存命中率。跨平台 CI 全部通过后，仍须按[发布指南](../RELEASE_GUIDE.md)固定候选提交、执行预发布/发布与回滚步骤。生产切换和发布由后续明确指令执行，本次没有部署。
