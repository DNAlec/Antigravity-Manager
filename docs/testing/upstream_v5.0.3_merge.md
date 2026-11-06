# 上游 v5.0.3 合并记录

上游：`lbjlaq/Antigravity-Manager`，正式版 `v5.0.3`（`0144b0f2`）。
本地起点：`d05f15c0`；合并分支：`codex/merge-upstream-v5.0.3`。
按维护者要求，同一问题的修复优先采用上游；保留未被上游覆盖的独立功能。

## 冲突取舍

首次试合并共出现四个文件、19 个文本冲突块。

| 文件 | 重叠问题与采用的实现 | 保留的独立能力 |
| --- | --- | --- |
| `handlers/common.rs`（2 处） | 采用上游 Pipeline 重试、终态和预算导出，删除本地旧位置实现 | 单次尝试诊断控制迁入 `pipeline/rotation.rs`，继续统一覆盖四种协议 |
| `upstream/client.rs`（7 处） | 用上游上传／首字节分阶段超时、请求总预算和重放护栏替换旧响应头看门狗 | 单次尝试开关限制账号、端点、Grace Retry、403 降级重发和自动续跑 |
| `qos_guardian.rs`（6 处） | 采用上游自动隔离开关、冷启动分桶及 15 秒／100K 输入补偿；扫描与写入分离遵循上游修复 | 实例级阈值、配置热更新、探针策略快照、原子领取和旧结果防覆盖 |
| `token_manager.rs`（4 处） | 采用上游极简探针信封、配额优先选模和端点回退规则；扫描集中到 Guardian | 领取前检查凭据／模型、账号代理复用、全链截止时间、有效文本判定、删除后清理 |

QoS 的并发保护并非仅替代上游自锁修复：它还阻止旧探针覆盖手工重置、删除、重新隔离或新领取结果。
因此在上游“先结束迭代、再写状态”的流程上继续保留领取凭证，并在准备凭据后重新校验当前隔离状态。
探针已知零配额、非文本或受保护模型不会被选中；缺少配额表时仍允许授权模型列表兜底。
Docker 的有界 curl 健康检查及实例级配置 API 保留。

关联本地 PR：[#1](https://github.com/DNAlec/Antigravity-Manager/pull/1)、
[#2](https://github.com/DNAlec/Antigravity-Manager/pull/2)、
[#3](https://github.com/DNAlec/Antigravity-Manager/pull/3)、
[#4](https://github.com/DNAlec/Antigravity-Manager/pull/4)。
上游相关：[自锁 #3749](https://github.com/lbjlaq/Antigravity-Manager/issues/3749)、
[超时 PR #3817](https://github.com/lbjlaq/Antigravity-Manager/pull/3817)、
[重放护栏 PR #3839](https://github.com/lbjlaq/Antigravity-Manager/pull/3839)。

## 升级行为变化

- 自动 QoS 隔离按上游默认关闭，使用 `proxy.qos_auto_jail` 显式开启。阈值配置本身不会开启隔离。
- `ANTIGRAVITY_UPSTREAM_HEADER_PRIMARY_SECS` 和 `ANTIGRAVITY_UPSTREAM_HEADER_FALLBACK_SECS` 退役；检测到时会记录迁移警告，实际使用上游预算。
  网络等待改由 `proxy.stream_timeouts.initial_ttft_secs` 与 `proxy.upload_timeouts` 控制，并受上游全局请求预算约束。
- `ANTIGRAVITY_UPSTREAM_SINGLE_ATTEMPT` 保留，默认关闭。
- 探针报文跟随上游，输出上限为 10，不再依赖已移除的旧思考预算接口。
- 冷启动达到 60K 未缓存输入时使用同桶基线，不足三个样本只记录、不判罚；详细配置见 [QoS 策略](../qos-policy.md)。

## 本地验证

Rust 1.96.0，按仓库锁文件离线编译 library 测试目标。定向运行 **87 项通过，0 失败**：
QoS 状态机与配置、单 worker HTTP 存活、探针代理／截止时间／SSE、冷启动边界、端点回退分类、
重试策略、上传及首字节预算、上游探针选模与信封。

```bash
CARGO_BUILD_JOBS=2 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 CARGO_INCREMENTAL=0 \
cargo +1.96.0 test --manifest-path src-tauri/Cargo.toml --locked --offline --no-default-features --lib -- \
  qos_ proxy::upstream::client::tests proxy::upstream::request_policy::tests \
  proxy::pipeline::rotation::tests test_select_probe_model test_build_probe_envelope --test-threads=1
```

上游 `test_first_byte_budget_merges_ttft_and_media_offsets` 仍用约 500KB 的旧测试输入断言 15 秒保底，
而生产实现已将门槛调整到 4MB。本次将测试输入对齐 `REPLAY_GUARD_BODY_BYTES`，未修改上游预算算法。
配置保存回归按 JSON 内容比较，避免 HashMap 序列化顺序变化导致假失败。

测试使用本机 HTTP/SSE stub，没有调用真实 Google 接口。未执行全量测试、前端打包、桌面打包、
远程 CI 或生产验证。本地适配相对上游通过 `git diff v5.0.3 --check`；没有新增依赖。
整体导入的上游在 `src/App.css`、`src/components/navbar/NavMenu.tsx` 和 `vite.config.ts`
自带尾随空白，保留上游内容，因此相对旧 main 的全量 whitespace check 会报告这些既有格式问题。

## 回退

本地 `main` 通过独立合并提交引入该分支。需要回退时可对该合并提交执行 `git revert -m 1`，
保留共享历史；原 `d05f15c0` 提交及上游原始提交均保留。回退后恢复旧响应头环境变量的解释方式。
