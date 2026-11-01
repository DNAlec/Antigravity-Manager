# QoS 冷却到期自锁与健康探针回归

本轮在 `codex/upstream-header-budget-validation` 的 `a98deb705c65ef95c0ab874f3744dc993ee1891d` 上继续修复。自锁代码在原基线 `0067fe045ddd77f4462ea8e2c25384839d869f32` 中也存在，不是响应头预算补丁新增。

## 现场关联与确认边界

运维提供的 UTC 时间线：2026-11-01 13:21:08.905 账号进入 Level 0 隔离，等待 120 秒，名义到期时间为 13:23:08.905。最后可见 HTTP 配置请求在 13:22:42；未见到期后发送 Hello 探针的日志，13:25、13:30 仍有后台任务输出。

该时间线与自锁触发条件吻合，但最后一条 HTTP 日志不能当作精确卡死时刻。远程用户态线程栈仍未取得，因此现场根因保持“高可信候选”。本轮确认的是本地可复现的代码缺陷及修复，不改写远程取证结论。

## 修复行为

`TokenManager::run_qos_probing_tick` 原先遍历 `QoSGuardian.states` 时调用 `mark_probing`。DashMap 迭代器和条目持有分片读锁，同一账号的 `insert` 又请求写锁，导致当前线程自锁。该路径在实际网络探针发出之前就阻塞；仅 `drop(entry)` 不足以释放迭代器持有的锁。

现在先收集到期账号 ID 并结束整个迭代，再取得账号凭据。`try_mark_probing` 在一个写 guard 内检查**当前**状态仍为 `Jailed` 且已到期，然后原子转换为 `Probing`，返回带当前隔离级别和唯一标识的领取凭证。所有 guard 均在发起网络任务之前释放。

- 手工恢复、删除、尚未到期的重新隔离不会被陈旧候选快照覆盖。
- 并发扫描只有一次能够领取同一账号。
- 凭据缺失、即将过期或没有未受保护的授权 Gemini 文本模型时保留隔离状态，等待凭据/模型元数据更新，避免把准备条件不足记成响应缓慢。
- 完成探针时再次原子检查领取标识；手工重置、账号删除、后来隔离或重新领取后，旧结果不再覆盖状态或惩罚记忆。实际删除账号路径同步清理关联 QoS 数据。
- 原来的 8 秒成功阈值和退避阶梯保留。探针复用共享上游客户端及账号代理，发送包含模型、项目和内部请求的有效信封；保持纯 Hello 请求，隔离用户全局提示词和业务思考配置。
- 8 秒截止覆盖客户端获取、发送和 SSE 读取。仅实际生成的非思考文本算成功；保活包、空流、错误事件和纯思考输出不算恢复证据。

三个 Dockerfile 的健康检查统一改为直接执行 curl：连接超时 2 秒、总超时 4 秒，低于 Docker 外层 5 秒超时，取消额外 shell 层。HTTP 失败和超时仍返回非零退出码。

## 本地回归结果

Rust/Cargo 1.96.0；按原 `Cargo.lock` 离线编译，未更改依赖。

1. **修改前复现**：真实 `run_qos_probing_tick` 在单 worker 运行时扫描到期账号，父进程在 10 秒硬截止后终止并回收卡死子进程。测试不提供 token，不可能发起真实模型调用。
2. **第一轮修改后通过**：15 项 QoS 测试全部通过，包含 7 项新增回归及 8 项原有状态机测试。合并前审查进一步补充探针请求与生命周期修复，最终结果见后续记录。死锁风险用例由独立父进程提供 20 秒截止，不依赖可能被阻塞的 Tokio 定时器。
3. **真实 HTTP 路由回归**：独立数据目录、单 worker、真实 `AxumServer`、空账号池；三轮执行空池 503、debug 配置 POST/GET、到期扫描和新 TCP 连接上的 `/health`。所有请求成功完成预期断言，健康接口保持 200，未访问真实上游。
4. **有凭据的领取路径**：使用假 token 仅调用纯状态领取函数，不启动网络探针；验证多个到期账号均可领取、未到期状态不变、后续扫描不重复领取。
5. **Docker 健康检查**：以原候选镜像临时派生测试镜像，在隔离网络中运行假服务。各场景连续完成 5 次检查，测试容器和派生镜像均已清理。

| 假服务场景 | curl 退出码 | 单次耗时 | Docker 状态 | 检查完成后残留 curl |
| --- | --- | --- | --- | --- |
| 正常 HTTP 200 | 0 | 约 0.03–0.05 秒 | healthy | 0 |
| 接受连接但不返回 HTTP | 28 | 约 4.02–4.03 秒 | unhealthy | 0 |
| 端口未监听 | 7 | 约 0.02–0.03 秒 | unhealthy | 0 |

Docker 测试仅将探针间隔缩短为 2 秒、启动宽限设为 0；命令和外层 5 秒超时与修改后的 Dockerfile 一致。采样期间最多同时存在一个 curl，没有存活超过外层超时的 curl。

针对性编译、测试命令（仓库根目录，需要允许本机临时端口）：

```bash
export CARGO_BUILD_JOBS=2 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 CARGO_INCREMENTAL=0
cargo +1.96.0 test --manifest-path src-tauri/Cargo.toml --locked --offline --no-default-features --lib qos_ -- --nocapture
cargo +1.96.0 build --manifest-path src-tauri/Cargo.toml --locked --offline --no-default-features --bin antigravity-tools
```

本地取证日志位于忽略目录 `.build-tmp/`：`qos-before-fix.log`、`qos-after-fix-build.log`、`qos-after-fix-tests.log`、`qos-after-fix-backend-build.log`；Docker 脚本及结果位于 `.build-tmp/healthcheck-regression/`。这些临时文件不属于 Git 仓库。

## 合并前复审与最终回归

三名子代理分别审查响应头策略、QoS 并发与探针、测试及 Docker 健康检查。审查发现解除自锁会激活旧探针的无效信封与代理绕过问题，随后补齐有效请求、业务设置隔离、模型保护检查和过期完成校验。独立复审未发现剩余合并阻塞项。

最终 **29 项 QoS 测试通过**（0 失败，约 8 秒），测试目标及 headless 后端编译通过。新增本地代理 stub 验证实际发送的 Gemini 3.8 Flash 请求信封和账号代理选择；错误 HTTP/SSE、空流、保活及纯思考输出不会误计为成功，持续保活也在 8 秒截止。隔离子进程开启全局提示词和高业务思考预算，验证 Gemini Lite、Flash、Pro、Argon 探针仍采用独立最小请求。没有发送真实 Google 请求。

最终日志：`.build-tmp/qos-final-build.log`、`qos-final-tests.log`、`qos-final-backend-build.log`。编译存在仓库既有的未使用代码等警告，未更新依赖或混入无关告警清理。

## 尚未覆盖的边界

- 没有取得远程用户态栈，没有重启或修改远程测试/生产实例。
- 没有验证真实 Google Hello 探针的成功率或费用；修复解除阻塞后，已有后台探针路径可以继续执行。
- 仅有 Claude/GPT、模型元数据为空或所有 Gemini 文本模型受保护的账号，仍需要元数据更新或手动解禁；Gemini 探针通过只进入 HalfOpen，还需实际业务请求连续通过才恢复健康。
- 155 条工具参数异常、同名并行工具调用边界、流终止检测和失败时原始流日志保存均不属于本轮修复。
- 本地验证不等于新候选镜像已经发布，也不等于生产验收通过。
- 创建早期草稿时，本 fork 的 GitHub Actions 尚未运行；后续正式 PR 已触发三平台 CI。此文记录本地验证，远程检查的最终结果以 PR 页面为准，不将“无检查”写成“检查通过”。
