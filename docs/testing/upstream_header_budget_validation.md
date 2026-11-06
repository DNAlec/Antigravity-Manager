# v5.0.3 合并后的适用范围

本文下方保留早期候选版本的验证记录。合入 v5.0.3 后，响应头看门狗已被上游的
上传／首字节分阶段预算替代，`ANTIGRAVITY_UPSTREAM_HEADER_PRIMARY_SECS` 和
`ANTIGRAVITY_UPSTREAM_HEADER_FALLBACK_SECS` 不再生效。请使用
`proxy.stream_timeouts.initial_ttft_secs` 和 `proxy.upload_timeouts` 配置网络等待；
仍受上游全局请求预算约束。`ANTIGRAVITY_UPSTREAM_SINGLE_ATTEMPT` 保留。
下方旧版本测试结论不代表 v5.0.3 的运行验证。

# 大工具响应头等待：验证补丁

基线：`0067fe045ddd77f4462ea8e2c25384839d869f32`（v5.0.0）。
开发分支：`codex/upstream-header-budget-validation`。

本补丁用于验证：小输入、大工具参数输出的请求，是否在原先 30 秒响应头看门狗之后才能返回。它提供实验控制和日志；真实上游因果关系仍需重放确认。

## 配置

环境变量在进程第一次读取策略时固定，修改后需重启候选实例。适用于 GUI 启动环境和 headless/CLI，无需修改管理 API 或已有配置文件。

| 环境变量 | 未设置时 | 有效值与作用 |
| --- | --- | --- |
| `ANTIGRAVITY_UPSTREAM_HEADER_PRIMARY_SECS` | 按输入大小 30～45 秒 | 5～600 的整数；覆盖 `streamGenerateContent` 首选端点响应头等待 |
| `ANTIGRAVITY_UPSTREAM_HEADER_FALLBACK_SECS` | 25 秒 | 5～600 的整数；覆盖备用端点响应头等待 |
| `ANTIGRAVITY_UPSTREAM_SINGLE_ATTEMPT` | `false` | `true`/`1` 开启；`false`/`0` 关闭；对该进程的共享生成路径限制为一次账号尝试、一次端点发送 |

无效值会记录警告并回退到原默认策略。两个时间覆盖项只作用于 `streamGenerateContent`，不改变 `countTokens` 等辅助方法的时间参数。

端点链预算从首选和备用的时间之和计算。例如 180＋60 对应 240 秒，不再受原 55 秒预算钳制。已明确返回容量不足时仍最多探测 15 秒。单次尝试开启时仅保留首选预算，不执行备用端点、403 去项目头重发、账号轮换、Grace Retry 或纯思考自动续跑。

原默认时间参数保持不变；剩余预算不足 5 秒时不再额外补足 5 秒，且预算耗尽后也不能通过 403 重试重新进入首选端点。HTTP 客户端 600 秒总超时、连接超时、Gemini 300 秒首块/空闲等待、思考参数和输出上限均保持原值。该预算仅管理端点响应头等待，不能视作完整客户端请求的总截止时间；错误响应体读取及各协议自己的截止机制仍独立存在。

## 对照方法

使用同一候选构建启动两个先后运行的隔离实例，保持账号、代理、模型、配置、请求 payload 和单次尝试开关一致。可在隔离测试数据目录只加载一个经授权测试账号，避免为了固定账号而停用生产号池。端点健康排序仍生效，必须核对两次日志实际选择了同一个端点。

对照组示例（候选二进制所在目录）：

```bash
ANTIGRAVITY_UPSTREAM_HEADER_PRIMARY_SECS=30 \
ANTIGRAVITY_UPSTREAM_HEADER_FALLBACK_SECS=25 \
ANTIGRAVITY_UPSTREAM_SINGLE_ATTEMPT=true \
./antigravity-tools --headless
```

实验组只改变首选响应头等待：

```bash
ANTIGRAVITY_UPSTREAM_HEADER_PRIMARY_SECS=180 \
ANTIGRAVITY_UPSTREAM_HEADER_FALLBACK_SECS=25 \
ANTIGRAVITY_UPSTREAM_SINGLE_ATTEMPT=true \
./antigravity-tools --headless
```

1. 用交接证据中的原始 `turn-2-request.json` 重放，通过原 sub2api 兼容路径接入候选 AM。不要重写历史、换模型或修改工具参数。先确认 sub2api 的目标地址确实指向隔离候选，测试客户端关闭自动重试。
2. 下游读取上限需足以观察响应头与完整工具输出；例如设置 660 秒的有限读取窗口，以免先于现有 AM 600 秒总超时结束。记录这一设置，不把客户端取消误判为服务端超时。
3. 在 INFO 日志中配对 `Upstream response-header wait started` 和 `Upstream response headers received`：核对 `call_id`、信封 `request_id`、账号 ID、端点、输入长度、实际预算及 `header_elapsed_secs`。信封请求 ID 不等同于 sub2api 请求 ID，应同时保留外层采集器记录。代理日志仅记录代理 ID或默认配置路径，不输出密码。另记录脱敏后的代理配置标识。
4. 区分 HTTP 响应头、首个 SSE 事件、工具参数完整返回这三个时刻。AM 会预读首块再向下游返回，客户端首响应时间不等于上游响应头时间。
5. 只有同条件下对照组在 30 秒失败、实验组在超过 30 秒后收到成功响应头并完成合法工具参数，才有力支持“原响应头预算不足”。若账号/代理/端点不同，或只有失败后重试才成功，应保留这些混杂因素。
6. 180 秒仍未返回应记录失败，继续定位上游、协议或网络；不要靠无界重放堆成功。每次真实调用会消耗额度。

单次尝试开关作用于整个候选进程，不是按某个客户端请求定向启用。它保留第一次失败，牺牲正常容错能力；应在隔离验证实例使用。它不锁定账号，不限制 sub2api 或调用方自身的重试，也不会清除现有端点健康/账号冷却状态。恢复时移除这三个环境变量并重启候选实例即可。

## 候选镜像准备

推送此开发分支不会自动发布镜像：现有 CI 的 push 入口只覆盖 main/master/beta，发布工作流另有版本标签和分支门禁。本次使用独立候选镜像标签，不触发正式版发布流程。

在干净的构建检出中，先拉取并固定开发分支的提交，再使用现有后端 Dockerfile 构建。该 Dockerfile 在 Debian Bookworm 构建环境中编译 release 后端，并复用指定镜像的前端资源；本机 Ubuntu 的 debug 可执行文件仅用于本地编译验证。

```bash
git fetch origin codex/upstream-header-budget-validation
AM_TEST_REV=$(git rev-parse origin/codex/upstream-header-budget-validation)
git switch --detach "$AM_TEST_REV"
docker build -f docker/Dockerfile.backend \
  --build-arg FRONTEND_IMAGE=lbjlaq/antigravity-manager:v5.0.0 \
  --label "org.opencontainers.image.revision=$AM_TEST_REV" \
  -t "antigravity-manager:header-budget-${AM_TEST_REV:0:12}" .
```

示例中的前端镜像标签应核对为现网使用版本；有 registry manifest digest 时优先使用 `仓库@sha256:...` 固定它，并记录实际镜像标识。容器 image ID 不能直接当作 registry manifest digest。构建、发布和实际切换候选容器是后续操作，本地 Rust 编译通过不等同于该候选镜像已经构建通过。

候选容器的实验组环境变量：

```yaml
environment:
  ANTIGRAVITY_UPSTREAM_HEADER_PRIMARY_SECS: "180"
  ANTIGRAVITY_UPSTREAM_HEADER_FALLBACK_SECS: "25"
  ANTIGRAVITY_UPSTREAM_SINGLE_ATTEMPT: "true"
  RUST_LOG: info
```

对照组只把首选等待改为 `"30"`。Compose 中必须显式传入这些变量，仅在宿主 shell 中设置不会自动进入容器。单次尝试模式的影响范围是整个候选进程，使用隔离实例和独立数据目录测试。切换前保留原镜像、Compose/环境配置和数据备份；回滚时恢复原镜像及本轮修改的环境变量，避免覆盖其他并发配置修改。

## 针对性测试

项目锁定的 `yaml-rt 0.3.1` 要求 Rust 1.96。安装 rustup 工具链及项目系统依赖后，从仓库根目录执行：

```bash
. "$HOME/.cargo/env"
cargo +1.96.0 test --manifest-path src-tauri/Cargo.toml --locked --no-default-features --lib proxy::upstream::
cargo +1.96.0 test --manifest-path src-tauri/Cargo.toml --locked --no-default-features --lib proxy::handlers::common::tests::task_short_429
cargo +1.96.0 build --manifest-path src-tauri/Cargo.toml --locked --no-default-features --bin antigravity-tools
```

执行测试前移除上述实验变量，以免进程级配置影响正常策略测试。策略测试直接构造配置，不修改全局环境变量；HTTP 测试使用本机临时端口和毫秒级延迟，验证等待响应头失败/成功以及收到响应头后继续读流的边界，不访问 Google、不消耗额度。

本轮使用 Rust/Cargo 1.96.0、Ubuntu 26.04、2 个编译任务，关闭 dev/test 调试信息和增量编译以降低资源占用。依赖按 `Cargo.lock` 下载后离线构建，未更新依赖版本。后端测试目标和 `src-tauri/target/debug/antigravity-tools` 可执行文件均编译通过，存在仓库既有的未使用代码等警告。`--no-default-features` 用于后端开发构建，不嵌入前端发布资源；桌面应用/发布包仍需另行构建和验证。

已通过上游模块 24 项测试和共享 429/Grace Retry 回归 1 项，共 25 项。3 项本机 HTTP 测试首次因沙箱禁止监听端口失败，在允许 loopback 通信的环境重跑后全部通过。这些测试不能替代真实大工具请求验收；该次本地测试阶段尚未执行生产重放、构建/发布候选镜像或修改生产配置。

后续已发布公开候选镜像 `ghcr.io/dnalec/antigravity-manager:header-budget-a98deb705c65`（linux/amd64，digest `sha256:2ff0e522a81b911075403af734b611b87919294c5cef8adef68e779a8179f9bf`），并验证匿名拉取。远程合成对照中，30 秒组超时，180 秒组约 44.65 秒收到响应头且完整返回 240 条。这不是原始失败请求的重放；该镜像不包含后续 QoS 自锁或 Docker 健康检查修复。
