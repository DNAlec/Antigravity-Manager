# 上游 v5.0.4-beta.7 合并记录

上游：`lbjlaq/Antigravity-Manager` 的 `v5.0.4-beta.7`（`3f759ae9`）。
合并起点：`origin/main` 的 `f5609a80`。按维护者要求直接集成到本仓库 main，
保留上游提交历史及 beta 版本号；本次不创建标签或发布正式版。

关联本仓库 PR：[#1](https://github.com/DNAlec/Antigravity-Manager/pull/1)、
[#2](https://github.com/DNAlec/Antigravity-Manager/pull/2)、
[#3](https://github.com/DNAlec/Antigravity-Manager/pull/3)、
[#4](https://github.com/DNAlec/Antigravity-Manager/pull/4)。
上游相关：[上传策略 PR #3906](https://github.com/lbjlaq/Antigravity-Manager/pull/3906)、
[连接重置 PR #3899](https://github.com/lbjlaq/Antigravity-Manager/pull/3899)、
[429 #3901](https://github.com/lbjlaq/Antigravity-Manager/issues/3901)、
[流尾部 #3911](https://github.com/lbjlaq/Antigravity-Manager/issues/3911)。

## 冲突与行为变化

- 三个 Dockerfile：保留有连接和总时限的 exec-form curl 健康检查，采用上游 30 秒启动宽限和 tini 入口。
- Claude、Gemini、OpenAI 两条路径：采用上游限流错误响应和统一终态重试策略；原账号重试携带本地会话租约，提交和解绑继续校验代际，保留原始 usage 诊断。
- TokenManager：采用上游 CacheFirst 等待、额度归零迁移及账号锁定规则；将涉及会话解绑的操作适配到本地租约机制，保留实例级 QoS 配置、代理复用、有界探针及账号删除保护。
- QoSGuardian：新增原子的 HalfOpen 转换以支持 CacheFirst，保留探针领取凭证；旧探针完成不能覆盖恢复后的状态，已删除账号不会被重新创建。
- 上传客户端和超时界面：采用上游进展驱动软预算、慢速样本放宽、Sandbox 选择及预算展示，替代本仓库旧硬预算宽限修复；同步中英文说明。单次尝试环境开关仍限制端点、账号、403 降级及新增终态重试。
- 空账号池的本地 HTTP 存活测试对齐上游 429 响应，原先断言为 503。

## 验证范围

使用 Rust 1.96.0、锁定依赖、离线 library 测试目标，定向覆盖 QoS 配置与探针、
会话绑定、CacheFirst/Balance、原始 usage、上传预算、单次尝试、终态重试和流生命周期。
本机 HTTP/SSE stub 需要允许监听 localhost；设置 `NO_PROXY` 和 `no_proxy` 为
`127.0.0.1,localhost`，避免开发机代理干扰测试。未访问真实 Google 服务。

定向测试 **135 项通过，0 失败**。前端 `tsc --noEmit` 通过；相对上游的
`git diff --check` 通过。相对旧 main 的检查仅报告上游
`scripts/install-termux.sh:82` 自带的文件末尾空行，保持原样。
未执行全量测试、发布前预检、桌面打包、跨平台构建或真实账号端到端验证。

## 回退

对本次合并提交执行 `git revert -m 1 <merge-commit>` 可回到合并前的功能状态，
保留公开历史。上游版本包含账号管理和配置扩展，回退二进制前应备份数据目录。
