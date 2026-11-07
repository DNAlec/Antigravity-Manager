# 实例级 QoS 策略

QoS 用于识别慢响应账号并控制隔离、探针和恢复。配置位于数据目录的
`gui_config.json.proxy.qos`，通过配置文件或现有管理 API 管理，暂无配置页面。
每个服务实例持有自己的策略；修改测试实例不会改变另一个实例的策略。

合入上游 v5.0.3 后，自动隔离遵循上游的 `proxy.qos_auto_jail` 开关，默认关闭。
需要自动判罚和后台恢复探针时，通过设置页面或配置文件将其开启；
仅调整 `proxy.qos` 阈值不会开启自动隔离。关闭开关会释放隔离与惩罚记忆。

## 默认值与约束

以下是原有行为对应的默认值，单位均为秒。升级时省略整个 `qos` 或其中的字段，
会使用该字段的默认值。输入补偿与冷启动判定采用 v5.0.3 的上游算法。

```json
{
  "normal_threshold_secs": 20.0,
  "deep_thinking_threshold_secs": 35.0,
  "recovery_threshold_secs": 18.5,
  "no_peer_grace_secs": 10.0,
  "probe_timeout_secs": 8.0,
  "probe_success_threshold_secs": 8.0,
  "jail_ladder_secs": [120, 300, 600, 900, 1200, 1800]
}
```

| 字段 | 含义 | 有效范围 |
| --- | --- | --- |
| `normal_threshold_secs` | 普通请求的基础慢响应阈值 | 1–600 |
| `deep_thinking_threshold_secs` | 深度思考请求的基础慢响应阈值 | 1–600 |
| `recovery_threshold_secs` | Suspicious／HalfOpen 真实请求的基础恢复阈值 | 1–600 |
| `no_peer_grace_secs` | 没有可用同侪样本时的额外宽限 | 0–600 |
| `probe_timeout_secs` | Hello 探针的整体截止时间 | 0.1–120 |
| `probe_success_threshold_secs` | 探针通过所需的首段有效文本耗时上限 | 0.1–120 |
| `jail_ladder_secs` | 六级隔离时长 | 每项 1–86400，非递减 |

所有数值必须有限，且满足 `recovery < normal <= deep_thinking`、
`probe_success <= probe_timeout`。错误配置会被拒绝，不会静默截断为其他值。
保存接口的 JSON/schema 验证失败返回 4xx，原配置和运行策略保持不变。
磁盘配置非法时会导致配置加载失败，应修正后重启。

## 保存、热更新与回读

- 文件方式：备份当前配置，在保留其他字段的前提下修改 `proxy.qos`，然后重启服务。
  直接编辑文件不会自动热更新。
- API 方式：使用管理端认证读取 `GET /api/config`，修改返回完整配置中的
  `proxy.qos`，再以 `{"config": <完整配置>}` 调用 `POST /api/config`。
  此接口是完整配置保存，不是局部 PATCH。成功保存会同时更新当前实例的 QoS 策略。
- `GET /api/config` 回读磁盘配置；`GET /api/proxy/qos/config` 回读当前实例正在
  使用的完整 QoS 策略。后者沿用管理 API 认证，不返回账号或认证信息。
- 桌面已有配置保存命令也会更新共享 TokenManager，包括仅启动管理服务的情况。
  随后启用转发会保留该实例当前的 QoS 策略，不会用较早的启动参数覆盖已保存的新值。

热更新保留账号当前状态、历史 strike、惩罚记忆和已有隔离到期时间。
已领取的探针使用领取时的截止时间、通过阈值和失败隔离阶梯，避免执行途中改变规则；
后续请求判定和新领取的探针使用新策略。修改配置不会立即解封账号。
进程重启沿用已有行为：从文件恢复策略，账号的内存 QoS 状态不持久化。

## 判定含义

基础处罚阈值按普通／深度思考选择，加上估算的未缓存输入补偿
`uncached_tokens / 100000 × 15 秒`。普通请求没有可用同侪时再加 `no_peer_grace_secs`。
未缓存输入达到 60K 时按冷启动处理：以 20K 为桶宽，只比较同等级、健康账号的
相邻规模桶样本；不足三个样本时只采样、不判罚。有基线时，处罚阈值取
基础阈值加输入补偿与同桶平均耗时的 1.5 倍中较大者。
观测耗时严格大于最终阈值才记 strike；等于阈值不处罚。
同侪全局拥堵豁免采用上游规则。

Suspicious／HalfOpen 的真实请求耗时严格小于 `recovery_threshold_secs + 输入补偿` 才记一次恢复，
两次通过恢复 Healthy；处于恢复与处罚阈值之间时保持中性。
HalfOpen 一次 strike 即升级隔离；Healthy 仍是五分钟窗口内两次 strike 才隔离。
默认惩罚记忆与级别选择保持原样，因此不保证每次隔离都从第一级开始。

Hello 探针必须收到非思考的有效文本，且耗时严格小于 `probe_success_threshold_secs`；
仅有 keepalive、空流或错误均不通过。`probe_timeout_secs` 覆盖请求及流读取。
成功探针只进入 HalfOpen，仍需真实请求确认恢复。
探针采用上游极简信封与端点回退规则，同时复用当前实例的账号代理。
网络错误、HTTP 404/408/5xx 或 HTTP 200 但响应体为零字节时可回退；400/401/403/429 直接失败。
所有端点共享一次探针的总截止时间，但恢复阈值只比较成功端点从发送到收到有效文本的耗时。
探针仅发送当前授权列表中的实际模型 ID，不发送配额汇总名；优先选择有正配额的文本 Flash，
已知零配额或受保护模型不参与探测。缺少配额表时可用授权模型列表兜底。

现有 QoS 采样口径保留：不同协议／流式模式的采样点可能是上游响应头、首个有效流块，
或非流式完整响应；以当前上游尝试开始时间起算。这些值不能统一解释为模型首个输出 token
的精确耗时。debug 日志会显示基础阈值、输入补偿、同侪可用性、最终阈值和恢复阈值，
状态转换日志记录变化原因及实际处罚边界。

响应头与首字节等待统一由上游 `proxy.stream_timeouts.initial_ttft_secs` 及自适应预算控制，
上传参数位于 `proxy.upload_timeouts`。旧实验变量 `ANTIGRAVITY_UPSTREAM_HEADER_PRIMARY_SECS`
和 `ANTIGRAVITY_UPSTREAM_HEADER_FALLBACK_SECS` 不再生效。
`ANTIGRAVITY_UPSTREAM_SINGLE_ATTEMPT` 仍可限制单次上游尝试，用于隔离诊断。
本策略也不根据 `max_tokens` 自动扩展阈值，不修复工具参数内容偏差。
建议先在隔离测试实例分别验证正常快请求、正常慢请求和真实异常请求，再确定生产参数。
