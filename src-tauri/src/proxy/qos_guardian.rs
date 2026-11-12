use dashmap::DashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use crate::proxy::qos_config::QoSConfig;
use crate::proxy::token_manager::ProxyToken;

/// 账号 QoS 运行时健康状态
#[derive(Debug, Clone, PartialEq)]
pub enum AccountQoSState {
    /// 纯健康：参与基线计算，正常分配新会话
    Healthy,
    /// 瞬态可疑：从同行基线分母中瞬态剔除，调度器开启“只出不进”，暂缓新会话分配
    Suspicious {
        first_strike_at: Instant,
        strike_count: u8,
        consecutive_healthy: u8,
    },
    /// 小黑屋隔离：会话已强制解绑，彻底暂停业务调度，进入阶梯自愈
    Jailed {
        level: usize,
        jailed_at: Instant,     // 本级开始冷却的时间戳 (关键：每次从 0 计时)
        current_wait: Duration, // 本级等待总时长
    },
    /// 探针执行中：后台轻量探针正在发出“你好”测速，防止并发重复探测
    Probing {
        level: usize,
        started_at: Instant,
        claim_id: uuid::Uuid,
    },
    /// 缓刑观察期：探针通过，单路试探真实请求，连续 2 次真实请求通过才转 Healthy，再次失败从严惩罚
    HalfOpen {
        level: usize,
        consecutive_healthy: u8,
        entered_at: Instant,
    },
}

/// A probe may complete only the transition that it claimed, even if the account
/// is reset and reaches the same jail level again before the request finishes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ProbeClaim {
    pub level: usize,
    pub config: QoSConfig,
    claim_id: uuid::Uuid,
}

impl ProbeClaim {
    pub fn accepts_ttft(self, ttft_ms: f64) -> bool {
        ttft_ms < self.config.probe_success_threshold_secs * 1000.0
    }

    fn matches(self, state: &AccountQoSState) -> bool {
        matches!(
            state,
            AccountQoSState::Probing { level, claim_id, .. }
                if *level == self.level && *claim_id == self.claim_id
        )
    }
}

/// 单次请求的 TTFT 观测记录
#[derive(Debug, Clone)]
pub struct TtftSample {
    pub ttft_ms: f64,
    pub timestamp: Instant,
    pub is_thinking: bool,
    pub is_cold_start: bool,
    /// 未缓存 token 的分桶（桶宽 20K），用于同规模对比，避免拿热缓存小请求当基准
    pub uncached_bucket: u8,
}

/// 冷启动样本分桶宽度（未缓存 token）
const COLD_BUCKET_TOKENS: u32 = 20_000;

/// 判定冷启动是否违规所需的最少同桶基线样本数（低于它只记样本、不判罚）
const MIN_COLD_PEER_SAMPLES: usize = 3;

fn cold_bucket_of(uncached_tokens: u32) -> u8 {
    ((uncached_tokens / COLD_BUCKET_TOKENS).min(15)) as u8
}

/// 裁决动作
#[derive(Debug, Clone, PartialEq)]
pub enum QoSAction {
    None,
    TaggedSuspicious,
    Vindicated,
    TriggerJailedAndUnbind {
        level: usize,
        wait_duration: Duration,
    },
    HalfOpenPassed,
    HalfOpenFailed {
        escalated_level: usize,
        wait_duration: Duration,
    },
}

/// QoS 守护管理器
pub struct QoSGuardian {
    /// 账号状态映射: account_id -> AccountQoSState
    pub states: Arc<DashMap<String, AccountQoSState>>,
    /// 账号最近有效 TTFT 滑动窗口采样: account_id -> Vec<TtftSample> (保留最近 15 分钟)
    pub samples: Arc<DashMap<String, Vec<TtftSample>>>,
    /// 会话上下文水位跟踪: (account_id, session_id) -> (last_input_tokens, last_seen_at)
    pub session_watermarks: Arc<DashMap<(String, String), (u32, Instant)>>,
    /// 惩罚历史记忆: account_id -> (last_level, updated_at)，保留 30 分钟防短时高频复发
    pub penalty_memory: Arc<DashMap<String, (usize, Instant)>>,
    /// Each operation copies this policy before acquiring any account-state lock.
    config: RwLock<QoSConfig>,
    pub auto_jail_enabled: Arc<AtomicBool>,
}

impl Default for QoSGuardian {
    fn default() -> Self {
        Self::new()
    }
}

impl QoSGuardian {
    pub fn new() -> Self {
        Self {
            states: Arc::new(DashMap::new()),
            samples: Arc::new(DashMap::new()),
            session_watermarks: Arc::new(DashMap::new()),
            penalty_memory: Arc::new(DashMap::new()),
            auto_jail_enabled: Arc::new(AtomicBool::new(false)),
            config: RwLock::new(QoSConfig::default()),
        }
    }

    pub fn config_snapshot(&self) -> QoSConfig {
        *self
            .config
            .read()
            .unwrap_or_else(|error| error.into_inner())
    }

    /// Apply a validated policy without rewriting cooldowns or probe claims.
    pub fn update_config(&self, config: QoSConfig) -> Result<(), String> {
        config.validate()?;
        *self
            .config
            .write()
            .unwrap_or_else(|error| error.into_inner()) = config;
        Ok(())
    }

    /// 查询自动限流进小黑屋机制是否启用
    pub fn is_auto_jail_enabled(&self) -> bool {
        self.auto_jail_enabled.load(Ordering::Relaxed)
    }

    /// 设置自动限流进小黑屋机制开关
    pub fn set_auto_jail_enabled(&self, enabled: bool) {
        let prev = self.auto_jail_enabled.swap(enabled, Ordering::SeqCst);
        if prev && !enabled {
            // 当开关从开启转为关闭时，立即解封清空所有小黑屋与惩罚记忆
            self.release_all();
            tracing::info!("🔓 [QoS Guardian] 自动限流进小黑屋已关闭，已清空并解封所有受限账号");
        } else if !prev && enabled {
            tracing::info!("🔒 [QoS Guardian] 自动限流进小黑屋已开启");
        }
    }

    /// 解封清空所有小黑屋隔离与惩罚记忆，使全池账号恢复为 Healthy
    pub fn release_all(&self) {
        self.states.clear();
        self.penalty_memory.clear();
    }

    /// 获取账号当前 QoS 状态（未记录或未启用小黑屋则默认 Healthy）
    pub fn get_state(&self, account_id: &str) -> AccountQoSState {
        if !self.is_auto_jail_enabled() {
            return AccountQoSState::Healthy;
        }
        self.states
            .get(account_id)
            .map(|r| r.value().clone())
            .unwrap_or(AccountQoSState::Healthy)
    }

    /// 判断账号是否处于小黑屋或探针探测中（完全不可用于业务调度）
    pub fn is_jailed_or_probing(&self, account_id: &str) -> bool {
        if !self.is_auto_jail_enabled() {
            return false;
        }
        matches!(
            self.get_state(account_id),
            AccountQoSState::Jailed { .. } | AccountQoSState::Probing { .. }
        )
    }

    /// 判断账号是否处于瞬态可疑状态（开启只出不进）
    pub fn is_suspicious(&self, account_id: &str) -> bool {
        if !self.is_auto_jail_enabled() {
            return false;
        }
        matches!(
            self.get_state(account_id),
            AccountQoSState::Suspicious { .. }
        )
    }

    /// 判断账号是否处于缓刑期（HalfOpen）
    pub fn is_half_open(&self, account_id: &str) -> bool {
        if !self.is_auto_jail_enabled() {
            return false;
        }
        matches!(self.get_state(account_id), AccountQoSState::HalfOpen { .. })
    }

    /// 获取账号当前小黑屋冷却剩余时长 (单位: 秒)
    pub fn get_jailed_remaining_seconds(&self, account_id: &str) -> Option<u64> {
        if !self.is_auto_jail_enabled() {
            return None;
        }
        if let Some(state) = self.states.get(account_id) {
            if let AccountQoSState::Jailed {
                jailed_at,
                current_wait,
                ..
            } = state.value()
            {
                let elapsed = jailed_at.elapsed();
                if elapsed < *current_wait {
                    return Some((*current_wait - elapsed).as_secs());
                } else {
                    return Some(0);
                }
            }
        }
        None
    }

    /// 手动解除账号的小黑屋与 QoS 隔离限制，重置为 Healthy
    pub fn release_from_jail(&self, account_id: &str) {
        self.states
            .insert(account_id.to_string(), AccountQoSState::Healthy);
        self.penalty_memory.remove(account_id);
    }

    /// 判断账号是否处于纯健康状态
    pub fn is_healthy(&self, account_id: &str) -> bool {
        if !self.is_auto_jail_enabled() {
            return true;
        }
        matches!(self.get_state(account_id), AccountQoSState::Healthy)
    }

    /// 计算纯净同侪基线均值（Pure Peer Baseline Extraction）
    /// 严格排除：自身、非同级Tier、本地禁用、429限流/零配额、小黑屋(Jailed)、探测中(Probing)、瞬态可疑(Suspicious)、缓刑(HalfOpen)
    pub fn calculate_pure_peer_avg<F>(
        &self,
        target_account_id: &str,
        target_tier: &str,
        all_accounts: &[ProxyToken],
        is_unhealthy_or_limited: F,
    ) -> Option<f64>
    where
        F: Fn(&str) -> bool,
    {
        let now = Instant::now();
        let sample_ttl = Duration::from_secs(15 * 60); // 仅采纳近 15 分钟内的采样
        let norm_target_tier = target_tier.trim().to_uppercase();

        let mut valid_ttfts: Vec<f64> = Vec::new();

        for acc in all_accounts {
            // 过滤 1: 必须是同类账号 (Tier 对齐，如 PRO vs PRO)
            let acc_tier = acc
                .subscription_tier
                .as_deref()
                .unwrap_or("FREE")
                .trim()
                .to_uppercase();
            if acc_tier != norm_target_tier {
                continue;
            }

            // 过滤 2: 严格排除自身，杜绝自相关性
            if acc.account_id == target_account_id {
                continue;
            }

            // 过滤 3: 排除外部已判定为非健康的账号 (本地已禁用 / 429 限流中 / 配额 <= 0%)
            if is_unhealthy_or_limited(&acc.account_id) {
                continue;
            }

            // 过滤 4: 排除任何非纯 Healthy 的账号！杜绝劣币污染基准
            if let Some(state) = self.states.get(&acc.account_id) {
                match state.value() {
                    AccountQoSState::Healthy => {}
                    _ => continue,
                }
            }

            // 提取该纯净同行近 15 分钟内的非思考、非冷启动样本均值
            if let Some(samples) = self.samples.get(&acc.account_id) {
                let fresh_samples: Vec<f64> = samples
                    .iter()
                    .filter(|s| {
                        now.duration_since(s.timestamp) <= sample_ttl
                            && !s.is_thinking
                            && !s.is_cold_start
                    })
                    .map(|s| s.ttft_ms)
                    .collect();

                if !fresh_samples.is_empty() {
                    let avg = fresh_samples.iter().sum::<f64>() / fresh_samples.len() as f64;
                    valid_ttfts.push(avg);
                }
            }
        }

        if valid_ttfts.is_empty() {
            None
        } else {
            Some(valid_ttfts.iter().sum::<f64>() / valid_ttfts.len() as f64)
        }
    }

    /// 冷启动同桶基线均值（毫秒）
    ///
    /// 只采纳：同 Tier、纯 Healthy、`is_cold_start` 且未缓存桶相差 ±1 的样本。
    /// 样本数不足 [`MIN_COLD_PEER_SAMPLES`] 时返回 None（上层据此“只采样不判罚”）。
    fn calculate_cold_peer_avg<F>(
        &self,
        target_account_id: &str,
        target_tier: &str,
        target_bucket: u8,
        all_accounts: &[ProxyToken],
        is_unhealthy_or_limited: F,
    ) -> Option<f64>
    where
        F: Fn(&str) -> bool,
    {
        let now = Instant::now();
        let sample_ttl = Duration::from_secs(15 * 60);
        let norm_target_tier = target_tier.trim().to_uppercase();
        let mut values: Vec<f64> = Vec::new();

        for acc in all_accounts {
            let acc_tier = acc
                .subscription_tier
                .as_deref()
                .unwrap_or("FREE")
                .trim()
                .to_uppercase();
            if acc_tier != norm_target_tier || acc.account_id == target_account_id {
                continue;
            }
            if is_unhealthy_or_limited(&acc.account_id) {
                continue;
            }
            if let Some(state) = self.states.get(&acc.account_id) {
                if !matches!(state.value(), AccountQoSState::Healthy) {
                    continue;
                }
            }
            if let Some(samples) = self.samples.get(&acc.account_id) {
                for s in samples.iter() {
                    if now.duration_since(s.timestamp) > sample_ttl
                        || !s.is_cold_start
                        || s.uncached_bucket.abs_diff(target_bucket) > 1
                    {
                        continue;
                    }
                    values.push(s.ttft_ms);
                }
            }
        }

        if values.len() < MIN_COLD_PEER_SAMPLES {
            return None;
        }
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }

    /// 评估单次请求的 TTFT 并流转状态机
    pub fn record_and_evaluate<F>(
        &self,
        account_id: &str,
        session_id: Option<&str>,
        tier: &str,
        ttft_ms: f64,
        is_deep_thinking: bool,
        input_tokens: Option<u32>,
        all_accounts: &[ProxyToken],
        is_unhealthy_or_limited: F,
    ) -> QoSAction
    where
        F: Fn(&str) -> bool,
    {
        // 若未启用自动限流进小黑屋机制，直接忽略校验时长，不标记疑似限速，不进小黑屋，不解绑会话
        if !self.is_auto_jail_enabled() {
            return QoSAction::None;
        }

        let config = self.config_snapshot();
        let ttft_secs = ttft_ms / 1000.0;
        let now = Instant::now();

        // 1. 未缓存 Token 规模计算 (Uncached Tokens)
        // 区分大 Token 冷启动 (Cold Start Prefill) 与 热缓存交互 (Hot Cache Turn)
        let uncached_tokens = if let Some(sid) = session_id.filter(|s| !s.is_empty()) {
            let key = (account_id.to_string(), sid.to_string());
            let uncached = match self.session_watermarks.get(&key) {
                Some(prev) => {
                    let (prev_toks, prev_time) = *prev.value();
                    // 若距离上次请求超过 5 分钟，上游 KV Cache 大概率已失效淘汰，按全量冷启动计算
                    if now.duration_since(prev_time) > Duration::from_secs(5 * 60) {
                        input_tokens.unwrap_or(0)
                    } else {
                        input_tokens.unwrap_or(0).saturating_sub(prev_toks)
                    }
                }
                None => {
                    // 该会话在该账号上的首次请求 (冷启动 Prefill)
                    input_tokens.unwrap_or(0)
                }
            };
            if let Some(tokens) = input_tokens {
                self.session_watermarks.insert(key, (tokens, now));
            }
            uncached
        } else {
            input_tokens.unwrap_or(0)
        };

        if self.session_watermarks.len() > 2000 {
            self.session_watermarks
                .retain(|_, (_, ts)| now.duration_since(*ts).as_secs() < 15 * 60);
        }

        // 2. 动态预算补偿算法：
        // - 每 100K 未缓存 Token 额外补偿 +15.0s 物理 Prefill 计算耗时
        // - 基础阈值：按实例策略选择普通或深度思考阈值
        let uncached_bonus_secs = (uncached_tokens as f64 / 100_000.0) * 15.0;
        let base_threshold_secs = if is_deep_thinking {
            config.deep_thinking_threshold_secs
        } else {
            config.normal_threshold_secs
        };
        let threshold_secs = base_threshold_secs + uncached_bonus_secs;

        // 恢复阈值同样支持未缓存增量补偿
        let vindicate_threshold_secs = config.recovery_threshold_secs + uncached_bonus_secs;

        // 标记是否为大 Token 冷启动 (未缓存 >= 60K)
        let is_cold_start = uncached_tokens >= 60_000;
        let bucket = cold_bucket_of(uncached_tokens);

        // 冷启动不搞“一刀切豁免”（真被限速也测不出来），而是**只和同桶冷启动样本比**：
        // - 有 ≥3 个同桶基线 → 正常参与 strike 判定；
        // - 没有基线 → 只记样本、不判罚（首次遇到该规模时绝不误杀）。
        let cold_peer_avg = if is_cold_start {
            self.calculate_cold_peer_avg(
                account_id,
                tier,
                bucket,
                all_accounts,
                &is_unhealthy_or_limited,
            )
        } else {
            None
        };

        if is_cold_start && cold_peer_avg.is_none() {
            tracing::debug!(
                "[QoS] cold-start without same-bucket baseline | account={} ttft={:.2}s uncached={} bucket={} strike_threshold={:.1}s",
                account_id,
                ttft_secs,
                uncached_tokens,
                bucket,
                threshold_secs
            );
            self.push_sample_with_bucket(account_id, ttft_ms, is_deep_thinking, true, bucket);
            return QoSAction::None;
        }

        // 3. 提取同侪均值 (秒)：冷启动对比同桶冷启动，常规对比热缓存样本
        let peer_avg_secs = if is_cold_start {
            cold_peer_avg
        } else {
            self.calculate_pure_peer_avg(account_id, tier, all_accounts, &is_unhealthy_or_limited)
        }
        .map(|ms| ms / 1000.0);

        // 3.1 冷启动的相对阈值：有同桶基线时，拿“同桶均值的 1.5 倍”与绝对阈值取大。
        // 否则 150K 冷启动的 42.5s 绝对线会卡在实测 36~44s 的抖动区间，依旧可能误杀。
        let threshold_secs = match (is_cold_start, peer_avg_secs) {
            (true, Some(peer_avg)) => threshold_secs.max(peer_avg * 1.5),
            _ => threshold_secs,
        };
        let no_peer_grace_secs = if peer_avg_secs.is_none() {
            config.no_peer_grace_secs
        } else {
            0.0
        };
        let final_threshold_secs = threshold_secs + no_peer_grace_secs;
        tracing::debug!(
            account_id, ttft_secs, is_deep_thinking, base_threshold_secs,
            uncached_bonus_secs, peer_avg_secs = ?peer_avg_secs, no_peer_grace_secs,
            final_threshold_secs, recovery_threshold_secs = vindicate_threshold_secs,
            "Evaluating local QoS latency"
        );

        // 4. 同侪均值也达到本次阈值时，按全局拥堵豁免单号处罚
        if let Some(peer_avg) = peer_avg_secs {
            if peer_avg >= threshold_secs {
                tracing::warn!(
                    "🌐 [QoS Guardian] Global upstream congestion detected (peer_avg={:.2}s >= {:.2}s). Bypassing strike for account {}",
                    peer_avg, threshold_secs, account_id
                );
                // 记录采样后退出
                self.push_sample_with_bucket(
                    account_id,
                    ttft_ms,
                    is_deep_thinking,
                    is_cold_start,
                    bucket,
                );
                return QoSAction::None;
            }
        }

        // 5. 判定是否构成一次卡顿打击 (Strike)
        let is_strike = ttft_secs > final_threshold_secs;

        let mut current_state = self
            .states
            .entry(account_id.to_string())
            .or_insert(AccountQoSState::Healthy);

        let action = match current_state.value().clone() {
            AccountQoSState::Healthy => {
                if is_strike {
                    // 第 1 次超标：打上瞬态可疑标记！
                    tracing::warn!(
                        "⚠️ [QoS Guardian] Account {} tagged as SUSPICIOUS (TTFT={:.2}s > {:.2}s, uncached_tokens={}, bonus=+{:.2}s)",
                        account_id,
                        ttft_secs,
                        final_threshold_secs,
                        uncached_tokens,
                        uncached_bonus_secs
                    );
                    *current_state = AccountQoSState::Suspicious {
                        first_strike_at: now,
                        strike_count: 1,
                        consecutive_healthy: 0,
                    };
                    QoSAction::TaggedSuspicious
                } else {
                    QoSAction::None
                }
            }
            AccountQoSState::Suspicious {
                first_strike_at,
                mut strike_count,
                mut consecutive_healthy,
            } => {
                // 检查第 1 次 Strike 是否在 5 分钟滑动窗口内
                let window_valid =
                    now.duration_since(first_strike_at) <= Duration::from_secs(5 * 60);

                if is_strike {
                    if window_valid {
                        strike_count += 1;
                    } else {
                        strike_count = 1;
                    }

                    if strike_count >= 2 {
                        // 5 分钟内累计 2 次超标：坐实被上游降速！关入小黑屋并触发会话解绑
                        let level = self.resolve_initial_jail_level(account_id, &config);
                        let wait_duration = Duration::from_secs(config.jail_ladder_secs[level]);

                        tracing::error!(
                            "🚨 [QoS Guardian] Account {} exceeded the local QoS latency threshold twice in the strike window. Jailing Level {} ({:?}) & unbinding session.",
                            account_id, level, wait_duration
                        );

                        // 记录惩罚历史（保留 30 分钟记忆）
                        self.penalty_memory
                            .insert(account_id.to_string(), (level, now));

                        *current_state = AccountQoSState::Jailed {
                            level,
                            jailed_at: now, // 从 0 开始重新计时
                            current_wait: wait_duration,
                        };

                        QoSAction::TriggerJailedAndUnbind {
                            level,
                            wait_duration,
                        }
                    } else {
                        *current_state = AccountQoSState::Suspicious {
                            first_strike_at: now,
                            strike_count,
                            consecutive_healthy: 0,
                        };
                        QoSAction::TaggedSuspicious
                    }
                } else if ttft_secs < vindicate_threshold_secs {
                    // 表现良好 (TTFT < 宽限线)：双重确认平反
                    consecutive_healthy += 1;
                    if consecutive_healthy >= 2 || !window_valid {
                        tracing::info!(
                            "✅ [QoS Guardian] Account {} vindicated (consecutive healthy passes), restored to HEALTHY",
                            account_id
                        );
                        *current_state = AccountQoSState::Healthy;
                        QoSAction::Vindicated
                    } else {
                        *current_state = AccountQoSState::Suspicious {
                            first_strike_at,
                            strike_count,
                            consecutive_healthy,
                        };
                        QoSAction::None
                    }
                } else {
                    // vindicate_threshold_secs ~ threshold_secs 处于迟滞死区 (Deadband)：不惩罚也不平反，保持原状态
                    if !window_valid {
                        // 超过 5 分钟无再次违规，TTL 衰减平反
                        tracing::info!(
                            "🕒 [QoS Guardian] Account {} suspicious flag expired via TTL, restored to HEALTHY",
                            account_id
                        );
                        *current_state = AccountQoSState::Healthy;
                        QoSAction::Vindicated
                    } else {
                        QoSAction::None
                    }
                }
            }
            AccountQoSState::HalfOpen {
                level,
                mut consecutive_healthy,
                ..
            } => {
                if is_strike {
                    // 缓刑期间真实业务请求再次卡顿：判定假康复，从严惩罚！
                    let next_level = (level + 1).min(config.jail_ladder_secs.len() - 1);
                    let next_wait = Duration::from_secs(config.jail_ladder_secs[next_level]);

                    tracing::error!(
                        "💥 [QoS Guardian] Account {} FAILED Half-Open probation (TTFT={:.2}s). Escalating to Level {} ({:?})!",
                        account_id, ttft_secs, next_level, next_wait
                    );

                    self.penalty_memory
                        .insert(account_id.to_string(), (next_level, now));

                    *current_state = AccountQoSState::Jailed {
                        level: next_level,
                        jailed_at: now, // 从 0 重新计时
                        current_wait: next_wait,
                    };

                    QoSAction::HalfOpenFailed {
                        escalated_level: next_level,
                        wait_duration: next_wait,
                    }
                } else if ttft_secs < vindicate_threshold_secs {
                    // 缓刑期间真实请求达标 (< vindicate_threshold_secs)
                    consecutive_healthy += 1;
                    if consecutive_healthy >= 2 {
                        tracing::info!(
                            "🎉 [QoS Guardian] Account {} fully PASSED Half-Open probation (2 consecutive healthy real turns). Restored to HEALTHY!",
                            account_id
                        );
                        *current_state = AccountQoSState::Healthy;
                        QoSAction::HalfOpenPassed
                    } else {
                        *current_state = AccountQoSState::HalfOpen {
                            level,
                            consecutive_healthy,
                            entered_at: now,
                        };
                        QoSAction::None
                    }
                } else {
                    // 死区：不记通过，也不算失败
                    QoSAction::None
                }
            }
            AccountQoSState::Jailed { .. } | AccountQoSState::Probing { .. } => {
                // 小黑屋与探测中的账号通常不承接生产流量
                QoSAction::None
            }
        };

        // 记录采样 (冷启动样本同时记入其规模桶，供后续同桶对比)
        self.push_sample_with_bucket(account_id, ttft_ms, is_deep_thinking, is_cold_start, bucket);

        action
    }

    /// 确定入狱初始阶梯：30 分钟内二次复发直接继承并升级阶梯
    fn resolve_initial_jail_level(&self, account_id: &str, config: &QoSConfig) -> usize {
        let memory_ttl = Duration::from_secs(30 * 60);
        if let Some(mem) = self.penalty_memory.get(account_id) {
            let (last_level, recorded_at) = *mem.value();
            if recorded_at.elapsed() <= memory_ttl {
                let escalated = (last_level + 1).min(config.jail_ladder_secs.len() - 1);
                tracing::info!(
                    "🧠 [QoS Guardian] Account {} has recent penalty memory (Lv {}), escalating to Lv {}",
                    account_id, last_level, escalated
                );
                return escalated;
            }
        }
        0 // 初犯从 Level 0 (2分钟) 开始
    }

    /// 记录历史滑动采样 (保留最近 15 分钟)
    pub fn push_sample(
        &self,
        account_id: &str,
        ttft_ms: f64,
        is_thinking: bool,
        is_cold_start: bool,
    ) {
        self.push_sample_with_bucket(account_id, ttft_ms, is_thinking, is_cold_start, 0);
    }

    /// 带分桶的采样入口
    fn push_sample_with_bucket(
        &self,
        account_id: &str,
        ttft_ms: f64,
        is_thinking: bool,
        is_cold_start: bool,
        uncached_bucket: u8,
    ) {
        let mut entry = self.samples.entry(account_id.to_string()).or_default();
        entry.push(TtftSample {
            ttft_ms,
            timestamp: Instant::now(),
            is_thinking,
            is_cold_start,
            uncached_bucket,
        });
        let ttl = Duration::from_secs(15 * 60);
        entry.retain(|s| s.timestamp.elapsed() <= ttl);
    }

    /// 只有仍持有本次探针领取权时，才将账号置为 HalfOpen。
    pub(crate) fn mark_probe_passed(&self, account_id: &str, claim: ProbeClaim) -> bool {
        let Some(mut state) = self.states.get_mut(account_id) else {
            return false;
        };
        if !claim.matches(state.value()) {
            return false;
        }

        *state = AccountQoSState::HalfOpen {
            level: claim.level,
            consecutive_healthy: 0,
            entered_at: Instant::now(),
        };
        true
    }

    /// Resume only an expired jail; an active probe owns its own transition.
    pub fn promote_to_half_open(&self, account_id: &str) {
        let Some(mut state) = self.states.get_mut(account_id) else {
            return;
        };
        let level = match *state {
            AccountQoSState::Jailed {
                level,
                jailed_at,
                current_wait,
            } if jailed_at.elapsed() >= current_wait => level,
            _ => return,
        };
        *state = AccountQoSState::HalfOpen {
            level,
            consecutive_healthy: 0,
            entered_at: Instant::now(),
        };
    }

    /// 本次探针失败升级阶梯；已重置、移除或被新探针替换的结果不再回写。
    pub(crate) fn mark_probe_failed(
        &self,
        account_id: &str,
        claim: ProbeClaim,
    ) -> Option<(usize, Duration)> {
        let mut state = self.states.get_mut(account_id)?;
        if !claim.matches(state.value()) {
            return None;
        }

        let next_level = (claim.level + 1).min(claim.config.jail_ladder_secs.len() - 1);
        let next_wait = Duration::from_secs(claim.config.jail_ladder_secs[next_level]);
        let now = Instant::now();

        self.penalty_memory
            .insert(account_id.to_string(), (next_level, now));

        *state = AccountQoSState::Jailed {
            level: next_level,
            jailed_at: now, // 从 0 开始重新计时
            current_wait: next_wait,
        };

        Some((next_level, next_wait))
    }

    /// 原子领取已到期的隔离账号；候选快照中的状态可能已被重置或重新隔离。
    #[cfg(test)]
    pub(crate) fn try_mark_probing(&self, account_id: &str) -> Option<ProbeClaim> {
        self.try_mark_probing_with_config(account_id, self.config_snapshot())
    }

    pub(crate) fn try_mark_probing_with_config(
        &self,
        account_id: &str,
        config: QoSConfig,
    ) -> Option<ProbeClaim> {
        if !self.is_auto_jail_enabled() {
            return None;
        }
        let mut state = self.states.get_mut(account_id)?;
        let AccountQoSState::Jailed {
            level,
            jailed_at,
            current_wait,
        } = *state
        else {
            return None;
        };
        if jailed_at.elapsed() < current_wait {
            return None;
        }

        let claim = ProbeClaim {
            level,
            config,
            claim_id: uuid::Uuid::new_v4(),
        };
        *state = AccountQoSState::Probing {
            level,
            started_at: Instant::now(),
            claim_id: claim.claim_id,
        };
        Some(claim)
    }

    /// Collect a snapshot before writing to DashMap (upstream Fixes #3749).
    /// Prepare credentials without state guards, then atomically claim the current
    /// expired jail. Claims retain the local policy and reject stale completions.
    pub(crate) fn take_expired_jails_for_probing<T>(
        &self,
        mut prepare: impl FnMut(&str, QoSConfig) -> Option<T>,
    ) -> Vec<(T, ProbeClaim)> {
        if !self.is_auto_jail_enabled() {
            return Vec::new();
        }
        let candidates: Vec<String> = self
            .states
            .iter()
            .filter_map(|entry| match *entry.value() {
                AccountQoSState::Jailed {
                    jailed_at,
                    current_wait,
                    ..
                } if jailed_at.elapsed() >= current_wait => Some(entry.key().clone()),
                _ => None,
            })
            .collect();
        let mut expired = Vec::with_capacity(candidates.len());
        for account_id in candidates {
            let config = self.config_snapshot();
            if let Some(prepared) = prepare(&account_id, config) {
                if let Some(claim) = self.try_mark_probing_with_config(&account_id, config) {
                    expired.push((prepared, claim));
                }
            }
        }
        expired
    }

    /// 强制恢复健康 (如在 UI 上手动重置)
    pub fn reset_account(&self, account_id: &str) {
        self.states
            .insert(account_id.to_string(), AccountQoSState::Healthy);
        self.samples.remove(account_id);
        self.penalty_memory.remove(account_id);
        self.session_watermarks
            .retain(|(acc, _), _| acc != account_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};
    use std::path::PathBuf;

    fn configurable_policy() -> QoSConfig {
        QoSConfig {
            normal_threshold_secs: 12.0,
            deep_thinking_threshold_secs: 22.0,
            recovery_threshold_secs: 10.0,
            no_peer_grace_secs: 3.0,
            probe_timeout_secs: 3.0,
            probe_success_threshold_secs: 2.0,
            jail_ladder_secs: [10, 20, 30, 40, 50, 60],
        }
    }

    #[test]
    fn qos_cold_start_requires_bucket_baseline_and_keeps_configured_threshold() {
        let guardian = QoSGuardian::new();
        guardian.set_auto_jail_enabled(true);
        guardian.update_config(configurable_policy()).unwrap();
        let accounts = vec![
            create_test_token("account", "PRO"),
            create_test_token("peer", "PRO"),
        ];
        // A hot-cache peer must not cause a cold-start false positive.
        guardian.push_sample("peer", 4000.0, false, false);
        assert_eq!(
            guardian.record_and_evaluate(
                "account",
                None,
                "PRO",
                100_000.0,
                false,
                Some(100_000),
                &accounts,
                |_| false
            ),
            QoSAction::None
        );
        for _ in 0..3 {
            guardian.push_sample_with_bucket("peer", 4000.0, false, true, cold_bucket_of(100_000));
        }
        // Local base 12s + upstream prefill compensation 15s = 27s.
        assert_eq!(
            guardian.record_and_evaluate(
                "account",
                None,
                "PRO",
                27_000.0,
                false,
                Some(100_000),
                &accounts,
                |_| false
            ),
            QoSAction::None
        );
        assert_eq!(
            guardian.record_and_evaluate(
                "account",
                None,
                "PRO",
                27_001.0,
                false,
                Some(100_000),
                &accounts,
                |_| false
            ),
            QoSAction::TaggedSuspicious
        );
    }

    #[test]
    fn qos_snapshot_claim_preserves_reset_during_credential_preparation() {
        let guardian = QoSGuardian::new();
        guardian.set_auto_jail_enabled(true);
        guardian.states.insert(
            "account".into(),
            AccountQoSState::Jailed {
                level: 0,
                jailed_at: Instant::now(),
                current_wait: Duration::ZERO,
            },
        );
        let claims = guardian.take_expired_jails_for_probing(|id, _| {
            guardian.reset_account(id);
            Some(id.to_string())
        });
        assert!(claims.is_empty());
        assert_eq!(guardian.get_state("account"), AccountQoSState::Healthy);
    }

    #[test]
    fn qos_configured_strike_thresholds_preserve_strict_boundaries_and_input_compensation() {
        for (thinking, peer, input_tokens, boundary_secs) in [
            (false, true, 0, 12.0),
            (true, true, 0, 22.0),
            (false, false, 0, 15.0),
            (true, false, 0, 25.0),
            (false, true, 40_000, 18.0),
            (true, true, 40_000, 28.0),
            (false, false, 40_000, 21.0),
            (true, false, 40_000, 31.0),
        ] {
            for (offset_ms, expected) in
                [(0.0, QoSAction::None), (1.0, QoSAction::TaggedSuspicious)]
            {
                let guardian = QoSGuardian::new();
                guardian.set_auto_jail_enabled(true);
                guardian.update_config(configurable_policy()).unwrap();
                let accounts = vec![
                    create_test_token("account", "PRO"),
                    create_test_token("peer", "PRO"),
                ];
                if peer {
                    guardian.push_sample("peer", 4000.0, false, false);
                }
                let action = guardian.record_and_evaluate(
                    "account",
                    None,
                    "PRO",
                    boundary_secs * 1000.0 + offset_ms,
                    thinking,
                    Some(input_tokens),
                    &accounts,
                    |_| false,
                );
                assert_eq!(
                    action, expected,
                    "thinking={thinking}, peer={peer}, input={input_tokens}, offset={offset_ms}"
                );
            }
        }
    }

    #[test]
    fn qos_configured_peer_brake_uses_the_same_request_threshold() {
        for (thinking, input, threshold) in
            [(false, 0, 12.0), (true, 0, 22.0), (false, 40_000, 18.0)]
        {
            for (peer_offset_ms, expected) in
                [(0.0, QoSAction::None), (-1.0, QoSAction::TaggedSuspicious)]
            {
                let guardian = QoSGuardian::new();
                guardian.set_auto_jail_enabled(true);
                guardian.update_config(configurable_policy()).unwrap();
                let accounts = vec![
                    create_test_token("account", "PRO"),
                    create_test_token("peer", "PRO"),
                ];
                guardian.push_sample("peer", threshold * 1000.0 + peer_offset_ms, false, false);
                assert_eq!(
                    guardian.record_and_evaluate(
                        "account",
                        None,
                        "PRO",
                        100_000.0,
                        thinking,
                        Some(input),
                        &accounts,
                        |_| false
                    ),
                    expected
                );
            }
        }
    }

    #[test]
    fn qos_configured_recovery_threshold_preserves_deadband_and_two_pass_rule() {
        for (half_open, input_tokens, boundary_secs) in [
            (false, 0, 10.0),
            (true, 0, 10.0),
            (false, 40_000, 16.0),
            (true, 40_000, 16.0),
        ] {
            let guardian = QoSGuardian::new();
            guardian.set_auto_jail_enabled(true);
            guardian.update_config(configurable_policy()).unwrap();
            let state = if half_open {
                AccountQoSState::HalfOpen {
                    level: 0,
                    consecutive_healthy: 0,
                    entered_at: Instant::now(),
                }
            } else {
                AccountQoSState::Suspicious {
                    first_strike_at: Instant::now(),
                    strike_count: 1,
                    consecutive_healthy: 0,
                }
            };
            guardian.states.insert("account".into(), state.clone());
            let accounts = vec![
                create_test_token("account", "PRO"),
                create_test_token("peer", "PRO"),
            ];
            guardian.push_sample("peer", 4000.0, false, false);
            assert_eq!(
                guardian.record_and_evaluate(
                    "account",
                    None,
                    "PRO",
                    boundary_secs * 1000.0,
                    false,
                    Some(input_tokens),
                    &accounts,
                    |_| false
                ),
                QoSAction::None
            );
            assert_eq!(guardian.get_state("account"), state);
            assert_eq!(
                guardian.record_and_evaluate(
                    "account",
                    None,
                    "PRO",
                    boundary_secs * 1000.0 - 1.0,
                    false,
                    Some(input_tokens),
                    &accounts,
                    |_| false
                ),
                QoSAction::None
            );
            assert_eq!(
                guardian.record_and_evaluate(
                    "account",
                    None,
                    "PRO",
                    boundary_secs * 1000.0 - 1.0,
                    false,
                    Some(input_tokens),
                    &accounts,
                    |_| false
                ),
                if half_open {
                    QoSAction::HalfOpenPassed
                } else {
                    QoSAction::Vindicated
                }
            );
            assert!(guardian.is_healthy("account"));
        }
    }

    #[test]
    fn qos_config_update_preserves_existing_state_and_is_instance_local() {
        let guardian = QoSGuardian::new();
        guardian.set_auto_jail_enabled(true);
        let other = QoSGuardian::new();
        let now = Instant::now();
        let jailed = AccountQoSState::Jailed {
            level: 3,
            jailed_at: now,
            current_wait: Duration::from_secs(900),
        };
        guardian.states.insert("account".into(), jailed.clone());
        guardian.penalty_memory.insert("account".into(), (3, now));
        guardian.push_sample("account", 4000.0, false, false);
        guardian
            .session_watermarks
            .insert(("account".into(), "session".into()), (100, now));

        guardian.update_config(configurable_policy()).unwrap();

        assert_eq!(guardian.config_snapshot(), configurable_policy());
        assert_eq!(other.config_snapshot(), QoSConfig::default());
        let fresh_accounts = vec![create_test_token("fresh", "PRO")];
        assert_eq!(
            guardian.record_and_evaluate(
                "fresh",
                None,
                "PRO",
                20_000.0,
                false,
                None,
                &fresh_accounts,
                |_| false
            ),
            QoSAction::TaggedSuspicious
        );
        assert_eq!(
            other.record_and_evaluate(
                "fresh",
                None,
                "PRO",
                20_000.0,
                false,
                None,
                &fresh_accounts,
                |_| false
            ),
            QoSAction::None
        );
        assert_eq!(guardian.get_state("account"), jailed);
        assert_eq!(guardian.try_mark_probing("account"), None);
        assert_eq!(*guardian.penalty_memory.get("account").unwrap(), (3, now));
        assert_eq!(guardian.samples.get("account").unwrap().len(), 1);
        assert_eq!(
            *guardian
                .session_watermarks
                .get(&("account".into(), "session".into()))
                .unwrap(),
            (100, now)
        );

        let mut invalid = configurable_policy();
        invalid.normal_threshold_secs = f64::NAN;
        assert!(guardian.update_config(invalid).is_err());
        assert_eq!(guardian.config_snapshot(), configurable_policy());
        assert_eq!(guardian.get_state("account"), jailed);
    }

    #[test]
    fn qos_probe_claim_keeps_policy_and_ladder_until_completion() {
        for level in 0..6 {
            let guardian = QoSGuardian::new();
            guardian.set_auto_jail_enabled(true);
            let original = configurable_policy();
            guardian.update_config(original).unwrap();
            let claim = jail_for_probe(&guardian, level);
            let mut changed = original;
            changed.probe_timeout_secs = 5.0;
            changed.probe_success_threshold_secs = 4.0;
            changed.jail_ladder_secs = [11, 22, 33, 44, 55, 66];
            guardian.update_config(changed).unwrap();

            assert_eq!(claim.config, original);
            assert!(claim.accepts_ttft(1999.0));
            assert!(!claim.accepts_ttft(2000.0));
            assert!(!claim.accepts_ttft(3000.0));
            let next_level = (level + 1).min(5);
            assert_eq!(
                guardian.mark_probe_failed("account", claim),
                Some((
                    next_level,
                    Duration::from_secs(original.jail_ladder_secs[next_level])
                ))
            );
            let next_claim = expire_jail_and_claim(&guardian, "account");
            assert_eq!(next_claim.config, changed);
            assert!(next_claim.accepts_ttft(3000.0));
        }
    }

    #[test]
    fn qos_configured_ladder_applies_to_new_jail_and_half_open_failure() {
        let guardian = QoSGuardian::new();
        guardian.set_auto_jail_enabled(true);
        guardian.update_config(configurable_policy()).unwrap();
        let accounts = vec![create_test_token("account", "PRO")];
        guardian.record_and_evaluate(
            "account",
            None,
            "PRO",
            30_000.0,
            false,
            None,
            &accounts,
            |_| false,
        );
        assert_eq!(
            guardian.record_and_evaluate(
                "account",
                None,
                "PRO",
                30_000.0,
                false,
                None,
                &accounts,
                |_| false
            ),
            QoSAction::TriggerJailedAndUnbind {
                level: 0,
                wait_duration: Duration::from_secs(10)
            }
        );
        let claim = expire_jail_and_claim(&guardian, "account");
        assert!(guardian.mark_probe_passed("account", claim));
        let mut changed = configurable_policy();
        changed.jail_ladder_secs = [11, 22, 33, 44, 55, 66];
        guardian.update_config(changed).unwrap();
        assert_eq!(
            guardian.record_and_evaluate(
                "account",
                None,
                "PRO",
                30_000.0,
                false,
                None,
                &accounts,
                |_| false
            ),
            QoSAction::HalfOpenFailed {
                escalated_level: 1,
                wait_duration: Duration::from_secs(22)
            }
        );
    }

    #[test]
    fn qos_config_updates_do_not_wait_for_account_state_locks() {
        let guardian = QoSGuardian::new();
        guardian.set_auto_jail_enabled(true);
        guardian
            .states
            .insert("account".into(), AccountQoSState::Healthy);
        std::thread::scope(|scope| {
            let state = guardian.states.get_mut("account").unwrap();
            let (tx, rx) = std::sync::mpsc::channel();
            let guardian_ref = &guardian;
            scope.spawn(move || {
                tx.send(guardian_ref.update_config(configurable_policy()))
                    .unwrap();
            });
            let result = rx.recv_timeout(Duration::from_secs(2));
            drop(state);
            result
                .expect("config update must not acquire account state locks")
                .unwrap();
        });
    }

    #[test]
    fn probe_claim_uses_current_expired_jail_level_once() {
        let guardian = QoSGuardian::new();
        guardian.set_auto_jail_enabled(true);
        guardian.states.insert(
            "account".into(),
            AccountQoSState::Jailed {
                level: 3,
                jailed_at: Instant::now(),
                current_wait: Duration::ZERO,
            },
        );

        assert_eq!(guardian.try_mark_probing("account").unwrap().level, 3);
        assert!(matches!(
            guardian.get_state("account"),
            AccountQoSState::Probing { level: 3, .. }
        ));
        assert_eq!(guardian.try_mark_probing("account"), None);
    }

    #[test]
    fn probe_claim_preserves_states_changed_after_candidate_scan() {
        let guardian = QoSGuardian::new();
        guardian.set_auto_jail_enabled(true);
        let now = Instant::now();
        for current in [
            AccountQoSState::Healthy,
            AccountQoSState::Suspicious {
                first_strike_at: now,
                strike_count: 1,
                consecutive_healthy: 0,
            },
            AccountQoSState::Jailed {
                level: 2,
                jailed_at: now,
                current_wait: Duration::from_secs(120),
            },
            AccountQoSState::Probing {
                level: 1,
                started_at: now,
                claim_id: uuid::Uuid::new_v4(),
            },
            AccountQoSState::HalfOpen {
                level: 1,
                consecutive_healthy: 0,
                entered_at: now,
            },
        ] {
            // A candidate ID alone must not authorize overwriting its current state.
            guardian.states.insert("account".into(), current.clone());
            assert_eq!(guardian.try_mark_probing("account"), None, "{current:?}");
            assert_eq!(guardian.get_state("account"), current);
        }
    }

    #[test]
    fn probe_claim_does_not_recreate_a_removed_account() {
        let guardian = QoSGuardian::new();
        guardian.set_auto_jail_enabled(true);
        assert_eq!(guardian.try_mark_probing("removed"), None);
        assert!(!guardian.states.contains_key("removed"));
    }

    #[test]
    fn probe_claim_has_one_winner_across_concurrent_scans() {
        let guardian = QoSGuardian::new();
        guardian.set_auto_jail_enabled(true);
        guardian.states.insert(
            "account".into(),
            AccountQoSState::Jailed {
                level: 0,
                jailed_at: Instant::now(),
                current_wait: Duration::ZERO,
            },
        );
        let barrier = std::sync::Barrier::new(4);
        let winners = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        barrier.wait();
                        guardian.try_mark_probing("account").is_some()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| usize::from(handle.join().unwrap()))
                .sum::<usize>()
        });
        assert_eq!(winners, 1);
    }

    fn expire_jail_and_claim(guardian: &QoSGuardian, account_id: &str) -> ProbeClaim {
        {
            let mut state = guardian.states.get_mut(account_id).unwrap();
            let AccountQoSState::Jailed {
                jailed_at,
                current_wait,
                ..
            } = state.value_mut()
            else {
                panic!("expected a jailed account before probing");
            };
            *jailed_at = Instant::now() - *current_wait;
        }
        guardian.try_mark_probing(account_id).unwrap()
    }

    fn jail_for_probe(guardian: &QoSGuardian, level: usize) -> ProbeClaim {
        guardian.states.insert(
            "account".into(),
            AccountQoSState::Jailed {
                level,
                jailed_at: Instant::now(),
                current_wait: Duration::ZERO,
            },
        );
        guardian.try_mark_probing("account").unwrap()
    }

    #[test]
    fn probe_completion_preserves_manual_reset_and_release() {
        for reset in [QoSGuardian::reset_account, QoSGuardian::release_from_jail] {
            let guardian = QoSGuardian::new();
            guardian.set_auto_jail_enabled(true);
            let claim = jail_for_probe(&guardian, 1);
            guardian
                .penalty_memory
                .insert("account".into(), (1, Instant::now()));

            reset(&guardian, "account");

            assert!(!guardian.mark_probe_passed("account", claim));
            assert_eq!(guardian.mark_probe_failed("account", claim), None);
            assert_eq!(guardian.get_state("account"), AccountQoSState::Healthy);
            assert!(!guardian.penalty_memory.contains_key("account"));
        }
    }

    #[test]
    fn probe_completion_does_not_recreate_a_removed_account() {
        let guardian = QoSGuardian::new();
        guardian.set_auto_jail_enabled(true);
        let claim = jail_for_probe(&guardian, 1);
        guardian.states.remove("account");

        assert!(!guardian.mark_probe_passed("account", claim));
        assert_eq!(guardian.mark_probe_failed("account", claim), None);
        assert!(!guardian.states.contains_key("account"));
        assert!(!guardian.penalty_memory.contains_key("account"));
    }

    #[test]
    fn stale_probe_completion_preserves_a_new_claim_at_the_same_level() {
        let guardian = QoSGuardian::new();
        guardian.set_auto_jail_enabled(true);
        let old_claim = jail_for_probe(&guardian, 1);
        guardian.reset_account("account");
        let new_claim = jail_for_probe(&guardian, 1);
        let current = guardian.get_state("account");
        assert_ne!(old_claim, new_claim);

        assert!(!guardian.mark_probe_passed("account", old_claim));
        assert_eq!(guardian.mark_probe_failed("account", old_claim), None);
        assert_eq!(guardian.get_state("account"), current);
        assert!(!guardian.penalty_memory.contains_key("account"));
        assert!(guardian.mark_probe_passed("account", new_claim));
        assert!(guardian.is_half_open("account"));
    }

    #[test]
    fn stale_probe_completion_preserves_a_later_jail_and_penalty() {
        let guardian = QoSGuardian::new();
        guardian.set_auto_jail_enabled(true);
        let claim = jail_for_probe(&guardian, 0);
        let now = Instant::now();
        let current = AccountQoSState::Jailed {
            level: 3,
            jailed_at: now,
            current_wait: Duration::from_secs(guardian.config_snapshot().jail_ladder_secs[3]),
        };
        guardian.states.insert("account".into(), current.clone());
        guardian.penalty_memory.insert("account".into(), (3, now));

        assert!(!guardian.mark_probe_passed("account", claim));
        assert_eq!(guardian.mark_probe_failed("account", claim), None);
        assert_eq!(guardian.get_state("account"), current);
        assert_eq!(*guardian.penalty_memory.get("account").unwrap(), (3, now));
    }

    #[test]
    fn cache_first_promotion_preserves_live_jails_and_probes() {
        let guardian = QoSGuardian::new();
        guardian.set_auto_jail_enabled(true);
        let claim = jail_for_probe(&guardian, 1);
        let probing = guardian.get_state("account");
        guardian.promote_to_half_open("account");
        assert_eq!(guardian.get_state("account"), probing);
        assert!(guardian.mark_probe_passed("account", claim));

        let jailed = AccountQoSState::Jailed {
            level: 2,
            jailed_at: Instant::now(),
            current_wait: Duration::from_secs(600),
        };
        guardian.states.insert("account".into(), jailed.clone());
        guardian.promote_to_half_open("account");
        assert_eq!(guardian.get_state("account"), jailed);

        guardian.states.insert(
            "account".into(),
            AccountQoSState::Jailed {
                level: 2,
                jailed_at: Instant::now(),
                current_wait: Duration::ZERO,
            },
        );
        guardian.promote_to_half_open("account");
        assert!(guardian.is_half_open("account"));
        guardian.states.remove("account");
        guardian.promote_to_half_open("account");
        assert!(!guardian.states.contains_key("account"));
    }

    #[test]
    fn probe_completion_applies_only_once() {
        for passes in [true, false] {
            let guardian = QoSGuardian::new();
            guardian.set_auto_jail_enabled(true);
            let claim = jail_for_probe(&guardian, 1);
            if passes {
                assert!(guardian.mark_probe_passed("account", claim));
            } else {
                assert_eq!(
                    guardian.mark_probe_failed("account", claim),
                    Some((2, Duration::from_secs(10 * 60)))
                );
            }
            let current = guardian.get_state("account");
            let penalty = guardian.penalty_memory.get("account").map(|entry| *entry);

            assert!(!guardian.mark_probe_passed("account", claim));
            assert_eq!(guardian.mark_probe_failed("account", claim), None);
            assert_eq!(guardian.get_state("account"), current);
            assert_eq!(
                guardian.penalty_memory.get("account").map(|entry| *entry),
                penalty
            );
        }
    }

    fn create_test_token(account_id: &str, tier: &str) -> ProxyToken {
        ProxyToken {
            account_id: account_id.to_string(),
            priority: 50,
            model_priorities: HashMap::new(),
            access_token: "token".to_string(),
            refresh_token: "refresh".to_string(),
            expires_in: 3600,
            timestamp: chrono::Utc::now().timestamp() + 3600,
            email: format!("{}@test.com", account_id),
            account_path: PathBuf::from("/tmp/test"),
            project_id: None,
            subscription_tier: Some(tier.to_string()),
            remaining_quota: Some(100),
            protected_models: HashSet::new(),
            forbidden_models: std::collections::HashSet::new(),
            health_score: 1.0,
            reset_time: None,
            weekly_reset_time: None,
            validation_blocked: false,
            validation_blocked_until: 0,
            validation_url: None,
            model_quotas: HashMap::new(),
            model_limits: HashMap::new(),
            available_models: HashSet::new(),
            model_metadata: HashMap::new(),
            model_reset_times: HashMap::new(),
            model_weekly_reset_times: HashMap::new(),
            model_quota_windows: HashMap::new(),
        }
    }

    #[test]
    fn test_pure_peer_baseline_filtering() {
        let guardian = QoSGuardian::new();
        guardian.set_auto_jail_enabled(true);
        let acc_a = create_test_token("acc_a", "PRO");
        let acc_b = create_test_token("acc_b", "PRO");
        let acc_c = create_test_token("acc_c", "PRO");
        let acc_ultra = create_test_token("acc_ultra", "ULTRA");
        let all_accounts = vec![
            acc_a.clone(),
            acc_b.clone(),
            acc_c.clone(),
            acc_ultra.clone(),
        ];

        // 注入采样数据
        guardian.push_sample("acc_b", 4000.0, false, false); // 4.0s
        guardian.push_sample("acc_c", 6000.0, false, false); // 6.0s
        guardian.push_sample("acc_ultra", 2000.0, false, false); // ULTRA 不计入 PRO

        // 正常情况下，计算 acc_a 的同类 (PRO) 纯净基线应为 (4000 + 6000) / 2 = 5000ms
        let avg = guardian.calculate_pure_peer_avg("acc_a", "PRO", &all_accounts, |_| false);
        assert_eq!(avg, Some(5000.0));

        // 若 acc_b 被标记为 Suspicious，acc_b 必须被瞬态排除
        guardian.states.insert(
            "acc_b".to_string(),
            AccountQoSState::Suspicious {
                first_strike_at: Instant::now(),
                strike_count: 1,
                consecutive_healthy: 0,
            },
        );
        let avg_after_b_suspicious =
            guardian.calculate_pure_peer_avg("acc_a", "PRO", &all_accounts, |_| false);
        assert_eq!(avg_after_b_suspicious, Some(6000.0)); // 只有 acc_c 的 6000

        // 若 acc_c 被放入小黑屋 Jailed，同行无纯净健康可用账号，返回 None
        guardian.states.insert(
            "acc_c".to_string(),
            AccountQoSState::Jailed {
                level: 0,
                jailed_at: Instant::now(),
                current_wait: Duration::from_secs(120),
            },
        );
        let avg_none = guardian.calculate_pure_peer_avg("acc_a", "PRO", &all_accounts, |_| false);
        assert_eq!(avg_none, None);
    }

    #[test]
    fn test_thinking_model_dynamic_compensation() {
        let guardian = QoSGuardian::new();
        guardian.set_auto_jail_enabled(true);
        let acc_a = create_test_token("acc_a", "PRO");
        let acc_b = create_test_token("acc_b", "PRO");
        let all = vec![acc_a.clone(), acc_b.clone()];

        // 注入同行基线 4.0s
        guardian.push_sample("acc_b", 4000.0, false, false);

        // 深度思考模型，TTFT = 25s (25000ms)，由于阈值上浮至 35s，不触发 Strike
        let action = guardian.record_and_evaluate(
            "acc_a",
            None,
            "PRO",
            25000.0,
            true, // is_deep_thinking = true
            None,
            &all,
            |_| false,
        );
        assert_eq!(action, QoSAction::None);
        assert_eq!(guardian.get_state("acc_a"), AccountQoSState::Healthy);

        // 深度思考模型，TTFT = 40s (40000ms) > 35s，此时触发 Strike
        let action_strike =
            guardian
                .record_and_evaluate("acc_a", None, "PRO", 40000.0, true, None, &all, |_| false);
        assert_eq!(action_strike, QoSAction::TaggedSuspicious);
        assert!(guardian.is_suspicious("acc_a"));
    }

    #[test]
    fn test_global_outage_brake() {
        let guardian = QoSGuardian::new();
        guardian.set_auto_jail_enabled(true);
        let acc_a = create_test_token("acc_a", "PRO");
        let acc_b = create_test_token("acc_b", "PRO");
        let all = vec![acc_a.clone(), acc_b.clone()];

        // 同行均值已经达到 25s (25000ms >= 20s)，说明上游全局拥堵
        guardian.push_sample("acc_b", 25000.0, false, false);

        // 此时 acc_a 即使耗时 26s，也触发全局制动刹车，不误判 Strike
        let action =
            guardian
                .record_and_evaluate("acc_a", None, "PRO", 26000.0, false, None, &all, |_| false);
        assert_eq!(action, QoSAction::None);
        assert_eq!(guardian.get_state("acc_a"), AccountQoSState::Healthy);
    }

    #[test]
    fn test_hysteresis_deadband() {
        let guardian = QoSGuardian::new();
        guardian.set_auto_jail_enabled(true);
        let acc_a = create_test_token("acc_a", "PRO");
        let acc_b = create_test_token("acc_b", "PRO");
        let all = vec![acc_a.clone(), acc_b.clone()];
        guardian.push_sample("acc_b", 4000.0, false, false);

        // 第 1 次超标打成 Suspicious
        guardian.record_and_evaluate("acc_a", None, "PRO", 22000.0, false, None, &all, |_| false);
        assert!(guardian.is_suspicious("acc_a"));

        // 下一次耗时 19s（处于 18.5s ~ 20s 迟滞死区），保持状态不变，不加重也不平反
        let action =
            guardian
                .record_and_evaluate("acc_a", None, "PRO", 19000.0, false, None, &all, |_| false);
        assert_eq!(action, QoSAction::None);
        assert!(guardian.is_suspicious("acc_a"));
    }

    #[test]
    fn test_sliding_window_strike_and_jail_handover() {
        let guardian = QoSGuardian::new();
        guardian.set_auto_jail_enabled(true);
        let acc_a = create_test_token("acc_a", "PRO");
        let acc_b = create_test_token("acc_b", "PRO");
        let all = vec![acc_a.clone(), acc_b.clone()];
        guardian.push_sample("acc_b", 4000.0, false, false);

        // 第一次超标 -> TaggedSuspicious
        let action1 =
            guardian
                .record_and_evaluate("acc_a", None, "PRO", 22000.0, false, None, &all, |_| false);
        assert_eq!(action1, QoSAction::TaggedSuspicious);
        assert!(guardian.is_suspicious("acc_a"));

        // 第二次超标 -> TriggerJailedAndUnbind
        let action2 =
            guardian
                .record_and_evaluate("acc_a", None, "PRO", 25000.0, false, None, &all, |_| false);
        assert_eq!(
            action2,
            QoSAction::TriggerJailedAndUnbind {
                level: 0,
                wait_duration: Duration::from_secs(120),
            }
        );
        assert!(guardian.is_jailed_or_probing("acc_a"));
    }

    #[test]
    fn test_ladder_probing_escalation_and_probation() {
        let guardian = QoSGuardian::new();
        guardian.set_auto_jail_enabled(true);
        let acc_a = create_test_token("acc_a", "PRO");
        let acc_b = create_test_token("acc_b", "PRO");
        let all = vec![acc_a.clone(), acc_b.clone()];
        guardian.push_sample("acc_b", 4000.0, false, false);

        // 触发入狱 Level 0 (2m)
        guardian.record_and_evaluate("acc_a", None, "PRO", 22000.0, false, None, &all, |_| false);
        guardian.record_and_evaluate("acc_a", None, "PRO", 22000.0, false, None, &all, |_| false);
        assert!(guardian.is_jailed_or_probing("acc_a"));

        // 探针失败 -> 升级为 Level 1 (5m)，时间从 0 重新计时
        let failed_claim = expire_jail_and_claim(&guardian, "acc_a");
        let (next_lv, wait) = guardian.mark_probe_failed("acc_a", failed_claim).unwrap();
        assert_eq!(next_lv, 1);
        assert_eq!(wait, Duration::from_secs(5 * 60));

        // 下一次探针成功 (<8s) -> 转移到 HalfOpen 缓刑期
        let passed_claim = expire_jail_and_claim(&guardian, "acc_a");
        assert!(guardian.mark_probe_passed("acc_a", passed_claim));
        assert!(guardian.is_half_open("acc_a"));

        // 缓刑期间真实业务请求连续 2 次通过 (<8s) -> 正式满血出狱
        let act1 =
            guardian
                .record_and_evaluate("acc_a", None, "PRO", 5000.0, false, None, &all, |_| false);
        assert_eq!(act1, QoSAction::None);
        assert!(guardian.is_half_open("acc_a"));

        let act2 =
            guardian
                .record_and_evaluate("acc_a", None, "PRO", 4000.0, false, None, &all, |_| false);
        assert_eq!(act2, QoSAction::HalfOpenPassed);
        assert_eq!(guardian.get_state("acc_a"), AccountQoSState::Healthy);
    }

    #[test]
    fn test_penalty_memory_retention() {
        let guardian = QoSGuardian::new();
        guardian.set_auto_jail_enabled(true);
        let acc_a = create_test_token("acc_a", "PRO");
        let acc_b = create_test_token("acc_b", "PRO");
        let all = vec![acc_a.clone(), acc_b.clone()];
        guardian.push_sample("acc_b", 4000.0, false, false);

        // 模拟前次已被关押到 Level 2 (10分钟) 并刚康复
        guardian
            .penalty_memory
            .insert("acc_a".to_string(), (2, Instant::now()));

        // 短期内再次复发违规 2 次
        guardian.record_and_evaluate("acc_a", None, "PRO", 22000.0, false, None, &all, |_| false);
        let action =
            guardian
                .record_and_evaluate("acc_a", None, "PRO", 22000.0, false, None, &all, |_| false);

        // 验证：直接继承并升级至 Level 3 (15分钟)，而不是又从 Level 0 重新来
        assert_eq!(
            action,
            QoSAction::TriggerJailedAndUnbind {
                level: 3,
                wait_duration: Duration::from_secs(15 * 60),
            }
        );
    }

    #[test]
    fn test_cold_start_uncached_tokens_bonus_and_isolation() {
        let guardian = QoSGuardian::new();
        guardian.set_auto_jail_enabled(true);
        let acc_a = create_test_token("acc_a", "PRO");
        let acc_b = create_test_token("acc_b", "PRO");
        let all = vec![acc_a.clone(), acc_b.clone()];

        // 注入同行基线 4.0s
        guardian.push_sample("acc_b", 4000.0, false, false);

        // 模拟场景：346K 零缓存冷启动 (大 Token Prefill)
        // 按照用户规则：每 100K 未缓存 Token 补偿 +7.0s
        // 346K tokens 补偿 = 3.46 * 7.0s = 24.22s
        // 深度思考基础阈值 35.0s -> 提升至 35.0 + 24.22 = 59.22s
        // 若实际首字为 55.0s (55000ms)，未超过 59.22s，应当豁免，不触发 Strike
        let act = guardian.record_and_evaluate(
            "acc_a",
            Some("session_large_1"),
            "PRO",
            55000.0,
            true, // is_deep_thinking
            Some(346_000),
            &all,
            |_| false,
        );
        assert_eq!(act, QoSAction::None);
        assert_eq!(guardian.get_state("acc_a"), AccountQoSState::Healthy);

        // 验证冷启动样本隔离：acc_a 的该 55s 冷启动样本不应该污染同行基准计算
        // 只有 acc_b 的 4.0s 参与计算，基线仍然是 4.0s (4000ms)
        let peer_avg = guardian.calculate_pure_peer_avg("acc_b", "PRO", &all, |_| false);
        assert_eq!(peer_avg, None); // acc_a 只有一个冷启动样本，被过滤后无有效热样本

        // 紧接着第二轮连续交互 (热缓存)：总 Token 350K，上一轮 346K
        // 未缓存增量仅为 4K tokens，补偿 = 0.04 * 7s = 0.28s
        // 阈值恢复为 35.28s。如果此时再次耗时 40s (超标)，则判定为真卡顿，打上 Suspicious
        let act2 = guardian.record_and_evaluate(
            "acc_a",
            Some("session_large_1"),
            "PRO",
            40000.0,
            true,
            Some(350_000),
            &all,
            |_| false,
        );
        assert_eq!(act2, QoSAction::TaggedSuspicious);
        assert!(guardian.is_suspicious("acc_a"));
    }

    #[test]
    fn test_multiturn_thinking_model_prevents_false_strikes() {
        let guardian = QoSGuardian::new();
        guardian.set_auto_jail_enabled(true);
        let acc_a = create_test_token("acc_a", "PRO");
        let acc_b = create_test_token("acc_b", "PRO");
        let all = vec![acc_a.clone(), acc_b.clone()];

        // 注入同行基准 4.0s
        guardian.push_sample("acc_b", 4000.0, false, false);

        // Turn 1: 50K tokens
        let _ = guardian.record_and_evaluate(
            "acc_a",
            Some("session_multi_turn"),
            "PRO",
            12000.0,
            true,
            Some(50_000),
            &all,
            |_| false,
        );

        // Turn 2: 51K tokens (热缓存增量仅 1K)，思考耗时 28s。
        // 若 is_deep_thinking = true (思考模型)，基础阈值为 35.0s，因此 28s 不触发 Strike；
        // 若错误判定为 false，20.15s 阈值将导致误判 Suspicious。
        let act2 = guardian.record_and_evaluate(
            "acc_a",
            Some("session_multi_turn"),
            "PRO",
            28000.0,
            true, // is_deep_thinking
            Some(51_000),
            &all,
            |_| false,
        );
        assert_eq!(act2, QoSAction::None);
        assert_eq!(guardian.get_state("acc_a"), AccountQoSState::Healthy);
    }

    #[test]
    fn test_take_expired_jails_for_probing_does_not_deadlock() {
        let guardian = Arc::new(QoSGuardian::new());
        guardian.set_auto_jail_enabled(true);

        // 设置到期账号 (50ms 前入狱，刑期 10ms)
        let expired_at = Instant::now() - Duration::from_millis(50);
        guardian.states.insert(
            "acc_expired".to_string(),
            AccountQoSState::Jailed {
                level: 0,
                jailed_at: expired_at,
                current_wait: Duration::from_millis(10),
            },
        );

        // 设置未到期账号 (刑期 1 小时)
        guardian.states.insert(
            "acc_waiting".to_string(),
            AccountQoSState::Jailed {
                level: 1,
                jailed_at: Instant::now(),
                current_wait: Duration::from_secs(3600),
            },
        );

        // 设置普通健康账号
        guardian
            .states
            .insert("acc_healthy".to_string(), AccountQoSState::Healthy);

        // 在单独线程中运行，配合 5 秒超时保护检测死锁
        let g = guardian.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let expired = g.take_expired_jails_for_probing(|id, _| Some(id.to_string()));
            let _ = tx.send(expired);
        });

        let result = rx
            .recv_timeout(Duration::from_secs(5))
            .expect("take_expired_jails_for_probing deadlocked or timed out!");

        // 断言：仅已到期账号被取出
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, "acc_expired");
        assert_eq!(result[0].1.level, 0);

        // 断言：到期账号状态已原子转换为 Probing
        match guardian.get_state("acc_expired") {
            AccountQoSState::Probing { level, .. } => assert_eq!(level, 0),
            other => panic!("Expected Probing state, got {:?}", other),
        }

        // 断言：未到期账号依然保持 Jailed
        match guardian.get_state("acc_waiting") {
            AccountQoSState::Jailed { level, .. } => assert_eq!(level, 1),
            other => panic!("Expected Jailed state, got {:?}", other),
        }

        // 断言：再次调用不会重复取出（已在 Probing 中）
        let second_run = guardian.take_expired_jails_for_probing(|id, _| Some(id.to_string()));
        assert!(second_run.is_empty());
    }

    #[test]
    fn test_auto_jail_disabled_ignores_duration_and_jail() {
        let guardian = QoSGuardian::new();
        assert!(!guardian.is_auto_jail_enabled());

        let acc_a = create_test_token("acc_a", "PRO");
        let acc_b = create_test_token("acc_b", "PRO");
        let all = vec![acc_a.clone(), acc_b.clone()];
        guardian.push_sample("acc_b", 4000.0, false, false);

        // 默认关闭状态：即使耗时高达 120s (120000ms)，直接忽略校验时长，不产生 Strike 判定
        let act =
            guardian
                .record_and_evaluate("acc_a", None, "PRO", 120000.0, false, None, &all, |_| false);
        assert_eq!(act, QoSAction::None);
        assert_eq!(guardian.get_state("acc_a"), AccountQoSState::Healthy);
        assert!(!guardian.is_suspicious("acc_a"));
        assert!(!guardian.is_jailed_or_probing("acc_a"));
        assert!(guardian.is_healthy("acc_a"));
        assert_eq!(guardian.get_jailed_remaining_seconds("acc_a"), None);

        // 动态开启开关：再次超标应该被标记为 Suspicious
        guardian.set_auto_jail_enabled(true);
        assert!(guardian.is_auto_jail_enabled());
        let act_strike =
            guardian
                .record_and_evaluate("acc_a", None, "PRO", 30000.0, false, None, &all, |_| false);
        assert_eq!(act_strike, QoSAction::TaggedSuspicious);
        assert!(guardian.is_suspicious("acc_a"));

        // 动态关闭开关：所有限制应立即解封，状态清空恢复为 Healthy
        guardian.set_auto_jail_enabled(false);
        assert!(!guardian.is_auto_jail_enabled());
        assert_eq!(guardian.get_state("acc_a"), AccountQoSState::Healthy);
        assert!(!guardian.is_suspicious("acc_a"));
        assert!(guardian.is_healthy("acc_a"));
    }
}
