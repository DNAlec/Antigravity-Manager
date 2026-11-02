export interface UpstreamProxyConfig {
    enabled: boolean;
    url: string;
}

export interface ProxyConfig {
    enabled: boolean;
    allow_lan_access?: boolean;
    auth_mode?: 'off' | 'strict' | 'all_except_health' | 'auto';
    port: number;
    api_key: string;
    admin_password?: string;
    auto_start: boolean;
    custom_mapping?: Record<string, string>;
    request_timeout: number;
    enable_logging: boolean;
    capture_health_logs?: boolean;
    log_retention?: LogRetentionConfig;
    internal_error_log_retention?: InternalErrorLogRetentionConfig;
    debug_logging?: DebugLoggingConfig;
    upstream_proxy: UpstreamProxyConfig;
    scheduling?: StickySessionConfig;
    qos?: QoSConfig;
    experimental?: ExperimentalConfig;
    user_agent_override?: string;
    saved_user_agent?: string;
    hub_version?: string;
    hub_cl?: string;
    thinking_budget?: ThinkingBudgetConfig;
    global_system_prompt?: GlobalSystemPromptConfig;
    image_thinking_mode?: 'enabled' | 'disabled'; // [NEW] 图像思维模式开关
    only_raw_quota_models?: boolean; // [NEW] 是否只暴露真实配额模型
    proxy_pool?: ProxyPoolConfig;
    multimodal?: MultimodalConfig;
    stream_timeouts?: StreamTimeoutConfig;
}

export interface StreamTimeoutConfig {
    initial_ttft_secs?: number;
    transition_secs?: number;
    streaming_sliding_secs?: number;
}

/** Instance QoS policy; omitted fields retain backend defaults. */
export interface QoSConfig {
    normal_threshold_secs?: number;
    deep_thinking_threshold_secs?: number;
    recovery_threshold_secs?: number;
    no_peer_grace_secs?: number;
    probe_timeout_secs?: number;
    probe_success_threshold_secs?: number;
    jail_ladder_secs?: [number, number, number, number, number, number];
}

export interface MultimodalConfig {
    enable_sliding_window?: boolean;
    strategy?: 'count' | 'memory';
    max_fresh_images?: number;
    strip_remote_urls?: boolean;
    max_total_image_mb?: number;
}

export interface LogRetentionConfig {
    max_body_age_hours: number;
    max_storage_gb: number;
    max_disk_mb?: number;
    max_rows: number;
    max_age_days?: number;
}

export interface InternalErrorLogRetentionConfig {
    max_storage_mb: number;
}


// ============================================================================
// Thinking Budget 配置 (控制 AI 深度思考时的 Token 预算)
// ============================================================================

/** 思考预算控制权归属 */
export type ThinkingControlSource = 'gateway' | 'client';

/** Thinking Budget 处理模式 */
export type ThinkingBudgetMode = 'default' | 'custom' | 'auto' | 'passthrough' | 'adaptive';

/** Thinking Effort 等级 (仅 adaptive 模式) */
export type ThinkingEffort = 'low' | 'medium' | 'high';

/** Thinking Budget 配置 */
export interface ThinkingBudgetConfig {
    /** 控制权大选择：网关权威控制 (gateway) vs 客户端直接控制 (client) */
    control_source?: ThinkingControlSource;

    // --- Gemini Flash 系列配置 ---
    flash_mode?: ThinkingBudgetMode;
    flash_low?: number;       // 默认 1000
    flash_medium?: number;    // 默认 4000
    flash_high?: number;      // 默认 -1，走官方模型结构体
    flash_tiered?: number;    // 默认 -1

    // --- Gemini Pro 系列配置（官方仅 Low 与 High 两档） ---
    pro_mode?: ThinkingBudgetMode;
    pro_low?: number;         // 默认 1001
    pro_high?: number;        // 默认 10001

    // --- Claude 系列配置 ---
    claude_mode?: ThinkingBudgetMode;
    claude_budget?: number;    // 统一思考预算 (默认 16000, 填 -1 自适应)
    claude_low?: number;       // 默认 1024
    claude_medium?: number;    // 默认 4096
    claude_high?: number;      // 默认 16000

    // --- 旧版兼容字段 ---
    mode?: ThinkingBudgetMode;
    custom_value?: number;
    effort?: ThinkingEffort;
    custom_low?: number;
    custom_medium?: number;
    custom_high?: number;
    custom_tiered?: number;
}

// ============================================================================
// 全局系统提示词配置
// ============================================================================

/** 全局系统提示词配置 */
export interface GlobalSystemPromptConfig {
    /** 是否启用 */
    enabled: boolean;
    /** 提示词内容 */
    content: string;
}

export interface DebugLoggingConfig {
    enabled: boolean;
    output_dir?: string;
}

export type SchedulingMode = 'CacheFirst' | 'Balance' | 'PerformanceFirst';

export interface StickySessionConfig {
    mode: SchedulingMode;
    max_wait_seconds: number;
    max_spillover_accounts?: number;
}

export interface ScheduledWarmupConfig {
    enabled: boolean;
    monitored_models: string[];
}

export interface QuotaProtectionConfig {
    enabled: boolean;
    threshold_percentage: number; // 1-99
    monitored_models: string[];
}

export interface PinnedQuotaModelsConfig {
    models: string[];
}

export interface ExperimentalConfig {
    enable_usage_scaling: boolean;
    payload_storage_mode?: 'simple' | 'full';
    log_retention_days?: number;
    thinking_store_enabled?: boolean;
    thinking_retention_days?: number;
    thinking_max_memory_turns?: number;
    enable_cowork_auto_compact?: boolean;
    cowork_compact_threshold?: number;
    enable_cowork_manual_compact?: boolean;
    enable_global_compaction_strip?: boolean;
}

export interface CircuitBreakerConfig {
    enabled: boolean;
    backoff_steps: number[];
    lock_on_zero_quota?: boolean;
}

export interface ClassicAutoSwitchConfig {
    enabled: boolean;
    threshold_percentage: number;
    monitored_models: string[];
}

export interface AppConfig {
    language: string;
    theme: string;
    auto_refresh: boolean;
    refresh_interval: number;
    auto_sync: boolean;
    sync_interval: number;
    auto_switch_classic_depleted?: boolean; // [NEW] 兼容旧版开关
    classic_auto_switch?: ClassicAutoSwitchConfig; // [NEW] 经典款配额自动热切换配置
    default_export_path?: string;
    antigravity_executable?: string; // [NEW] 手动指定的反重力程序路径
    antigravity_ide_executable?: string; // [NEW] 手动指定的 Antigravity IDE 程序路径
    antigravity_cli_executable?: string; // [NEW] 手动指定的 Antigravity CLI (agy) 路径
    antigravity_args?: string[]; // [NEW] Antigravity 启动参数
    auto_launch?: boolean; // 开机自动启动
    quiet_autostart?: boolean; // 开机自启时留在托盘。缺省视为开启
    auto_check_update?: boolean; // 自动检查更新
    update_check_interval?: number; // 更新检查间隔（小时）
    update_channel?: 'stable' | 'beta'; // 更新通道：正式版 vs 预览版
    accounts_page_size?: number; // 账号列表每页显示数量,默认 0 表示自动计算
    hidden_menu_items?: string[]; // 隐藏的菜单项路径列表
    scheduled_warmup: ScheduledWarmupConfig;
    quota_protection: QuotaProtectionConfig; // [NEW] 配额保护配置
    pinned_quota_models: PinnedQuotaModelsConfig; // [NEW] 配额关注列表
    circuit_breaker: CircuitBreakerConfig; // [NEW] 熔断器配置
    proxy: ProxyConfig;
    cloudflared: CloudflaredConfig; // [NEW] Cloudflared 配置
    lightweight_mode?: boolean; // [NEW] 轻量模式：关闭到托盘时释放 WebView
    suggestion_delete_thinking_store?: boolean; // [NEW] 建议删除历史思考块缓存开关
    thinking_cleanup_dismissed?: boolean; // [NEW] 用户是否已确认/忽略该建议
    dismissed_thinking_cleanup_version?: string; // [NEW] 用户已确认或忽略建议的目标版本号
}

// ============================================================================
// Cloudflared (CF隧道) 类型定义
// ============================================================================

export type TunnelMode = 'quick' | 'auth';

export interface CloudflaredConfig {
    enabled: boolean;
    mode: TunnelMode;
    port: number;
    token?: string;
    use_http2: boolean;
}

export interface CloudflaredStatus {
    installed: boolean;
    version?: string;
    running: boolean;
    url?: string;
    error?: string;
}

// ============================================================================
// 代理池类型定义
// ============================================================================

export interface ProxyAuth {
    username: string;
    password?: string;
}

export interface ProxyEntry {
    id: string;
    name: string;
    url: string;
    auth?: ProxyAuth;
    enabled: boolean;
    priority: number;
    tags: string[];
    max_accounts?: number;
    health_check_url?: string;
    last_check_time?: number;
    is_healthy: boolean;
    latency?: number; // [NEW] 延迟 (毫秒)
}

// export type ProxyPoolMode = 'global' | 'per_account' | 'hybrid'; // [REMOVED]

export type ProxySelectionStrategy = 'round_robin' | 'random' | 'priority' | 'least_connections' | 'weighted_round_robin';

export interface ProxyPoolConfig {
    enabled: boolean;
    // mode: ProxyPoolMode; // [REMOVED]
    proxies: ProxyEntry[];
    health_check_interval: number;
    auto_failover: boolean;
    strategy: ProxySelectionStrategy;
    account_bindings?: Record<string, string>;
}

export interface HubFingerprint {
    version: string;
    cl: string;
    user_agent: string;
    source: 'custom' | 'remote' | 'local' | 'baseline';
}

