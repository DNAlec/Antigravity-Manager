use serde_json;
use std::{fs, path::Path, sync::Mutex};

use super::account::get_data_dir;
use crate::models::AppConfig;

const CONFIG_FILE: &str = "gui_config.json";

// Disk transactions are short and synchronous. Runtime publishers additionally hold
// CONFIG_OPERATION_LOCK across persistence and their asynchronous hot updates.
static CONFIG_FILE_LOCK: Mutex<()> = Mutex::new(());
static CONFIG_OPERATION_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

pub async fn lock_config_operation() -> Result<tokio::sync::MutexGuard<'static, ()>, String> {
    tokio::time::timeout(
        std::time::Duration::from_secs(30),
        CONFIG_OPERATION_LOCK.lock(),
    )
    .await
    .map_err(|_| "Timed out waiting for configuration update".to_string())
}

pub fn is_config_conflict(error: &str) -> bool {
    error.starts_with("CONFIG_CONFLICT:")
}

/// Load configuration under the same lock as read-modify-write transactions.
pub fn load_app_config() -> Result<AppConfig, String> {
    let _guard = CONFIG_FILE_LOCK
        .lock()
        .map_err(|_| "Configuration lock poisoned")?;
    load_at(&get_data_dir()?.join(CONFIG_FILE))
}

fn load_at(path: &Path) -> Result<AppConfig, String> {
    if !path.exists() {
        return write_at(path, AppConfig::new());
    }
    let content =
        fs::read_to_string(path).map_err(|e| format!("failed_to_read_config_file: {}", e))?;
    let (config, modified) = parse_and_migrate_config(&content)?;
    // Upgrade legacy files before exposing them to a UI. Uninitialized forms have
    // revision zero and must never be accepted as a replacement for this config.
    if modified || config.config_revision == 0 {
        return write_at(path, config);
    }
    Ok(config)
}

fn write_at(path: &Path, config: AppConfig) -> Result<AppConfig, String> {
    // Return exactly the normalized representation that subsequent reads expose.
    let serialized = serde_json::to_string(&config).map_err(|e| e.to_string())?;
    let (mut config, _) = parse_and_migrate_config(&serialized)?;
    config.config_revision = config
        .config_revision
        .checked_add(1)
        .ok_or("Configuration revision exhausted")?;
    let content = serde_json::to_vec_pretty(&config)
        .map_err(|e| format!("failed_to_serialize_config: {}", e))?;
    crate::utils::fs::write_atomic(path, &content)
        .map_err(|e| format!("failed_to_save_config: {}", e))?;
    Ok(config)
}

fn replace_at(path: &Path, config: &AppConfig) -> Result<AppConfig, String> {
    let current = if path.exists() {
        Some(load_at(path)?)
    } else {
        None
    };
    let revision = current.as_ref().map_or(0, |c| c.config_revision);
    if revision != config.config_revision {
        return Err("CONFIG_CONFLICT: Configuration changed; reload before saving".into());
    }
    write_at(path, config.clone())
}

/// Compare-and-swap for complete snapshots supplied by clients.
pub fn save_app_config_versioned(config: &AppConfig) -> Result<AppConfig, String> {
    let _guard = CONFIG_FILE_LOCK
        .lock()
        .map_err(|_| "Configuration lock poisoned")?;
    replace_at(&get_data_dir()?.join(CONFIG_FILE), config)
}

/// Apply a narrow server-side mutation without releasing the disk transaction lock.
pub fn update_app_config(
    update: impl FnOnce(&mut AppConfig) -> Result<(), String>,
) -> Result<AppConfig, String> {
    update_at(&get_data_dir()?.join(CONFIG_FILE), update)
}

fn update_at(
    path: &Path,
    update: impl FnOnce(&mut AppConfig) -> Result<(), String>,
) -> Result<AppConfig, String> {
    let _guard = CONFIG_FILE_LOCK
        .lock()
        .map_err(|_| "Configuration lock poisoned")?;
    let mut config = load_at(path)?;
    update(&mut config)?;
    write_at(path, config)
}

/// Parse, migrate and normalize raw configuration JSON string.
/// Returns (AppConfig, modified_flag).
pub fn parse_and_migrate_config(content: &str) -> Result<(AppConfig, bool), String> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        tracing::warn!("Configuration content is empty, generating default configuration");
        return Ok((AppConfig::new(), true));
    }

    let mut v: serde_json::Value =
        serde_json::from_str(trimmed).map_err(|e| format!("failed_to_parse_config_file: {}", e))?;

    let mut modified = false;

    // Migration logic
    if let Some(proxy) = v.get_mut("proxy") {
        // [FIX #1738] Enhanced type checking for custom_mapping
        // Ensures the field is always parsed as an object, preventing type mismatch errors
        let mut custom_mapping = match proxy.get("custom_mapping") {
            Some(m) if m.is_object() => m.as_object().unwrap().clone(),
            Some(m) => {
                // If custom_mapping is not an object type (e.g., string), log warning and reset to empty
                tracing::warn!(
                    "Invalid custom_mapping type (expected object, got {:?}), resetting to empty",
                    m
                );
                serde_json::Map::new()
            }
            None => serde_json::Map::new(),
        };

        // Migrate Anthropic mapping
        if let Some(anthropic) = proxy
            .get_mut("anthropic_mapping")
            .and_then(|m| m.as_object_mut())
        {
            for (k, v) in anthropic.iter() {
                // Only move non-series fields, as series fields are now handled by Preset logic or builtin tables
                if !k.ends_with("-series") {
                    if !custom_mapping.contains_key(k) {
                        custom_mapping.insert(k.clone(), v.clone());
                    }
                }
            }
            // Remove old field
            proxy.as_object_mut().unwrap().remove("anthropic_mapping");
            modified = true;
        }

        // Migrate OpenAI mapping
        if let Some(openai) = proxy
            .get_mut("openai_mapping")
            .and_then(|m| m.as_object_mut())
        {
            for (k, v) in openai.iter() {
                if !k.ends_with("-series") {
                    if !custom_mapping.contains_key(k) {
                        custom_mapping.insert(k.clone(), v.clone());
                    }
                }
            }
            // Remove old field
            proxy.as_object_mut().unwrap().remove("openai_mapping");
            modified = true;
        }

        // 旧版启动时注入的精确映射会挡住 3.x Flash 裸模型的档位路由。
        // 只删除仍等于当时默认值的条目；用户改过的目标保留。
        for (k, old_default) in [
            ("gemini-3.6-flash", "gemini-3.6-flash-tiered"),
            ("gemini-3.7-flash", "gemini-3.7-flash-tiered"),
            ("gemini-3.8-flash", "gemini-3.8-flash-tiered"),
        ] {
            if custom_mapping.get(k).and_then(|v| v.as_str()) == Some(old_default) {
                custom_mapping.remove(k);
                modified = true;
            }
        }

        // 清理已废弃的旧内置规则 gemini-3.x-flash（已由纯数据驱动 DynamicTierRouter 取代）
        if custom_mapping.contains_key("gemini-3.x-flash") {
            custom_mapping.remove("gemini-3.x-flash");
            modified = true;
        }

        // 检查 builtin_mapping_version 并进行内置自适应暗路由规则版本迁移 (防复活保障)
        let builtin_version = proxy
            .get("builtin_mapping_version")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let mut custom_mapping_expose = match proxy.get("custom_mapping_expose") {
            Some(m) if m.is_object() => m.as_object().unwrap().clone(),
            _ => serde_json::Map::new(),
        };

        if builtin_version < 1 {
            // 首次升级至版本 1：注入 3 条自适应暗路由规则（默认 expose: false）
            let builtins = [
                ("claude-fable-*", "auto:claude-opus", false),
                ("claude-opus-4-7*", "auto:claude-opus-4-7", false),
                ("claude-haiku-*", "auto:gemini-flash-low", false),
            ];
            for (pattern, target, expose) in builtins {
                if !custom_mapping.contains_key(pattern) {
                    custom_mapping.insert(
                        pattern.to_string(),
                        serde_json::Value::String(target.to_string()),
                    );
                }
                if !custom_mapping_expose.contains_key(pattern) {
                    custom_mapping_expose
                        .insert(pattern.to_string(), serde_json::Value::Bool(expose));
                }
            }
            // 存量自定义映射如果未显式声明 expose，通配与自适应规则默认 false，精确自定义规则默认 true
            for (k, v) in custom_mapping.iter() {
                if !custom_mapping_expose.contains_key(k) {
                    let is_auto = v.as_str().map_or(false, |s| s.starts_with("auto:"));
                    let default_exp = !k.contains('*') && !is_auto;
                    custom_mapping_expose.insert(k.clone(), serde_json::Value::Bool(default_exp));
                }
            }
            proxy.as_object_mut().unwrap().insert(
                "builtin_mapping_version".to_string(),
                serde_json::Value::Number(1.into()),
            );
            proxy.as_object_mut().unwrap().insert(
                "custom_mapping_expose".to_string(),
                serde_json::Value::Object(custom_mapping_expose.clone()),
            );
            modified = true;
        }

        // 旧出厂默认 flash_high = 16384 或 32768。只在首次启动且模式为 default 时改成官方 -1。
        if let Some(tb) = proxy
            .get_mut("thinking_budget")
            .and_then(|t| t.as_object_mut())
        {
            let migrated_32k = tb
                .get("thinking_budget_32k_legacy_migrated")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if !migrated_32k {
                let flash_mode_is_default = tb
                    .get("flash_mode")
                    .and_then(|v| v.as_str())
                    .map(|s| s.eq_ignore_ascii_case("default"))
                    .unwrap_or(false);
                let pro_mode_is_default = tb
                    .get("pro_mode")
                    .and_then(|v| v.as_str())
                    .map(|s| s.eq_ignore_ascii_case("default"))
                    .unwrap_or(false);

                if flash_mode_is_default {
                    if let Some(val) = tb.get("flash_high").and_then(|v| v.as_i64()) {
                        if val == 16384 || val == 32768 {
                            tb.insert("flash_high".to_string(), serde_json::Value::from(-1));
                        }
                    }
                }

                if pro_mode_is_default {
                    if let Some(val) = tb.get("pro_high").and_then(|v| v.as_i64()) {
                        if val == 32768 {
                            tb.insert("pro_high".to_string(), serde_json::Value::from(-1));
                        }
                    }
                }

                tb.insert(
                    "thinking_budget_32k_legacy_migrated".to_string(),
                    serde_json::Value::from(true),
                );
                modified = true;
            }

            let migrated = tb
                .get("flash_high_legacy_migrated")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if !migrated {
                if tb.get("flash_high").and_then(|v| v.as_i64()) == Some(16384) {
                    tb.insert("flash_high".to_string(), serde_json::Value::from(-1));
                }
                tb.insert(
                    "flash_high_legacy_migrated".to_string(),
                    serde_json::Value::from(true),
                );
                modified = true;
            }
        }
        if let Some(log_retention) = proxy
            .get_mut("log_retention")
            .and_then(|m| m.as_object_mut())
        {
            if let Some(max_disk_mb) = log_retention.get("max_disk_mb").and_then(|v| v.as_u64()) {
                if max_disk_mb == 0 {
                    log_retention.insert("max_disk_mb".to_string(), serde_json::Value::from(1024));
                    modified = true;
                }
            }
        }

        // Migrate legacy User-Agent in user_agent_override and saved_user_agent to >= 4.3.0
        // to prevent upstream Google 404/429 model rejections
        for ua_field in ["user_agent_override", "saved_user_agent"] {
            if let Some(ua_val) = proxy.get(ua_field).and_then(|v| v.as_str()) {
                let sanitized = crate::constants::sanitize_egress_user_agent(ua_val);
                if sanitized != ua_val {
                    tracing::info!(
                        field = %ua_field,
                        old = %ua_val,
                        new = %sanitized,
                        "Migrating legacy User-Agent config to supported stable floor"
                    );
                    proxy
                        .as_object_mut()
                        .unwrap()
                        .insert(ua_field.to_string(), serde_json::Value::String(sanitized));
                    modified = true;
                }
            }
        }

        // [2026-10-29] 思考控制权已收敛为「网关 + 自适应」：
        // 旧配置中的 client（客户端直控）统一迁移回 gateway，避免 UI 显示已不存在的模式。
        if let Some(tb) = proxy
            .get_mut("thinking_budget")
            .and_then(|t| t.as_object_mut())
        {
            if tb.get("control_source").and_then(|v| v.as_str()) == Some("client") {
                tb.insert(
                    "control_source".to_string(),
                    serde_json::Value::String("gateway".to_string()),
                );
                tracing::info!(
                    "Migrating thinking_budget.control_source: client -> gateway (client control mode removed)"
                );
                modified = true;
            }
        }

        // 自动迁移/初始化 prompt_rewrite 配置：若旧配置中缺失该字段，则注入默认开启并搭载官方推荐预设
        if let Some(proxy_obj) = proxy.as_object_mut() {
            if !proxy_obj.contains_key("prompt_rewrite") {
                let default_prompt_rewrite =
                    serde_json::to_value(crate::proxy::config::PromptRewriteConfig::default())
                        .unwrap_or(serde_json::Value::Null);
                if !default_prompt_rewrite.is_null() {
                    proxy_obj.insert("prompt_rewrite".to_string(), default_prompt_rewrite);
                    tracing::info!("Initializing default prompt_rewrite config (enabled with recommended presets)");
                    modified = true;
                }
            }
        }

        if modified {
            proxy.as_object_mut().unwrap().insert(
                "custom_mapping".to_string(),
                serde_json::Value::Object(custom_mapping),
            );
        }
    }

    let config: AppConfig = serde_json::from_value(v)
        .map_err(|e| format!("failed_to_convert_config_after_migration: {}", e))?;

    Ok((config, modified))
}

/// Compatibility entry point for internal full snapshots; stale writes still fail.
pub fn save_app_config(config: &AppConfig) -> Result<(), String> {
    save_app_config_versioned(config).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_snapshot_cannot_overwrite_credentials_or_bindings() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        let initial = load_at(&path).unwrap();
        let mut changed = initial.clone();
        changed.proxy.api_key = "synthetic-new-key".into();
        changed
            .proxy
            .proxy_pool
            .account_bindings
            .insert("account".into(), "proxy".into());
        let saved = replace_at(&path, &changed).unwrap();
        let mut stale = initial;
        stale.theme = "dark".into();
        assert!(is_config_conflict(&replace_at(&path, &stale).unwrap_err()));
        assert_eq!(load_at(&path).unwrap().proxy.api_key, saved.proxy.api_key);
        assert_eq!(
            load_at(&path)
                .unwrap()
                .proxy
                .proxy_pool
                .account_bindings
                .len(),
            1
        );
    }

    #[test]
    fn concurrent_narrow_updates_preserve_other_fields() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        let initial = load_at(&path).unwrap();
        std::thread::scope(|scope| {
            for index in 0..8 {
                let path = &path;
                scope.spawn(move || {
                    update_at(path, |config| {
                        config
                            .proxy
                            .custom_mapping
                            .insert(format!("test-{index}"), "model".into());
                        Ok(())
                    })
                    .unwrap();
                });
            }
        });
        let saved = load_at(&path).unwrap();
        assert_eq!(saved.config_revision, initial.config_revision + 8);
        for index in 0..8 {
            assert!(saved
                .proxy
                .custom_mapping
                .contains_key(&format!("test-{index}")));
        }
        assert_eq!(saved.proxy.api_key, initial.proxy.api_key);
    }

    #[test]
    fn failed_update_keeps_disk_and_revision_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        load_at(&path).unwrap();
        let before = fs::read(&path).unwrap();
        assert!(update_at(&path, |config| {
            config.proxy.api_key.clear();
            Err("validation failed".into())
        })
        .is_err());
        assert!(
            fs::read(&path).unwrap() == before,
            "failed transaction modified the file"
        );
    }

    #[test]
    fn legacy_configuration_gets_a_revision_before_exposure() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        let legacy = AppConfig::new();
        fs::write(&path, serde_json::to_vec(&legacy).unwrap()).unwrap();
        assert!(load_at(&path).unwrap().config_revision > 0);
        assert!(is_config_conflict(&replace_at(&path, &legacy).unwrap_err()));
    }

    #[test]
    fn test_parse_and_migrate_empty_or_whitespace_config() {
        // [FIX #3548] Ensure empty or whitespace-only config file does not trigger EOF error in headless mode
        let (empty_cfg, modified_empty) =
            parse_and_migrate_config("").expect("empty config should parse to default");
        assert!(modified_empty);
        assert!(!empty_cfg.proxy.api_key.is_empty());

        let (ws_cfg, modified_ws) = parse_and_migrate_config("   \n\t  \r\n  ")
            .expect("whitespace config should parse to default");
        assert!(modified_ws);
        assert!(!ws_cfg.proxy.api_key.is_empty());
    }

    #[test]
    fn test_parse_and_migrate_valid_and_invalid_json() {
        let invalid = parse_and_migrate_config("{ invalid_json: ");
        assert!(invalid.is_err());
        assert!(invalid.unwrap_err().contains("failed_to_parse_config_file"));

        let default_json = serde_json::to_string(&AppConfig::new()).unwrap();
        let (valid_cfg, _) =
            parse_and_migrate_config(&default_json).expect("valid config should parse");
        assert!(!valid_cfg.proxy.api_key.is_empty());
    }
}
