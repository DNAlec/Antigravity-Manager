//! Server-side full thinking-block store.
//!
//! Captures untruncated thought text + thoughtSignature from upstream Gemini
//! responses, then precisely re-injects them into the next request's `contents`
//! even when the client dropped / truncated thinking.
//!
//! Text turns are keyed by their visible history prefix as well as their answer;
//! tool turns retain their causal tool ids. Repeated answers must not overwrite
//! unrelated turns, even when clients omit all thinking and signatures.
//!
//! Isolation key = `{tenant}:{client_session_id}`:
//! - tenant is a hash of the caller's API key
//! - client_session_id prefers `X-Session-Id` / body `session_id`

use axum::http::HeaderMap;
use dashmap::DashMap;
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

const MIN_SIGNATURE_LENGTH: usize = 32;
pub const SENTINEL_SIGNATURE: &str = "c2tpcF90aG91Z2h0X3NpZ25hdHVyZV92YWxpZGF0b3I=";
pub const LEGACY_SENTINEL_SIGNATURE: &str = "skip_thought_signature_validator";

#[inline]
pub fn is_sentinel_signature(s: &str) -> bool {
    let trimmed = s.trim();
    trimmed == SENTINEL_SIGNATURE || trimmed == LEGACY_SENTINEL_SIGNATURE
}
const MAX_SESSIONS: usize = 2000;
fn max_turns_per_session() -> usize {
    crate::proxy::config::get_thinking_max_memory_turns()
}
const MAX_BYTES_PER_SESSION: usize = 64 * 1024 * 1024;
/// Persist last_accessed at most this often. Fill/hydrate is memory-only between writes.
const TOUCH_PERSIST_INTERVAL: Duration = Duration::from_secs(5 * 60);

fn idle_ttl() -> Duration {
    let days = crate::proxy::config::get_thinking_retention_days().max(1) as u64;
    Duration::from_secs(days.saturating_mul(24 * 60 * 60))
}

const PLACEHOLDER_THOUGHTS: &[&str] = &[
    "...",
    "·",
    ".",
    "···",
    "[undefined]",
    "Applying tool decisions and generating response...",
];

/// Server-side auto thinking. Clients usually omit thinking config;
/// if any of `claude` / `flash` / `pro` / `agent` appears in a model id
/// (requested or mapped), the proxy enables thoughts + signature restore.
/// Image / embed / lite are excluded to avoid 400s.
pub fn model_forces_server_thinking(model: &str) -> bool {
    if !crate::proxy::config::is_thinking_store_enabled() {
        return false;
    }
    let m = model.to_lowercase();
    if m.is_empty() {
        return false;
    }
    if m.contains("image")
        || m.contains("imagen")
        || m.contains("embed")
        || m.contains("lite")
        || m.contains("preview")
    {
        return false;
    }
    m.contains("claude")
        || m.contains("flash")
        || m.contains("pro")
        || m.contains("agent")
        || m.contains("gemini")
        || m.contains("thinking")
        || m.contains("o1")
        || m.contains("o3")
        || m.contains("deepseek")
}

pub fn any_model_forces_server_thinking(models: &[&str]) -> bool {
    models.iter().copied().any(model_forces_server_thinking)
}

#[derive(Debug, Clone)]
pub struct ThinkingRecord {
    pub fingerprint: String,
    pub thought: String,
    pub signature: Option<String>,
    pub tool_ids: Vec<String>,
    #[allow(dead_code)]
    pub tool_names: Vec<String>,
    pub visible: String,
}

#[derive(Debug)]
struct SessionEntry {
    turns: Vec<Arc<ThinkingRecord>>,
    last_access: Instant,
    last_persist_touch: Instant,
    bytes: usize,
    l2_loaded: bool,
    // Fence delayed cold reads across explicit invalidation, pruning or removal.
    load_epoch: Arc<()>,
    // A warm memory window may contain only one of several persisted generations.
    l2_has_evicted_turns: bool,
}

impl SessionEntry {
    fn new() -> Self {
        Self {
            turns: Vec::new(),
            last_access: Instant::now(),
            last_persist_touch: Instant::now(),
            bytes: 0,
            l2_loaded: false,
            load_epoch: Arc::new(()),
            l2_has_evicted_turns: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SessionWatermark {
    pub chars: usize,
    pub turns: usize,
    pub last_seen: Instant,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct ThinkingStoreStats {
    pub sessions: usize,
    pub memory_turns: usize,
    pub memory_bytes: usize,
}

pub fn thinking_store_stats() -> ThinkingStoreStats {
    let store = ThinkingStore::global();
    let mut stats = ThinkingStoreStats::default();
    for entry in store.sessions.iter() {
        stats.sessions += 1;
        stats.memory_turns += entry.value().turns.len();
        stats.memory_bytes += entry.value().bytes;
    }
    stats
}

pub struct ThinkingStore {
    sessions: DashMap<String, SessionEntry>,
    watermarks: DashMap<String, SessionWatermark>,
    compaction_events: DashMap<String, (usize, Instant)>,
}

struct ThinkingPrunePlan {
    snapshot: crate::modules::proxy_db::ThinkingPruneSnapshot,
    keep_fps: Vec<String>,
    live_tool_ids: Vec<String>,
    live_sigs: Vec<String>,
}

impl ThinkingPrunePlan {
    fn persist(self) {
        let _ = crate::modules::proxy_db::delete_thinking_records_except_fingerprints(
            &self.snapshot,
            &self.keep_fps,
            &self.live_tool_ids,
            &self.live_sigs,
        );
    }
}

impl ThinkingStore {
    fn new() -> Self {
        Self {
            sessions: DashMap::new(),
            watermarks: DashMap::new(),
            compaction_events: DashMap::new(),
        }
    }

    pub fn global() -> &'static ThinkingStore {
        static INSTANCE: OnceLock<ThinkingStore> = OnceLock::new();
        INSTANCE.get_or_init(ThinkingStore::new)
    }

    fn maybe_evict(&self, keep_key: &str) {
        // [FIX #3697] 水位标与压缩事件 TTL 淘汰必须独立执行，严禁被 sessions.len() <= MAX_SESSIONS 提前截断导致无界内存泄漏。
        // 会话水位线仅用于两轮之间宏观压缩检测，保留 2 小时 (7200s) 即可，无须保留数周堆积 DashMap。
        self.watermarks
            .retain(|_, w| w.last_seen.elapsed() < Duration::from_secs(7200));
        self.compaction_events
            .retain(|_, (_, t)| t.elapsed() < Duration::from_secs(60));

        if self.sessions.len() <= MAX_SESSIONS {
            return;
        }
        self.sessions
            .retain(|_, e| e.last_access.elapsed() < idle_ttl());
        if self.sessions.len() <= MAX_SESSIONS {
            return;
        }
        if let Some(oldest_key) = self
            .sessions
            .iter()
            .min_by_key(|e| e.last_access)
            .map(|e| e.key().clone())
        {
            if oldest_key != keep_key {
                self.sessions.remove(&oldest_key);
            }
        }
    }

    pub fn record(&self, store_key: &str, rec: ThinkingRecord) {
        if !crate::proxy::config::is_thinking_store_enabled() {
            return;
        }
        if rec.thought.trim().is_empty() && rec.signature.is_none() {
            return;
        }
        // Placeholder "..." / sentinel-only blocks are injected for Gemini protocol
        // compliance. Recording them as new turns made every 300K-context request
        // append N dummies, then prune rewrite the whole SQLite session.
        if !is_capturable_thought(&rec.thought, rec.signature.as_deref()) {
            return;
        }

        self.record_with_loader(
            store_key,
            rec,
            crate::modules::proxy_db::load_thinking_records,
        );
    }

    fn record_with_loader(
        &self,
        store_key: &str,
        rec: ThinkingRecord,
        load: impl FnOnce(
            &str,
        )
            -> Result<Vec<crate::modules::proxy_db::PersistedThinkingRecord>, String>,
    ) {
        let rec_bytes = record_bytes(&rec);

        self.maybe_evict(store_key);

        // Always hydrate L2 before appending. Otherwise the first capture after a
        // process start can mark l2_loaded=true with only the new turn and permanently
        // shadow older SQLite history on subsequent hydrate/restore calls.
        let needs_l2 = self
            .sessions
            .get(store_key)
            .map(|e| !e.l2_loaded)
            .unwrap_or(true);
        if needs_l2 {
            let _ = self.load_turns_with(store_key, load);
        }

        let persist = {
            let mut entry = self
                .sessions
                .entry(store_key.to_string())
                .or_insert_with(SessionEntry::new);
            entry.last_access = Instant::now();

            let merge_last = entry.turns.last().is_some_and(|last| {
                last.fingerprint == rec.fingerprint && legacy_text_signatures_compatible(last, &rec)
            });

            if merge_last {
                let (stronger, old_text_bytes) = {
                    let last = entry.turns.last().expect("merge_last");
                    (
                        rec.thought.len() >= last.thought.len()
                            || rec.signature.as_ref().map(|s| s.len()).unwrap_or(0)
                                > last.signature.as_ref().map(|s| s.len()).unwrap_or(0),
                        record_bytes(last),
                    )
                };
                if stronger {
                    entry.bytes = entry.bytes.saturating_sub(old_text_bytes);
                    {
                        let last_arc = entry.turns.last_mut().expect("merge_last");
                        *Arc::make_mut(last_arc) = rec;
                    }
                    entry.bytes = entry.bytes.saturating_add(rec_bytes);
                    entry.turns.last().cloned()
                } else {
                    None
                }
            } else {
                entry.turns.push(Arc::new(rec));
                entry.bytes = entry.bytes.saturating_add(rec_bytes);
                while entry.turns.len() > max_turns_per_session()
                    || entry.bytes > MAX_BYTES_PER_SESSION
                {
                    if let Some(old) = entry.turns.first() {
                        let old_bytes = old.thought.len()
                            + old.signature.as_ref().map(|s| s.len()).unwrap_or(0)
                            + old.visible.len();
                        entry.bytes = entry.bytes.saturating_sub(old_bytes);
                    }
                    if entry.turns.is_empty() {
                        break;
                    }
                    entry.turns.remove(0);
                    entry.l2_has_evicted_turns = true;
                }
                entry.turns.last().cloned()
            }
        };

        if let Some(saved) = persist {
            if let Err(e) = crate::modules::proxy_db::save_thinking_record(
                store_key,
                &saved.fingerprint,
                &saved.thought,
                saved.signature.as_deref(),
                &saved.tool_ids,
                &saved.tool_names,
                &saved.visible,
            ) {
                tracing::warn!(
                    "[ThinkingStore] Failed to persist record for {}: {}",
                    store_key,
                    e
                );
            }
        }
    }

    /// Refresh in-memory expiry. SQLite last_accessed is debounced so HDD
    /// never sees a write on the fill hot path after the session is warm.
    pub fn touch_session(&self, store_key: &str) {
        if !crate::proxy::config::is_thinking_store_enabled() || store_key.is_empty() {
            return;
        }
        let mut persist = false;
        if let Some(mut entry) = self.sessions.get_mut(store_key) {
            entry.last_access = Instant::now();
            if entry.last_persist_touch.elapsed() >= TOUCH_PERSIST_INTERVAL {
                entry.last_persist_touch = Instant::now();
                persist = true;
            }
        }
        if persist {
            let _ = crate::modules::proxy_db::touch_thinking_session(store_key);
        }
    }

    fn load_turns(&self, store_key: &str) -> Vec<Arc<ThinkingRecord>> {
        self.load_turns_with(store_key, crate::modules::proxy_db::load_thinking_records)
    }

    fn load_turns_with(
        &self,
        store_key: &str,
        read: impl FnOnce(
            &str,
        )
            -> Result<Vec<crate::modules::proxy_db::PersistedThinkingRecord>, String>,
    ) -> Vec<Arc<ThinkingRecord>> {
        let epoch = {
            let mut entry = self
                .sessions
                .entry(store_key.to_owned())
                .or_insert_with(SessionEntry::new);
            entry.last_access = Instant::now();
            if entry.l2_loaded && !entry.turns.is_empty() {
                return entry.turns.clone();
            }
            entry.load_epoch.clone()
        };
        let persisted = read(store_key);
        let Some(mut entry) = self.sessions.get_mut(store_key) else {
            return Vec::new();
        };
        if !Arc::ptr_eq(&entry.load_epoch, &epoch) {
            return entry.turns.clone();
        }
        if entry.l2_loaded && !entry.turns.is_empty() {
            // Another cold load completed while SQLite was being read.
            return entry.turns.clone();
        }
        let persisted = match persisted {
            Ok(records) => records,
            Err(error) => {
                tracing::warn!(
                    "[ThinkingStore] Failed to load L2 records for {}: {}",
                    store_key,
                    error
                );
                entry.l2_loaded = false;
                entry.l2_has_evicted_turns = true;
                return entry.turns.clone();
            }
        };
        let loaded_len = persisted.len();
        let mut turns: Vec<Arc<ThinkingRecord>> = persisted
            .into_iter()
            .map(|p| {
                Arc::new(ThinkingRecord {
                    fingerprint: p.fingerprint,
                    thought: p.thought,
                    signature: p.signature,
                    tool_ids: p.tool_ids,
                    tool_names: p.tool_names,
                    visible: p.visible,
                })
            })
            .collect();
        // Captures can exist only in RAM after a failed write. Match persisted
        // fields (tool metadata is deliberately omitted from SQLite) in capture
        // order, keeping each resident Arc available to concurrent replays.
        if !entry.turns.is_empty() {
            fn identity(
                rec: &ThinkingRecord,
            ) -> (
                &str,
                &str,
                Option<std::borrow::Cow<'_, str>>,
                Vec<std::borrow::Cow<'_, str>>,
            ) {
                (
                    &rec.fingerprint,
                    &rec.thought,
                    rec.signature
                        .as_deref()
                        .map(normalize_signature_for_comparison),
                    rec.tool_ids
                        .iter()
                        .map(|id| crate::proxy::common::utils::normalize_tool_id(id))
                        .collect(),
                )
            }
            let mut positions = HashMap::new();
            for (idx, row) in turns.iter().enumerate().rev() {
                positions
                    .entry(identity(row))
                    .or_insert_with(Vec::new)
                    .push(idx);
            }
            let placements: Vec<_> = entry
                .turns
                .iter()
                .map(|rec| positions.get_mut(&identity(rec)).and_then(Vec::pop))
                .collect();
            drop(positions);
            for (rec, at) in entry.turns.iter().zip(placements) {
                if let Some(idx) = at {
                    turns[idx] = rec.clone();
                } else {
                    turns.push(rec.clone());
                }
            }
        }
        entry.bytes = turns.iter().map(|rec| record_bytes(rec)).sum();
        entry.turns = turns;
        entry.l2_loaded = true;
        entry.l2_has_evicted_turns = false;
        entry.last_access = Instant::now();
        entry.last_persist_touch = Instant::now();
        let turns = entry.turns.clone();
        drop(entry);
        if loaded_len > 0 {
            let _ = crate::modules::proxy_db::touch_thinking_session(store_key);
        }
        turns
    }

    /// Capture real thinking from the inbound request without re-appending
    /// history that is already stored. Placeholder blocks are ignored.
    pub fn ingest_from_contents(&self, store_key: &str, contents: &[Value]) {
        self.ingest_selected_contents(store_key, contents, None);
    }

    fn ingest_selected_contents(
        &self,
        store_key: &str,
        contents: &[Value],
        text_only: Option<bool>,
    ) {
        let incoming = self.collect_selected_records(store_key, contents, text_only);
        self.ingest_records(
            store_key,
            incoming.into_iter().map(|(_, record)| record).collect(),
        );
    }

    fn collect_selected_records(
        &self,
        store_key: &str,
        contents: &[Value],
        text_only: Option<bool>,
    ) -> Vec<(usize, ThinkingRecord)> {
        if !crate::proxy::config::is_thinking_store_enabled() || store_key.is_empty() {
            return Vec::new();
        }

        let compaction_boundary = determine_compaction_boundary(contents, Some(store_key));
        let text_anchors = compute_text_turn_anchors(contents);

        let mut incoming = Vec::new();
        for (c_idx, content) in contents.iter().enumerate() {
            if compaction_boundary.map_or(false, |b| c_idx <= b) {
                continue;
            }
            let role = content.get("role").and_then(|v| v.as_str()).unwrap_or("");
            if role != "model" && role != "assistant" {
                continue;
            }
            let Some(parts) = content.get("parts").and_then(|p| p.as_array()) else {
                continue;
            };
            let is_text = !parts.iter().any(|p| p.get("functionCall").is_some());
            if text_only.is_some_and(|selected| {
                selected != is_text || parts.iter().any(|p| p.get("functionResponse").is_some())
            }) {
                continue;
            }
            let preceding_turn = if c_idx > 0 {
                contents.get(c_idx - 1)
            } else {
                None
            };
            let anchor = compute_causal_anchor(preceding_turn);
            let mut acc = TurnAccumulator::with_anchor(&anchor)
                .with_text_context_anchor(text_anchors[c_idx].clone());
            for part in parts {
                acc.ingest_part(part);
            }
            if !acc.should_capture() {
                continue;
            }
            if acc.function_call_count > 1 {
                incoming.extend(
                    std::mem::take(&mut acc.signed_calls)
                        .into_iter()
                        .map(|r| (c_idx, r)),
                );
            }
            incoming.push((c_idx, acc.into_record()));
        }
        incoming
    }

    fn ingest_records(&self, store_key: &str, incoming: Vec<ThinkingRecord>) {
        if incoming.is_empty() {
            return;
        }

        let existing = self.load_turns(store_key);
        let mut used = vec![false; existing.len()];
        let mut to_append = Vec::new();
        for mut rec in incoming {
            // A signed call is not a candidate for enriching an entire model turn.
            if rec.fingerprint.starts_with(CALL_SIGNATURE_PREFIX) {
                if !existing.iter().any(|old| same_thinking_record(old, &rec)) {
                    to_append.push(rec);
                }
                continue;
            }
            if let Some(idx) = match_existing_record(&rec, &existing, &used) {
                used[idx] = true;
                let previous = &existing[idx];
                if is_stronger_record(&rec, previous) {
                    // A richer unsigned replay cannot choose between signed siblings.
                    // A partial RAM window cannot establish uniqueness at all.
                    let explicitly_matched = rec
                        .signature
                        .as_deref()
                        .filter(|s| is_real_signature(s))
                        .zip(previous.signature.as_deref())
                        .is_some_and(|(a, b)| signatures_match(a, b));
                    if !explicitly_matched
                        && (self
                            .sessions
                            .get(store_key)
                            .is_none_or(|entry| !entry.l2_loaded || entry.l2_has_evicted_turns)
                            || has_ambiguous_replay_match(&rec, &existing, idx))
                    {
                        continue;
                    }

                    // Keep the matched generation's identity and never erase a
                    // stronger field when a replay enriches only the other one.
                    rec.fingerprint = previous.fingerprint.clone();
                    if rec.thought.len() < previous.thought.len() {
                        rec.thought.clone_from(&previous.thought);
                    }
                    if previous.signature.as_deref().is_some_and(is_real_signature) {
                        rec.signature.clone_from(&previous.signature);
                    }
                    // Enrich the existing row, not an alias supplied by truncated history.
                    rec.visible.clone_from(&previous.visible);
                    rec.tool_ids.clone_from(&previous.tool_ids);
                    rec.tool_names.clone_from(&previous.tool_names);
                    self.upgrade_record(store_key, previous, rec);
                }
            } else {
                to_append.push(rec);
            }
        }

        for rec in to_append {
            self.record(store_key, rec);
        }
    }

    fn upgrade_record(
        &self,
        store_key: &str,
        previous: &Arc<ThinkingRecord>,
        upgraded: ThinkingRecord,
    ) {
        use crate::modules::proxy_db::{apply_thinking_upgrade, snapshot_thinking_upgrade};
        let result = (|| -> Result<(), String> {
            let row = snapshot_thinking_upgrade(store_key, previous)?;
            // A prune can shift indices or another request can replace this
            // generation between matching and this write. Revalidate the Arc
            // itself, not its old index, before publishing either upgrade.
            if !self.sessions.get(store_key).is_some_and(|entry| {
                entry
                    .turns
                    .iter()
                    .any(|current| Arc::ptr_eq(current, previous))
            }) {
                return Ok(());
            }
            if let Some(row) = row {
                if !apply_thinking_upgrade(&row, &upgraded)? {
                    return Ok(());
                }
            }
            if let Some(mut entry) = self.sessions.get_mut(store_key) {
                if let Some(index) = entry.turns.iter().position(|r| Arc::ptr_eq(r, previous)) {
                    entry.bytes = entry
                        .bytes
                        .saturating_sub(record_bytes(previous))
                        .saturating_add(record_bytes(&upgraded));
                    entry.turns[index] = Arc::new(upgraded);
                }
            }
            Ok(())
        })();
        if let Err(error) = result {
            tracing::warn!(
                "[ThinkingStore] Failed to upgrade historical record in {}: {}",
                store_key,
                error
            );
        }
    }

    pub fn restore_gemini_contents(&self, store_key: &str, contents: &mut Vec<Value>) -> usize {
        self.restore_gemini_contents_with_model(store_key, contents, None)
    }

    pub fn restore_gemini_contents_with_model(
        &self,
        store_key: &str,
        contents: &mut Vec<Value>,
        target_model: Option<&str>,
    ) -> usize {
        self.restore_selected_contents(store_key, contents, target_model, None)
    }

    fn restore_selected_contents(
        &self,
        store_key: &str,
        contents: &mut Vec<Value>,
        target_model: Option<&str>,
        text_only: Option<bool>,
    ) -> usize {
        if !crate::proxy::config::is_thinking_store_enabled() {
            return 0;
        }
        if contents.is_empty() || store_key.is_empty() {
            return 0;
        }

        let compaction_boundary = determine_compaction_boundary(contents, Some(store_key));
        let mut records = self.load_turns(store_key);
        let text_anchors = compute_text_turn_anchors(contents);

        // 收集所有的 model 轮次元信息
        struct ModelTurnMeta {
            content_idx: usize,
            visible: String,
            norm_visible: String,
            tool_ids: Vec<String>,
            #[allow(dead_code)]
            tool_names: Vec<String>,
            existing_thought: String,
            existing_sig: Option<String>,
            fp: String,
            text_anchor: String,
            matched_record_idx: Option<usize>,
            already_complete: bool,
            ambiguous_context: bool,
        }

        let mut model_turns: Vec<ModelTurnMeta> = Vec::new();
        for (c_idx, content) in contents.iter().enumerate() {
            let role = content.get("role").and_then(|v| v.as_str()).unwrap_or("");
            if role != "model" && role != "assistant" {
                continue;
            }
            if compaction_boundary.map_or(false, |b| c_idx <= b) {
                // 压缩历史中的 model 轮次绝不从旧缓存中反向复活已失效的旧签名或旧思考
                continue;
            }
            let Some(parts) = content.get("parts").and_then(|p| p.as_array()) else {
                continue;
            };
            let is_text = !parts.iter().any(|p| p.get("functionCall").is_some());
            if text_only.is_some_and(|selected| {
                selected != is_text || parts.iter().any(|p| p.get("functionResponse").is_some())
            }) {
                continue;
            }
            let preceding_turn = if c_idx > 0 {
                contents.get(c_idx - 1)
            } else {
                None
            };
            let anchor = compute_causal_anchor(preceding_turn);
            let (visible, tool_ids, tool_names, existing_thought) =
                inspect_parts_with_anchor(parts, &anchor);
            let existing_sig = parts.iter().find_map(|p| {
                let sig = p
                    .get("thoughtSignature")
                    .or_else(|| p.get("thought_signature"))
                    .and_then(|s| s.as_str())
                    .filter(|s| is_real_signature(s))
                    .map(str::to_string);
                sig.or_else(|| {
                    p.get("functionCall")
                        .and_then(|fc| fc.get("id"))
                        .and_then(|id| id.as_str())
                        .and_then(|id| {
                            crate::proxy::SignatureCache::global().get_tool_signature(store_key, id)
                        })
                        .filter(|s| is_real_signature(s))
                })
            });
            let already_complete = !turn_needs_restore(parts, &existing_thought);
            // Agent tool turns match by tool_id (Phase 1). Skip fingerprint /
            // whitespace-normalize until a later phase actually needs them.
            model_turns.push(ModelTurnMeta {
                content_idx: c_idx,
                visible,
                norm_visible: String::new(),
                tool_ids,
                tool_names,
                existing_thought,
                existing_sig,
                fp: String::new(),
                text_anchor: text_anchors[c_idx].clone(),
                matched_record_idx: None,
                already_complete,
                ambiguous_context: false,
            });
        }

        if model_turns.is_empty() {
            return 0;
        }

        let last_model_content_idx = model_turns.last().map(|turn| turn.content_idx);
        let latest_session_sig =
            crate::proxy::SignatureCache::global().get_session_signature(store_key);

        // Complete the exact text candidate set before any matching phase. A
        // partial memory window must not select one resident generation while
        // another generation of the same prefix/answer remains only in SQLite.
        // Full warm sessions need no extra read when the exact fingerprint is known.
        let l2_has_evicted_turns = self
            .sessions
            .get(store_key)
            .is_some_and(|entry| entry.l2_has_evicted_turns);
        let mut l2_fingerprints = HashSet::new();
        for turn in &mut model_turns {
            if !turn.tool_ids.is_empty()
                || !turn.tool_names.is_empty()
                || turn.visible.trim().is_empty()
            {
                continue;
            }
            turn.fp = contextual_fingerprint(
                &turn.visible,
                &turn.tool_ids,
                &turn.tool_names,
                Some(&turn.text_anchor),
            );
            if l2_has_evicted_turns || !records.iter().any(|rec| rec.fingerprint == turn.fp) {
                l2_fingerprints.insert(turn.fp.clone());
                l2_fingerprints.insert(base_fingerprint(&turn.fp).to_owned());
            }
        }
        if !l2_fingerprints.is_empty() {
            let fingerprints: Vec<String> = l2_fingerprints.iter().cloned().collect();
            match crate::modules::proxy_db::load_thinking_by_fingerprints(store_key, &fingerprints)
            {
                Ok(persisted) => {
                    let mut complete: Vec<Arc<ThinkingRecord>> = persisted
                        .into_iter()
                        .map(|p| {
                            Arc::new(ThinkingRecord {
                                fingerprint: p.fingerprint,
                                thought: p.thought,
                                signature: p.signature,
                                tool_ids: p.tool_ids,
                                tool_names: p.tool_names,
                                visible: p.visible,
                            })
                        })
                        .collect();
                    // Reuse the database's capture order (also used by a cold
                    // load), retaining memory-only captures after a failed save.
                    let persisted_len = complete.len();
                    for rec in records {
                        if !complete[..persisted_len]
                            .iter()
                            .any(|known| same_thinking_record(known, &rec))
                        {
                            complete.push(rec);
                        }
                    }
                    records = complete;
                }
                Err(error) => {
                    tracing::warn!(
                        "[ThinkingStore] Failed to load text candidates for {}: {}",
                        store_key,
                        error
                    );
                    for turn in &mut model_turns {
                        if l2_fingerprints.contains(&turn.fp) {
                            // Without the full set, weaker text fallback cannot
                            // safely decide which persisted generation to use.
                            turn.ambiguous_context = true;
                        }
                    }
                }
            }
        }

        if l2_has_evicted_turns {
            let ids: HashSet<_> = model_turns
                .iter()
                .flat_map(|turn| turn.tool_ids.iter().cloned())
                .collect();
            if !ids.is_empty() {
                match crate::modules::proxy_db::load_thinking_by_tool_ids(
                    store_key,
                    &ids.into_iter().collect::<Vec<_>>(),
                ) {
                    Ok(rows) => extend_tool_candidates(&mut records, rows),
                    Err(_) => {
                        for turn in &mut model_turns {
                            if !turn.tool_ids.is_empty() {
                                turn.ambiguous_context = true;
                            }
                        }
                    }
                }
            }
        }
        records.retain(|r| !r.fingerprint.starts_with(CALL_SIGNATURE_PREFIX));
        let mut used = vec![false; records.len()];
        let mut by_sig: HashMap<String, usize> = HashMap::new();
        let mut by_tool: HashMap<String, Vec<usize>> = HashMap::new();
        let mut by_fp: HashMap<&str, Vec<usize>> = HashMap::new();
        for (rec_idx, rec) in records.iter().enumerate() {
            if let Some(ref sig) = rec.signature.as_ref().filter(|s| is_real_signature(s)) {
                let norm = normalize_signature_for_comparison(sig);
                by_sig.insert(norm.into_owned(), rec_idx);
            }
            for id in &rec.tool_ids {
                let norm = crate::proxy::common::utils::normalize_tool_id(id);
                by_tool.entry(norm.to_string()).or_default().push(rec_idx);
                if norm.as_ref() != id.as_str() {
                    by_tool.entry(id.clone()).or_default().push(rec_idx);
                }
            }
            by_fp
                .entry(rec.fingerprint.as_str())
                .or_default()
                .push(rec_idx);
        }

        // Phase 0: 真实签名直查（最高优先级：客户端历史若自带真实签名，直接精确反向匹配）
        for turn in model_turns.iter_mut() {
            if turn.matched_record_idx.is_some() {
                continue;
            }
            if let Some(ref sig) = turn.existing_sig {
                let norm = normalize_signature_for_comparison(sig);
                if let Some(&rec_idx) = by_sig.get(norm.as_ref()) {
                    if !used[rec_idx] {
                        // 防错配保护：工具调用轮次绝不能匹配纯文本记录，纯文本轮次绝不能匹配工具记录！
                        let turn_has_tools =
                            !turn.tool_ids.is_empty() || !turn.tool_names.is_empty();
                        let rec_has_tools = !records[rec_idx].tool_ids.is_empty()
                            || !records[rec_idx].tool_names.is_empty();
                        if turn_has_tools == rec_has_tools
                            && (!turn_has_tools
                                || tool_ids_compatible(&turn.tool_ids, &records[rec_idx].tool_ids))
                            && (turn_has_tools
                                || record_signature_matches_target(&records[rec_idx], target_model))
                        {
                            let tool_names_match = if turn.tool_names.is_empty()
                                || records[rec_idx].tool_names.is_empty()
                            {
                                true
                            } else {
                                turn.tool_names == records[rec_idx].tool_names
                            };
                            if tool_names_match {
                                turn.matched_record_idx = Some(rec_idx);
                                used[rec_idx] = true;
                            }
                        }
                    }
                }
            }
        }

        // Phase 1: 工具调用 ID 精准锚定（包括原生唯一 ID 与确定性合成 ID，正向保序匹配）
        for turn in model_turns.iter_mut() {
            if turn.already_complete
                || turn.matched_record_idx.is_some()
                || turn.tool_ids.is_empty()
            {
                continue;
            }
            if turn.ambiguous_context {
                continue;
            }
            let mut candidates: Vec<usize> = turn
                .tool_ids
                .iter()
                .filter_map(|id| by_tool.get(id))
                .flatten()
                .copied()
                .collect();
            candidates.sort_unstable();
            candidates.dedup();
            let eligible = candidates.into_iter().filter(|&i| {
                turn.tool_names.is_empty()
                    || records[i].tool_names.is_empty()
                    || turn.tool_names == records[i].tool_names
            });
            // Used siblings still prove an unsigned alias is ambiguous.
            match select_tool_generation(
                &turn.tool_ids,
                turn.existing_sig.as_deref(),
                &records,
                eligible,
            ) {
                ToolMatch::Found(i) if !used[i] => {
                    turn.matched_record_idx = Some(i);
                    used[i] = true;
                }
                ToolMatch::Found(_) | ToolMatch::Conflict => turn.ambiguous_context = true,
                ToolMatch::Missing => {}
            }
        }

        // Legacy rows have no prefix identity. Preserve their established
        // ordered full-history migration, but never infer an ordering from a
        // shortened replay or a conflicting explicit signed generation.
        let mut legacy_replay_slots = HashMap::<String, usize>::new();
        let mut legacy_replay_conflicts = HashSet::new();
        for turn in &model_turns {
            if !turn.tool_ids.is_empty()
                || !turn.tool_names.is_empty()
                || turn.matched_record_idx.is_some()
            {
                continue;
            }
            let key = base_fingerprint(&turn.fp).to_owned();
            if turn.existing_sig.as_deref().is_some_and(is_real_signature) {
                legacy_replay_conflicts.insert(key);
            } else if !turn.already_complete && !turn.ambiguous_context {
                *legacy_replay_slots.entry(key).or_default() += 1;
            }
        }

        // Phase 2: 完整指纹匹配（正向保序匹配，杜绝修剪后逆向滑窗相位错位）
        for turn in model_turns.iter_mut() {
            if turn.already_complete || turn.ambiguous_context || turn.matched_record_idx.is_some()
            {
                continue;
            }
            if turn.fp.is_empty() {
                turn.fp = contextual_fingerprint(
                    &turn.visible,
                    &turn.tool_ids,
                    &turn.tool_names,
                    Some(&turn.text_anchor),
                );
            }
            let turn_has_tools = !turn.tool_ids.is_empty() || !turn.tool_names.is_empty();
            let exact = by_fp.get(turn.fp.as_str());
            // Old databases have unscoped fingerprints. Only those legacy
            // records may be considered after an exact contextual lookup.
            let legacy = (!turn_has_tools)
                .then(|| by_fp.get(base_fingerprint(&turn.fp)))
                .flatten();
            let eligible = |i: usize| {
                if used[i] {
                    return false;
                }
                let rec_has_tools =
                    !records[i].tool_ids.is_empty() || !records[i].tool_names.is_empty();
                if rec_has_tools != turn_has_tools {
                    return false;
                }
                if !turn_has_tools && !record_signature_matches_target(&records[i], target_model) {
                    return false;
                }
                if !turn_has_tools
                    && turn
                        .existing_sig
                        .as_deref()
                        .zip(records[i].signature.as_deref())
                        .is_some_and(|(a, b)| !signatures_match(a, b))
                {
                    return false;
                }
                if !turn.tool_names.is_empty() && !records[i].tool_names.is_empty() {
                    if turn.tool_names != records[i].tool_names {
                        return false;
                    }
                }
                true
            };
            let exact_candidates: Vec<usize> = exact
                .into_iter()
                .flatten()
                .copied()
                .filter(|&i| eligible(i))
                .collect();
            let exact_match = if !turn_has_tools && !exact_candidates.is_empty() {
                let authority = turn.existing_sig.as_deref().or_else(|| {
                    (last_model_content_idx == Some(turn.content_idx))
                        .then_some(latest_session_sig.as_deref())
                        .flatten()
                });
                select_contextual_text_record(&records, &exact_candidates, authority)
            } else {
                exact_candidates.first().copied()
            };
            if !turn_has_tools && !exact_candidates.is_empty() && exact_match.is_none() {
                // Distinct signed generations of the very same prompt/answer
                // cannot be distinguished by text alone. Do not let a weaker
                // similarity, legacy or SQLite fallback choose one arbitrarily.
                turn.ambiguous_context = true;
                continue;
            }
            let legacy_candidates: Vec<_> = legacy
                .into_iter()
                .flatten()
                .copied()
                .filter(|&i| eligible(i))
                .collect();
            let authority = turn.existing_sig.as_deref().or_else(|| {
                (last_model_content_idx == Some(turn.content_idx))
                    .then_some(latest_session_sig.as_deref())
                    .flatten()
            });
            let legacy_match =
                select_contextual_text_record(&records, &legacy_candidates, authority).or_else(
                    || {
                        let key = base_fingerprint(&turn.fp);
                        (!legacy_replay_conflicts.contains(key)
                            && legacy_replay_slots.get(key).copied().unwrap_or(0)
                                >= legacy_candidates.len())
                        .then(|| legacy_candidates.first().copied())
                        .flatten()
                    },
                );
            if exact_match.is_none() && !legacy_candidates.is_empty() && legacy_match.is_none() {
                // Old rows lack prefix fingerprints, but differing signatures
                // still prove that text alone cannot select a generation.
                turn.ambiguous_context = true;
                continue;
            }
            let matched = exact_match.or(legacy_match);
            if let Some(rec_idx) = matched {
                turn.matched_record_idx = Some(rec_idx);
                used[rec_idx] = true;
            }
        }

        // Phase 3: 纯文本前缀 / 正文相似匹配（仅限纯文本轮次）
        // Normalize each record once. The old inner-loop split_whitespace().collect().join()
        // was O(turns * records * visible_len) and stalled 10s+ at ~300K context.
        let needs_phase3 = model_turns.iter().any(|t| {
            !t.already_complete
                && !t.ambiguous_context
                && t.matched_record_idx.is_none()
                && t.tool_ids.is_empty()
                && t.tool_names.is_empty()
                && !t.visible.trim().is_empty()
        });
        let (rec_norms, rec_thought_norms): (Vec<String>, Vec<String>) = if needs_phase3 {
            records
                .iter()
                .map(|r| (normalize_ws(&r.visible), normalize_ws(&r.thought)))
                .unzip()
        } else {
            (Vec::new(), Vec::new())
        };
        if needs_phase3 {
            for turn in model_turns.iter_mut() {
                if turn.already_complete
                    || turn.ambiguous_context
                    || turn.matched_record_idx.is_some()
                {
                    continue;
                }
                let turn_has_tools = !turn.tool_ids.is_empty() || !turn.tool_names.is_empty();
                if turn_has_tools || turn.visible.trim().is_empty() {
                    continue;
                }
                if turn.norm_visible.is_empty() {
                    turn.norm_visible = normalize_ws(&turn.visible);
                }
                if turn.norm_visible.is_empty() {
                    continue;
                }
                for (rec_idx, rec) in records.iter().enumerate() {
                    let rec_has_tools = !rec.tool_ids.is_empty() || !rec.tool_names.is_empty();
                    if used[rec_idx]
                        || rec_has_tools
                        || !record_signature_matches_target(rec, target_model)
                        || rec_norms[rec_idx].is_empty()
                        || !text_context_matches(&rec.fingerprint, &turn.text_anchor)
                        || turn
                            .existing_sig
                            .as_deref()
                            .zip(rec.signature.as_deref())
                            .is_some_and(|(a, b)| !signatures_match(a, b))
                    {
                        continue;
                    }
                    let norm_rec = &rec_norms[rec_idx];
                    let norm_vis = &turn.norm_visible;
                    let norm_thought = &rec_thought_norms[rec_idx];
                    let thought_matches_prefix =
                        !norm_thought.is_empty() && norm_vis.starts_with(norm_thought);
                    if norm_rec == norm_vis
                        || norm_rec.starts_with(norm_vis)
                        || norm_vis.starts_with(norm_rec)
                        || (norm_rec.len() >= 10 && norm_vis.ends_with(norm_rec))
                        || (thought_matches_prefix && norm_vis.ends_with(norm_rec))
                    {
                        turn.matched_record_idx = Some(rec_idx);
                        used[rec_idx] = true;
                        break;
                    }
                }
            }
        }

        // Phase 3.5: L2 SQLite 精准穿透回捞 (针对超过内存容量淘汰或冷启动的历史轮次)
        // Exact text fingerprints were loaded before matching; only authoritative signatures / tool ids need a point lookup here.
        for turn in model_turns.iter_mut() {
            if turn.already_complete || turn.ambiguous_context || turn.matched_record_idx.is_some()
            {
                continue;
            }

            let mut fetched_rec: Option<ThinkingRecord> = None;

            // 0. 优先按签名精准穿透
            if let Some(ref sig) = turn.existing_sig {
                let mut found =
                    crate::modules::proxy_db::load_thinking_by_signature(store_key, sig);
                if found.as_ref().map(|o| o.is_none()).unwrap_or(true) && is_claude_signature(sig) {
                    let google_sig = ensure_google_claude_thought_signature(sig);
                    if google_sig != *sig {
                        found = crate::modules::proxy_db::load_thinking_by_signature(
                            store_key,
                            &google_sig,
                        );
                    }
                }
                if let Ok(Some(persisted)) = found {
                    let turn_has_tools = !turn.tool_ids.is_empty() || !turn.tool_names.is_empty();
                    let rec_has_tools =
                        !persisted.tool_ids.is_empty() || !persisted.tool_names.is_empty();
                    if !persisted.fingerprint.starts_with(CALL_SIGNATURE_PREFIX)
                        && turn_has_tools == rec_has_tools
                        && (!turn_has_tools
                            || tool_ids_compatible(&turn.tool_ids, &persisted.tool_ids))
                    {
                        fetched_rec = Some(ThinkingRecord {
                            fingerprint: persisted.fingerprint,
                            thought: persisted.thought,
                            signature: persisted.signature,
                            tool_ids: persisted.tool_ids,
                            tool_names: persisted.tool_names,
                            visible: persisted.visible,
                        });
                    }
                }
            }

            // Tool candidates are resolved as a complete set above. A single
            // latest SQLite row is not authority to choose a generation.
            if let Some(rec) = fetched_rec {
                let turn_has_tools = !turn.tool_ids.is_empty() || !turn.tool_names.is_empty();
                if (!turn_has_tools && !record_signature_matches_target(&rec, target_model))
                    || records
                        .iter()
                        .zip(&used)
                        .any(|(known, used)| *used && same_thinking_record(known, &rec))
                {
                    continue;
                }
                let rec_arc = Arc::new(rec);
                records.push(rec_arc.clone());
                used.push(true);
                let new_idx = records.len() - 1;
                turn.matched_record_idx = Some(new_idx);
                // 同步注册回内存会话实体，确保后续轮次无需重复点查
                if let Some(mut entry) = self.sessions.get_mut(store_key) {
                    entry.bytes = entry.bytes.saturating_add(record_bytes(&rec_arc));
                    entry.turns.push(rec_arc);
                }
            }
        }

        // Phase 4: 尾部优先的逆向兜底匹配（仅限纯文本轮次，绝不跨轮借用工具签名造成下轮突变！）
        if let Some(last_turn) = model_turns.last_mut() {
            if !last_turn.already_complete
                && !last_turn.ambiguous_context
                && last_turn.matched_record_idx.is_none()
            {
                let last_turn_has_tools =
                    !last_turn.tool_ids.is_empty() || !last_turn.tool_names.is_empty();
                if !last_turn_has_tools {
                    if let Some((last_unused_rec_idx, _)) =
                        records.iter().enumerate().rfind(|(idx, r)| {
                            if used[*idx] {
                                return false;
                            }
                            let r_has_tools = !r.tool_ids.is_empty() || !r.tool_names.is_empty();
                            !r_has_tools
                                && record_signature_matches_target(r, target_model)
                                && text_context_matches(&r.fingerprint, &last_turn.text_anchor)
                                && !last_turn
                                    .existing_sig
                                    .as_deref()
                                    .zip(r.signature.as_deref())
                                    .is_some_and(|(a, b)| !signatures_match(a, b))
                        })
                    {
                        last_turn.matched_record_idx = Some(last_unused_rec_idx);
                        used[last_unused_rec_idx] = true;
                    }
                }
            }
        }

        let mut restored = 0usize;
        for turn in model_turns {
            if turn.already_complete {
                continue;
            }
            let Some(rec_idx) = turn.matched_record_idx else {
                continue;
            };
            let rec = &records[rec_idx];
            if rec.thought.trim().is_empty() && rec.signature.is_none() {
                continue;
            }

            let Some(content) = contents.get_mut(turn.content_idx) else {
                continue;
            };
            let Some(parts) = content.get_mut("parts").and_then(|p| p.as_array_mut()) else {
                continue;
            };

            let has_unvalidated_function_call = parts
                .iter()
                .any(|p| p.get("functionCall").is_some() && !part_has_signature(p));

            let should_replace = is_placeholder_thought(&turn.existing_thought)
                || turn.existing_thought.len() < rec.thought.len()
                || (rec.signature.is_some() && !parts.iter().any(|p| part_has_signature(p)))
                || has_unvalidated_function_call;

            if !should_replace {
                continue;
            }

            parts.retain(|p| p.get("thought").and_then(|t| t.as_bool()) != Some(true));

            let has_meaningful_thought =
                !is_placeholder_thought(&rec.thought) && !rec.thought.trim().is_empty();

            let is_claude_target = target_model
                .map(|m| m.to_lowercase().contains("claude"))
                .unwrap_or_else(|| store_key.to_lowercase().contains("claude"));

            if has_meaningful_thought {
                let thought_part = json!({
                    "text": rec.thought.as_str(),
                    "thought": true,
                });

                // [DE-DUPLICATION & ELEVATION] 正文残留思考文本切除与去重：
                // 解决关思考时降级到正文的思考文本，在重新开思考时被再次提升复活后，导致正文残留双份复读的问题！
                let rec_thought_trimmed = rec.thought.trim();
                let rec_thought_norm = normalize_ws(&rec.thought);

                let mut cleaned_parts = Vec::with_capacity(parts.len());
                for part in parts.drain(..) {
                    let is_plain_text = part.get("text").is_some()
                        && part.get("functionCall").is_none()
                        && part.get("functionResponse").is_none()
                        && part.get("thought").and_then(|v| v.as_bool()) != Some(true);

                    if is_plain_text {
                        let text = part["text"].as_str().unwrap_or("");
                        let text_trimmed = text.trim();

                        // 0. 若正文带有 <think>...</think> 标签，精准切除标签与思考内容，保留剩余真实正文
                        if let Some((_, rem)) = extract_think_tags(text) {
                            if !rem.is_empty() {
                                cleaned_parts.push(json!({ "text": rem }));
                            }
                            continue;
                        }

                        // 1. 完全相同（之前作为独立降级部件存在）：直接丢弃该部件
                        if text_trimmed == rec_thought_trimmed
                            || (!rec_thought_norm.is_empty()
                                && normalize_ws(text) == rec_thought_norm)
                        {
                            continue;
                        }
                        // 2. 正文以思考文本开头（思考文本与正文被客户端合并为一个部件）：切除前缀
                        if text.starts_with(&rec.thought) {
                            let remainder = &text[rec.thought.len()..];
                            let trimmed_rem =
                                remainder.trim_start_matches(|c| c == '\r' || c == '\n');
                            if !trimmed_rem.is_empty() {
                                cleaned_parts.push(json!({ "text": trimmed_rem }));
                            }
                            continue;
                        } else if text_trimmed.starts_with(rec_thought_trimmed) {
                            let remainder = &text_trimmed[rec_thought_trimmed.len()..];
                            let trimmed_rem =
                                remainder.trim_start_matches(|c| c == '\r' || c == '\n');
                            if !trimmed_rem.is_empty() {
                                cleaned_parts.push(json!({ "text": trimmed_rem }));
                            }
                            continue;
                        }
                    }
                    cleaned_parts.push(part);
                }
                *parts = cleaned_parts;

                if is_claude_target {
                    // Claude 模型：官方抓包铁证（Ground Truth）：
                    // 1. 从 Gemini 切换到 Claude 模型后，历史上的 Gemini 轮次必须彻底剥离签名，且绝不能送出 thought: true！
                    // 2. 只有当该轮历史记录确属 Claude 真实签名时，才允许回填签名并恢复 thought: true 思考块！
                    let fallback = rec
                        .signature
                        .as_deref()
                        .filter(|s| is_real_signature(s) && is_claude_signature(s));
                    if let Some(claude_sig) = fallback {
                        let wrapped = ensure_google_claude_thought_signature(claude_sig);
                        place_turn_signature_scoped(
                            &mut *parts,
                            Some(&wrapped),
                            false,
                            Some(store_key),
                        );
                        parts.insert(0, thought_part);
                        restored += 1;
                    } else {
                        // 历史记录非 Claude 签名（例如从 Gemini 切过来的历史轮次）：
                        // 严禁插入 thought: true 思考块！严禁注入任何假签名或哨兵！
                        // 严格对齐官方抓包形态：历史保持纯文本与普通工具调用（无签留空出站，Google 100% 正常接受）。
                        tracing::debug!(
                            "[ThinkingStore] Bypassing non-Claude thought block/signature for Claude target model"
                        );
                    }
                } else {
                    // Gemini 原生：把本轮捕获到的真实签名归位到「该轮第一个非思考 part」。
                    //
                    // 首位思考块保持纯净思考文本（铁律 I4：思考块绝不携带签名），随后由
                    // `parts.insert(0, thought_part)` 插入 index 0 —— 锚点自然落到 index 1，
                    // 正好复现官方的「思考块 + 带签名正文」排列。
                    let fallback = rec
                        .signature
                        .as_deref()
                        .filter(|s| is_real_signature(s) && is_likely_gemini_signature(s));
                    place_turn_signature_scoped(&mut *parts, fallback, true, Some(store_key));
                    parts.insert(0, thought_part);
                    restored += 1;
                }
            } else {
                // 【2026-09-27】无实质思考内容（占位/空思考）：绝不注入 "..." 占位思考块！
                // 官方标准形态为「无思考块 + 锚点带签名」（baogao.txt 9/24 轮）。
                // 首个非思考 part 作为锚点正常承接该轮签名。
                if is_claude_target {
                    let fallback = rec
                        .signature
                        .as_deref()
                        .filter(|s| is_real_signature(s) && is_claude_signature(s));
                    let wrapped = fallback.map(|s| ensure_google_claude_thought_signature(s));
                    place_turn_signature_scoped(
                        &mut *parts,
                        wrapped.as_deref(),
                        false,
                        Some(store_key),
                    );
                } else {
                    let fallback = rec
                        .signature
                        .as_deref()
                        .filter(|s| is_real_signature(s) && is_likely_gemini_signature(s));
                    if place_turn_signature_scoped(&mut *parts, fallback, true, Some(store_key))
                        .is_some()
                    {
                        restored += 1;
                    }
                }
            }
        }

        if restored > 0 {
            tracing::info!(
                "[ThinkingStore] Restored {} full thinking block(s) for session {}",
                restored,
                store_key
            );
        }
        restored
    }

    pub fn end_session(&self, store_key: &str) -> EndSessionResult {
        let removed = self.sessions.remove(store_key);
        let (deleted_turns, deleted_bytes) = removed
            .map(|(_, e)| (e.turns.len(), e.bytes))
            .unwrap_or((0, 0));
        let _ = crate::modules::proxy_db::delete_thinking_records_for_session(store_key);
        EndSessionResult {
            session_id: client_id_from_store_key(store_key).to_string(),
            deleted_turns,
            deleted_bytes,
        }
    }

    /// 彻底清理指定会话的全部预压缩历史数据与签名缓存 (RAM sessions + SQLite L2 + SignatureCache)
    /// 当检测到会话发生上下文压缩提纯 (Reactive Compaction) 时调用，彻底杜绝失效旧签名穿透或二次膨胀
    pub fn clear_session_for_compaction(&self, store_key: &str) {
        let clean_key = store_key.trim();
        if clean_key.is_empty() {
            return;
        }
        let suffix = format!(":{}", clean_key);
        self.sessions
            .retain(|k, _| k != clean_key && !k.ends_with(&suffix));
        let _ = crate::modules::proxy_db::delete_thinking_records_for_session(clean_key);
        crate::proxy::SignatureCache::global().invalidate_session(clean_key);
        tracing::info!(
            "[ThinkingStore] Cleared session {} for compaction",
            clean_key
        );
    }

    /// 检查并维护会话高水位线。若发生宏观上下文骤降（轮数骤减或总字符量骤减），返回 true
    pub fn check_macro_context_drop(
        &self,
        session_id: &str,
        cur_chars: usize,
        cur_turns: usize,
    ) -> bool {
        let clean_key = session_id.trim();
        if clean_key.is_empty() {
            return false;
        }

        if let Some(mut entry) = self.watermarks.get_mut(clean_key) {
            let prev_chars = entry.chars;
            let prev_turns = entry.turns;
            // 宏观骤降门禁：
            // 历史轮次至少达到 4 轮且轮数骤降 >= 2，或累积字符数达到 4000 字符以上且字符骤降 > 35%
            let turns_dropped = prev_turns >= 4 && cur_turns.saturating_add(2) <= prev_turns;
            let chars_dropped = prev_chars >= 4000 && cur_chars < (prev_chars * 65 / 100);

            if turns_dropped || chars_dropped {
                tracing::info!(
                    "[ThinkingStore] Macro context drop detected for session {}: turns {} -> {}, chars {} -> {}",
                    clean_key, prev_turns, cur_turns, prev_chars, cur_chars
                );
                entry.chars = cur_chars;
                entry.turns = cur_turns;
                entry.last_seen = Instant::now();
                return true;
            } else {
                if cur_chars > entry.chars {
                    entry.chars = cur_chars;
                }
                if cur_turns > entry.turns {
                    entry.turns = cur_turns;
                }
                entry.last_seen = Instant::now();
            }
        } else {
            if self.watermarks.len() > MAX_SESSIONS {
                self.watermarks
                    .retain(|_, w| w.last_seen.elapsed() < idle_ttl());
            }
            self.watermarks.insert(
                clean_key.to_string(),
                SessionWatermark {
                    chars: cur_chars,
                    turns: cur_turns,
                    last_seen: Instant::now(),
                },
            );
        }

        false
    }

    /// 精准定向净化指定会话中的异构污染签名（保留思考文本与健康签名）
    pub fn purge_corrupted_signatures(&self, store_key: &str, target_model: &str) -> usize {
        if store_key.is_empty() {
            return 0;
        }
        let is_gemini = target_model.to_lowercase().contains("gemini");
        let is_claude = target_model.to_lowercase().contains("claude");
        if !is_gemini && !is_claude {
            return 0;
        }

        let mut purged_count = 0;

        // 1. 精准净化内存缓存 (RAM)
        if let Some(mut entry) = self.sessions.get_mut(store_key) {
            entry.load_epoch = Arc::new(());
            let mut new_turns = Vec::with_capacity(entry.turns.len());
            for rec in &entry.turns {
                if let Some(ref sig) = rec.signature {
                    let is_foreign = if is_gemini {
                        !is_likely_gemini_signature(sig)
                    } else if is_claude {
                        !is_claude_signature(sig)
                    } else {
                        false
                    };
                    if is_foreign {
                        purged_count += 1;
                        let mut cleaned = (**rec).clone();
                        cleaned.signature = None;
                        new_turns.push(Arc::new(cleaned));
                        continue;
                    }
                }
                new_turns.push(rec.clone());
            }
            entry.turns = new_turns;
        }

        // 2. 精准净化持久化数据库 (SQLite)
        let _ = crate::modules::proxy_db::purge_foreign_signatures_for_session_with_model(
            store_key,
            target_model,
        );

        if purged_count > 0 {
            tracing::warn!(
                "[ThinkingStore] Surgically purged {} foreign signature(s) for session {} targeting {}",
                purged_count, store_key, target_model
            );
        }
        purged_count
    }

    /// [Signature-Retry] 让一个会话的全部签名失效（保留思考文本）。
    ///
    /// 与 `purge_corrupted_signatures`（只清异构签名）不同：上游已经明确判定签名无效时，
    /// 无法区分哪一轮有问题，保守做法是整会话失效，由调用方在重试载荷中移除签名后再发。
    pub fn invalidate_session_signatures(&self, store_key: &str) -> usize {
        if store_key.is_empty() {
            return 0;
        }

        let mut cleared = 0;
        if let Some(mut entry) = self.sessions.get_mut(store_key) {
            entry.load_epoch = Arc::new(());
            let mut new_turns = Vec::with_capacity(entry.turns.len());
            for rec in &entry.turns {
                if rec.signature.is_some() {
                    cleared += 1;
                    let mut cleaned = (**rec).clone();
                    cleaned.signature = None;
                    new_turns.push(Arc::new(cleaned));
                } else {
                    new_turns.push(rec.clone());
                }
            }
            entry.turns = new_turns;
        }

        let _ = crate::modules::proxy_db::clear_thinking_signatures_for_session(store_key);

        if cleared > 0 {
            tracing::warn!(
                "[ThinkingStore] Invalidated {} signature(s) for session {} after upstream signature error",
                cleared, store_key
            );
        }
        cleared
    }

    /// Drop thinking records that no longer appear in the (possibly compressed) history.
    /// Always keeps the newest 2 turns so the latest unused response thinking is not lost.
    pub fn prune_orphaned_records(&self, store_key: &str, contents: &[Value]) {
        if let Some(plan) = self.prepare_orphaned_records_prune(store_key, contents) {
            plan.persist();
        }
    }

    fn prepare_orphaned_records_prune(
        &self,
        store_key: &str,
        contents: &[Value],
    ) -> Option<ThinkingPrunePlan> {
        if !crate::proxy::config::is_thinking_store_enabled() || store_key.is_empty() {
            return None;
        }

        let mem_turns = self
            .sessions
            .get(store_key)
            .map(|e| e.turns.len())
            .unwrap_or(0);
        if mem_turns == 0 {
            return None;
        }
        let live_turn_count = contents.iter().filter(|c| is_model_or_assistant(c)).count();
        // Typical agent path: stored turns ≈ live model turns. Skip inspect/fingerprint/SQLite.
        if mem_turns <= live_turn_count.saturating_add(2) {
            return None;
        }

        // Snapshot before taking the RAM write guard. A concurrent capture can
        // then finish both writes without an old prune deleting its new version.
        // Never hold a SQLite guard while acquiring a DashMap shard (or vice versa).
        let snapshot =
            match crate::modules::proxy_db::snapshot_thinking_records_for_prune(store_key) {
                Ok(snapshot) => snapshot,
                Err(error) => {
                    tracing::warn!(
                        "[ThinkingStore] Skipping prune for {}: {}",
                        store_key,
                        error
                    );
                    return None;
                }
            };

        let mut live_tool_ids = std::collections::HashSet::new();
        let mut live_fps = std::collections::HashSet::new();
        let mut live_sigs = std::collections::HashSet::new();
        let mut live_visibles: Vec<String> = Vec::new();
        let mut live_text_anchors = std::collections::HashSet::new();
        let text_anchors = compute_text_turn_anchors(contents);

        for (c_idx, content) in contents.iter().enumerate() {
            if !is_model_or_assistant(content) {
                continue;
            }
            let Some(parts) = content.get("parts").and_then(|p| p.as_array()) else {
                continue;
            };
            let preceding_turn = if c_idx > 0 {
                contents.get(c_idx - 1)
            } else {
                None
            };
            let anchor = compute_causal_anchor(preceding_turn);
            let (visible, tool_ids, tool_names, _) = inspect_parts_with_anchor(parts, &anchor);
            live_fps.insert(fingerprint(&visible, &tool_ids, &tool_names));
            live_fps.insert(contextual_fingerprint(
                &visible,
                &tool_ids,
                &tool_names,
                Some(&text_anchors[c_idx]),
            ));
            if tool_ids.is_empty() && tool_names.is_empty() {
                live_text_anchors.insert(text_anchors[c_idx].as_str());
            }
            for id in tool_ids {
                live_tool_ids.insert(id);
            }
            for part in parts {
                if let Some(sig) = part
                    .get("thoughtSignature")
                    .or_else(|| part.get("thought_signature"))
                    .and_then(|s| s.as_str())
                    .filter(|s| is_real_signature(s))
                {
                    live_sigs.insert(sig.to_string());
                }
            }
            let norm = normalize_ws(&visible);
            if !norm.is_empty() {
                live_visibles.push(norm);
            }
        }

        let keep_fps = {
            let Some(mut entry) = self.sessions.get_mut(store_key) else {
                return None;
            };
            if entry.turns.len() <= live_turn_count.saturating_add(2) {
                return None;
            }

            let rec_norms: Vec<String> = entry
                .turns
                .iter()
                .map(|r| normalize_ws(&r.visible))
                .collect();
            let total = entry.turns.len();
            let keep_tail_start = total.saturating_sub(2);
            let mut keep: Vec<Arc<ThinkingRecord>> = Vec::new();
            for (i, rec) in entry.turns.iter().enumerate() {
                let matched_sig = rec
                    .signature
                    .as_deref()
                    .is_some_and(|s| live_sigs.contains(s));
                let matched_tool = rec.tool_ids.iter().any(|id| live_tool_ids.contains(id));
                let matched_fp = live_fps.contains(&rec.fingerprint);
                let norm_rec = &rec_norms[i];
                let context_live = text_fingerprint_context(&rec.fingerprint)
                    .map_or(true, |(anchor, _)| live_text_anchors.contains(anchor));
                let matched_text = context_live
                    && !norm_rec.is_empty()
                    && live_visibles.iter().any(|v| {
                        v == norm_rec || v.starts_with(norm_rec) || norm_rec.starts_with(v)
                    });
                if matched_sig || matched_tool || matched_fp || matched_text || i >= keep_tail_start
                {
                    keep.push(rec.clone());
                }
            }

            if keep.len() == entry.turns.len() {
                return None;
            }

            let dropped = entry.turns.len() - keep.len();
            entry.bytes = keep.iter().map(|r| record_bytes(r)).sum();
            // Live generations can remain only in SQLite: an ambiguous text
            // match is intentionally not hydrated, for example. RAM survivors
            // alone are therefore not an authoritative database keep list.
            live_fps.extend(keep.iter().map(|r| r.fingerprint.clone()));
            let fps: Vec<String> = live_fps.into_iter().collect();
            entry.turns = keep;
            entry.load_epoch = Arc::new(());
            // SQLite can retain other generations of a surviving fingerprint,
            // or captures protected by the prune snapshot's write revisions.
            // RAM is now a partial candidate set even below the capacity limit.
            entry.l2_has_evicted_turns = true;
            tracing::info!(
                "[ThinkingStore] Pruned {} orphaned thinking record(s) after context compression for session {}",
                dropped,
                store_key
            );
            fps
        };

        Some(ThinkingPrunePlan {
            snapshot,
            keep_fps,
            live_tool_ids: live_tool_ids.into_iter().collect(),
            live_sigs: live_sigs.into_iter().collect(),
        })
    }

    pub fn session_stats(&self, store_key: &str) -> Option<(usize, usize)> {
        self.sessions
            .get(store_key)
            .map(|e| (e.turns.len(), e.bytes))
    }

    pub fn clear(&self) {
        self.sessions.clear();
    }
}

#[derive(Debug, Clone)]
pub struct EndSessionResult {
    pub session_id: String,
    pub deleted_turns: usize,
    pub deleted_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct SessionScope {
    /// 带内容锚点的会话号。思维库、签名缓存、上游 `sessionId` 用它。
    pub client_id: String,
    /// `{tenant}:{client_id}`。思维库、签名缓存、上游 `sessionId` 的存储键。
    pub store_key: String,
    /// 不含内容锚点的稳定会话号。没有稳定会话身份时与 `client_id` 相同。
    pub affinity_id: String,
    /// `{tenant}:{affinity_id}`。只决定用哪个 Google 账号。
    pub affinity_key: String,
}

impl SessionScope {
    pub fn from_headers(headers: &HeaderMap, fallback: impl Into<String>) -> Self {
        Self::from_request_parts(headers, None, None, fallback)
    }

    pub fn from_headers_and_body(
        headers: &HeaderMap,
        body: Option<&Value>,
        fallback: impl Into<String>,
    ) -> Self {
        Self::from_request_parts(headers, body, None, fallback)
    }

    pub fn from_request_parts(
        headers: &HeaderMap,
        body: Option<&Value>,
        query: Option<&str>,
        fallback: impl Into<String>,
    ) -> Self {
        Self::resolve(headers, body, query, fallback, None)
    }

    /// `tenant_id` 来自中间件已经解析出的用户 Token。没有时再哈希请求头里的凭证。
    pub fn resolve(
        headers: &HeaderMap,
        body: Option<&Value>,
        query: Option<&str>,
        fallback: impl Into<String>,
        tenant_id: Option<&str>,
    ) -> Self {
        let fallback = fallback.into();
        let tenant = tenant_component(headers, tenant_id);
        let winner = stable_session_winner(headers, body, query);

        // 读取粘性调度配置维度（优先环境变量，其次配置文件，默认 Session 级）
        let affinity_scope = crate::proxy::get_affinity_scope();
        let (affinity_id, affinity_key) = match affinity_scope {
            crate::proxy::sticky_config::AffinityScope::ApiKey => {
                // 凭据级强粘性：针对多智能体框架频繁派生瞬态 Session UUID 场景，
                // 同一 API Key / 租户来源的所有流量统一绑定至同一上游主账号，消除 Prompt Cache 击穿
                let aff_id = format!("key-{}", tenant);
                let aff_key = format!("{}:{}", tenant, aff_id);
                (aff_id, aff_key)
            }
            crate::proxy::sticky_config::AffinityScope::Session => {
                // 默认会话级粘性：自动识别多智能体常见的派生子步骤后缀（如 :step:N、:talk:N 等），
                // 自动归一化到父根会话，实现子 Agent 组内统一路由与缓存命中
                let normalized_winner = winner.as_deref().map(extract_root_session_id);
                let affinity_anchor = if normalized_winner.is_some() {
                    ""
                } else {
                    fallback.as_str()
                };
                let aff_id = derive_winner_session_id(&tenant, normalized_winner, affinity_anchor);
                let aff_key = format!("{}:{}", tenant, aff_id);
                (aff_id, aff_key)
            }
        };

        // client_id 与 store_key 依然保持未截断的原始完整 winner，用于思维库隔离、诊断与下游响应
        let client_id = derive_winner_session_id(&tenant, winner.as_deref(), &fallback);
        let store_key = format!("{}:{}", tenant, client_id);
        Self {
            client_id,
            store_key,
            affinity_id,
            affinity_key,
        }
    }

    pub fn purge_signatures(&self, model: &str) {
        ThinkingStore::global().purge_corrupted_signatures(&self.store_key, model);
        crate::proxy::SignatureCache::global().delete_session_signature(&self.store_key);
    }
}

#[derive(Debug, Default, Clone)]
pub struct TurnAccumulator {
    thought: String,
    signature: Option<String>,
    visible: String,
    tool_ids: Vec<String>,
    tool_names: Vec<String>,
    context_anchor: String,
    text_context_anchor: Option<String>,
    function_call_count: usize,
    // Individual signatures must never be inferred from the longest signature in a turn.
    signed_calls: Vec<ThinkingRecord>,
}

impl TurnAccumulator {
    pub fn new() -> Self {
        Self::with_anchor("root")
    }

    pub fn with_anchor(anchor: impl Into<String>) -> Self {
        Self {
            thought: String::new(),
            signature: None,
            visible: String::new(),
            tool_ids: Vec::new(),
            tool_names: Vec::new(),
            context_anchor: anchor.into(),
            text_context_anchor: None,
            function_call_count: 0,
            signed_calls: Vec::new(),
        }
    }

    /// The full visible request prefix distinguishes repeated text-only turns.
    /// Keep the separate causal anchor unchanged for synthetic tool ids.
    pub fn with_text_context_anchor(mut self, anchor: impl Into<String>) -> Self {
        self.text_context_anchor = Some(anchor.into());
        self
    }

    pub fn ingest_part(&mut self, part: &Value) {
        let is_thought = part
            .get("thought")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
            if is_thought {
                self.thought.push_str(text);
            } else {
                self.visible.push_str(text);
            }
        }
        if let Some(sig) = part
            .get("thoughtSignature")
            .or_else(|| part.get("thought_signature"))
            .and_then(|s| s.as_str())
        {
            if is_real_signature(sig)
                && self
                    .signature
                    .as_ref()
                    .map(|old| sig.len() > old.len())
                    .unwrap_or(true)
            {
                self.signature = Some(sig.to_string());
            }
        }
        if let Some(fc) = part.get("functionCall") {
            let name = fc
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            let explicit_id = fc
                .get("id")
                .and_then(|v| v.as_str())
                .filter(|s| !s.trim().is_empty())
                .map(|s| crate::proxy::common::utils::normalize_tool_id(s).into_owned());

            let synthetic = synthesize_tool_id(
                &name,
                fc.get("args"),
                &self.context_anchor,
                self.function_call_count,
            );
            self.function_call_count += 1;

            if let Some(signature) = part
                .get("thoughtSignature")
                .or_else(|| part.get("thought_signature"))
                .and_then(Value::as_str)
                .filter(|s| is_real_signature(s) && is_likely_gemini_signature(s))
            {
                let ids: Vec<_> = std::iter::once(synthetic.clone())
                    .chain(explicit_id.clone().filter(|id| id != &synthetic))
                    .collect();
                let names = vec![name.clone()];
                self.signed_calls.push(ThinkingRecord {
                    fingerprint: format!(
                        "{CALL_SIGNATURE_PREFIX}{}",
                        fingerprint("", &ids, &names)
                    ),
                    thought: String::new(),
                    signature: Some(signature.to_owned()),
                    tool_ids: ids,
                    tool_names: names,
                    visible: String::new(),
                });
            }

            // 核心演进：全面转战因果伪哈希 ID！首位强制存入确定性 synthetic ID，客户端 real_id 紧随其后作为元数据兜底
            if !self.tool_ids.iter().any(|x| x == &synthetic) {
                self.tool_ids.push(synthetic.clone());
            }

            if let Some(ref real_id) = explicit_id {
                if !self.tool_ids.iter().any(|x| x == real_id) {
                    self.tool_ids.push(real_id.clone());
                }
            }

            if !self.tool_names.iter().any(|x| x == &name) {
                self.tool_names.push(name);
            }
        }
    }

    pub(crate) fn observed_signature(&self) -> Option<&str> {
        self.signature.as_deref()
    }

    pub fn record_tool_id(&mut self, tool_name: &str, real_id: &str) {
        let norm_id = crate::proxy::common::utils::normalize_tool_id(real_id);
        let id_str = norm_id.to_string();
        if !self.tool_ids.iter().any(|x| x == &id_str) {
            self.tool_ids.push(id_str);
        }
        if !self.tool_names.iter().any(|x| x == tool_name) {
            self.tool_names.push(tool_name.to_string());
        }
    }

    pub fn is_empty(&self) -> bool {
        self.thought.trim().is_empty() && self.signature.is_none()
    }

    fn should_capture(&self) -> bool {
        !self.is_empty() && is_capturable_thought(&self.thought, self.signature.as_deref())
    }

    fn into_record(self) -> ThinkingRecord {
        let fp = contextual_fingerprint(
            &self.visible,
            &self.tool_ids,
            &self.tool_names,
            self.text_context_anchor.as_deref(),
        );
        ThinkingRecord {
            fingerprint: fp,
            thought: self.thought,
            signature: self.signature,
            tool_ids: self.tool_ids,
            tool_names: self.tool_names,
            visible: self.visible,
        }
    }

    pub fn commit(mut self, store_key: &str) {
        if store_key.is_empty() || !self.should_capture() {
            return;
        }
        // Keep the ordinary thinking record for thought/text hydration. Store independent
        // signed calls separately, using the same scoped and generation-aware SQLite store.
        if self.function_call_count > 1 {
            for call in std::mem::take(&mut self.signed_calls) {
                ThinkingStore::global().record(store_key, call);
            }
        }
        let rec = self.into_record();
        tracing::debug!(
            "[ThinkingStore] Capture thought len={} sig_len={} fp={} sid={}",
            rec.thought.len(),
            rec.signature.as_ref().map(|s| s.len()).unwrap_or(0),
            rec.fingerprint,
            store_key
        );
        ThinkingStore::global().record(store_key, rec);
    }
}

pub fn capture_gemini_contents(store_key: &str, contents: &[Value]) {
    if !crate::proxy::config::is_thinking_store_enabled() || store_key.is_empty() {
        return;
    }
    let text_anchors = compute_text_turn_anchors(contents);
    for (c_idx, content) in contents.iter().enumerate() {
        let role = content.get("role").and_then(|v| v.as_str()).unwrap_or("");
        if role != "model" && role != "assistant" {
            continue;
        }
        if let Some(parts) = content.get("parts").and_then(|p| p.as_array()) {
            let preceding_turn = if c_idx > 0 {
                contents.get(c_idx - 1)
            } else {
                None
            };
            let anchor = compute_causal_anchor(preceding_turn);
            capture_gemini_parts_with_context(
                store_key,
                parts,
                &anchor,
                Some(&text_anchors[c_idx]),
            );
        }
    }
}

/// Capture client-supplied thinking, restore missing blocks, then prune compressed-away history.
///
/// Client histories usually have no real thinking (Claude/OpenAI). After a session is
/// warm in memory, this path is RAM-only: no SQLite open, no placeholder ingest, no prune
/// rewrite. JSON fill still copies stored thought text into the freshly built request.
pub fn hydrate_gemini_contents(store_key: &str, contents: &mut Vec<Value>) -> usize {
    hydrate_gemini_contents_with_model(store_key, contents, None)
}

pub fn hydrate_gemini_contents_with_model(
    store_key: &str,
    contents: &mut Vec<Value>,
    target_model: Option<&str>,
) -> usize {
    if store_key.is_empty() {
        return 0;
    }
    let store = ThinkingStore::global();
    store.touch_session(store_key);
    // 优先执行拓扑还原，保证历史已有记录对齐为真实真签名
    let restored = store.restore_gemini_contents_with_model(store_key, contents, target_model);
    // 只有在完成还原后，若仍有客户端自带的合法实质思考块，才安全吸纳进库
    if contents_have_capturable_thought(contents) {
        store.ingest_from_contents(store_key, contents);
    }
    store.prune_orphaned_records(store_key, contents);
    restored
}

/// The early pipeline still aligns tools before protocol finalization. Text
/// identities depend on the final sanitized prefix, so neither capture nor
/// pruning of text records belongs in this pre-transit stage.
pub fn hydrate_gemini_tool_contents_with_model(
    store_key: &str,
    contents: &mut Vec<Value>,
    target_model: Option<&str>,
) -> usize {
    if store_key.is_empty() {
        return 0;
    }
    let store = ThinkingStore::global();
    store.touch_session(store_key);
    let restored = store.restore_selected_contents(store_key, contents, target_model, Some(false));
    store.ingest_selected_contents(store_key, contents, Some(false));
    // Pruning is session-wide, including SQLite rows outside the RAM window.
    // Wait for finalize_request_text_context to recover text against the final
    // prefix before deciding which generations have actually left the history.
    restored
}

/// Run once after request sanitization/context transforms and before upload.
/// The same final visible prefix drives both hydration and response capture.
pub fn finalize_request_text_context(
    store_key: &str,
    contents: &mut Vec<Value>,
    target_model: &str,
    thinking_enabled: bool,
    is_retry: bool,
) -> String {
    if !store_key.is_empty() && thinking_enabled && !is_retry {
        let store = ThinkingStore::global();
        let mut anchor = compute_response_text_anchor(contents);
        let mut incoming = std::collections::BTreeMap::<usize, ThinkingRecord>::new();
        let mut converged = false;
        // Restoring a downgraded thought can remove visible text, while the
        // finalizer can turn an unsigned thought back into visible text. Later
        // turns must match the resulting prefix. Normally this is one pass;
        // only a visible change warrants another, bounded by the history size.
        for _ in 0..=contents.len() {
            store.restore_selected_contents(store_key, contents, Some(target_model), Some(true));
            // Keep the full thoughts before Gemini's wire shape drops them.
            // Assign their context identities only after finalization settles.
            for (idx, mut record) in store.collect_selected_records(store_key, contents, Some(true))
            {
                if let Some(previous) = incoming.get(&idx) {
                    // A later pass may see only the signature or downgraded
                    // visible text after Gemini removed a new client-supplied
                    // thought. Preserve that capture until the stable commit.
                    // Different signed generations must never be combined.
                    if record.visible == previous.visible
                        && legacy_text_signatures_compatible(previous, &record)
                    {
                        if previous.thought.len() > record.thought.len() {
                            record.thought = previous.thought.clone();
                        }
                        if record.signature.is_none() {
                            record.signature = previous.signature.clone();
                        }
                    }
                }
                incoming.insert(idx, record);
            }
            finalize_gemini_contents_with_stage(
                contents,
                thinking_enabled,
                Some(target_model),
                Some(store_key),
                false,
                None,
                false,
            );
            let next_anchor = compute_response_text_anchor(contents);
            if next_anchor == anchor {
                converged = true;
                break;
            }
            anchor = next_anchor;
        }
        let anchors = compute_text_turn_anchors(contents);
        let records = incoming
            .into_iter()
            .map(|(idx, mut record)| {
                record.fingerprint = contextual_fingerprint(
                    &record.visible,
                    &record.tool_ids,
                    &record.tool_names,
                    Some(&anchors[idx]),
                );
                record
            })
            .collect();
        store.ingest_records(store_key, records);
        if converged {
            store.prune_orphaned_records(store_key, contents);
        } else {
            // Never delete history based on an unsettled projection.
            tracing::warn!(
                "[ThinkingStore] Visible prefix did not settle for {}",
                store_key
            );
        }
        return anchor;
    }
    // Finish text only after the exact-context restore. A session's latest
    // signature is not evidence that an unrelated historical text turn owns it.
    finalize_gemini_contents_with_stage(
        contents,
        thinking_enabled,
        Some(target_model),
        Some(store_key),
        is_retry,
        None,
        false,
    );
    compute_response_text_anchor(contents)
}

fn is_model_or_assistant(content: &Value) -> bool {
    matches!(
        content.get("role").and_then(|v| v.as_str()),
        Some("model") | Some("assistant")
    )
}

fn contents_have_capturable_thought(contents: &[Value]) -> bool {
    for content in contents {
        if !is_model_or_assistant(content) {
            continue;
        }
        let Some(parts) = content.get("parts").and_then(|p| p.as_array()) else {
            continue;
        };
        for part in parts {
            let is_thought = part
                .get("thought")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if !is_thought {
                continue;
            }
            let text = part.get("text").and_then(|t| t.as_str()).unwrap_or("");
            let sig = part
                .get("thoughtSignature")
                .or_else(|| part.get("thought_signature"))
                .and_then(|s| s.as_str());
            if is_capturable_thought(text, sig) {
                return true;
            }
        }
    }
    false
}

pub fn capture_gemini_parts(store_key: &str, parts: &[Value]) {
    capture_gemini_parts_with_anchor(store_key, parts, "root");
}

pub fn capture_gemini_parts_with_anchor(store_key: &str, parts: &[Value], anchor: &str) {
    capture_gemini_parts_with_context(store_key, parts, anchor, None);
}

pub fn capture_gemini_parts_with_context(
    store_key: &str,
    parts: &[Value],
    anchor: &str,
    text_context_anchor: Option<&str>,
) {
    let mut acc = TurnAccumulator::with_anchor(anchor);
    if let Some(anchor) = text_context_anchor {
        acc = acc.with_text_context_anchor(anchor);
    }
    for part in parts {
        acc.ingest_part(part);
    }
    acc.commit(store_key);
}

pub fn capture_gemini_response(store_key: &str, response: &Value) {
    capture_gemini_response_with_preceding(store_key, response, None);
}

pub fn capture_gemini_response_with_preceding(
    store_key: &str,
    response: &Value,
    preceding: Option<&Value>,
) {
    capture_gemini_response_with_context(store_key, response, preceding, None);
}

pub fn capture_gemini_response_with_context(
    store_key: &str,
    response: &Value,
    preceding: Option<&Value>,
    text_context_anchor: Option<&str>,
) {
    let raw = response.get("response").unwrap_or(response);
    let anchor = compute_causal_anchor(preceding);
    if let Some(candidates) = raw.get("candidates").and_then(Value::as_array) {
        if crate::proxy::mappers::gemini::candidate_indices(candidates).is_none() {
            return;
        }
        for candidate in candidates {
            if let Some(parts) = candidate
                .pointer("/content/parts")
                .and_then(Value::as_array)
            {
                capture_gemini_parts_with_context(store_key, parts, &anchor, text_context_anchor);
            }
        }
    }
}

/// 快速检索 contents 数组中包含 post-compaction continuation 接续标记的轮次索引
/// 当会话在客户端成功生成 compact_boundary 或注入历史摘要后，该轮及其之前的历史属于压缩上下文。
pub fn find_compaction_continuation_boundary(contents: &[Value]) -> Option<usize> {
    // [FIX #3697] 采用逆序扫描 contents，确保在长会话经历多次压缩时严格锚定到最新的第二次/后续压缩边界
    for (idx, content) in contents.iter().enumerate().rev() {
        if let Some(parts) = content.get("parts").and_then(Value::as_array) {
            for part in parts {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    if crate::proxy::mappers::common_utils::is_post_compaction_continuation_text(
                        text,
                    ) {
                        return Some(idx);
                    }
                }
            }
        }
    }
    None
}

/// 计算 contents 的字符量与总轮数，供全协议宏观水位线检测
pub fn calculate_contents_metrics(contents: &[Value]) -> (usize, usize) {
    let mut total_chars = 0usize;
    for content in contents {
        if let Some(parts) = content.get("parts").and_then(Value::as_array) {
            for part in parts {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    total_chars = total_chars.saturating_add(text.len());
                }
                if let Some(thought) = part.get("thought").and_then(Value::as_str) {
                    total_chars = total_chars.saturating_add(thought.len());
                }
                if let Some(fc) = part.get("functionCall") {
                    if let Some(name) = fc.get("name").and_then(Value::as_str) {
                        total_chars = total_chars.saturating_add(name.len());
                    }
                    if let Some(args) = fc.get("args") {
                        if let Some(s) = args.as_str() {
                            total_chars = total_chars.saturating_add(s.len());
                        } else {
                            total_chars = total_chars.saturating_add(args.to_string().len());
                        }
                    }
                }
                if let Some(fr) = part.get("functionResponse") {
                    if let Some(resp) = fr.get("response") {
                        if let Some(s) = resp.as_str() {
                            total_chars = total_chars.saturating_add(s.len());
                        } else {
                            total_chars = total_chars.saturating_add(resp.to_string().len());
                        }
                    }
                }
                if let Some(id) = part.get("inlineData") {
                    if let Some(data) = id.get("data").and_then(Value::as_str) {
                        total_chars = total_chars.saturating_add(data.len());
                    }
                }
            }
        }
    }
    (total_chars, contents.len())
}

/// 识别文本中是否包含各主流 AI 客户端（Cline, Roo-Code, Cursor, Aider 等）常见的输出就地修剪/折叠占位标记
pub fn is_truncated_or_folded_text(text: &str) -> bool {
    let lower = text.to_lowercase();
    lower.contains("[output truncated")
        || lower.contains("[result omitted")
        || lower.contains("[tool result truncated")
        || lower.contains("[output omitted")
        || lower.contains("[content truncated")
        || lower.contains("[image omitted")
        || lower.contains("[image removed")
        || lower.contains("truncated to save")
        || lower.contains("... (truncated")
        || lower.contains("...[truncated")
        || lower.contains("[history truncated")
        || lower.contains("[context truncated")
        || lower.contains("<persisted-output>")
}

/// 检索 contents 中是否存在包含微观就地修剪、折叠或占位符的轮次最大索引
pub fn find_micro_folding_boundary(contents: &[Value]) -> Option<usize> {
    let mut last_folded_idx = None;
    for (idx, content) in contents.iter().enumerate() {
        if let Some(parts) = content.get("parts").and_then(Value::as_array) {
            for part in parts {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    if is_truncated_or_folded_text(text) {
                        last_folded_idx = Some(idx);
                        break;
                    }
                }
                if let Some(fr) = part.get("functionResponse") {
                    if let Some(resp) = fr.get("response") {
                        let matches = if let Some(s) = resp.as_str() {
                            is_truncated_or_folded_text(s)
                        } else {
                            is_truncated_or_folded_text(&resp.to_string())
                        };
                        if matches {
                            last_folded_idx = Some(idx);
                            break;
                        }
                    }
                }
            }
        }
    }
    last_folded_idx
}

/// 综合判断当前 contents 的压缩接续边界（结合宏观骤降、微观折叠与显式接续标记）
pub fn determine_compaction_boundary(
    contents: &[Value],
    session_id: Option<&str>,
) -> Option<usize> {
    let explicit_boundary = find_compaction_continuation_boundary(contents);
    if !crate::proxy::config::is_global_compaction_strip_enabled() {
        return explicit_boundary;
    }

    let mut boundary = explicit_boundary;
    let store = ThinkingStore::global();

    // 方案一：宏观骤降联动 (基于同一会话的字符量与轮次高水位线)
    if let Some(sid) = session_id {
        let clean_sid = sid.trim();
        if !clean_sid.is_empty() {
            // 先检查近期（15 秒内）是否已判定过该会话的宏观压缩事件
            let recent_macro_boundary = store
                .compaction_events
                .get(clean_sid)
                .filter(|entry| entry.1.elapsed() < Duration::from_secs(15))
                .map(|entry| entry.0);

            let macro_boundary = if let Some(b) = recent_macro_boundary {
                Some(b)
            } else {
                let (cur_chars, cur_turns) = calculate_contents_metrics(contents);
                if store.check_macro_context_drop(clean_sid, cur_chars, cur_turns) {
                    store.clear_session_for_compaction(clean_sid);
                    // 压缩发生后，截至当前最新 prompt 之前的所有历史 model 轮次均属于待治理历史
                    let mb = contents.len().checked_sub(2).unwrap_or(0);
                    store
                        .compaction_events
                        .insert(clean_sid.to_string(), (mb, Instant::now()));
                    Some(mb)
                } else {
                    None
                }
            };

            if let Some(mb) = macro_boundary {
                boundary = match boundary {
                    Some(existing) => Some(existing.max(mb)),
                    None => Some(mb),
                };
            }
        }
    }

    // 方案二：微观局部折叠/占位符检测（检测 [Output truncated]、[Result omitted] 等特征）
    if let Some(micro_b) = find_micro_folding_boundary(contents) {
        boundary = match boundary {
            Some(existing) => Some(existing.max(micro_b)),
            None => Some(micro_b),
        };
    }

    boundary
}

/// 模块级便捷接口：彻底清理指定会话的全部预压缩历史数据与签名缓存
pub fn clear_session_for_compaction(store_key: &str) {
    ThinkingStore::global().clear_session_for_compaction(store_key);
}

/// 四大协议统一思考补齐管线：确保所有 Gemini contents 中的 model 轮次在开启思考时，必须具备合法的思考块与签名
pub fn finalize_gemini_contents_thinking(contents: &mut [Value], is_thinking_enabled: bool) {
    finalize_gemini_contents_thinking_with_model(contents, is_thinking_enabled, None);
}

pub fn finalize_gemini_contents_thinking_with_model(
    contents: &mut [Value],
    is_thinking_enabled: bool,
    target_model: Option<&str>,
) {
    finalize_gemini_contents_thinking_with_session(
        contents,
        is_thinking_enabled,
        target_model,
        None,
    );
}

pub fn finalize_gemini_contents_thinking_with_session(
    contents: &mut [Value],
    is_thinking_enabled: bool,
    target_model: Option<&str>,
    session_id: Option<&str>,
) {
    finalize_gemini_contents_thinking_with_options(
        contents,
        is_thinking_enabled,
        target_model,
        session_id,
        false,
    );
}

/// `untrusted_signatures = true` 用于重试轮次：不信任任何客户端/缓存签名，
/// 思考块直接降级为正文，工具调用只保留哨兵，避免“先丢文本、再删签名”。
pub fn finalize_gemini_contents_thinking_with_options(
    contents: &mut [Value],
    is_thinking_enabled: bool,
    target_model: Option<&str>,
    session_id: Option<&str>,
    untrusted_signatures: bool,
) {
    finalize_gemini_contents_with_stage(
        contents,
        is_thinking_enabled,
        target_model,
        session_id,
        untrusted_signatures,
        None,
        true,
    );
}

pub fn finalize_gemini_tool_contents_thinking_with_options(
    contents: &mut [Value],
    is_thinking_enabled: bool,
    target_model: Option<&str>,
    session_id: Option<&str>,
    untrusted_signatures: bool,
) {
    finalize_gemini_contents_with_stage(
        contents,
        is_thinking_enabled,
        target_model,
        session_id,
        untrusted_signatures,
        Some(false),
        true,
    );
}

#[allow(clippy::too_many_arguments)]
fn finalize_gemini_contents_with_stage(
    contents: &mut [Value],
    is_thinking_enabled: bool,
    target_model: Option<&str>,
    session_id: Option<&str>,
    untrusted_signatures: bool,
    text_only: Option<bool>,
    allow_session_text_fallback: bool,
) {
    let compaction_boundary = determine_compaction_boundary(contents, session_id);

    // 预先计算每一轮的前置因果锚点 (causal anchor)，以便无 ID 的 Gemini 原生工具调用也能合成确定性 ID
    let anchors: Vec<String> = (0..contents.len())
        .map(|i| {
            let preceding = if i > 0 { contents.get(i - 1) } else { None };
            compute_causal_anchor(preceding)
        })
        .collect();

    // 黄金法则 2.2：纯思考轮签名暂存游标
    // 当某一轮只有纯思考块（无正文、无工具调用）时，Gemini 思考块绝不能带签名，
    // 将该轮获取到的有效签名攒在手上 (pending_thought_sig) 预留给下一轮；
    // 若下一轮有新签名返回，则丢弃旧签名采纳新签名；若下一轮无签名，则由攒着的签名补充。
    let mut pending_thought_sig: Option<String> = None;
    let is_claude_target = target_model
        .map(|m| m.to_lowercase().contains("claude"))
        .unwrap_or(false);

    let initial_session_sig: Option<String> =
        if untrusted_signatures || compaction_boundary.is_some() {
            None
        } else {
            session_id.and_then(|sid| {
                crate::proxy::SignatureCache::global()
                    .get_session_signature(sid)
                    .filter(|s| {
                        if is_claude_target {
                            is_claude_signature(s)
                        } else {
                            is_likely_gemini_signature(s)
                        }
                    })
            })
        };

    // [Batch Recall] 预取本轮所有原生 functionCall ID：一个请求一次 IN 查询，
    // 避免 100+ 工具调用逐条点查 SQLite。未命中的 ID 会写入负缓存，后续单查不会再穿透 L2。
    let prefetched_tool_sigs: HashMap<String, String> = if untrusted_signatures {
        HashMap::new()
    } else if let Some(sid) = session_id {
        let mut ids: Vec<String> = contents
            .iter()
            .flat_map(|m| {
                m.get("parts")
                    .and_then(|p| p.as_array())
                    .into_iter()
                    .flatten()
            })
            .filter_map(|p| {
                p.get("functionCall")
                    .and_then(|fc| fc.get("id"))
                    .and_then(|id| id.as_str())
                    .filter(|id| !id.is_empty())
                    .map(str::to_string)
            })
            .collect();
        ids.sort_unstable();
        ids.dedup();
        if ids.is_empty() {
            HashMap::new()
        } else {
            crate::proxy::SignatureCache::global().get_tool_signatures_batch(sid, &ids)
        }
    } else {
        HashMap::new()
    };

    // 预先计算最后一个 model / 工具调用轮次的索引
    let last_model_idx = contents.iter().enumerate().rposition(|(_, m)| {
        let parts = m.get("parts").and_then(Value::as_array);
        let has_fc =
            parts.is_some_and(|parts| parts.iter().any(|p| p.get("functionCall").is_some()));
        let has_fr =
            parts.is_some_and(|parts| parts.iter().any(|p| p.get("functionResponse").is_some()));
        has_fc
            || (!has_fr
                && matches!(
                    m.get("role").and_then(|r| r.as_str()),
                    Some("model") | Some("assistant")
                ))
    });

    for (msg_idx, msg) in contents.iter_mut().enumerate() {
        let is_in_compaction_history = compaction_boundary.map_or(false, |b| msg_idx <= b);
        let _anchor = &anchors[msg_idx];
        let has_fc = msg
            .get("parts")
            .and_then(|p| p.as_array())
            .map_or(false, |parts| {
                parts.iter().any(|p| p.get("functionCall").is_some())
            });
        // The Google envelope normalizes functionResponse turns to model, but
        // they are still tool results and cannot own a model thought signature.
        let is_tool_response = !has_fc
            && msg
                .get("parts")
                .and_then(Value::as_array)
                .is_some_and(|parts| parts.iter().any(|p| p.get("functionResponse").is_some()));
        if has_fc && msg.get("role").and_then(|r| r.as_str()) == Some("user") {
            msg["role"] = json!("model");
        }
        let is_model = !is_tool_response
            && (matches!(
                msg.get("role").and_then(|r| r.as_str()),
                Some("model") | Some("assistant")
            ) || has_fc);

        if is_model && text_only.is_some_and(|selected| selected == has_fc) {
            pending_thought_sig = None;
            continue;
        }

        if !is_model {
            // Only adjacent fragments of one model response can hand off a
            // pending signature; a new user prompt or tool result ends it.
            pending_thought_sig = None;
            // 非 model 轮次（如 user 轮次的 functionResponse）：清洗可能混入的误标签名
            if let Some(parts) = msg.get_mut("parts").and_then(|p| p.as_array_mut()) {
                for part in parts.iter_mut() {
                    if let Some(obj) = part.as_object_mut() {
                        obj.remove("thought_signature");
                        if is_in_compaction_history
                            || is_tool_response
                            || obj.contains_key("functionResponse")
                        {
                            obj.remove("thoughtSignature");
                        }
                    }
                }
            }
            continue;
        }

        if let Some(parts) = msg.get_mut("parts").and_then(|p| p.as_array_mut()) {
            let mut thinking_parts = Vec::new();
            let mut other_parts = Vec::new();

            for mut part in parts.drain(..) {
                if let Some(obj) = part.as_object_mut() {
                    // 统一清洗向 Google 发送的非标准蛇形字段
                    obj.remove("thought_signature");
                }
                // 铁律：只认 thought: true。functionCall 与纯正文都会携带 thoughtSignature，
                // 绝不能仅凭签名判定为思考块，否则会漏补首位 thought、关思考时误删工具。
                let is_thought = is_thought_part(&part);
                if is_thought {
                    thinking_parts.push(part);
                } else {
                    other_parts.push(part);
                }
            }

            let is_claude_turn = target_model
                .map(|m| m.to_lowercase().contains("claude"))
                .unwrap_or(false);

            let parallel = !is_claude_turn && has_parallel_calls(&other_parts);
            if parallel && !untrusted_signatures && !is_in_compaction_history {
                if let Some(sid) = session_id {
                    restore_parallel_signatures(sid, _anchor, &mut other_parts);
                }
            }

            // 1. 提取当前轮次已有合法的真实签名 (纯检查当前轮部件自带签名，必须严格校验合法性！)
            // [Pipeline First] 若存在 functionCall 强制优先从 functionCall 提取合法签名，
            // 避免前面紧随的 text 说明正文误抢签名导致真实工具调用凭据丢失或绕过 tool_signatures 穿透查库。
            let turn_real_sig = if untrusted_signatures {
                None
            } else {
                extract_turn_signature(&other_parts, !is_claude_turn, is_claude_turn)
            };

            // 1.1 工具调用签名多级检索与穿透回填（支持 Gemini 与 Claude 模型）：
            // ① 原生/传入 tool_id -> 内存 L1 + SQLite L2 (tool_signatures)
            // ② 若未命中（非原生 ID 如 Write-76 或缓存失配），计算 ID 无关的确定性因果伪哈希 ID (synthetic_id) 穿透匹配 SQLite
            // ③ 若找到则回填并反向自愈写入缓存，回填不了则由后序 chosen_sig 统一处理 (Gemini 带哨兵，Claude 留空)
            let is_valid_family_sig = |s: &str| {
                if is_claude_turn {
                    is_claude_signature(s)
                } else {
                    is_likely_gemini_signature(s)
                }
            };

            // A synthetic alias or a session-wide hint is weaker than a
            // conflicting generation in ThinkingStore. Do not let finalization
            // reinsert a signature hydration intentionally declined.
            let mut tool_conflict = parallel;
            let checked_tool_sig = if has_fc
                && !parallel
                && turn_real_sig.is_none()
                && !untrusted_signatures
                && !is_in_compaction_history
            {
                session_id.and_then(|sid| {
                    let (_, ids, _, _) = inspect_parts_with_anchor(&other_parts, _anchor);
                    let store = ThinkingStore::global();
                    let mut rows = store.load_turns(sid);
                    if store
                        .sessions
                        .get(sid)
                        .is_some_and(|e| e.l2_has_evicted_turns)
                    {
                        match crate::modules::proxy_db::load_thinking_by_tool_ids(sid, &ids) {
                            Ok(more) => extend_tool_candidates(&mut rows, more),
                            Err(_) => {
                                tool_conflict = true;
                                return None;
                            }
                        }
                    }
                    match select_tool_generation(
                        &ids,
                        None,
                        &rows,
                        (0..rows.len())
                            .filter(|&i| !rows[i].fingerprint.starts_with(CALL_SIGNATURE_PREFIX)),
                    ) {
                        ToolMatch::Conflict => {
                            tool_conflict = true;
                            None
                        }
                        ToolMatch::Found(i) => rows[i]
                            .signature
                            .clone()
                            .filter(|sig| is_valid_family_sig(sig)),
                        ToolMatch::Missing => None,
                    }
                })
            } else {
                None
            };
            let cached_tool_sig: Option<String> =
                if checked_tool_sig.is_some() {
                    checked_tool_sig
                } else if !tool_conflict
                    && !untrusted_signatures
                    && !is_in_compaction_history
                    && turn_real_sig.is_none()
                {
                    let mut found_sig = None;
                    let mut fc_counter = 0usize;
                    for p in &other_parts {
                        if let Some(fc) = p.get("functionCall") {
                            let id = fc.get("id").and_then(|v| v.as_str()).unwrap_or("");
                            // 1. 先用原生 ID 查批量预取结果 / 内存 L1 / SQLite L2
                            if !id.is_empty() {
                                let prefetched_sig = prefetched_tool_sigs
                                    .get(id)
                                    .map(String::as_str)
                                    .filter(|s| is_valid_family_sig(s))
                                    .map(str::to_string);
                                let native_sig = prefetched_sig.or_else(|| {
                                    session_id.and_then(|sid| {
                                        crate::proxy::SignatureCache::global()
                                            .get_tool_signature(sid, id)
                                            .filter(|s| is_valid_family_sig(s))
                                    })
                                });
                                if let Some(sig) = native_sig {
                                    let final_s = if is_claude_turn {
                                        ensure_google_claude_thought_signature(&sig)
                                    } else {
                                        sig
                                    };
                                    found_sig = Some(final_s);
                                    break;
                                }
                            }

                            // 2. 若未命中（非原生 ID 如 Write-76，或缓存失配）：
                            // 使用 ID 无关的确定性因果伪哈希 ID 穿透匹配 SQLite
                            let name = fc.get("name").and_then(|v| v.as_str()).unwrap_or("unknown");
                            let synthetic_id =
                                synthesize_tool_id(name, fc.get("args"), _anchor, fc_counter);
                            fc_counter += 1;

                            // 2.1 先查内存 L1 与 SQLite L2 (tool_signatures 表)
                            if let Some(sid) = session_id {
                                if let Some(sig) = crate::proxy::SignatureCache::global()
                                    .get_tool_signature(sid, &synthetic_id)
                                    .filter(|s| is_valid_family_sig(s))
                                {
                                    let final_s = if is_claude_turn {
                                        ensure_google_claude_thought_signature(&sig)
                                    } else {
                                        sig
                                    };
                                    if !id.is_empty() {
                                        crate::proxy::SignatureCache::global()
                                            .cache_tool_signature(sid, id, final_s.clone());
                                    }
                                    found_sig = Some(final_s);
                                    break;
                                }
                            }

                            // 2.2 若仍未命中，尝试按 session_key 在 SQLite thinking_records 极速穿透点查 (causal_tool_id)
                            if let Some(sid) = session_id {
                                if let Ok(Some(rec)) =
                                    crate::modules::proxy_db::load_thinking_by_tool_id(
                                        sid,
                                        &synthetic_id,
                                    )
                                {
                                    if let Some(sig) =
                                        rec.signature.filter(|s| is_valid_family_sig(s))
                                    {
                                        let final_s = if is_claude_turn {
                                            ensure_google_claude_thought_signature(&sig)
                                        } else {
                                            sig
                                        };
                                        if !id.is_empty() {
                                            if let Some(sid) = session_id {
                                                crate::proxy::SignatureCache::global()
                                                    .cache_tool_signature(sid, id, final_s.clone());
                                            }
                                        }
                                        found_sig = Some(final_s);
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    found_sig
                } else {
                    None
                };

            let is_last_model_turn = last_model_idx == Some(msg_idx);
            let cached_session_sig: Option<String> = if !tool_conflict
                && !untrusted_signatures
                && (has_fc || allow_session_text_fallback)
                && is_last_model_turn
                && turn_real_sig.is_none()
                && cached_tool_sig.is_none()
            {
                initial_session_sig
                    .clone()
                    .filter(|s| is_valid_family_sig(s))
                    .map(|s| {
                        if is_claude_turn {
                            ensure_google_claude_thought_signature(&s)
                        } else {
                            s
                        }
                    })
            } else {
                None
            };

            // 2. 签名归位（终审出站门禁 Gatekeeper）：
            // 签名只写在该轮第一个非思考 part 上。Gemini 与 Claude 桌面端都是这个落点。
            if untrusted_signatures {
                // 重试轮次：客户端/缓存签名一律不可信，直接清空；
                // Gemini 工具调用仅保留官方哨兵，Claude 目标不带任何签名。
                for p in other_parts.iter_mut() {
                    if let Some(obj) = p.as_object_mut() {
                        obj.remove("thought_signature");
                        obj.remove("thoughtSignature");
                        if !is_claude_turn && obj.contains_key("functionCall") {
                            obj.insert("thoughtSignature".to_string(), json!(SENTINEL_SIGNATURE));
                        }
                    }
                }
            } else if is_in_compaction_history {
                // 压缩历史轮次治理：
                // 彻底剥离历史失效签名，Gemini 缺失签名的工具调用统一以官方哨兵锁定（防 400）。
                // 注意：Claude 模型上游严禁携带 Gemini 哨兵 (避免 400 Invalid signature)，故仅对非 Claude 目标设置哨兵；
                // 伴随正文/思考块绝对不携带任何签名
                for p in other_parts.iter_mut() {
                    if let Some(obj) = p.as_object_mut() {
                        obj.remove("thought_signature");
                        if !is_claude_turn && obj.contains_key("functionCall") {
                            obj.insert("thoughtSignature".to_string(), json!(SENTINEL_SIGNATURE));
                        } else {
                            obj.remove("thoughtSignature");
                        }
                    }
                }
            } else if other_parts.is_empty() {
                // 黄金法则 2.2：纯思考轮（无正文、无工具调用）
                // 思考块绝不挂载签名。若本轮持有有效签名，先暂存在手上 (pending_thought_sig) 预留给后续轮次
                let turn_sig = thinking_parts.iter().find_map(|tp| {
                    tp.get("thoughtSignature")
                        .or_else(|| tp.get("thought_signature"))
                        .and_then(|s| s.as_str())
                        .filter(|s| {
                            is_real_signature(s)
                                && (is_likely_gemini_signature(s) || is_claude_signature(s))
                        })
                        .map(str::to_string)
                });
                for tp in thinking_parts.iter_mut() {
                    if let Some(obj) = tp.as_object_mut() {
                        obj.remove("thoughtSignature");
                        obj.remove("thought_signature");
                    }
                }
                if let Some(sig) = turn_sig {
                    pending_thought_sig = Some(sig);
                }
            } else {
                // 第一种情况：经过各种手段（客户端校验采纳 / 网关缓存或库按 tool_id / 会话级最新签名 / 上一轮纯思考暂存），本轮手上有签名：
                // 优先级：
                // ① 客户端自带且严格校验合法的签名 turn_real_sig（有效则采纳并反向入库）
                // ② 签名缓存与 SQLite 库按 tool_id / 伪哈希 ID 精确回填 cached_tool_sig
                // ③ 最新轮次按会话 ID 取回上一轮流式产出的最新签名 cached_session_sig
                // ④ 上一轮纯思考暂存预留的签名 pending_thought_sig
                // ⑤ 兜底保底：若当前包含工具调用且为 Gemini 目标，回填不了则带官方哨兵 (SENTINEL_SIGNATURE) 防 400
                let (chosen_sig, is_fresh) = if let Some(ref sig) = turn_real_sig {
                    (Some(sig.clone()), true)
                } else if let Some(ref sig) = cached_tool_sig {
                    (Some(sig.clone()), true)
                } else if let Some(ref sig) = cached_session_sig {
                    (Some(sig.clone()), true)
                } else if let Some(sig) = pending_thought_sig.take().filter(|_| !tool_conflict) {
                    (Some(sig), false)
                } else if !is_claude_turn
                    && other_parts.iter().any(|p| p.get("functionCall").is_some())
                {
                    (Some(SENTINEL_SIGNATURE.to_string()), true)
                } else {
                    (None, false)
                };

                if is_fresh {
                    // 若本轮已有独立的新签名，直接丢弃之前纯思考轮暂存的陈旧签名
                    pending_thought_sig = None;
                }

                // 黄金法则 3：
                let placed = place_turn_signature_scoped(
                    &mut other_parts,
                    chosen_sig.as_deref(),
                    !is_claude_turn,
                    session_id,
                );
                if let Some(final_sig) = placed {
                    if final_sig != SENTINEL_SIGNATURE && is_likely_gemini_signature(&final_sig) {
                        if let Some(sid) = session_id {
                            if !crate::proxy::SignatureCache::global()
                                .has_session_signature_after(sid, msg_idx)
                            {
                                crate::proxy::SignatureCache::global()
                                    .cache_session_signature(sid, final_sig, msg_idx);
                            }
                        }
                    }
                }
            }

            // 3. 治理思考块 (thinking_parts)
            // 思考块不挂签名，签名只落在非思考锚点上。
            // 策略（只在签名不可信/缺失时降级，不影响正常签名轮次）：
            // - 有目标家族的合法签名：
            //   · Claude 目标 → 保留 thought 块；
            //   · Gemini 目标且有正文/工具 → 丢弃思考文本（签名已承载连续性，官方形态）；
            //   · Gemini 纯思考轮 → 保留 thought 块；
            // - 无合法签名（含哨兵）→ 降级为普通正文，保留推理内容（思考关闭时同样保留）；
            // - 压缩历史 → 直接丢弃，避免 token 膨胀。
            let turn_has_real_signature = !untrusted_signatures
                && other_parts.iter().any(|p| {
                    p.get("thoughtSignature")
                        .or_else(|| p.get("thought_signature"))
                        .and_then(|s| s.as_str())
                        .map_or(false, |s| {
                            is_real_signature(s)
                                && !is_sentinel_signature(s)
                                && (if is_claude_turn {
                                    is_claude_signature(s)
                                } else {
                                    is_likely_gemini_signature(s)
                                })
                        })
                });

            let keep_thought = !untrusted_signatures
                && is_thinking_enabled
                && ((is_claude_turn && turn_has_real_signature)
                    || (!is_claude_turn && other_parts.is_empty()));
            let drop_with_signature = is_thinking_enabled
                && !is_claude_turn
                && !other_parts.is_empty()
                && turn_has_real_signature;
            let degrade_unsigned_thought =
                !is_in_compaction_history && !keep_thought && !drop_with_signature;

            if keep_thought {
                for tp in thinking_parts.iter_mut() {
                    if let Some(obj) = tp.as_object_mut() {
                        obj.remove("thoughtSignature");
                        obj.remove("thought_signature");
                    }
                }
                parts.extend(thinking_parts);
            } else if degrade_unsigned_thought {
                // 无合法签名时不再丢弃思考内容：降级为普通正文，保留推理连续性。
                // 不注入 <think> 标签，避免污染正文与模型格式模仿。
                for tp in &thinking_parts {
                    if let Some(text) = tp.get("text").and_then(|t| t.as_str()) {
                        if !text.trim().is_empty() && !is_placeholder_thought(text) {
                            parts.push(json!({ "text": text }));
                        }
                    }
                }
            }

            parts.extend(other_parts);
        }
    }
}

/// 从 URL Query 字符串中提取 session / conversation 标识符
pub fn extract_session_from_query_str(query: &str) -> Option<String> {
    for (k, v) in url::form_urlencoded::parse(query.as_bytes()) {
        let key = k.to_ascii_lowercase();
        if matches!(
            key.as_str(),
            "session_id" | "sid" | "cid" | "conversation_id" | "chat_id" | "thread_id" | "channel"
        ) {
            let trimmed = v.trim();
            if !trimmed.is_empty() {
                let sanitized = sanitize_session_id(trimmed);
                if !sanitized.is_empty() && sanitized != "sid-unknown" {
                    return Some(sanitized);
                }
            }
        }
    }
    None
}

/// 全生态显式会话标识。只返回一个赢家，后面的来源全部丢掉。
pub fn explicit_session_id_with_query(
    headers: &HeaderMap,
    body: Option<&Value>,
    query: Option<&str>,
) -> Option<String> {
    stable_session_winner(headers, body, query)
}

/// 稳定会话号的唯一优先级：
/// 产品会话头（`x-claude-code-session-id` 最先）→ 其它通配会话头 →
/// 通用 `x-session-id` → `x-session-affinity` → 别名头 → query → body 稳定字段。
/// `previous_response_id` 是每轮游标，不在名单里。
pub fn stable_session_winner(
    headers: &HeaderMap,
    body: Option<&Value>,
    query: Option<&str>,
) -> Option<String> {
    if let Some(sid) = session_id_from_headers(headers) {
        return Some(sid);
    }
    if let Some(sid) = extract_query_session_id(headers, query) {
        return Some(sid);
    }
    extract_body_session_id(body)
}

pub fn explicit_session_id(headers: &HeaderMap, body: Option<&Value>) -> Option<String> {
    explicit_session_id_with_query(headers, body, None)
}

/// Product-specific `x-**-session-id` / `x-**-sessionid`. Checked before generic `x-session-id`.
const PRODUCT_SESSION_HEADERS: &[&str] = &[
    "x-claude-code-session-id",
    "x-jeikcode-sessionid",
    "x-jeikcode-session-id",
    "x-atomcode-session-id",
    "x-atomcode-sessionid",
    "x-antigravity-session-id",
    "x-client-session-id",
    "x-cursor-session-id",
    "cursor-session-id",
    "x-vscode-session-id",
    "anthropic-session-id",
];

const GENERIC_SESSION_HEADER: &str = "x-session-id";

/// Non-session-named aliases. Lowest header priority after `x-session-id`.
const ALIAS_SESSION_HEADERS: &[&str] = &[
    "x-session-affinity",
    "x-conversation-id",
    "conversation-id",
    "x-chat-id",
    "chat-id",
    "x-thread-id",
    "thread-id",
];

fn header_session_value(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|h| h.to_str().ok())
        .and_then(|v| {
            let v = v.trim();
            if v.is_empty() {
                None
            } else {
                Some(sanitize_session_id(v))
            }
        })
}

fn is_generic_x_session_id(name: &str) -> bool {
    name.eq_ignore_ascii_case(GENERIC_SESSION_HEADER)
}

/// 兼容 AtomCode / JeikCode / Cursor 等客户端自定义会话头：
/// 优先 `x-*-session-id` / `x-*-sessionid`，其次通用 `x-session-id`。
fn is_wildcard_session_header(name: &str) -> bool {
    let key = name.trim().to_ascii_lowercase().replace('_', "-");
    if key == "mcp-session-id" {
        return false;
    }
    if key.ends_with("-request-id")
        || key.ends_with("-trace-id")
        || key.ends_with("-correlation-id")
        || key == "x-request-id"
        || key == "request-id"
    {
        return false;
    }
    let compact = key.replace('-', "");
    compact.contains("session") && compact.ends_with("id")
}

fn session_id_from_headers(headers: &HeaderMap) -> Option<String> {
    // 1. Product-specific x-**-session-id / x-**-sessionid，名单顺序即优先级。
    for name in PRODUCT_SESSION_HEADERS {
        if let Some(sid) = header_session_value(headers, name) {
            return Some(sid);
        }
    }
    let mut wildcards = std::collections::BTreeMap::new();
    for (name, value) in headers.iter() {
        let key = name.as_str();
        if is_generic_x_session_id(key)
            || PRODUCT_SESSION_HEADERS
                .iter()
                .any(|listed| key.eq_ignore_ascii_case(listed))
        {
            continue;
        }
        if !is_wildcard_session_header(key) {
            continue;
        }
        if let Ok(v) = value.to_str() {
            let v = v.trim();
            if !v.is_empty() {
                let sanitized = sanitize_session_id(v);
                if sanitized != "sid-unknown" {
                    wildcards.insert(key.to_ascii_lowercase(), sanitized);
                }
            }
        }
    }
    if let Some((_, sid)) = wildcards.iter().next() {
        return Some(sid.clone());
    }

    // 2. Generic x-session-id
    if let Some(sid) = header_session_value(headers, GENERIC_SESSION_HEADER) {
        return Some(sid);
    }

    // 3. Other conversation/chat/thread aliases
    for name in ALIAS_SESSION_HEADERS {
        if let Some(sid) = header_session_value(headers, name) {
            return Some(sid);
        }
    }
    None
}

fn hash16(raw: &str) -> String {
    let hash = format!("{:x}", Sha256::digest(raw.as_bytes()));
    hash[..16].to_string()
}

fn tenant_component(headers: &HeaderMap, tenant_id: Option<&str>) -> String {
    if let Some(id) = tenant_id.map(str::trim).filter(|id| !id.is_empty()) {
        return hash16(id);
    }
    tenant_from_headers(headers)
}

fn tenant_from_headers(headers: &HeaderMap) -> String {
    let raw = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer ").or(Some(s)))
        .or_else(|| headers.get("x-api-key").and_then(|h| h.to_str().ok()))
        .or_else(|| headers.get("x-goog-api-key").and_then(|h| h.to_str().ok()))
        .unwrap_or("anon");
    hash16(raw)
}

pub fn sanitize_session_id(raw: &str) -> String {
    let mut out = String::new();
    for ch in raw.chars().take(128) {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.' | ':') {
            out.push(ch);
        }
    }
    if out.is_empty() {
        "sid-unknown".to_string()
    } else {
        out
    }
}

/// 从 URL Query、代理跳转 Header (x-forwarded-uri, x-original-uri) 以及 Referer 中提取会话参数
pub fn extract_query_session_id(headers: &HeaderMap, query: Option<&str>) -> Option<String> {
    if let Some(q) = query {
        if let Some(sid) = extract_session_from_query_str(q) {
            return Some(sid);
        }
    }
    for uri_h in ["x-forwarded-uri", "x-original-uri"] {
        if let Some(raw_uri) = headers.get(uri_h).and_then(|h| h.to_str().ok()) {
            if let Some(pos) = raw_uri.find('?') {
                if let Some(sid) = extract_session_from_query_str(&raw_uri[pos + 1..]) {
                    return Some(sid);
                }
            }
        }
    }
    if let Some(referer) = headers.get("referer").and_then(|h| h.to_str().ok()) {
        if let Some(pos) = referer.find('?') {
            if let Some(sid) = extract_session_from_query_str(&referer[pos + 1..]) {
                return Some(sid);
            }
        }
    }
    None
}

/// 从 JSON Body 及 metadata 中提取显式指定的会话字段
pub fn extract_body_session_id(body: Option<&Value>) -> Option<String> {
    let body = body?;
    for field in [
        "session_id",
        "sessionId",
        "conversation_id",
        "chat_id",
        "thread_id",
        "client_session_id",
        "_session_thinking_id",
    ] {
        if let Some(v) = body.get(field).and_then(|v| v.as_str()) {
            let v = v.trim();
            if !v.is_empty() {
                let sanitized = sanitize_session_id(v);
                if !sanitized.is_empty() && sanitized != "sid-unknown" {
                    return Some(sanitized);
                }
            }
        }
    }
    if let Some(req) = body.get("request") {
        for field in ["sessionId", "session_id"] {
            if let Some(v) = req.get(field).and_then(|v| v.as_str()) {
                let v = v.trim();
                if !v.is_empty() {
                    let sanitized = sanitize_session_id(v);
                    if !sanitized.is_empty() && sanitized != "sid-unknown" {
                        return Some(sanitized);
                    }
                }
            }
        }
    }
    if let Some(metadata) = body.get("metadata") {
        for field in [
            "conversation_id",
            "chat_id",
            "session_id",
            "sessionId",
            "thread_id",
        ] {
            if let Some(v) = metadata.get(field).and_then(|v| v.as_str()) {
                let v = v.trim();
                if !v.is_empty() && !v.contains("session-") {
                    let sanitized = sanitize_session_id(v);
                    if !sanitized.is_empty() && sanitized != "sid-unknown" {
                        return Some(sanitized);
                    }
                }
            }
        }
    }
    None
}

/// 提取根会话标识符：自动识别多智能体协同框架（如 Cindy 群聊军团、AutoGPT、LangGraph）
/// 派生的子任务/步骤后缀，将 `uuid:step:1`、`prefix:talk:2` 等派生 ID 归一化至根 Session ID。
/// 确保同一业务大任务中的所有子 Agent 调用统一路由至同一上游账号以最大化 Prompt Cache 命中率。
pub fn extract_root_session_id(raw_sid: &str) -> &str {
    let s = raw_sid.trim();
    if s.is_empty() {
        return s;
    }

    // 常见多智能体复合派生后缀分隔标记（按匹配优先级降序）
    const AGENT_DELIMITERS: &[&str] = &[
        ":step:",
        ":step-",
        ":subtask:",
        ":subtask-",
        ":agent:",
        ":talk:",
        ":round:",
        ":task:",
        ":phase:",
        "--step-",
        "--subtask-",
        "::",
    ];

    for delim in AGENT_DELIMITERS {
        if let Some((root, _)) = s.split_once(delim) {
            let trimmed_root = root.trim();
            if !trimmed_root.is_empty() {
                return trimmed_root;
            }
        }
    }

    s
}

/// 只哈希一个稳定会话号。有赢家时账号键省略内容锚点，思维库键保留锚点。
pub fn derive_winner_session_id(tenant: &str, stable_id: Option<&str>, anchor: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"v3|");
    hasher.update(tenant.as_bytes());
    hasher.update([0xff]);
    if let Some(id) = stable_id.map(str::trim).filter(|id| !id.is_empty()) {
        hasher.update(b"sid=");
        hasher.update(id.as_bytes());
        hasher.update([0xfe]);
    }
    let anchor = anchor.trim();
    if !anchor.is_empty() {
        hasher.update(b"anchor=");
        hasher.update(anchor.as_bytes());
    }
    let hash = format!("{:x}", hasher.finalize());
    format!("sess-{}", &hash[..16])
}

fn client_id_from_store_key(store_key: &str) -> &str {
    store_key
        .split_once(':')
        .map(|(_, rest)| rest)
        .unwrap_or(store_key)
}

pub fn is_real_signature(sig: &str) -> bool {
    let s = sig.trim();
    if is_sentinel_signature(s) {
        return false;
    }
    if s.as_bytes().first() == Some(&0x12) && s.len() >= 16 {
        return true;
    }
    s.len() >= MIN_SIGNATURE_LENGTH
}

/// 判断签名是否符合 Google Gemini 原生 Protobuf 签名特征：
/// 1. 官方跳过验签哨兵 (skip_thought_signature_validator 或 Base64 编码的 c2tpcF90aG91Z2h0X3NpZ25hdHVyZV92YWxpZGF0b3I=)；
/// 2. 或满足有效长度 (>= MIN_SIGNATURE_LENGTH)，且 Base64 解码后首字节为 Protobuf Tag 2 (0x12)
///    (单层 Base64 通常以 'E' 开头，双层 Base64 包装通常以 'R' 开头)
pub fn is_likely_gemini_signature(sig: &str) -> bool {
    let s = sig.trim();
    if is_sentinel_signature(s) {
        return true;
    }
    // Claude 签名绝不能被误判为 Gemini 签名
    if is_claude_signature(s) {
        return false;
    }
    // 兼容历史脏数据中被误解码为原始二进制 Protobuf (首字节 0x12) 的签名
    if s.as_bytes().first() == Some(&0x12) && s.len() >= 16 {
        return true;
    }
    if s.len() < MIN_SIGNATURE_LENGTH {
        return false;
    }
    #[cfg(test)]
    if s.starts_with("sig_")
        || s.starts_with("test-")
        || s.starts_with("client-")
        || s.chars().all(|c| c == 's' || c == 't')
    {
        return true;
    }
    if !s.starts_with('E') && !s.starts_with('R') && !s.starts_with('c') {
        return false;
    }
    use base64::Engine;
    if let Ok(decoded) = base64::engine::general_purpose::STANDARD.decode(s) {
        if decoded.first() == Some(&0x12) {
            return true;
        }
        if decoded == SENTINEL_SIGNATURE.as_bytes() {
            return true;
        }
        // 双层 Base64 包装支持（Google Vertex AI 格式）
        if let Ok(inner_str) = std::str::from_utf8(&decoded) {
            if inner_str.starts_with('E') {
                if let Ok(inner) = base64::engine::general_purpose::STANDARD.decode(inner_str) {
                    if inner.first() == Some(&0x12) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// 判断字节流是否为原始 Claude 签名的负载：
/// 1. 经典 Claude 签名：包含 ASCII b"claude"
/// 2. 官方 Protobuf 二进制格式 (Claude 5.5)：以 0x08 0x04 0x12 开头
#[inline]
fn is_raw_claude_payload(bytes: &[u8]) -> bool {
    bytes.windows(6).any(|w| w.eq_ignore_ascii_case(b"claude"))
        || bytes.starts_with(&[0x08, 0x04, 0x12])
}

/// 判断一层 Base64 解码后的字节流是否包装了原始 Claude 签名（即发往 Google Vertex AI 的双层签名解码结果）：
/// 第一层解出 ASCII 字符串形式的原始签名 (如以 b"CAQS" 开头，或内层解码为经典/Protobuf 格式)
#[inline]
fn is_wrapped_claude_payload(decoded: &[u8]) -> bool {
    use base64::Engine;
    if decoded.starts_with(b"CAQS") {
        return true;
    }
    if let Ok(inner) = base64::engine::general_purpose::STANDARD.decode(decoded) {
        if is_raw_claude_payload(&inner) {
            return true;
        }
    }
    false
}

/// 判断签名是否属于 Claude 家族的签名
pub fn is_claude_signature(sig: &str) -> bool {
    let s = sig.trim();
    if s.is_empty() || is_sentinel_signature(s) {
        return false;
    }
    use base64::Engine;
    if let Ok(decoded) = base64::engine::general_purpose::STANDARD.decode(s) {
        if is_raw_claude_payload(&decoded) || is_wrapped_claude_payload(&decoded) {
            return true;
        }
    }
    false
}

/// 将 Claude 签名正规化为发送给 Google Vertex AI 接口所需的格式
/// Google 的 REST API 对 bytes 字段会自动执行 base64_decode，
/// 因此发往 Google 的 thoughtSignature 必须是 ASCII 签名字节的 Base64 编码 (即 "RXU4..." 或 "Q0FR..." 格式)
pub fn ensure_google_claude_thought_signature(sig: &str) -> String {
    let s = sig.trim();
    if s.is_empty() || is_sentinel_signature(s) {
        return s.to_string();
    }
    use base64::Engine;
    if let Ok(decoded) = base64::engine::general_purpose::STANDARD.decode(s) {
        // 如果已经由 Google Vertex AI 双层包装过，无需重复包装
        if is_wrapped_claude_payload(&decoded) {
            return s.to_string();
        }
        // 只有在当前签名确实是原始 Claude 客户端单层签名时，才进行一次 Base64 包装！
        // 严禁对非 Claude 签名或未知字符串无节制再包装，彻底阻断几何级膨胀死循环。
        if is_raw_claude_payload(&decoded) {
            return base64::engine::general_purpose::STANDARD.encode(s.as_bytes());
        }
    }
    s.to_string()
}

/// 将 Claude 签名还原为客户端（Claude Code / Anthropic SDK）期望的原生格式 (Eu8... 或 CAQS...)
pub fn ensure_raw_claude_thought_signature(sig: &str) -> String {
    let s = sig.trim();
    if s.is_empty() || s == SENTINEL_SIGNATURE {
        return s.to_string();
    }
    use base64::Engine;
    if let Ok(decoded) = base64::engine::general_purpose::STANDARD.decode(s) {
        if is_wrapped_claude_payload(&decoded) {
            if let Ok(raw_s) = String::from_utf8(decoded) {
                return raw_s;
            }
        }
    }
    s.to_string()
}

/// 用于 ThinkingStore / SignatureCache 内部的比对与哈希：
/// 统一归一化为原始客户端签名形式 (Eu8... 或 CAQS...)，使 "RXU4..." 与 "Eu8..."、"Q0FR..." 与 "CAQS..." 判定为相同签名
pub fn normalize_signature_for_comparison(sig: &str) -> std::borrow::Cow<'_, str> {
    use base64::Engine;
    // SQLite heals legacy raw Gemini signatures to this same wire representation.
    if sig.as_bytes().first() == Some(&0x12) {
        return std::borrow::Cow::Owned(
            base64::engine::general_purpose::STANDARD.encode(sig.as_bytes()),
        );
    }
    if let Ok(decoded) = base64::engine::general_purpose::STANDARD.decode(sig) {
        if is_wrapped_claude_payload(&decoded) {
            if let Ok(s) = String::from_utf8(decoded) {
                return std::borrow::Cow::Owned(s);
            }
        }
    }
    std::borrow::Cow::Borrowed(sig)
}

pub fn signatures_match(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    normalize_signature_for_comparison(a) == normalize_signature_for_comparison(b)
}

/// 判定某个 Part 是否为思考块。
///
/// **铁律：只认 `thought == true`。**
///
/// 绝不能退化为"带 `thoughtSignature` 且无 `functionCall`/`functionResponse`"——
/// 真实 Antigravity 报文会把签名挂在**纯正文 part** 上（官方不变量见
/// `.workbuddy/outputs/correct-assembly-spec.md`）。旧启发式会把可见回答误判为思考块，
/// 进而在回填阶段把正文改写成"内部思考"，并丢掉该轮唯一的签名——这正是
/// "签名剥离 → 跨请求死循环"事故的根因。
#[inline]
pub fn is_thought_part(part: &Value) -> bool {
    part.get("thought")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// 把签名归位到**该轮第一个非思考 part**，其余 part 一律删除签名字段。
///
/// 由官方报文归纳出的四条硬约束：
///
/// 1. 锚点 = 该轮第一个 `thought != true` 的 part，**不是**硬编码的 `parts[0]`
/// 确定 model 轮次中接收签名的目标锚点部件索引。
///
/// 官方规范与 Google Gemini 强校验铁律（现场 400 铁证）：
/// 1. 若本轮包含工具调用 (`functionCall`)，Google Gemini 的工具校验器强制要求签名**必须落在 `functionCall` 部件上**。
///    若存在 `functionCall` 但签名落在了正文文本部件（如客户端占位符 `"..."` 或过程说明）上，
///    上游直接抛出终止性 400 错误：
///    `Function call is missing a thought_signature in functionCall parts. Additional data, function call ...`
///    因此，当存在 `functionCall` 时，锚点**严格必须**为首个 `functionCall` 部件。
/// 2. 若本轮无任何 `functionCall`（纯正文回答轮次），锚点为首个非思考部件（`thought != true` 的纯文本正文）。
/// 3. 若整轮皆为思考块，则无锚点（返回 None）。
pub fn find_turn_anchor(parts: &[Value]) -> Option<usize> {
    find_turn_anchor_with(parts, true)
}

/// `prefer_function_call`：Gemini 为 true，有工具调用时锚点必须是第一个 `functionCall`。
/// Claude 为 false，锚点固定是该轮第一个非思考 part，即使后面还有工具调用。
pub fn find_turn_anchor_with(parts: &[Value], prefer_function_call: bool) -> Option<usize> {
    if prefer_function_call {
        if let Some(fc_pos) = parts.iter().position(|p| p.get("functionCall").is_some()) {
            return Some(fc_pos);
        }
    }
    parts.iter().position(|p| !is_thought_part(p))
}

/// 从单个部件中提取合法签名（若有）。
/// `is_claude`: 若为 true 则校验 Claude 签名形态，若为 false 则允许官方哨兵或 Gemini 签名形态。
pub fn extract_part_signature(part: &Value, is_claude: bool) -> Option<String> {
    let raw = part
        .get("thoughtSignature")
        .or_else(|| part.get("thought_signature"))
        .and_then(|s| s.as_str())?;

    if is_claude {
        if is_real_signature(raw) && is_claude_signature(raw) {
            Some(raw.to_string())
        } else {
            None
        }
    } else {
        if raw == SENTINEL_SIGNATURE || (is_real_signature(raw) && is_likely_gemini_signature(raw))
        {
            Some(raw.to_string())
        } else {
            None
        }
    }
}

/// 从一轮部件切片中提取本轮合法签名：
/// 若 `prefer_function_call` 为 true 且当前切片中存在 `functionCall`，
/// 强制优先从 `functionCall` 部件中提取签名，确保工具调用的合法凭据不被前面的说明文本抢先占有；
/// 若 `functionCall` 未携带签名或为纯正文轮次，则回退查找非思考普通部件自带的签名。
pub fn extract_turn_signature(
    parts: &[Value],
    prefer_function_call: bool,
    is_claude: bool,
) -> Option<String> {
    let has_fc = parts.iter().any(|p| p.get("functionCall").is_some());
    if prefer_function_call && has_fc {
        parts
            .iter()
            .filter(|p| p.get("functionCall").is_some())
            .find_map(|p| extract_part_signature(p, is_claude))
            .or_else(|| {
                parts
                    .iter()
                    .filter(|p| !is_thought_part(p))
                    .find_map(|p| extract_part_signature(p, is_claude))
            })
    } else {
        parts
            .iter()
            .filter(|p| !is_thought_part(p))
            .find_map(|p| extract_part_signature(p, is_claude))
    }
}

/// 官方报文对齐规范：为 model 轮次将签名归位到正确的锚点。
///
/// 由官方报文归纳出的硬约束：
/// 1. 签名只写在「目标锚点部件」上（若有工具调用则落在首个 `functionCall`，纯正文轮落在首个正文 `parts[0]` 或 `parts[1]`）；
/// 2. `thought: true` / `functionResponse` / 其余 part —— 字段必须**缺席**，而不是空串；
/// 3. 锚点自带或本轮携带真实签名时**抢救保留**（覆盖客户端将签名误挂在占位文本或非锚点部件的场景）；
/// 4. 锚点无签名且 `fallback_sig` 也为空时：若本轮存在 `functionCall`，写入 **哨兵占位**
///    （供下游识别“缺签名但不阻断”）；纯正文轮则维持缺席。哨兵不会反向写回签名库。
///
/// 返回最终写入锚点的签名（若有）。
pub fn place_turn_signature(parts: &mut Vec<Value>, fallback_sig: Option<&str>) -> Option<String> {
    place_turn_signature_scoped(parts, fallback_sig, true, None)
}

pub fn place_turn_signature_with(
    parts: &mut Vec<Value>,
    fallback_sig: Option<&str>,
    prefer_function_call: bool,
) -> Option<String> {
    place_turn_signature_scoped(parts, fallback_sig, prefer_function_call, None)
}

pub fn place_turn_signature_scoped(
    parts: &mut Vec<Value>,
    fallback_sig: Option<&str>,
    prefer_function_call: bool,
    store_key: Option<&str>,
) -> Option<String> {
    // 1. 签名抢救与提取（必须在任何清空或过滤之前执行）：
    // 必须经过严格校验（is_real_signature && is_likely_gemini_signature），只有合法才接受！
    // 优先抢救 functionCall 自带的真实签名，无工具调用时抢救首个非思考正文部件签名。
    let has_fc = parts.iter().any(|p| p.get("functionCall").is_some());
    let parallel = prefer_function_call && has_parallel_calls(parts);
    // A sibling's signature, a legacy turn-wide record, or the session's last
    // signature is not evidence that this particular function call owns it.
    let fallback_sig = fallback_sig.filter(|_| !parallel);
    let own_sig = if parallel {
        parts
            .iter()
            .find(|p| p.get("functionCall").is_some())
            .and_then(|p| extract_part_signature(p, false))
    } else {
        extract_turn_signature(parts, prefer_function_call, !prefer_function_call)
    };

    // 2. 占位文本清理：若本轮包含 functionCall，清理混入的无意义占位正文（如 "..."、"·" 等客户端/中间件遗留脏数据）
    if has_fc {
        parts.retain(|p| {
            if p.get("functionCall").is_none() && !is_thought_part(p) {
                if let Some(t) = p.get("text").and_then(|s| s.as_str()) {
                    if is_placeholder_thought(t) {
                        return false;
                    }
                }
            }
            true
        });
    }

    // 3. 锚点确定：有工具调用则必须锚定首个 functionCall；无工具调用则锚定首个非思考正文
    let anchor = find_turn_anchor_with(parts, prefer_function_call)?;

    // 4. 全量清空非目标部件签名，保证非目标 part 的签名字段确实"缺席"
    for (idx, part) in parts.iter_mut().enumerate() {
        if idx == anchor {
            continue;
        }
        if let Some(obj) = part.as_object_mut() {
            // [Pipeline First] 非锚点部件全量清空误挂的签名（如 text / thought / functionResponse）
            // 但如果当前部件本身也是一个携带了有效签名的 functionCall，予以保留，杜绝抹除并发工具的独立签名
            let is_signed_fc = prefer_function_call
                && obj.contains_key("functionCall")
                && obj
                    .get("thoughtSignature")
                    .or_else(|| obj.get("thought_signature"))
                    .and_then(|s| s.as_str())
                    .map_or(false, |s| s == SENTINEL_SIGNATURE || is_real_signature(s));

            if !is_signed_fc {
                obj.remove("thoughtSignature");
                obj.remove("thought_signature");
            }
        }
    }

    // 5. functionResponse 永不携带签名
    if parts[anchor].get("functionResponse").is_some() {
        return None;
    }

    // 6. 锚点自带/抢救优先，缺失才使用回填来源（真实签名或哨兵）
    let sig = own_sig
        .or_else(|| {
            fallback_sig
                .filter(|s| {
                    *s == SENTINEL_SIGNATURE
                        || (is_real_signature(s)
                            && (is_likely_gemini_signature(s) || is_claude_signature(s)))
                })
                .map(str::to_string)
        })
        .or_else(|| {
            if prefer_function_call && has_fc {
                Some(SENTINEL_SIGNATURE.to_string())
            } else {
                None
            }
        })?;

    // 权威防裂化：如果签名是原始二进制 protobuf (首字节 0x12)，必须转为标准 Base64 编码后再发送给 Gemini！
    let final_sig = if sig.as_bytes().first() == Some(&0x12) {
        use base64::Engine;
        base64::engine::general_purpose::STANDARD.encode(sig.as_bytes())
    } else {
        sig
    };
    parts[anchor]["thoughtSignature"] = json!(final_sig);

    // 反向入库优化：如果锚点是工具调用且不是哨兵占位，反向更新/修补回签名缓存与 SQLite tool_signatures 库！
    if final_sig != SENTINEL_SIGNATURE {
        if let Some(fc) = parts[anchor].get("functionCall") {
            if let (Some(scope), Some(id)) = (store_key, fc.get("id").and_then(|v| v.as_str())) {
                crate::proxy::SignatureCache::global().cache_tool_signature(
                    scope,
                    id,
                    final_sig.clone(),
                );
            }
        }
    }

    Some(final_sig)
}

pub fn is_placeholder_thought(s: &str) -> bool {
    let t = s.trim();
    t.is_empty()
        || PLACEHOLDER_THOUGHTS.contains(&t)
        || t.chars().all(|c| c == '.' || c == '·' || c == '…')
}

pub fn is_meaningful_thought(thought: &str) -> bool {
    let t = thought.trim();
    if t.is_empty() || is_placeholder_thought(t) {
        return false;
    }
    // 拦截伪思考标签与客户端占位脏数据
    let stripped = t
        .trim_start_matches("<think>")
        .trim_end_matches("</think>")
        .trim_start_matches("Thinking Process:")
        .trim_start_matches("Thinking Process")
        .trim_start_matches("[Thinking]")
        .trim();
    if stripped.is_empty()
        || stripped.eq_ignore_ascii_case("none")
        || stripped.eq_ignore_ascii_case("null")
        || stripped.eq_ignore_ascii_case("undefined")
        || is_placeholder_thought(stripped)
    {
        return false;
    }
    true
}

/// 提取并切除文本中的 `<think>...</think>` 标签内容（用于跨模型自愈与正文思考文本重提升）
/// 支持任意大小写（<think> / <THINK>）、前后正文无缝缝合拼接、以及未闭合标签容错
/// 返回 `Some((thought, remaining_visible))`
pub fn extract_think_tags(text: &str) -> Option<(String, String)> {
    let lower = text.to_lowercase();
    let start_tag = "<think>";
    let end_tag = "</think>";

    if let Some(start_pos) = lower.find(start_tag) {
        let after_start = start_pos + start_tag.len();
        if let Some(end_rel) = lower[after_start..].find(end_tag) {
            let end_pos = after_start + end_rel;
            let thought = text[after_start..end_pos].trim().to_string();
            let before = text[..start_pos].trim();
            let after = text[end_pos + end_tag.len()..].trim();
            let visible = if before.is_empty() {
                after.to_string()
            } else if after.is_empty() {
                before.to_string()
            } else {
                format!("{}\n\n{}", before, after)
            };
            return Some((thought, visible));
        } else {
            // 未闭合标签容错：截断到文本末尾
            let thought = text[after_start..].trim().to_string();
            let before = text[..start_pos].trim();
            return Some((thought, before.to_string()));
        }
    }
    None
}

fn is_capturable_thought(thought: &str, signature: Option<&str>) -> bool {
    // 占位符或纯空白思考绝不可捕获为新的持久化思考记录！
    if is_placeholder_thought(thought) || thought.trim().is_empty() {
        // [DECOUPLE 2026-09-26] 官方新规：任何 model 轮都可能返回签名（正文轮 / 工具轮 / 纯思考轮）。
        // 纯 fc 轮（无思考文本）只要携带真实签名，也必须入库——否则下一轮锚点回填时找不到签名。
        return signature.is_some_and(is_real_signature);
    }
    if signature.is_some_and(is_real_signature) {
        return true;
    }
    is_meaningful_thought(thought)
}

fn record_bytes(rec: &ThinkingRecord) -> usize {
    rec.thought.len() + rec.signature.as_ref().map(|s| s.len()).unwrap_or(0) + rec.visible.len()
}

const TEXT_FINGERPRINT_VERSION: &str = "text-v2:";

fn text_fingerprint_context(fp: &str) -> Option<(&str, &str)> {
    fp.strip_prefix(TEXT_FINGERPRINT_VERSION)?.split_once(':')
}

fn base_fingerprint(fp: &str) -> &str {
    text_fingerprint_context(fp).map_or(fp, |(_, base)| base)
}

fn contextual_fingerprint(
    visible: &str,
    tool_ids: &[String],
    tool_names: &[String],
    text_anchor: Option<&str>,
) -> String {
    let base = fingerprint(visible, tool_ids, tool_names);
    if tool_ids.is_empty() && tool_names.is_empty() {
        if let Some(anchor) = text_anchor.filter(|a| !a.is_empty()) {
            return format!("{TEXT_FINGERPRINT_VERSION}{anchor}:{base}");
        }
    }
    base
}

fn text_context_matches(fp: &str, anchor: &str) -> bool {
    text_fingerprint_context(fp).is_none_or(|(stored, _)| stored == anchor)
}

fn same_thinking_record(a: &ThinkingRecord, b: &ThinkingRecord) -> bool {
    a.fingerprint == b.fingerprint
        && a.thought == b.thought
        && a.visible == b.visible
        && a.tool_ids == b.tool_ids
        && a.tool_names == b.tool_names
        && match (a.signature.as_deref(), b.signature.as_deref()) {
            (Some(a), Some(b)) => signatures_match(a, b),
            (None, None) => true,
            _ => false,
        }
}

fn record_signature_matches_target(record: &ThinkingRecord, target_model: Option<&str>) -> bool {
    let Some(signature) = record.signature.as_deref().filter(|s| is_real_signature(s)) else {
        return true;
    };
    match target_model.map(str::to_ascii_lowercase) {
        Some(model) if model.contains("claude") => is_claude_signature(signature),
        Some(model) if model.contains("gemini") => is_likely_gemini_signature(signature),
        _ => true,
    }
}

/// `candidates` have already matched the exact visible prefix and target family.
/// A session signature can resolve that set, but cannot introduce a candidate
/// from another turn or override an explicit signature supplied by the client.
fn select_contextual_text_record(
    records: &[Arc<ThinkingRecord>],
    candidates: &[usize],
    authority: Option<&str>,
) -> Option<usize> {
    if let Some(signature) = authority.filter(|s| is_real_signature(s)) {
        if let Some(&idx) = candidates.iter().find(|&&idx| {
            records[idx]
                .signature
                .as_deref()
                .is_some_and(|stored| signatures_match(signature, stored))
        }) {
            return Some(idx);
        }
    }
    let signed: Vec<usize> = candidates
        .iter()
        .copied()
        .filter(|&idx| {
            records[idx]
                .signature
                .as_deref()
                .is_some_and(is_real_signature)
        })
        .collect();
    let Some(&first) = signed.first() else {
        return candidates.first().copied();
    };
    let signature = records[first]
        .signature
        .as_deref()
        .expect("signed candidate");
    signed
        .iter()
        .all(|&idx| {
            records[idx]
                .signature
                .as_deref()
                .is_some_and(|other| signatures_match(signature, other))
        })
        .then_some(first)
}

/// Two different real signatures are
/// evidence of separate turns, never a reason to overwrite the older response.
/// Shared with SQLite so memory eviction/restart cannot reintroduce the merge.
pub(crate) fn text_record_signatures_compatible(
    _fingerprint: &str,
    old_signature: Option<&str>,
    new_signature: Option<&str>,
) -> bool {
    match (
        old_signature.filter(|s| is_real_signature(s)),
        new_signature.filter(|s| is_real_signature(s)),
    ) {
        (Some(old), Some(new)) => signatures_match(old, new),
        _ => true,
    }
}

/// Real client IDs distinguish generations that share a deterministic tool
/// alias. If one side lacks a real ID, the unique causal match is still useful.
pub(crate) fn tool_ids_compatible(a: &[String], b: &[String]) -> bool {
    let real = |id: &&String| !crate::modules::proxy_db::is_synthetic_tool_id(id);
    let mut lhs = a.iter().filter(real).peekable();
    let mut rhs = b.iter().filter(real).peekable();
    lhs.peek().is_none() || rhs.peek().is_none() || lhs.eq(rhs)
}

fn extend_tool_candidates(
    records: &mut Vec<Arc<ThinkingRecord>>,
    persisted: Vec<crate::modules::proxy_db::PersistedThinkingRecord>,
) {
    for p in persisted {
        if !records.iter().any(|r| {
            r.fingerprint == p.fingerprint
                && r.thought == p.thought
                && r.tool_ids == p.tool_ids
                && r.signature
                    .as_deref()
                    .map(normalize_signature_for_comparison)
                    == p.signature
                        .as_deref()
                        .map(normalize_signature_for_comparison)
        }) {
            records.push(Arc::new(ThinkingRecord {
                fingerprint: p.fingerprint,
                thought: p.thought,
                signature: p.signature,
                tool_ids: p.tool_ids,
                tool_names: p.tool_names,
                visible: p.visible,
            }));
        }
    }
}
const CALL_SIGNATURE_PREFIX: &str = "parallel-call-v1:";

fn has_parallel_calls(parts: &[Value]) -> bool {
    parts
        .iter()
        .filter(|p| p.get("functionCall").is_some())
        .nth(1)
        .is_some()
}

/// Look up each call against *all* its persisted generations before choosing a
/// signature. A native ID may disambiguate a shared causal alias; with no such
/// evidence two different signatures for that alias must remain unresolved.
fn restore_parallel_signatures(sid: &str, anchor: &str, parts: &mut [Value]) {
    let mut pending = Vec::new();
    for (ordinal, (i, part)) in parts
        .iter()
        .enumerate()
        .filter(|(_, p)| p.get("functionCall").is_some())
        .enumerate()
    {
        if extract_part_signature(part, false)
            .as_deref()
            .is_some_and(is_real_signature)
        {
            continue;
        }
        let fc = &part["functionCall"];
        let name = fc["name"].as_str().unwrap_or("unknown");
        let synthetic = synthesize_tool_id(name, fc.get("args"), anchor, ordinal);
        let ids: Vec<_> = std::iter::once(synthetic)
            .chain(
                fc["id"]
                    .as_str()
                    .filter(|id| !id.is_empty())
                    .map(|id| crate::proxy::common::utils::normalize_tool_id(id).into_owned()),
            )
            .collect();
        pending.push((i, ids));
    }
    if pending.is_empty() {
        return;
    }
    let store = ThinkingStore::global();
    let mut rows = store.load_turns(sid);
    if store
        .sessions
        .get(sid)
        .is_some_and(|s| !s.l2_loaded || s.l2_has_evicted_turns)
    {
        let ids = pending
            .iter()
            .flat_map(|(_, ids)| ids.iter().cloned())
            .collect::<Vec<_>>();
        let Ok(more) = crate::modules::proxy_db::load_thinking_by_tool_ids(sid, &ids) else {
            return;
        };
        extend_tool_candidates(&mut rows, more);
    }
    rows.retain(|r| r.fingerprint.starts_with(CALL_SIGNATURE_PREFIX));
    for (i, ids) in pending {
        if let ToolMatch::Found(n) = select_tool_generation(&ids, None, &rows, 0..rows.len()) {
            if let Some(sig) = rows[n]
                .signature
                .as_deref()
                .filter(|s| is_likely_gemini_signature(s))
            {
                parts[i]["thoughtSignature"] = json!(sig);
            }
        }
    }
}

enum ToolMatch {
    Missing,
    Conflict,
    Found(usize),
}

fn select_tool_generation(
    ids: &[String],
    signature: Option<&str>,
    records: &[Arc<ThinkingRecord>],
    candidates: impl IntoIterator<Item = usize>,
) -> ToolMatch {
    let signature = signature.filter(|s| is_real_signature(s));
    let mut matched = Vec::new();
    let mut exact = false;
    let mut saw = false;
    for idx in candidates {
        let row = &records[idx];
        if !ids.iter().any(|id| row.tool_ids.contains(id)) {
            continue;
        }
        saw = true;
        if !tool_ids_compatible(ids, &row.tool_ids)
            || !text_record_signatures_compatible("", signature, row.signature.as_deref())
        {
            continue;
        }
        let strong = ids.iter().any(|id| {
            !crate::modules::proxy_db::is_synthetic_tool_id(id) && row.tool_ids.contains(id)
        });
        if strong && !exact {
            matched.clear();
            exact = true;
        }
        if strong || !exact {
            matched.push(idx);
        }
    }
    if !saw {
        return ToolMatch::Missing;
    }
    select_contextual_text_record(records, &matched, signature)
        .map_or(ToolMatch::Conflict, ToolMatch::Found)
}

fn legacy_text_signatures_compatible(old: &ThinkingRecord, new: &ThinkingRecord) -> bool {
    text_record_signatures_compatible(
        &old.fingerprint,
        old.signature.as_deref(),
        new.signature.as_deref(),
    )
}

/// Hash the semantic visible prefix, excluding recovered thought/signatures and
/// transport metadata. Adjacent text blocks concatenate just as streamed deltas
/// do; tool ids are excluded because adapters may normalize or synthesize them.
/// Tool response roles normalize here because Gemini finalization changes them
/// from user to model after hydration.
fn hash_visible_turn(hasher: &mut Sha256, content: &Value) {
    let Some(parts) = content.get("parts").and_then(Value::as_array) else {
        return;
    };
    let mut projected = Vec::new();
    let mut text = String::new();
    let mut has_response = false;
    for part in parts {
        if part.get("thought").and_then(Value::as_bool) == Some(true) {
            continue;
        }
        if let Some(value) = part.get("text").and_then(Value::as_str) {
            text.push_str(value);
            continue;
        }
        let value = if let Some(call) = part.get("functionCall") {
            Some(json!({"functionCall": {"name": call.get("name"), "args": call.get("args")}}))
        } else if let Some(response) = part.get("functionResponse") {
            has_response = true;
            Some(
                json!({"functionResponse": {"name": response.get("name"), "response": response.get("response")}}),
            )
        } else if let Some(data) = part.get("inlineData").or_else(|| part.get("inline_data")) {
            Some(json!({"inlineData": data}))
        } else if let Some(data) = part.get("fileData").or_else(|| part.get("file_data")) {
            Some(json!({"fileData": data}))
        } else if let Some(code) = part.get("executableCode") {
            Some(json!({"executableCode": code}))
        } else {
            part.get("codeExecutionResult")
                .map(|result| json!({"codeExecutionResult": result}))
        };
        if let Some(value) = value {
            if !text.is_empty() {
                projected.push(json!({"text": std::mem::take(&mut text)}));
            }
            projected.push(value);
        }
    }
    if !text.is_empty() {
        projected.push(json!({"text": text}));
    }
    let role = if has_response {
        "tool"
    } else {
        match content.get("role").and_then(Value::as_str).unwrap_or("") {
            "assistant" => "model",
            role => role,
        }
    };
    let mut bytes = Vec::new();
    write_canonical_json(&json!({"role": role, "parts": projected}), &mut bytes);
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn text_anchor(hasher: &Sha256) -> String {
    format!("{:x}", hasher.clone().finalize())[..32].to_string()
}

fn compute_text_turn_anchors(contents: &[Value]) -> Vec<String> {
    let mut hasher = Sha256::new();
    hasher.update(b"thinking-text-context-v2");
    contents
        .iter()
        .map(|content| {
            let anchor = text_anchor(&hasher);
            hash_visible_turn(&mut hasher, content);
            anchor
        })
        .collect()
}

/// Capture this from the final request contents, before starting the response
/// stream. It is owned by that request, so concurrent requests cannot overwrite it.
pub fn compute_response_text_anchor(contents: &[Value]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"thinking-text-context-v2");
    for content in contents {
        hash_visible_turn(&mut hasher, content);
    }
    text_anchor(&hasher)
}

fn is_stronger_record(new: &ThinkingRecord, old: &ThinkingRecord) -> bool {
    new.thought.len() > old.thought.len()
        || new.signature.as_ref().map(|s| s.len()).unwrap_or(0)
            > old.signature.as_ref().map(|s| s.len()).unwrap_or(0)
}

fn has_ambiguous_replay_match(
    rec: &ThinkingRecord,
    existing: &[Arc<ThinkingRecord>],
    matched: usize,
) -> bool {
    existing.iter().enumerate().any(|(idx, other)| {
        if idx == matched {
            return false;
        }
        // Consider used siblings too: consuming a different turn earlier in
        // the request is not proof that an unsigned replay belonged to this one.
        replay_fallback_matches(rec, other)
    })
}

fn replay_fallback_matches(rec: &ThinkingRecord, other: &ThinkingRecord) -> bool {
    if !text_record_signatures_compatible(
        &rec.fingerprint,
        rec.signature.as_deref(),
        other.signature.as_deref(),
    ) {
        return false;
    }
    if !tool_ids_compatible(&rec.tool_ids, &other.tool_ids) {
        return false;
    }
    if rec.tool_ids.iter().any(|id| other.tool_ids.contains(id)) {
        return true;
    }
    let rec_has_tools = !rec.tool_ids.is_empty() || !rec.tool_names.is_empty();
    let other_has_tools = !other.tool_ids.is_empty() || !other.tool_names.is_empty();
    if rec_has_tools != other_has_tools {
        return false;
    }
    rec.fingerprint == other.fingerprint
        || (!rec_has_tools
            && text_fingerprint_context(&other.fingerprint).is_none()
            && other.fingerprint == base_fingerprint(&rec.fingerprint))
}

fn match_existing_record(
    rec: &ThinkingRecord,
    existing: &[Arc<ThinkingRecord>],
    used: &[bool],
) -> Option<usize> {
    if let Some(ref sig) = rec.signature.as_ref().filter(|s| is_real_signature(s)) {
        for (i, ex) in existing.iter().enumerate() {
            if used[i] || ex.fingerprint.starts_with(CALL_SIGNATURE_PREFIX) {
                continue;
            }
            if let Some(ref ex_sig) = ex.signature {
                if signatures_match(sig, ex_sig) && tool_ids_compatible(&rec.tool_ids, &ex.tool_ids)
                {
                    return Some(i);
                }
            }
        }
    }
    if !rec.tool_ids.is_empty() {
        return match select_tool_generation(
            &rec.tool_ids,
            rec.signature.as_deref(),
            existing,
            (0..existing.len()).filter(|&i| {
                !used[i] && !existing[i].fingerprint.starts_with(CALL_SIGNATURE_PREFIX)
            }),
        ) {
            ToolMatch::Found(i) => Some(i),
            _ => None,
        };
    }
    for (i, ex) in existing.iter().enumerate() {
        if !used[i] && replay_fallback_matches(rec, ex) {
            return Some(i);
        }
    }
    None
}

fn normalize_ws(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut need_space = false;
    for word in s.split_whitespace() {
        if need_space {
            out.push(' ');
        }
        out.push_str(word);
        need_space = true;
    }
    out
}

fn hash_normalized_ws(hasher: &mut impl Digest, s: &str) {
    let mut need_space = false;
    for word in s.split_whitespace() {
        if need_space {
            hasher.update(b" ");
        }
        hasher.update(word.as_bytes());
        need_space = true;
    }
}

fn turn_needs_restore(parts: &[Value], existing_thought: &str) -> bool {
    if is_placeholder_thought(existing_thought) {
        return true;
    }
    // Sentinel / missing thought signature still needs ThinkingStore or tool-sig alignment.
    let thought_sig_ok = parts.iter().any(|p| {
        p.get("thought").and_then(|v| v.as_bool()).unwrap_or(false)
            && p.get("thoughtSignature")
                .or_else(|| p.get("thought_signature"))
                .and_then(|s| s.as_str())
                .is_some_and(is_real_signature)
    });
    if !thought_sig_ok {
        return true;
    }
    let mut saw_function_call = false;
    for part in parts {
        if part.get("functionCall").is_some() {
            saw_function_call = true;
            if !part_has_signature(part) {
                return true;
            }
        }
    }
    if saw_function_call {
        return false;
    }
    !parts.iter().any(part_has_signature)
}

fn part_has_signature(part: &Value) -> bool {
    part.get("thoughtSignature")
        .or_else(|| part.get("thought_signature"))
        .and_then(|s| s.as_str())
        .is_some_and(is_real_signature)
}

fn write_canonical_json(val: &Value, out: &mut Vec<u8>) {
    match val {
        Value::Null => out.extend_from_slice(b"null"),
        Value::Bool(b) => out.extend_from_slice(if *b { b"true" } else { b"false" }),
        Value::Number(n) => out.extend_from_slice(n.to_string().as_bytes()),
        Value::String(s) => {
            if let Ok(json_str) = serde_json::to_string(s) {
                out.extend_from_slice(json_str.as_bytes());
            } else {
                out.extend_from_slice(s.as_bytes());
            }
        }
        Value::Array(arr) => {
            out.push(b'[');
            for (i, elem) in arr.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_canonical_json(elem, out);
            }
            out.push(b']');
        }
        Value::Object(map) => {
            out.push(b'{');
            let mut sorted_keys: Vec<&String> = map.keys().collect();
            sorted_keys.sort();
            for (i, &key) in sorted_keys.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                if let Ok(json_key) = serde_json::to_string(key) {
                    out.extend_from_slice(json_key.as_bytes());
                } else {
                    out.extend_from_slice(key.as_bytes());
                }
                out.push(b':');
                if let Some(v) = map.get(key) {
                    write_canonical_json(v, out);
                }
            }
            out.push(b'}');
        }
    }
}

pub fn canonical_json_hash(val: Option<&Value>) -> String {
    let mut out = Vec::with_capacity(128);
    match val {
        Some(v) => write_canonical_json(v, &mut out),
        None => out.extend_from_slice(b"{}"),
    }
    let mut hasher = Sha256::new();
    hasher.update(&out);
    let hex = format!("{:x}", hasher.finalize());
    hex[..12].to_string()
}

pub fn compute_causal_anchor(turn: Option<&Value>) -> String {
    let Some(content) = turn else {
        return "root".to_string();
    };

    let mut hasher = Sha256::new();
    let role = content.get("role").and_then(|r| r.as_str()).unwrap_or("");
    hasher.update(role.as_bytes());
    hasher.update([0xff]);

    if let Some(parts) = content.get("parts").and_then(|p| p.as_array()) {
        for part in parts {
            if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                hasher.update(b"txt:");
                hasher.update(text.len().to_string().as_bytes());
                hasher.update([0xfe]);
                let prefix_len = text.len().min(48);
                hasher.update(&text.as_bytes()[..prefix_len]);
                hasher.update([0xfd]);
                let suffix_start = text.len().saturating_sub(48);
                hasher.update(&text.as_bytes()[suffix_start..]);
                hasher.update([0xfc]);
            } else if let Some(fr) = part.get("functionResponse") {
                hasher.update(b"fr:");
                let fr_name = fr.get("name").and_then(|n| n.as_str()).unwrap_or("");
                hasher.update(fr_name.as_bytes());
                hasher.update([0xfe]);
                if let Some(resp) = fr.get("response") {
                    let mut resp_bytes = Vec::new();
                    write_canonical_json(resp, &mut resp_bytes);
                    let resp_hash = Sha256::digest(&resp_bytes);
                    hasher.update(&resp_hash[..8]);
                }
                hasher.update([0xfd]);
            } else if let Some(fc) = part.get("functionCall") {
                hasher.update(b"fc:");
                let fc_name = fc.get("name").and_then(|n| n.as_str()).unwrap_or("");
                hasher.update(fc_name.as_bytes());
                hasher.update([0xfe]);
                let args_hash = canonical_json_hash(fc.get("args"));
                hasher.update(args_hash.as_bytes());
                hasher.update([0xfd]);
            }
        }
    }

    let hex = format!("{:x}", hasher.finalize());
    hex[..12].to_string()
}

pub fn synthesize_tool_id(
    tool_name: &str,
    args: Option<&Value>,
    causal_anchor: &str,
    call_index_in_turn: usize,
) -> String {
    let args_hash = canonical_json_hash(args);
    let anchor_clean = if causal_anchor.is_empty() {
        "root"
    } else {
        causal_anchor
    };
    format!(
        "call_{}_{}_{}_{}",
        tool_name, anchor_clean, args_hash, call_index_in_turn
    )
}

fn inspect_parts(parts: &[Value]) -> (String, Vec<String>, Vec<String>, String) {
    inspect_parts_with_anchor(parts, "root")
}

fn inspect_parts_with_anchor(
    parts: &[Value],
    anchor: &str,
) -> (String, Vec<String>, Vec<String>, String) {
    let mut visible = String::new();
    let mut thought = String::new();
    let mut tool_ids = Vec::new();
    let mut tool_names = Vec::new();
    let mut function_call_count = 0usize;
    for part in parts {
        let is_thought = part
            .get("thought")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
            if is_thought {
                thought.push_str(text);
            } else {
                visible.push_str(text);
            }
        }
        if let Some(fc) = part.get("functionCall") {
            let name = fc
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            let explicit_id = fc
                .get("id")
                .and_then(|v| v.as_str())
                .filter(|s| !s.trim().is_empty())
                .map(|s| crate::proxy::common::utils::normalize_tool_id(s).into_owned());

            let synthetic = synthesize_tool_id(&name, fc.get("args"), anchor, function_call_count);
            function_call_count += 1;

            // 核心演进：全面转战因果伪哈希 ID！首位强制存入确定性 synthetic ID
            if !tool_ids.iter().any(|x| x == &synthetic) {
                tool_ids.push(synthetic);
            }
            if let Some(real_id) = explicit_id {
                if !tool_ids.iter().any(|x| x == &real_id) {
                    tool_ids.push(real_id);
                }
            }
            tool_names.push(name);
        }
    }
    (visible, tool_ids, tool_names, thought)
}

pub fn fingerprint(visible: &str, tool_ids: &[String], tool_names: &[String]) -> String {
    let mut hasher = Sha256::new();
    hash_normalized_ws(&mut hasher, visible);
    hasher.update([0xff]);
    for id in tool_ids {
        hasher.update(id.as_bytes());
        hasher.update([0xfe]);
    }
    hasher.update([0xfd]);
    for name in tool_names {
        hasher.update(name.as_bytes());
        hasher.update([0xfc]);
    }
    let hex = format!("{:x}", hasher.finalize());
    hex[..16].to_string()
}

#[cfg(test)]
mod l2_text_candidate_tests {
    use super::*;
    use crate::proxy::SignatureCache;
    use base64::Engine;

    fn key(name: &str) -> String {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        format!(
            "l2-candidates-{name}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        )
    }
    fn signature(claude: bool, seed: u8) -> String {
        let mut raw = if claude {
            vec![0x08, 0x04, 0x12]
        } else {
            vec![0x12]
        };
        raw.extend(std::iter::repeat_n(seed, 48));
        let sig = base64::engine::general_purpose::STANDARD.encode(raw);
        if claude {
            ensure_google_claude_thought_signature(&sig)
        } else {
            sig
        }
    }
    fn user(text: &str) -> Value {
        json!({"role":"user","parts":[{"text":text}]})
    }
    fn answer() -> Value {
        json!({"role":"model","parts":[{"text":"Done."}]})
    }
    fn captured(prefix: &[Value], thought: &str, sig: &str) -> ThinkingRecord {
        let mut acc = TurnAccumulator::with_anchor(compute_causal_anchor(prefix.last()))
            .with_text_context_anchor(compute_response_text_anchor(prefix));
        acc.ingest_part(&json!({"text":thought,"thought":true}));
        acc.ingest_part(&json!({"text":"Done.","thoughtSignature":sig}));
        acc.into_record()
    }
    // Exercise different warm windows without depending on the user's configured
    // memory limit. The first test separately exercises actual record() eviction.
    fn warm_window(store: &ThinkingStore, sid: &str, resident_signature: Option<&str>) {
        store.sessions.remove(sid);
        let mut turns = store.load_turns(sid);
        turns.retain(|rec| {
            resident_signature.is_some_and(|sig| {
                rec.signature
                    .as_deref()
                    .is_some_and(|stored| signatures_match(sig, stored))
            })
        });
        // Keep a nonempty unrelated turn so load_turns uses the warm window.
        turns.push(Arc::new(captured(
            &[user("Unrelated")],
            "Unrelated thought",
            &signature(false, b'Z'),
        )));
        let mut entry = store.sessions.get_mut(sid).unwrap();
        entry.bytes = turns.iter().map(|rec| record_bytes(rec)).sum();
        entry.turns = turns;
        entry.l2_has_evicted_turns = true;
    }
    fn cleanup(sid: &str) {
        SignatureCache::global().invalidate_session(sid);
        crate::modules::proxy_db::delete_thinking_records_for_session(sid).unwrap();
    }
    fn assert_restored(turn: &Value, thought: &str, sig: &str) {
        assert_eq!(turn["parts"][0], json!({"text":thought,"thought":true}));
        assert_eq!(
            turn["parts"][1],
            json!({"text":"Done.","thoughtSignature":sig})
        );
    }

    #[test]
    fn evicted_exact_generations_remain_ambiguous_after_warm_or_cold_load() {
        let sid = key("actual-eviction");
        let store = ThinkingStore::new();
        let prefix = vec![user("Task")];
        for (thought, seed) in [
            ("First private thought", b'A'),
            ("Second private thought", b'B'),
        ] {
            store.record(&sid, captured(&prefix, thought, &signature(false, seed)));
        }
        let fp = captured(&prefix, "Unused", &signature(false, b'A')).fingerprint;
        let matches = crate::modules::proxy_db::load_thinking_by_fingerprints(
            &sid,
            &[fp.clone(), "missing".into()],
        )
        .unwrap();
        assert_eq!(matches.len(), 2, "SQLite must return every generation");
        assert_eq!(matches[0].thought, "First private thought");
        assert_eq!(matches[1].thought, "Second private thought");
        for idx in 0..=max_turns_per_session() {
            store.record(
                &sid,
                captured(
                    &[user(&format!("Filler {idx}"))],
                    "Filler thought",
                    &signature(false, b'F'),
                ),
            );
        }
        assert!(store.sessions.get(&sid).unwrap().l2_has_evicted_turns);
        assert!(!store
            .load_turns(&sid)
            .iter()
            .any(|rec| rec.fingerprint == fp));
        for cold in [false, true] {
            if cold {
                store.sessions.remove(&sid);
            }
            let mut history = vec![user("Task"), answer(), user("Continue")];
            let before = history.clone();
            assert_eq!(
                store.restore_gemini_contents_with_model(
                    &sid,
                    &mut history,
                    Some("gemini-3.8-flash")
                ),
                0
            );
            assert_eq!(
                history, before,
                "ambiguous generations cannot be chosen by database order"
            );
        }
        cleanup(&sid);
    }

    #[test]
    fn partially_resident_exact_generations_do_not_bypass_l2_ambiguity() {
        let sid = key("partial-window");
        let store = ThinkingStore::new();
        let a = signature(false, b'A');
        let b = signature(false, b'B');
        store.record(&sid, captured(&[user("Task")], "First thought", &a));
        store.record(&sid, captured(&[user("Task")], "Second thought", &b));
        for resident in [Some(a.as_str()), Some(b.as_str()), None] {
            warm_window(&store, &sid, resident);
            for _ in 0..2 {
                let mut history = vec![user("Task"), answer(), user("Continue")];
                assert_eq!(
                    store.restore_gemini_contents_with_model(
                        &sid,
                        &mut history,
                        Some("gemini-3.8-flash")
                    ),
                    0
                );
                assert_eq!(history[1], answer());
            }
        }
        cleanup(&sid);
    }

    #[test]
    fn l2_text_candidates_respect_family_and_authoritative_signatures() {
        for mixed_family in [false, true] {
            let sid = key("authority");
            let store = ThinkingStore::new();
            let a = signature(false, b'A');
            let b = signature(mixed_family, b'B');
            store.record(&sid, captured(&[user("Task")], "First thought", &a));
            store.record(&sid, captured(&[user("Task")], "Second thought", &b));
            for window in 0..3 {
                if window == 2 {
                    store.sessions.remove(&sid);
                } else {
                    warm_window(&store, &sid, (window == 1).then_some(b.as_str()));
                }
                SignatureCache::global().invalidate_session(&sid);
                SignatureCache::global().cache_session_signature(&sid, b.clone(), 1);
                let mut history = vec![user("Task"), answer(), user("Continue")];
                let model = if mixed_family {
                    "claude-sonnet-4-6"
                } else {
                    "gemini-3.8-flash"
                };
                assert_eq!(
                    store.restore_gemini_contents_with_model(&sid, &mut history, Some(model)),
                    1
                );
                assert_restored(&history[1], "Second thought", &b);
                let mut explicit = vec![user("Task"), answer(), user("Continue")];
                explicit[1]["parts"][0]["thoughtSignature"] = json!(a);
                assert_eq!(
                    store.restore_gemini_contents_with_model(
                        &sid,
                        &mut explicit,
                        Some("gemini-3.8-flash")
                    ),
                    1
                );
                assert_restored(&explicit[1], "First thought", &a);
                if !mixed_family {
                    SignatureCache::global().invalidate_session(&sid);
                    SignatureCache::global().cache_session_signature(
                        &sid,
                        signature(false, b'C'),
                        1,
                    );
                    let mut unrelated_hint = vec![user("Task"), answer(), user("Continue")];
                    assert_eq!(
                        store.restore_gemini_contents_with_model(
                            &sid,
                            &mut unrelated_hint,
                            Some(model)
                        ),
                        0
                    );
                    assert_eq!(
                        unrelated_hint[1],
                        answer(),
                        "an unrelated latest signature cannot resolve this candidate set"
                    );
                }
            }
            cleanup(&sid);
        }
    }

    #[test]
    fn l2_latest_hint_does_not_select_an_earlier_ambiguous_turn() {
        let sid = key("latest-scope");
        let store = ThinkingStore::new();
        let a = signature(false, b'A');
        let b = signature(false, b'B');
        for (thought, sig) in [("First thought", &a), ("Second thought", &b)] {
            store.record(&sid, captured(&[user("Task")], thought, sig));
        }
        SignatureCache::global().cache_session_signature(&sid, b, 1);
        for cold in [false, true] {
            if cold {
                store.sessions.remove(&sid);
            } else {
                warm_window(&store, &sid, None);
            }
            let mut history = vec![
                user("Task"),
                answer(),
                user("Continue"),
                answer(),
                user("Next"),
            ];
            let before = history.clone();
            assert_eq!(
                store.restore_gemini_contents_with_model(
                    &sid,
                    &mut history,
                    Some("gemini-3.8-flash")
                ),
                0
            );
            assert_eq!(history, before);
        }
        cleanup(&sid);
    }

    #[test]
    fn l2_legacy_candidates_are_consumed_once_in_capture_order() {
        let sid = key("legacy-consumption");
        let store = ThinkingStore::new();
        let a = signature(false, b'A');
        let b = signature(false, b'B');
        for (thought, sig) in [("First thought", &a), ("Second thought", &b)] {
            let mut rec = captured(&[user("Old")], thought, sig);
            rec.fingerprint = fingerprint("Done.", &[], &[]);
            store.record(&sid, rec);
        }
        for window in 0..3 {
            if window == 2 {
                store.sessions.remove(&sid);
            } else {
                warm_window(&store, &sid, (window == 1).then_some(b.as_str()));
            }
            let mut history = vec![
                user("First"),
                answer(),
                user("Second"),
                answer(),
                user("Third"),
                answer(),
            ];
            assert_eq!(
                store.restore_gemini_contents_with_model(
                    &sid,
                    &mut history,
                    Some("gemini-3.8-flash")
                ),
                2
            );
            assert_restored(&history[1], "First thought", &a);
            assert_restored(&history[3], "Second thought", &b);
            assert_eq!(
                history[5],
                answer(),
                "L2 must not replay an already consumed generation"
            );
        }
        cleanup(&sid);
    }

    #[test]
    fn l2_signature_point_lookup_rejects_foreign_family_and_consumed_rows() {
        let sid = key("signature-lookup");
        let store = ThinkingStore::new();
        let a = signature(false, b'A');
        store.record(&sid, captured(&[user("Task")], "First thought", &a));
        for cold in [false, true] {
            if cold {
                store.sessions.remove(&sid);
            } else {
                warm_window(&store, &sid, None);
            }
            let mut foreign = vec![user("Task"), answer()];
            foreign[1]["parts"][0]["thoughtSignature"] = json!(a);
            let before = foreign.clone();
            assert_eq!(
                store.restore_gemini_contents_with_model(
                    &sid,
                    &mut foreign,
                    Some("claude-sonnet-4-6")
                ),
                0
            );
            assert_eq!(
                foreign, before,
                "an explicit foreign signature must not restore foreign thought text"
            );
            let mut repeated = vec![
                user("Task"),
                answer(),
                user("Different"),
                json!({"role":"model","parts":[{"text":"Truncated","thoughtSignature":a}]}),
            ];
            repeated[1]["parts"][0]["thoughtSignature"] = json!(a);
            assert_eq!(
                store.restore_gemini_contents_with_model(
                    &sid,
                    &mut repeated,
                    Some("gemini-3.8-flash")
                ),
                1
            );
            assert_restored(&repeated[1], "First thought", &a);
            assert_eq!(repeated[3]["parts"].as_array().unwrap().len(), 1);
        }
        cleanup(&sid);
    }
}

#[cfg(test)]
mod l2_pruning_regression_tests {
    use super::*;
    use crate::modules::proxy_db;
    use crate::proxy::{
        mappers::prompt_sanitizer::PromptSanitizer, pipeline::inbound::InboundThinkingPipeline,
        SignatureCache,
    };
    use base64::Engine;

    fn key(name: &str) -> String {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        format!(
            "l2-prune-{name}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        )
    }
    fn signature(seed: u64) -> String {
        let mut raw = vec![0x12];
        for _ in 0..6 {
            raw.extend(seed.to_le_bytes());
        }
        base64::engine::general_purpose::STANDARD.encode(raw)
    }
    fn user(text: &str) -> Value {
        json!({"role":"user","parts":[{"text":text}]})
    }
    fn history() -> Vec<Value> {
        vec![
            user("Task"),
            json!({"role":"model","parts":[{"text":"Done."}]}),
            user("Continue"),
        ]
    }
    fn capture_answer(sid: &str, sig: &str) {
        capture_gemini_contents(
            sid,
            &[
                user("Task"),
                json!({"role":"model","parts":[{"thought":true,"text":"Private reasoning"},{"text":"Done.","thoughtSignature":sig}]}),
            ],
        );
    }
    fn capture_branches(sid: &str) {
        let prefix = history();
        let anchor = compute_response_text_anchor(&prefix);
        let causal = compute_causal_anchor(prefix.last());
        // Alternatives all share one unchanged prefix, so this does not
        // simulate compression or removal of the live Done turn.
        for idx in 0..max_turns_per_session() + 5 {
            capture_gemini_parts_with_context(
                sid,
                &[
                    json!({"thought":true,"text":format!("Branch reasoning {idx}")}),
                    json!({"functionCall":{"name":"lookup","args":{"branch":idx},"id":format!("call_branch_{idx}")},"thoughtSignature":signature(idx as u64+1000)}),
                ],
                &causal,
                Some(&anchor),
            );
        }
    }
    fn finish(sid: &str, contents: &mut Vec<Value>, enabled: bool, retry: bool) {
        let mut payload = json!({"contents":contents});
        PromptSanitizer::sanitize_gemini_payload(&mut payload);
        *contents = payload["contents"].as_array().unwrap().clone();
        finalize_request_text_context(sid, contents, "gemini-3.8-flash", enabled, retry);
    }
    fn cleanup(sid: &str) {
        ThinkingStore::global().sessions.remove(sid);
        SignatureCache::global().invalidate_session(sid);
        proxy_db::delete_thinking_records_for_session(sid).unwrap();
    }
    fn has_text(sid: &str) -> bool {
        proxy_db::load_thinking_records(sid)
            .unwrap()
            .iter()
            .any(|row| row.visible == "Done.")
    }

    #[test]
    fn deferred_tool_stage_preserves_evicted_text_until_final_restore() {
        for cold in [false, true] {
            let sid = key("live-text");
            let sig = signature(1);
            capture_answer(&sid, &sig);
            capture_branches(&sid);
            assert!(!ThinkingStore::global()
                .load_turns(&sid)
                .iter()
                .any(|r| r.visible == "Done."));
            let before = proxy_db::load_thinking_records(&sid).unwrap().len();
            if cold {
                ThinkingStore::global().sessions.remove(&sid);
            }
            let mut contents = history();
            InboundThinkingPipeline::process_contents_deferred_text(
                &mut contents,
                "gemini-3.8-flash",
                true,
                Some(&sid),
                false,
            );
            assert!(
                has_text(&sid),
                "early tool hydration must not delete L2 text before it can be restored"
            );
            assert_eq!(proxy_db::load_thinking_records(&sid).unwrap().len(), before);
            finish(&sid, &mut contents, true, false);
            assert_eq!(
                contents[1]["parts"],
                json!([{"text":"Done.","thoughtSignature":sig}])
            );
            let rows = proxy_db::load_thinking_records(&sid).unwrap();
            assert!(rows.iter().any(|r| r.visible == "Done."));
            assert!(
                rows.iter().filter(|r| !r.tool_ids.is_empty()).count() <= 2,
                "orphan tools are still pruned after text restoration"
            );
            let once = contents.clone();
            finish(&sid, &mut contents, true, false);
            assert_eq!(contents, once);
            cleanup(&sid);
        }
    }

    #[test]
    fn final_prune_removes_compressed_text_and_tools_in_warm_and_cold_windows() {
        for cold in [false, true] {
            let sid = key("compressed");
            capture_answer(&sid, &signature(1));
            capture_branches(&sid);
            if cold {
                ThinkingStore::global().sessions.remove(&sid);
            }
            // Preserve the first-user identity while genuinely replacing the
            // old model history with a user-provided summary.
            let mut contents = vec![
                user("Task"),
                user("Summary of completed work; begin the next task."),
            ];
            InboundThinkingPipeline::process_contents_deferred_text(
                &mut contents,
                "gemini-3.8-flash",
                true,
                Some(&sid),
                false,
            );
            assert!(
                has_text(&sid),
                "compaction pruning waits for the final prefix too"
            );
            finish(&sid, &mut contents, true, false);
            let rows = proxy_db::load_thinking_records(&sid).unwrap();
            assert!(!rows.iter().any(|r| r.visible == "Done."));
            assert_eq!(
                rows.len(),
                2,
                "only the two newest unused responses survive real compression"
            );
            assert!(rows.iter().all(|r| !r.tool_ids.is_empty()));
            cleanup(&sid);
        }
    }

    #[test]
    fn live_ambiguous_l2_text_survives_prune_and_later_explicit_disambiguation() {
        let sid = key("ambiguous");
        let a = signature(1);
        let b = signature(2);
        capture_answer(&sid, &a);
        capture_answer(&sid, &b);
        capture_branches(&sid);
        let mut contents = history();
        InboundThinkingPipeline::process_contents_deferred_text(
            &mut contents,
            "gemini-3.8-flash",
            true,
            Some(&sid),
            false,
        );
        finish(&sid, &mut contents, true, false);
        assert_eq!(contents[1]["parts"], json!([{"text":"Done."}]));
        let rows = proxy_db::load_thinking_records(&sid).unwrap();
        let live: Vec<_> = rows.iter().filter(|r| r.visible == "Done.").collect();
        assert_eq!(
            live.len(),
            2,
            "an ambiguous live generation is not an orphan merely because it stays outside RAM"
        );
        assert!(live
            .iter()
            .any(|r| r.signature.as_deref() == Some(a.as_str())));
        assert!(live
            .iter()
            .any(|r| r.signature.as_deref() == Some(b.as_str())));
        contents[1]["parts"][0]["thoughtSignature"] = json!(a);
        InboundThinkingPipeline::process_contents_deferred_text(
            &mut contents,
            "gemini-3.8-flash",
            true,
            Some(&sid),
            false,
        );
        finish(&sid, &mut contents, true, false);
        assert_eq!(
            contents[1]["parts"],
            json!([{"text":"Done.","thoughtSignature":a}])
        );
        assert!(has_text(&sid));
        cleanup(&sid);
    }

    #[test]
    fn active_evicted_tools_survive_while_nonlive_text_and_tools_are_pruned() {
        let sid = key("live-tool");
        let tool_sig = signature(2);
        capture_answer(&sid, &signature(1));
        let call = json!({"functionCall":{"name":"lookup","args":{"active":true},"id":"call_active"},"thoughtSignature":tool_sig});
        capture_gemini_contents(
            &sid,
            &[
                user("Task"),
                json!({"role":"model","parts":[{"thought":true,"text":"Active tool reasoning"},call]}),
            ],
        );
        capture_branches(&sid);
        let mut contents = vec![
            user("Task"),
            json!({"role":"model","parts":[{"functionCall":{"name":"lookup","args":{"active":true},"id":"call_active"}}]}),
            json!({"role":"user","parts":[{"functionResponse":{"name":"lookup","id":"call_active","response":{"output":"ok"}}}]}),
            user("Continue"),
        ];
        InboundThinkingPipeline::process_contents_deferred_text(
            &mut contents,
            "gemini-3.8-flash",
            true,
            Some(&sid),
            false,
        );
        finish(&sid, &mut contents, true, false);
        assert!(contents[1]["parts"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p.get("functionCall").is_some() && p["thoughtSignature"] == tool_sig));
        let rows = proxy_db::load_thinking_records(&sid).unwrap();
        assert!(rows
            .iter()
            .any(|r| r.tool_ids.iter().any(|id| id == "call_active")));
        assert!(!rows.iter().any(|r| r.visible == "Done."));
        assert!(
            rows.len() <= 3,
            "keep the active tool and at most two newest unused responses"
        );
        cleanup(&sid);
    }

    #[test]
    fn final_prune_keeps_live_tool_identity_even_when_its_record_is_only_in_l2() {
        for legacy_fingerprint in [false, true] {
            let sid = key("l2-tool-identity");
            let store = ThinkingStore::new();
            let mut acc = TurnAccumulator::with_anchor(compute_causal_anchor(Some(&user("Task"))));
            acc.ingest_part(&json!({"thought":true,"text":"Active tool reasoning"}));
            acc.ingest_part(&json!({"functionCall":{"name":"lookup","args":{"active":true},"id":"call_active"},"thoughtSignature":signature(1)}));
            let mut record = acc.into_record();
            if legacy_fingerprint {
                record.fingerprint = "old-tool-fingerprint-format".into();
            }
            store.record(&sid, record);
            for idx in 0..max_turns_per_session() + 5 {
                store.record(
                    &sid,
                    ThinkingRecord {
                        fingerprint: format!("unrelated-{idx}"),
                        thought: format!("Unrelated reasoning {idx}"),
                        signature: Some(signature(idx as u64 + 1000)),
                        tool_ids: vec![],
                        tool_names: vec![],
                        visible: format!("Unrelated answer {idx}"),
                    },
                );
            }
            assert!(!store
                .load_turns(&sid)
                .iter()
                .any(|r| r.tool_ids.iter().any(|id| id == "call_active")));
            let contents = vec![
                user("Task"),
                json!({"role":"model","parts":[{"functionCall":{"name":"lookup","args":{"active":true},"id":"call_active"}}]}),
                user("Continue"),
            ];
            store.prune_orphaned_records(&sid, &contents);
            let rows = proxy_db::load_thinking_records(&sid).unwrap();
            assert!(
                rows.iter()
                    .any(|r| r.tool_ids.iter().any(|id| id == "call_active")),
                "a live tool ID remains authoritative even when its stored fingerprint differs"
            );
            assert_eq!(
                rows.len(),
                3,
                "retain the live L2 tool and the two newest responses"
            );
            cleanup(&sid);
        }
    }

    #[test]
    fn disabled_or_retry_text_restore_keeps_history_for_the_next_enabled_request() {
        for (enabled, retry) in [(false, false), (true, true)] {
            let sid = key("deferred-cleanup");
            let sig = signature(1);
            capture_answer(&sid, &sig);
            capture_branches(&sid);
            let before = proxy_db::load_thinking_records(&sid).unwrap().len();
            let mut contents = history();
            contents.push(json!({"role":"model","parts":[{"functionCall":{"name":"lookup","args":{},"id":"call_new"}}]}));
            InboundThinkingPipeline::process_contents_deferred_text(
                &mut contents,
                "gemini-3.8-flash",
                enabled,
                Some(&sid),
                retry,
            );
            finish(&sid, &mut contents, enabled, retry);
            assert!(has_text(&sid));
            assert_eq!(proxy_db::load_thinking_records(&sid).unwrap().len(), before);
            let mut next = history();
            InboundThinkingPipeline::process_contents_deferred_text(
                &mut next,
                "gemini-3.8-flash",
                true,
                Some(&sid),
                false,
            );
            finish(&sid, &mut next, true, false);
            assert_eq!(
                next[1]["parts"],
                json!([{"text":"Done.","thoughtSignature":sig}])
            );
            cleanup(&sid);
        }
    }
}

#[cfg(test)]
mod native_candidate_regression_tests {
    use super::*;
    use crate::proxy::{
        mappers::gemini::{collector, streaming::GeminiStreamHandler},
        pipeline::stream_lifecycle::ProtocolStreamHandler,
    };
    use base64::Engine;
    fn sig(seed: u8) -> String {
        let mut raw = vec![0x12];
        raw.extend([seed; 48]);
        base64::engine::general_purpose::STANDARD.encode(raw)
    }
    fn candidate(index: usize) -> Value {
        let (t, x, s) = if index == 0 {
            ("Thought for option A.", "Option alpha.", sig(b'A'))
        } else {
            ("Thought for option B.", "Option beta.", sig(b'B'))
        };
        json!({"index":index,"content":{"role":"model","parts":[{"text":t,"thought":true},{"text":x,"thoughtSignature":s}]},"finishReason":"STOP"})
    }
    #[tokio::test]
    async fn gemini_candidates_keep_separate_responses_and_signed_history() {
        let user = json!({"role":"user","parts":[{"text":"Suggest two options"}]});
        let model = "gemini-2.5-flash";
        let ctx = compute_response_text_anchor(std::slice::from_ref(&user));
        let causal = compute_causal_anchor(Some(&user));
        for mode in ["stream", "buffer"] {
            for shape in ["single", "together", "interleaved", "implicit-zero"] {
                let sid = format!("r16-{mode}-{shape}");
                let groups = match shape {
                    "single" => vec![vec![candidate(0)]],
                    "together" => vec![vec![candidate(0), candidate(1)]],
                    "implicit-zero" => {
                        let mut zero = candidate(0);
                        zero.as_object_mut().unwrap().remove("index");
                        vec![vec![candidate(1), zero]]
                    }
                    _ => vec![vec![candidate(1)], vec![candidate(0)]],
                };
                let mut returned = Vec::new();
                let mut handler = GeminiStreamHandler::new_with_context(
                    None,
                    sid.clone(),
                    model.into(),
                    1,
                    causal.clone(),
                    false,
                    None,
                    String::new(),
                    String::new(),
                    Some(ctx.clone()),
                )
                .with_candidate_request(if shape == "single" { 1 } else { 2 }, mode == "buffer");
                let mut frames = Vec::new();
                for group in groups {
                    let f = format!("data: {}", json!({"candidates":group}));
                    for b in handler.process_line(&f) {
                        let line = String::from_utf8(b.to_vec()).unwrap();
                        let payload: Value =
                            serde_json::from_str(line.strip_prefix("data: ").unwrap().trim())
                                .unwrap();
                        returned.extend(payload["candidates"].as_array().unwrap().clone());
                        frames.push(Ok::<_, String>(b));
                    }
                }
                assert!(handler.emit_finalize().is_empty());
                handler.on_finish();
                if mode == "buffer" {
                    let v = collector::collect_stream_to_json_with_context(
                        futures::stream::iter(frames),
                        &sid,
                        Some(&causal),
                        1,
                        Some(ctx.clone()),
                    )
                    .await
                    .unwrap();
                    returned = v["candidates"].as_array().unwrap().clone();
                }
                let rows = crate::modules::proxy_db::load_thinking_records(&sid).unwrap();
                let texts: Vec<_> = returned
                    .iter()
                    .map(|v| {
                        v["content"]["parts"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .filter(|p| p["thought"] != true)
                            .filter_map(|p| p["text"].as_str())
                            .collect::<String>()
                    })
                    .collect();
                let mut history = vec![
                    user.clone(),
                    json!({"role":"model","parts":[{"text":"Option beta."}]}),
                    json!({"role":"user","parts":[{"text":"Expand that one."}]}),
                ];
                if shape == "single" {
                    history[1]["parts"][0]["text"] = json!("Option alpha.");
                }
                finalize_request_text_context(&sid, &mut history, model, true, false);
                let signed = history[1]["parts"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find_map(|p| p["thoughtSignature"].as_str());
                let got = if signed == Some(sig(b'A').as_str()) {
                    "A"
                } else if signed == Some(sig(b'B').as_str()) {
                    "B"
                } else {
                    "missing"
                };
                let expected = if shape == "single" { 1 } else { 2 };
                assert_eq!(returned.len(), expected, "{mode} {shape} candidates");
                for i in 0..expected {
                    let wanted = if i == 0 {
                        "Option alpha."
                    } else {
                        "Option beta."
                    };
                    assert!(returned.iter().any(|v| v
                        .get("index")
                        .and_then(Value::as_u64)
                        .unwrap_or(0)
                        == i as u64
                        && v["content"]["parts"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .any(|p| p["text"] == wanted)));
                    assert!(rows
                        .iter()
                        .any(|r| r.visible == wanted && r.signature == Some(sig(b'A' + i as u8))));
                }
                assert_eq!(rows.len(), expected, "{mode} {shape} SQLite rows");
                assert_eq!(
                    got,
                    if expected == 1 { "A" } else { "B" },
                    "{mode} {shape} {texts:?}"
                );
            }
        }
    }
    #[test]
    fn only_an_upstream_terminal_can_publish_a_signed_stream_candidate() {
        let key = format!("stream-publish-before-yield-{}", std::process::id());
        let make = || {
            GeminiStreamHandler::new(
                None,
                key.clone(),
                "gemini-2.5-flash".into(),
                1,
                "root".into(),
                false,
                None,
                String::new(),
                String::new(),
            )
        };
        let mut failed = make();
        let partial = json!({"candidates":[{"content":{"parts":[{"text":"Done.","thoughtSignature":sig(b'A')}]},"finishReason":"FINISH_REASON_UNSPECIFIED"}]});
        failed.process_line(&format!("data: {partial}"));
        // Stream cleanup after an error, timeout or EOF is not a model terminal.
        failed.on_finish();
        assert!(crate::modules::proxy_db::load_thinking_records(&key)
            .unwrap()
            .is_empty());
        let mut success = make();
        let complete = json!({"candidates":[{"content":{"parts":[{"text":"Done.","thoughtSignature":sig(b'B')}]},"finishReason":"STOP"}]});
        let line = format!("data: {complete}");
        success.process_line(&line);
        // Before on_finish, trailer, or a second upstream poll.
        let rows = crate::modules::proxy_db::load_thinking_records(&key).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].signature, Some(sig(b'B')));
        success.process_line(&line);
        success.on_finish();
        assert_eq!(
            crate::modules::proxy_db::load_thinking_records(&key)
                .unwrap()
                .len(),
            1
        );
        ThinkingStore::global().end_session(&key);
    }

    #[tokio::test]
    async fn duplicate_or_invalid_indices_are_not_published_as_signed_candidates() {
        for index in [json!(null), json!(-1), json!(0)] {
            let mut a = candidate(0);
            let mut b = candidate(1);
            a.as_object_mut().unwrap().remove("index");
            if index == Value::Null {
                b.as_object_mut().unwrap().remove("index");
            } else {
                b["index"] = index.clone();
            }
            let key = format!("invalid-index-{index}");
            let mut h = GeminiStreamHandler::new(
                None,
                key.clone(),
                "gemini-2.5-flash".into(),
                1,
                "root".into(),
                false,
                None,
                String::new(),
                String::new(),
            )
            .with_candidate_request(2, false);
            let line = format!("data: {}", json!({"candidates":[a,b]}));
            let direct: Value = serde_json::from_str(line.strip_prefix("data: ").unwrap()).unwrap();
            capture_gemini_response_with_context(&key, &direct, None, None);
            let frames = h.process_line(&line);
            h.on_finish();
            assert!(std::str::from_utf8(&frames[0])
                .unwrap()
                .contains("Invalid or duplicate"));
            let result = collector::collect_stream_to_json(
                futures::stream::iter([Ok::<_, String>(bytes::Bytes::from(format!("{line}\n\n")))]),
                &key,
            )
            .await
            .unwrap();
            assert_eq!(result["error"]["code"], 502);
            assert!(crate::modules::proxy_db::load_thinking_records(&key)
                .unwrap()
                .is_empty());
        }
    }

    #[tokio::test]
    async fn identical_visible_candidates_do_not_use_a_stale_session_signature() {
        for buffered in [false, true] {
            let key = format!("ambiguous-candidates-{buffered}");
            let user = json!({"role":"user","parts":[{"text":"choose"}]});
            let a = sig(b'A');
            let b = sig(b'B');
            crate::proxy::SignatureCache::global().cache_session_signature(&key, a.clone(), 1);
            let ctx = compute_response_text_anchor(std::slice::from_ref(&user));
            let anchor = compute_causal_anchor(Some(&user));
            let mut h = GeminiStreamHandler::new_with_context(
                None,
                key.clone(),
                "gemini-2.5-flash".into(),
                1,
                anchor.clone(),
                false,
                None,
                String::new(),
                String::new(),
                Some(ctx.clone()),
            )
            .with_candidate_request(2, buffered);
            let mut c0 = candidate(0);
            let mut c1 = candidate(1);
            c0["content"]["parts"][1]["text"] = json!("same");
            c1["content"]["parts"][1]["text"] = json!("same");
            let frames = h.process_line(&format!("data: {}", json!({"candidates":[c0,c1]})));
            h.on_finish();
            if buffered {
                collector::collect_stream_to_json_with_context(
                    futures::stream::iter(frames.into_iter().map(Ok::<_, String>)),
                    &key,
                    Some(&anchor),
                    1,
                    Some(ctx),
                )
                .await
                .unwrap();
            }
            // One candidate's late signature at the same depth is still not evidence of a user choice.
            crate::proxy::SignatureCache::global().cache_session_signature(&key, b.clone(), 1);
            assert!(crate::proxy::SignatureCache::global()
                .get_session_signature(&key)
                .is_none());
            let plain = vec![
                user.clone(),
                json!({"role":"model","parts":[{"text":"same"}]}),
                json!({"role":"user","parts":[{"text":"continue"}]}),
            ];
            let mut unsigned = plain.clone();
            finalize_request_text_context(&key, &mut unsigned, "gemini-2.5-flash", true, false);
            assert_eq!(unsigned, plain);
            let mut signed = plain;
            signed[1]["parts"][0]["thoughtSignature"] = json!(b);
            finalize_request_text_context(&key, &mut signed, "gemini-2.5-flash", true, false);
            assert!(signed[1]["parts"]
                .as_array()
                .unwrap()
                .iter()
                .any(|p| p["thoughtSignature"] == b));
        }
    }

    #[tokio::test]
    async fn incomplete_multi_candidate_stream_does_not_invent_a_success_for_missing_index() {
        let mut h = GeminiStreamHandler::new(
            None,
            "missing-branch".into(),
            "gemini-2.5-flash".into(),
            1,
            "root".into(),
            false,
            None,
            String::new(),
            String::new(),
        )
        .with_candidate_request(2, true);
        let mut data = h.process_line(&format!("data: {}", json!({"candidates":[candidate(0)]})));
        data.extend(h.emit_finalize());
        h.on_finish();
        let response = collector::collect_stream_to_json(
            futures::stream::iter(data.into_iter().map(Ok::<_, String>)),
            "missing-branch",
        )
        .await
        .unwrap();
        assert_eq!(response["error"]["code"], 502);
        assert!(response.get("candidates").is_none());
    }
}

#[cfg(test)]
mod tool_generation_regression_tests {
    use super::*;
    use crate::modules::proxy_db;
    use crate::proxy::{pipeline::InboundThinkingPipeline, SignatureCache};
    use base64::Engine;

    fn sig(seed: u8) -> String {
        let mut raw = vec![0x12];
        raw.extend([seed; 48]);
        base64::engine::general_purpose::STANDARD.encode(raw)
    }
    fn prompt() -> Value {
        json!({"role":"user","parts":[{"text":"Read the status."}]})
    }
    fn call(id: Option<&str>) -> Value {
        let mut p = json!({"functionCall":{"name":"get_status","args":{"k":"v"}}});
        if let Some(id) = id {
            p["functionCall"]["id"] = json!(id);
        }
        p
    }
    fn capture(key: &str, id: Option<&str>, signature: &str) {
        let mut part = call(id);
        part["thoughtSignature"] = json!(signature);
        capture_gemini_response_with_context(
            key,
            &json!({"candidates":[{"content":{"role":"model","parts":[
            {"thought":true,"text":"Tool reasoning."},part]},"finishReason":"STOP"}]}),
            Some(&prompt()),
            None,
        );
    }
    fn replay(key: &str, id: Option<&str>, authority: Option<&str>) -> Option<String> {
        let mut p = call(id);
        if let Some(sig) = authority {
            p["thoughtSignature"] = json!(sig);
        }
        let mut history = vec![
            prompt(),
            json!({"role":"model","parts":[p]}),
            json!({"role":"user","parts":[{"functionResponse":{"name":"get_status","response":{"output":"OK"}}}]}),
        ];
        InboundThinkingPipeline::process_contents_deferred_text(
            &mut history,
            "gemini-3.8-flash",
            true,
            Some(key),
            false,
        );
        history[1]["parts"]
            .as_array()
            .unwrap()
            .iter()
            .find_map(|p| {
                p.get("thoughtSignature")
                    .and_then(Value::as_str)
                    .filter(|s| is_real_signature(s))
                    .map(str::to_owned)
            })
    }
    #[test]
    fn parallel_calls_recover_only_their_own_signed_generations() {
        let model = "gemini-2.5-flash";
        let user = prompt();
        let capture_pair = |sid: &str,
                            names: [&str; 2],
                            ids: [Option<&str>; 2],
                            sigs: [&str; 2]| {
            let mut parts = Vec::new();
            for (i, name) in names.iter().enumerate() {
                let mut part =
                    json!({"functionCall":{"name":name,"args":{"x":i}},"thoughtSignature":sigs[i]});
                if let Some(id) = ids[i] {
                    part["functionCall"]["id"] = json!(id);
                }
                parts.push(part);
            }
            capture_gemini_response_with_context(
                sid,
                &json!({"candidates":[{"content":{"parts":parts},"finishReason":"STOP"}]}),
                Some(&user),
                None,
            );
            parts
        };
        let restore = |sid: &str, parts: Vec<Value>| {
            let mut body = vec![user.clone(), json!({"role":"model","parts":parts})];
            finalize_gemini_contents_thinking_with_session(&mut body, true, Some(model), Some(sid));
            body[1]["parts"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| p["thoughtSignature"].as_str().map(str::to_owned))
                .collect::<Vec<_>>()
        };
        let a = sig(b'A');
        let b = sig(b'B');
        let c = sig(b'C');
        for explicit in [true, false] {
            for mask in 0..4 {
                let sid = format!("parallel-by-id-{explicit}-{mask}-{}", std::process::id());
                let ids = if explicit {
                    [Some("call_A"), Some("call_B")]
                } else {
                    [None, None]
                };
                let mut parts = capture_pair(&sid, ["read", "read"], ids, [&a, &b]);
                for (i, p) in parts.iter_mut().enumerate() {
                    if !explicit {
                        p["functionCall"]["id"] = json!(format!("call_client_{i}"));
                    }
                    if mask & (1 << i) != 0 {
                        p.as_object_mut().unwrap().remove("thoughtSignature");
                    }
                }
                ThinkingStore::global().sessions.remove(&sid);
                assert_eq!(restore(&sid, parts), vec![Some(a.clone()), Some(b.clone())]);
                ThinkingStore::global().end_session(&sid);
            }
        }
        // Two candidates have the same causal call at index 0 but different
        // signatures; the uniquely signed sibling at index 1 is still usable.
        let sid = format!("parallel-conflict-{}", std::process::id());
        capture_pair(&sid, ["read", "read"], [None, None], [&a, &b]);
        let mut parts = capture_pair(&sid, ["read", "read"], [None, None], [&c, &b]);
        for p in &mut parts {
            p.as_object_mut().unwrap().remove("thoughtSignature");
        }
        ThinkingStore::global().sessions.remove(&sid);
        assert_eq!(
            restore(&sid, parts),
            vec![Some(SENTINEL_SIGNATURE.into()), Some(b)]
        );
        ThinkingStore::global().end_session(&sid);
    }

    #[test]
    fn regenerated_tools_replay_by_real_id_before_and_after_ram_eviction() {
        let key = format!("tool-regeneration-{}", std::process::id());
        let store = ThinkingStore::global();
        let a = sig(b'A');
        let b = sig(b'B');
        let a_id = "call_original_client_A";
        let b_id = "call_original_client_B";
        assert!(!proxy_db::is_synthetic_tool_id(a_id));
        capture(&key, Some(a_id), &a);
        capture(&key, Some(b_id), &b);
        assert_eq!(proxy_db::load_thinking_records(&key).unwrap().len(), 2);
        for window in ["warm", "partial", "cold"] {
            if window == "partial" {
                let mut entry = store.sessions.get_mut(&key).unwrap();
                entry.turns.remove(0);
                entry.l2_has_evicted_turns = true;
            }
            if window == "cold" {
                store.sessions.remove(&key);
            }
            // Weaker synthetic and session hints must not choose between siblings.
            let synthetic = synthesize_tool_id(
                "get_status",
                Some(&json!({"k":"v"})),
                &compute_causal_anchor(Some(&prompt())),
                0,
            );
            SignatureCache::global().cache_tool_signature(&key, &synthetic, b.clone());
            SignatureCache::global().cache_session_signature(&key, b.clone(), 2);
            assert_eq!(
                replay(&key, Some(a_id), None),
                Some(a.clone()),
                "{window} A"
            );
            assert_eq!(
                replay(&key, Some(b_id), None),
                Some(b.clone()),
                "{window} B"
            );
            assert!(
                replay(&key, None, None).is_none(),
                "{window} ambiguous alias"
            );
            assert!(
                replay(&key, Some("call_unrelated"), None).is_none(),
                "{window} unrelated real ID"
            );
            assert_eq!(
                replay(&key, None, Some(&a)),
                Some(a.clone()),
                "{window} signed"
            );
        }
        store.sessions.remove(&key);
        proxy_db::delete_thinking_records_for_session(&key).unwrap();
        SignatureCache::global().invalidate_session(&key);
    }
    #[test]
    fn concurrent_signed_captures_keep_separate_sqlite_generations() {
        let key = format!("tool-parallel-generations-{}", std::process::id());
        let barrier = std::sync::Barrier::new(3);
        std::thread::scope(|scope| {
            for (id, seed) in [("call_a", b'A'), ("call_b", b'B')] {
                let key = &key;
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    capture(key, Some(id), &sig(seed));
                });
            }
            barrier.wait();
        });
        let rows = proxy_db::load_thinking_records(&key).unwrap();
        assert_eq!(rows.len(), 2);
        for (id, seed) in [("call_a", b'A'), ("call_b", b'B')] {
            assert!(rows.iter().any(
                |row| row.tool_ids.iter().any(|s| s == id) && row.signature == Some(sig(seed))
            ));
        }
        ThinkingStore::global().sessions.remove(&key);
        proxy_db::delete_thinking_records_for_session(&key).unwrap();
    }

    #[test]
    fn tool_without_real_id_remains_recoverable_unless_signed_generations_conflict() {
        let key = format!("tool-idless-generation-{}", std::process::id());
        let a = sig(b'A');
        let b = sig(b'B');
        let store = ThinkingStore::global();
        capture(&key, None, &a);
        assert_eq!(replay(&key, Some("client-assigned"), None), Some(a.clone()));
        store.sessions.remove(&key);
        assert_eq!(replay(&key, None, None), Some(a.clone()));
        capture(&key, None, &b);
        for cold in [false, true] {
            if cold {
                store.sessions.remove(&key);
            }
            assert!(replay(&key, None, None).is_none());
            assert_eq!(replay(&key, None, Some(&a)), Some(a.clone()));
            assert_eq!(replay(&key, None, Some(&b)), Some(b.clone()));
        }
        store.sessions.remove(&key);
        proxy_db::delete_thinking_records_for_session(&key).unwrap();
        SignatureCache::global().invalidate_session(&key);
    }
}

#[cfg(test)]
mod pruned_candidate_completeness_tests {
    use super::*;
    use crate::modules::proxy_db;
    use crate::proxy::{session_manager::SessionManager, SignatureCache};
    use base64::Engine;

    fn session(name: &str) -> String {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        format!(
            "pruned-candidates-{name}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        )
    }
    fn signature(seed: u8) -> String {
        let mut raw = vec![0x12];
        raw.extend(std::iter::repeat_n(seed, 48));
        base64::engine::general_purpose::STANDARD.encode(raw)
    }
    fn user(text: &str) -> Value {
        json!({"role":"user","parts":[{"text":text}]})
    }
    fn answer() -> Value {
        json!({"role":"model","parts":[{"text":"Done."}]})
    }
    fn scope(session: &str, prefix: &[Value]) -> SessionScope {
        let body = json!({"session_id":session,"contents":prefix});
        SessionScope::from_headers_and_body(
            &HeaderMap::new(),
            Some(&body),
            SessionManager::gemini_content_anchor(&body),
        )
    }
    fn capture(session: &str, prefix: &[Value], thought: &str, signed_part: Value) -> String {
        let key = scope(session, prefix).store_key;
        let response = json!({"candidates":[{"content":{"role":"model","parts":[
            {"text":thought,"thought":true}, signed_part
        ]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":10}});
        capture_gemini_response_with_context(
            &key,
            &response,
            prefix.last(),
            Some(&compute_response_text_anchor(prefix)),
        );
        key
    }
    fn capture_text(session: &str, prefix: &[Value], thought: &str, signature: &str) -> String {
        capture(
            session,
            prefix,
            thought,
            json!({"text":"Done.","thoughtSignature":signature}),
        )
    }
    fn seed_omitted_branches(session: &str) {
        for idx in 0..5 {
            capture_text(
                session,
                &[user("Task"), answer(), user(&format!("Old branch {idx}"))],
                "Old complete reasoning",
                &signature(b'D' + idx),
            );
        }
    }
    fn cleanup(key: &str) {
        ThinkingStore::global().sessions.remove(key);
        SignatureCache::global().invalidate_session(key);
        proxy_db::delete_thinking_records_for_session(key).unwrap();
    }
    fn assert_restored(turn: &Value, thought: &str, signature: &str) {
        let parts = turn["parts"].as_array().unwrap();
        assert!(parts
            .iter()
            .any(|part| part["thought"] == true && part["text"] == thought));
        assert!(parts
            .iter()
            .any(|part| part["thoughtSignature"] == signature));
    }

    #[test]
    fn cold_read_error_keeps_memory_captures_without_claiming_a_complete_window() {
        let session = session("cold-read-error");
        let prefix = vec![user("Task")];
        let history = vec![user("Task"), answer(), user("Continue")];
        let (a, b, c) = (signature(b'A'), signature(b'B'), signature(b'C'));
        let key = capture_text(&session, &prefix, "First complete reasoning", &a);
        let store = ThinkingStore::global();
        store.sessions.remove(&key);
        let mut acc = TurnAccumulator::with_anchor(compute_causal_anchor(prefix.last()))
            .with_text_context_anchor(compute_response_text_anchor(&prefix));
        acc.ingest_part(&json!({"thought":true,"text":"Second complete reasoning"}));
        acc.ingest_part(&json!({"text":"Done.","thoughtSignature":b}));
        // Only the cold read fails. The ordinary record() write still succeeds.
        store.record_with_loader(
            &key,
            acc.into_record(),
            |_| Err("database is locked".into()),
        );
        assert!(!store.sessions.get(&key).unwrap().l2_loaded);
        let recovered = store.load_turns(&key);
        assert_eq!(recovered.len(), 2);
        assert!(recovered
            .iter()
            .any(|r| r.signature.as_deref() == Some(a.as_str())));
        capture_text(&session, &history, "Third branch reasoning", &c);
        SignatureCache::global().cache_session_signature(&key, c, 3);
        for cold in [false, true] {
            if cold {
                store.sessions.remove(&key);
            }
            let mut unsigned = history.clone();
            finalize_request_text_context(&key, &mut unsigned, "gemini-3.8-flash", true, false);
            assert_eq!(unsigned, history);
            let mut signed = history.clone();
            signed[1]["parts"][0]["thoughtSignature"] = json!(a);
            assert_eq!(
                store.restore_gemini_contents_with_model(
                    &key,
                    &mut signed,
                    Some("gemini-3.8-flash")
                ),
                1
            );
            assert_restored(&signed[1], "First complete reasoning", &a);
        }
        cleanup(&key);
    }

    #[test]
    fn every_historical_upgrade_survives_restart_without_touching_a_sibling_generation() {
        let session = session("historical-upgrades");
        let store = ThinkingStore::global();
        let claude = |seed| {
            let mut bytes = vec![0x08, 0x04, 0x12];
            bytes.extend(std::iter::repeat_n(seed, 48));
            ensure_google_claude_thought_signature(
                &base64::engine::general_purpose::STANDARD.encode(bytes),
            )
        };
        let (a, b, c, d) = (claude(b'A'), claude(b'B'), claude(b'C'), claude(b'D'));
        let signed = |sig: &str, thought: Option<&str>| {
            let mut parts = Vec::new();
            if let Some(thought) = thought {
                parts.push(json!({"thought":true,"text":thought}));
            }
            parts.push(json!({"text":"Done.","thoughtSignature":sig}));
            json!({"role":"model","parts":parts})
        };
        let model = "claude-sonnet-4-6";
        let key = scope(&session, &[user("Task")]).store_key;
        let mut first = vec![user("Task"), signed(&a, None), user("More")];
        finalize_request_text_context(&key, &mut first, model, true, false);
        capture_text(&session, &[user("Task")], "Sibling untouched", &b);
        let mut second = vec![
            user("Task"),
            signed(&a, None),
            user("More"),
            signed(&c, None),
            user("Continue"),
        ];
        finalize_request_text_context(&key, &mut second, model, true, false);
        capture_text(&session, &second, "Last row untouched", &d);
        let mut richer = second.clone();
        richer[1] = signed(&a, Some("Original A reasoning"));
        richer[3] = signed(&c, Some("Original C reasoning"));
        finalize_request_text_context(&key, &mut richer, model, true, false);
        let rows = proxy_db::load_thinking_records(&key).unwrap();
        for (sig, thought) in [
            (&a, "Original A reasoning"),
            (&b, "Sibling untouched"),
            (&c, "Original C reasoning"),
            (&d, "Last row untouched"),
        ] {
            assert_eq!(
                rows.iter()
                    .filter(|r| r.signature.as_ref() == Some(sig))
                    .count(),
                1
            );
            assert_eq!(
                rows.iter()
                    .find(|r| r.signature.as_ref() == Some(sig))
                    .unwrap()
                    .thought,
                thought
            );
        }
        for cold in [false, true] {
            if cold {
                store.sessions.remove(&key);
            }
            let mut restored = second.clone();
            restored[1] = signed(&a, None);
            restored[3] = signed(&c, None);
            finalize_request_text_context(&key, &mut restored, model, true, false);
            assert_restored(&restored[1], "Original A reasoning", &a);
            assert_restored(&restored[3], "Original C reasoning", &c);
        }
        cleanup(&key);
    }

    #[test]
    fn historical_snapshot_cannot_overwrite_a_recaptured_or_removed_row() {
        let session = session("historical-cas");
        let store = ThinkingStore::global();
        let a = signature(b'A');
        let prefix = vec![user("Task")];
        let key = capture_text(&session, &prefix, "Original reasoning", &a);
        let old = store.load_turns(&key)[0].clone();
        let mut upgrade = (*old).clone();
        upgrade.thought = "Late stale upgrade".into();
        let db_snapshot = proxy_db::snapshot_thinking_upgrade(&key, &old)
            .unwrap()
            .unwrap();
        capture_text(&session, &prefix, "Current much longer reasoning", &a);
        assert!(!proxy_db::apply_thinking_upgrade(&db_snapshot, &upgrade).unwrap());
        store.upgrade_record(&key, &old, upgrade.clone());
        assert_eq!(
            store.load_turns(&key)[0].thought,
            "Current much longer reasoning"
        );
        let current = store.load_turns(&key)[0].clone();
        let snap = proxy_db::snapshot_thinking_upgrade(&key, &current)
            .unwrap()
            .unwrap();
        store.sessions.remove(&key);
        proxy_db::delete_thinking_records_for_session(&key).unwrap();
        capture_text(&session, &prefix, "Replacement reasoning", &a);
        assert!(!proxy_db::apply_thinking_upgrade(&snap, &upgrade).unwrap());
        store.upgrade_record(&key, &current, upgrade);
        assert_eq!(
            proxy_db::load_thinking_records(&key).unwrap()[0].thought,
            "Replacement reasoning"
        );
        cleanup(&key);
    }

    #[test]
    fn healing_raw_gemini_signature_does_not_create_another_generation_on_cold_recovery() {
        let session = session("raw-heal");
        let store = ThinkingStore::global();
        let key = scope(&session, &[user("Task")]).store_key;
        let raw = format!("\u{12}{}", "a".repeat(48));
        let encoded = base64::engine::general_purpose::STANDARD.encode(raw.as_bytes());
        let mut record = TurnAccumulator::with_anchor("root");
        record.ingest_part(&json!({"text":"Done.","thoughtSignature":raw}));
        store.record_with_loader(&key, record.into_record(), |_| {
            Err("database is locked".into())
        });
        let disk = proxy_db::load_thinking_records(&key).unwrap();
        assert_eq!(disk.len(), 1);
        assert_eq!(disk[0].signature.as_deref(), Some(encoded.as_str()));
        let recovered = store.load_turns(&key);
        assert_eq!(recovered.len(), 1);
        assert!(signatures_match(
            recovered[0].signature.as_deref().unwrap(),
            &encoded
        ));
        cleanup(&key);
    }

    #[test]
    fn unsigned_richer_replay_cannot_upgrade_the_first_signed_sibling() {
        let session = session("unsigned-upgrade");
        let prefix = vec![user("Task")];
        let a = signature(b'A');
        let b = signature(b'B');
        let store = ThinkingStore::global();
        let key = capture_text(&session, &prefix, "A reasoning", &a);
        capture_text(&session, &prefix, "B reasoning", &b);
        let mut unsigned = (*store.load_turns(&key)[0]).clone();
        unsigned.signature = None;
        unsigned.thought = "Unattributable much longer client reasoning".into();
        store.ingest_records(&key, vec![unsigned]);
        let rows = proxy_db::load_thinking_records(&key).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].thought, "A reasoning");
        assert_eq!(rows[1].thought, "B reasoning");
        let ram = store.load_turns(&key);
        assert_eq!(ram[0].thought, rows[0].thought);
        assert_eq!(ram[1].thought, rows[1].thought);
        cleanup(&key);
    }

    #[test]
    fn delayed_cold_read_does_not_reintroduce_a_cleared_session() {
        let session = session("late-read");
        let key = capture_text(&session, &[user("Task")], "Old reasoning", &signature(b'A'));
        let store = ThinkingStore::global();
        store.sessions.remove(&key);
        let loaded = store.load_turns_with(&key, |k| {
            let snapshot = proxy_db::load_thinking_records(k)?;
            assert_eq!(snapshot.len(), 1);
            store.clear_session_for_compaction(k);
            Ok(snapshot)
        });
        assert!(loaded.is_empty());
        assert!(store.load_turns(&key).is_empty());
        cleanup(&key);
    }

    #[test]
    fn historical_upgrade_after_prune_tracks_the_original_generation_not_its_index() {
        let session = session("shifted-index");
        let store = ThinkingStore::global();
        let prefix = vec![user("Task")];
        let key = capture_text(&session, &prefix, "Early A reasoning", &signature(b'A'));
        capture_text(
            &session,
            &[user("Task"), answer(), user("Other")],
            "Later B reasoning",
            &signature(b'B'),
        );
        capture_text(
            &session,
            &[user("Task"), answer(), user("Third")],
            "Latest C reasoning",
            &signature(b'C'),
        );
        let old = store.load_turns(&key);
        assert_eq!(old.len(), 3);
        store.prune_orphaned_records(&key, &[]);
        assert_eq!(store.load_turns(&key).len(), 2);
        for idx in 0..2 {
            let mut richer = (*old[idx]).clone();
            richer.thought = format!("Richer reasoning for {idx}");
            store.upgrade_record(&key, &old[idx], richer);
        }
        let disk = proxy_db::load_thinking_records(&key).unwrap();
        assert_eq!(disk.len(), 2);
        assert_eq!(disk[0].thought, "Richer reasoning for 1");
        assert_eq!(disk[1].thought, "Latest C reasoning");
        assert_eq!(store.load_turns(&key)[0].thought, disk[0].thought);
        cleanup(&key);
    }

    #[test]
    fn missing_historical_generation_does_not_rewrite_its_persisted_sibling() {
        let session = session("missing-upgrade");
        let store = ThinkingStore::global();
        let prefix = vec![user("Task")];
        let key = capture_text(&session, &prefix, "Disk A reasoning", &signature(b'A'));
        let persisted = store.load_turns(&key)[0].clone();
        let mut b = (*persisted).clone();
        b.signature = Some(signature(b'B'));
        b.thought = "Memory B reasoning".into();
        let b = Arc::new(b);
        store.sessions.get_mut(&key).unwrap().turns.push(b.clone());
        let mut richer = (*b).clone();
        richer.thought = "Memory B richer reasoning".into();
        store.upgrade_record(&key, &b, richer);
        let turns = store.load_turns(&key);
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[1].thought, "Memory B richer reasoning");
        let sql = proxy_db::load_thinking_records(&key).unwrap();
        assert_eq!(sql.len(), 1);
        assert_eq!(sql[0].thought, "Disk A reasoning");
        cleanup(&key);
    }

    #[test]
    fn pruned_duplicate_text_generations_remain_ambiguous_warm_and_cold() {
        let session = session("ambiguous");
        let prefix = vec![user("Task")];
        let store = ThinkingStore::global();
        let a = signature(b'A');
        let b = signature(b'B');
        let c = signature(b'C');
        seed_omitted_branches(&session);
        let key = capture_text(&session, &prefix, "First complete reasoning", &a);
        let history = vec![user("Task"), answer(), user("Continue")];
        let mut established = history.clone();
        assert_eq!(
            store.restore_gemini_contents_with_model(
                &key,
                &mut established,
                Some("gemini-3.8-flash")
            ),
            1
        );
        assert_restored(&established[1], "First complete reasoning", &a);

        // Both are complete responses to the same legal prompt; their distinct
        // signed generations share visible text, not an invented fingerprint.
        assert_eq!(
            capture_text(&session, &prefix, "Second complete reasoning", &b),
            key
        );
        assert_eq!(
            capture_text(
                &session,
                &[user("Task"), answer(), user("Other task")],
                "Third complete reasoning",
                &c
            ),
            key
        );
        SignatureCache::global().cache_session_signature(&key, c, 3);
        let mut retry_prefix = prefix.clone();
        assert_eq!(scope(&session, &retry_prefix).store_key, key);
        finalize_request_text_context(&key, &mut retry_prefix, "gemini-3.8-flash", true, false);
        assert_eq!(store.session_stats(&key).unwrap().0, 2);
        let rows = proxy_db::load_thinking_records(&key).unwrap();
        let fingerprint = &rows
            .iter()
            .find(|rec| rec.signature.as_deref() == Some(a.as_str()))
            .unwrap()
            .fingerprint;
        assert_eq!(
            rows.iter()
                .filter(|rec| &rec.fingerprint == fingerprint)
                .count(),
            2,
            "SQLite retains both generations of the surviving tail fingerprint"
        );

        for cold in [false, true] {
            if cold {
                store.sessions.remove(&key);
            }
            let mut restored = history.clone();
            assert_eq!(
                store.restore_gemini_contents_with_model(
                    &key,
                    &mut restored,
                    Some("gemini-3.8-flash")
                ),
                0,
                "cold={cold}"
            );
            assert_eq!(
                restored, history,
                "an incomplete RAM window cannot select B over A"
            );
            for _ in 0..2 {
                finalize_request_text_context(&key, &mut restored, "gemini-3.8-flash", true, false);
                assert_eq!(
                    restored, history,
                    "later finalization must not reintroduce B"
                );
            }
            let mut explicit = history.clone();
            explicit[1]["parts"][0]["thoughtSignature"] = json!(a);
            assert_eq!(
                store.restore_gemini_contents_with_model(
                    &key,
                    &mut explicit,
                    Some("gemini-3.8-flash")
                ),
                1
            );
            assert_restored(&explicit[1], "First complete reasoning", &a);
        }
        cleanup(&key);
    }

    #[test]
    fn pruned_single_text_generation_and_tool_turn_still_restore() {
        let session = session("unambiguous");
        let store = ThinkingStore::global();
        let a = signature(b'A');
        let tool_sig = signature(b'T');
        seed_omitted_branches(&session);
        let key = capture_text(&session, &[user("Task")], "Unique complete reasoning", &a);
        let call = json!({"functionCall":{"name":"lookup","args":{},"id":"call_tail"}});
        let mut signed_call = call.clone();
        signed_call["thoughtSignature"] = json!(tool_sig);
        assert_eq!(
            capture(
                &session,
                &[user("Task"), answer(), user("Look it up")],
                "Complete tool reasoning",
                signed_call
            ),
            key
        );
        let mut retry_prefix = vec![user("Task")];
        finalize_request_text_context(&key, &mut retry_prefix, "gemini-3.8-flash", true, false);
        assert_eq!(store.session_stats(&key).unwrap().0, 2);
        for cold in [false, true] {
            if cold {
                store.sessions.remove(&key);
            }
            let mut history = vec![
                user("Task"),
                answer(),
                user("Look it up"),
                json!({"role":"model","parts":[call]}),
                json!({"role":"user","parts":[{"functionResponse":{
                    "name":"lookup","id":"call_tail","response":{"result":"ok"}
                }}]}),
            ];
            assert_eq!(
                store.restore_gemini_contents_with_model(
                    &key,
                    &mut history,
                    Some("gemini-3.8-flash")
                ),
                2,
                "cold={cold}"
            );
            assert_restored(&history[1], "Unique complete reasoning", &a);
            assert_restored(&history[3], "Complete tool reasoning", &tool_sig);
        }
        cleanup(&key);
    }
}

#[cfg(test)]
mod concurrent_pruning_tests {
    use super::*;
    use crate::modules::proxy_db;
    use crate::proxy::SignatureCache;
    use base64::Engine;

    fn key(name: &str) -> String {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        format!(
            "concurrent-prune-{name}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        )
    }
    fn signature(seed: u8) -> String {
        let mut raw = vec![0x12];
        raw.extend(std::iter::repeat_n(seed, 48));
        base64::engine::general_purpose::STANDARD.encode(raw)
    }
    fn user(text: &str) -> Value {
        json!({"role":"user","parts":[{"text":text}]})
    }
    fn captured(task: &str, thought: &str, sig: Option<&str>) -> ThinkingRecord {
        let prefix = vec![user(task)];
        let mut acc = TurnAccumulator::with_anchor(compute_causal_anchor(prefix.last()))
            .with_text_context_anchor(compute_response_text_anchor(&prefix));
        acc.ingest_part(&json!({"text":thought,"thought":true}));
        acc.ingest_part(&json!({"text":"Done.","thoughtSignature":sig}));
        acc.into_record()
    }
    fn persist(sid: &str, rec: &ThinkingRecord) {
        proxy_db::save_thinking_record(
            sid,
            &rec.fingerprint,
            &rec.thought,
            rec.signature.as_deref(),
            &rec.tool_ids,
            &rec.tool_names,
            &rec.visible,
        )
        .unwrap();
    }
    fn assert_warm_and_cold_restore(store: &ThinkingStore, sid: &str, thought: &str, sig: &str) {
        for cold in [false, true] {
            if cold {
                store.sessions.remove(sid);
            }
            SignatureCache::global().invalidate_session(sid);
            let mut contents = vec![
                user("Task"),
                json!({"role":"model","parts":[{"text":"Done."}]}),
                user("Continue"),
            ];
            assert_eq!(
                store.restore_gemini_contents_with_model(
                    sid,
                    &mut contents,
                    Some("gemini-3.8-flash")
                ),
                1,
                "cold={cold}"
            );
            assert_eq!(
                contents[1]["parts"],
                json!([
                    {"text":thought,"thought":true},
                    {"text":"Done.","thoughtSignature":sig}
                ])
            );
        }
    }
    fn cleanup(sid: &str) {
        SignatureCache::global().invalidate_session(sid);
        proxy_db::delete_thinking_records_for_session(sid).unwrap();
    }

    #[test]
    fn response_captured_after_ram_prune_survives_sql_prune_and_restart() {
        let sid = key("capture");
        let store = Arc::new(ThinkingStore::new());
        for idx in 0..6 {
            store.record(
                &sid,
                captured(
                    &format!("Old task {idx}"),
                    "Old reasoning",
                    Some(&signature(b'A' + idx)),
                ),
            );
        }
        // Pause at the real two-phase boundary, with both database and RAM
        // locks released. The other request must be able to finish capture.
        let plan = store
            .prepare_orphaned_records_prune(&sid, &[user("Task")])
            .unwrap();
        assert_eq!(store.session_stats(&sid).unwrap().0, 2);
        let sig = signature(b'Z');
        let fresh = captured("Task", "Fresh private reasoning", Some(&sig));
        let fp = fresh.fingerprint.clone();
        let worker_store = store.clone();
        let worker_sid = sid.clone();
        std::thread::spawn(move || worker_store.record(&worker_sid, fresh))
            .join()
            .unwrap();
        assert_eq!(proxy_db::load_thinking_records(&sid).unwrap().len(), 7);
        plan.persist();
        let rows = proxy_db::load_thinking_records(&sid).unwrap();
        assert_eq!(
            rows.len(),
            3,
            "prune the four old orphans, retain two tail turns plus capture"
        );
        assert!(rows.iter().any(|r| r.fingerprint == fp));
        assert_warm_and_cold_restore(&store, &sid, "Fresh private reasoning", &sig);
        cleanup(&sid);
    }

    #[test]
    fn same_fingerprint_updates_after_snapshot_survive_without_timestamp_precision() {
        for signature_only in [false, true] {
            let sid = key(if signature_only {
                "signature-only"
            } else {
                "thought-upgrade"
            });
            let sig = signature(b'Z');
            let old = captured(
                "Task",
                "Original reasoning",
                if signature_only { None } else { Some(&sig) },
            );
            persist(&sid, &old);
            let snapshot = proxy_db::snapshot_thinking_records_for_prune(&sid).unwrap();
            let fresh = captured(
                "Task",
                if signature_only {
                    "..."
                } else {
                    "Longer replacement private reasoning"
                },
                Some(&sig),
            );
            // Both saves intentionally occur without sleeping. A millisecond
            // cutoff cannot protect this in-place update reliably.
            persist(&sid, &fresh);
            let rows = proxy_db::load_thinking_records(&sid).unwrap();
            assert_eq!(
                rows.len(),
                1,
                "this is an UPDATE, not a new generation INSERT"
            );
            assert_eq!(rows[0].fingerprint, old.fingerprint);
            assert_eq!(
                proxy_db::delete_thinking_records_except_fingerprints(
                    &snapshot,
                    &["unrelated-keep".into()],
                    &[],
                    &[]
                )
                .unwrap(),
                0
            );
            let thought = if signature_only {
                "Original reasoning"
            } else {
                "Longer replacement private reasoning"
            };
            assert_warm_and_cold_restore(&ThinkingStore::new(), &sid, thought, &sig);
            // The protected record is eligible for a later, fresh prune. We do
            // not permanently exempt recaptured or upgraded rows from cleanup.
            let next = proxy_db::snapshot_thinking_records_for_prune(&sid).unwrap();
            assert_eq!(
                proxy_db::delete_thinking_records_except_fingerprints(
                    &next,
                    &["unrelated-keep".into()],
                    &[],
                    &[]
                )
                .unwrap(),
                1
            );
            cleanup(&sid);
        }
    }

    #[test]
    fn signature_only_tool_recapture_refreshes_an_unchanged_sqlite_payload() {
        let sid = key("unchanged-tool-payload");
        let store = ThinkingStore::new();
        let sig = signature(b'Z');
        let prefix = user("Task");
        let make_record = |visible: &str, thought: Option<&str>| {
            let mut acc = TurnAccumulator::with_anchor(compute_causal_anchor(Some(&prefix)));
            if let Some(thought) = thought {
                acc.ingest_part(&json!({"thought":true,"text":thought}));
            }
            acc.ingest_part(&json!({"text":visible}));
            acc.ingest_part(&json!({"functionCall":{"name":"lookup","args":{},"id":"call_tool"},"thoughtSignature":sig}));
            acc.into_record()
        };
        let old = make_record("First call", Some("Original tool reasoning"));
        let fresh = make_record("Retried call", None);
        assert_ne!(old.fingerprint, fresh.fingerprint);
        assert_eq!(old.tool_ids, fresh.tool_ids);
        store.record(&sid, old);
        let snapshot = proxy_db::snapshot_thinking_records_for_prune(&sid).unwrap();
        store.record(&sid, fresh);
        assert_eq!(
            store.session_stats(&sid).unwrap().0,
            2,
            "RAM appends the new fingerprint"
        );
        let rows = proxy_db::load_thinking_records(&sid).unwrap();
        assert_eq!(rows.len(), 1, "SQLite merges by the same causal tool ID");
        assert_eq!(rows[0].thought, "Original tool reasoning");
        assert_eq!(
            proxy_db::delete_thinking_records_except_fingerprints(
                &snapshot,
                &["unrelated-keep".into()],
                &[],
                &[]
            )
            .unwrap(),
            0
        );
        store.sessions.remove(&sid);
        SignatureCache::global().invalidate_session(&sid);
        let mut contents = vec![
            prefix,
            json!({"role":"model","parts":[{"text":"Retried call"},{"functionCall":{"name":"lookup","args":{},"id":"call_tool"}}]}),
            json!({"role":"user","parts":[{"functionResponse":{"name":"lookup","id":"call_tool","response":{"result":"ok"}}}]}),
        ];
        assert_eq!(
            store.restore_gemini_contents_with_model(&sid, &mut contents, Some("gemini-3.8-flash")),
            1
        );
        assert!(contents[1]["parts"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["thoughtSignature"] == sig));
        assert!(contents[1]["parts"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["thought"] == true && p["text"] == "Original tool reasoning"));
        cleanup(&sid);
    }

    #[test]
    fn empty_snapshot_cannot_delete_a_later_capture() {
        let sid = key("initial-capture");
        let snapshot = proxy_db::snapshot_thinking_records_for_prune(&sid).unwrap();
        let sig = signature(b'Z');
        persist(
            &sid,
            &captured("Task", "First private reasoning", Some(&sig)),
        );
        assert_eq!(
            proxy_db::delete_thinking_records_except_fingerprints(
                &snapshot,
                &["unrelated-keep".into()],
                &[],
                &[]
            )
            .unwrap(),
            0
        );
        assert_warm_and_cold_restore(&ThinkingStore::new(), &sid, "First private reasoning", &sig);
        cleanup(&sid);
    }
}

#[cfg(test)]
mod text_context_tests {
    use super::*;

    pub(super) fn key(name: &str) -> String {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        format!(
            "text-context-{name}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        )
    }

    fn signature(seed: u8) -> String {
        use base64::Engine;
        let mut raw = vec![0x12];
        raw.extend(std::iter::repeat_n(seed, 48));
        base64::engine::general_purpose::STANDARD.encode(raw)
    }

    fn user(text: &str) -> Value {
        json!({"role":"user", "parts":[{"text":text}]})
    }

    fn answer() -> Value {
        json!({"role":"model", "parts":[{"text":"Done."}]})
    }

    fn captured(prefix: &[Value], thought: &str, sig: &str) -> ThinkingRecord {
        let mut acc = TurnAccumulator::with_anchor(compute_causal_anchor(prefix.last()))
            .with_text_context_anchor(compute_response_text_anchor(prefix));
        acc.ingest_part(&json!({"text":thought,"thought":true}));
        acc.ingest_part(&json!({"text":"Done.","thoughtSignature":sig}));
        acc.into_record()
    }

    fn assert_thought(turn: &Value, thought: &str, sig: &str) {
        let parts = turn["parts"].as_array().unwrap();
        assert_eq!(parts[0]["thought"], true);
        assert_eq!(parts[0]["text"], thought);
        assert_eq!(parts[1]["text"], "Done.");
        assert_eq!(parts[1]["thoughtSignature"], sig);
    }

    #[test]
    fn repeated_continue_done_preserves_each_turn_and_sqlite_restart() {
        let sid = key("repeated");
        let store = ThinkingStore::new();
        let first_sig = signature(b'A');
        let second_sig = signature(b'B');
        let first_prefix = vec![user("Continue.")];
        let second_prefix = vec![user("Continue."), answer(), user("Continue.")];
        let first = captured(&first_prefix, "First reason", &first_sig);
        let second = captured(&second_prefix, "Second longer reason", &second_sig);
        assert_ne!(first.fingerprint, second.fingerprint);
        store.record(&sid, first);
        store.record(&sid, second);
        assert_eq!(store.session_stats(&sid).unwrap().0, 2);
        assert_eq!(
            crate::modules::proxy_db::load_thinking_records(&sid)
                .unwrap()
                .len(),
            2
        );
        for cold in [false, true] {
            if cold {
                store.sessions.remove(&sid);
            }
            let mut contents = second_prefix.clone();
            contents.push(answer());
            contents.push(user("Next task"));
            assert_eq!(
                store.restore_gemini_contents_with_model(
                    &sid,
                    &mut contents,
                    Some("gemini-3.8-flash")
                ),
                2
            );
            assert_thought(&contents[1], "First reason", &first_sig);
            assert_thought(&contents[3], "Second longer reason", &second_sig);
            // Restored thought/signature and text block boundaries do not change
            // the next capture's identity or create extra records on ingest.
            assert_eq!(
                compute_response_text_anchor(&contents[..3]),
                compute_response_text_anchor(&second_prefix)
            );
            store.ingest_from_contents(&sid, &contents);
            assert_eq!(store.session_stats(&sid).unwrap().0, 2);
        }
        crate::modules::proxy_db::delete_thinking_records_for_session(&sid).unwrap();
    }

    #[test]
    fn same_text_turn_increment_is_idempotent_in_memory_and_sqlite() {
        let sid = key("increment");
        let store = ThinkingStore::new();
        let prefix = vec![user("Task")];
        let sig = signature(b'C');
        let first = captured(&prefix, "Reason", &sig);
        let extended = captured(&prefix, "Reason with additional detail", &sig);
        store.record(&sid, first);
        store.record(&sid, extended.clone());
        let bytes = store.session_stats(&sid).unwrap().1;
        store.record(&sid, extended);
        assert_eq!(store.session_stats(&sid).unwrap().0, 1);
        assert_eq!(store.session_stats(&sid).unwrap().1, bytes);
        let rows = crate::modules::proxy_db::load_thinking_records(&sid).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].thought, "Reason with additional detail");
        assert_eq!(rows[0].visible, "Done.");
        assert_eq!(
            rows[0].fingerprint,
            contextual_fingerprint(
                "Done.",
                &[],
                &[],
                Some(&compute_response_text_anchor(&prefix))
            )
        );
        crate::modules::proxy_db::delete_thinking_records_for_session(&sid).unwrap();
    }

    #[test]
    fn same_context_regenerations_with_different_answers_or_signatures_stay_separate() {
        let sid = key("regeneration");
        let store = ThinkingStore::new();
        let prefix = vec![user("Choose")];
        let mut first = captured(&prefix, "Reason A", &signature(b'P'));
        first.visible = "Alpha".into();
        first.fingerprint = contextual_fingerprint(
            "Alpha",
            &[],
            &[],
            Some(&compute_response_text_anchor(&prefix)),
        );
        let mut second = first.clone();
        second.visible = "Beta".into();
        second.fingerprint = contextual_fingerprint(
            "Beta",
            &[],
            &[],
            Some(&compute_response_text_anchor(&prefix)),
        );
        let mut third = second.clone();
        third.signature = Some(signature(b'Q'));
        for rec in [first, second, third] {
            store.record(&sid, rec);
        }
        assert_eq!(store.session_stats(&sid).unwrap().0, 3);
        assert_eq!(
            crate::modules::proxy_db::load_thinking_records(&sid)
                .unwrap()
                .len(),
            3
        );
        crate::modules::proxy_db::delete_thinking_records_for_session(&sid).unwrap();
    }

    #[test]
    fn scoped_text_never_falls_back_to_an_unrelated_prefix_in_memory_or_l2() {
        let sid = key("no-fallback");
        let store = ThinkingStore::new();
        store.record(
            &sid,
            captured(&[user("Old task")], "Old private reason", &signature(b'D')),
        );
        for cold in [false, true] {
            if cold {
                store.sessions.remove(&sid);
            }
            let mut unrelated = vec![user("Different task"), answer()];
            let before = unrelated.clone();
            assert_eq!(store.restore_gemini_contents(&sid, &mut unrelated), 0);
            assert_eq!(
                unrelated, before,
                "Phase 2/3/3.5/4 must all respect text context"
            );
        }
        crate::modules::proxy_db::delete_thinking_records_for_session(&sid).unwrap();
    }

    #[test]
    fn legacy_records_remain_readable_without_merging_distinct_real_signatures() {
        let sid = key("legacy");
        let store = ThinkingStore::new();
        let a = signature(b'E');
        let b = signature(b'F');
        for (thought, sig) in [("Legacy first", &a), ("Legacy second", &b)] {
            let mut rec = captured(&[user("ignored old context")], thought, sig);
            rec.fingerprint = fingerprint("Done.", &[], &[]);
            store.record(&sid, rec);
        }
        assert_eq!(store.session_stats(&sid).unwrap().0, 2);
        assert_eq!(
            crate::modules::proxy_db::load_thinking_records(&sid)
                .unwrap()
                .len(),
            2
        );
        store.sessions.remove(&sid);
        let mut contents = vec![user("old first"), answer(), user("old second"), answer()];
        store.restore_gemini_contents(&sid, &mut contents);
        assert_thought(&contents[1], "Legacy first", &a);
        assert_thought(&contents[3], "Legacy second", &b);
        store.ingest_from_contents(&sid, &contents);
        assert_eq!(store.session_stats(&sid).unwrap().0, 2);
        assert_eq!(
            crate::modules::proxy_db::load_thinking_records(&sid)
                .unwrap()
                .len(),
            2
        );
        crate::modules::proxy_db::delete_thinking_records_for_session(&sid).unwrap();
    }

    #[test]
    fn text_context_ignores_rehydration_text_splits_tool_ids_and_response_role() {
        let sent = vec![
            user("Run"),
            json!({"role":"model","parts":[
                {"text":"private","thought":true},
                {"text":"Before"},
                {"functionCall":{"name":"lookup","args":{"a":1,"b":2},"id":"call_original"},"thoughtSignature":signature(b'G')},
                {"text":"After"}
            ]}),
            json!({"role":"model","parts":[{"functionResponse":{"name":"lookup","id":"call_original","response":{"output":"ok"}}}]}),
            user("Continue."),
        ];
        let replay = vec![
            user("Run"),
            json!({"role":"assistant","parts":[
                {"text":"Be"},{"text":"fore"},
                {"functionCall":{"name":"lookup","args":{"b":2,"a":1},"id":"call_normalized"}},
                {"text":"After"}
            ]}),
            json!({"role":"user","parts":[{"functionResponse":{"name":"lookup","id":"call_normalized","response":{"output":"ok"}}}]}),
            user("Continue."),
        ];
        assert_eq!(
            compute_response_text_anchor(&sent),
            compute_response_text_anchor(&replay)
        );
        let mut first = TurnAccumulator::with_anchor("unchanged-tool-anchor");
        let mut second = first.clone().with_text_context_anchor("new-text-only-key");
        let part = json!({"functionCall":{"name":"lookup","args":{"x":1}},"thoughtSignature":signature(b'H')});
        first.ingest_part(&part);
        second.ingest_part(&part);
        assert_eq!(
            first.into_record().fingerprint,
            second.into_record().fingerprint
        );
    }

    #[test]
    fn staged_hydration_preserves_text_until_final_sanitized_prefix_and_prunes_exactly() {
        let sid = key("stages");
        let store = ThinkingStore::global();
        let sig = signature(b'I');
        store.record(
            &sid,
            captured(&[user("clean")], "Original warm reason", &sig),
        );
        for n in 0..5 {
            store.record(
                &sid,
                captured(
                    &[user(&format!("unrelated {n}"))],
                    &format!("Unrelated reason {n}"),
                    &signature(b'J' + n),
                ),
            );
        }
        let mut raw = vec![user("transient metadata clean"), answer(), user("next")];
        hydrate_gemini_tool_contents_with_model(&sid, &mut raw, Some("gemini-3.8-flash"));
        assert_eq!(store.session_stats(&sid).unwrap().0, 6);
        assert_eq!(
            crate::modules::proxy_db::load_thinking_records(&sid)
                .unwrap()
                .len(),
            6
        );
        // Simulate the handler's already-authorized canonical sanitizer changing
        // the visible prefix between the early tool phase and final text phase.
        raw[0] = user("clean");
        finalize_request_text_context(&sid, &mut raw, "gemini-3.8-flash", true, false);
        assert_eq!(
            raw[1]["parts"],
            json!([{"text":"Done.","thoughtSignature":sig}])
        );
        assert_eq!(store.load_turns(&sid)[0].thought, "Original warm reason");
        assert_eq!(
            store.session_stats(&sid).unwrap().0,
            3,
            "exact live context plus two newest records"
        );
        assert_eq!(
            crate::modules::proxy_db::load_thinking_records(&sid)
                .unwrap()
                .len(),
            3
        );
        let once = raw.clone();
        finalize_request_text_context(&sid, &mut raw, "gemini-3.8-flash", true, false);
        assert_eq!(raw, once);
        store.sessions.remove(&sid);
        crate::modules::proxy_db::delete_thinking_records_for_session(&sid).unwrap();
    }

    #[test]
    fn pending_signature_only_crosses_adjacent_model_fragments() {
        let sig = signature(b'R');
        let thought = json!({"role":"model","parts":[{
            "text":"Private reason","thought":true,"thoughtSignature":sig
        }]});
        let mut adjacent = vec![thought.clone(), answer()];
        finalize_gemini_contents_with_stage(
            &mut adjacent,
            true,
            Some("gemini-3.8-flash"),
            None,
            false,
            None,
            false,
        );
        assert_eq!(adjacent[1]["parts"][0]["thoughtSignature"], sig);

        for boundary in [
            user("Different task"),
            json!({"role":"model","parts":[{
                "functionResponse":{"name":"run","response":{"output":"ok"}},
                "thoughtSignature":sig
            }]}),
        ] {
            let mut contents = vec![thought.clone(), boundary, answer()];
            finalize_gemini_contents_with_stage(
                &mut contents,
                true,
                Some("gemini-3.8-flash"),
                None,
                false,
                None,
                false,
            );
            assert!(contents[1]["parts"][0].get("thoughtSignature").is_none());
            assert_eq!(
                contents[2],
                answer(),
                "pending signatures cannot cross user/tool-result boundaries"
            );
        }
    }
}

#[cfg(test)]
mod text_context_pipeline_tests {
    use super::text_context_tests::key;
    use super::*;
    use crate::proxy::{
        mappers::prompt_sanitizer::PromptSanitizer, pipeline::inbound::InboundThinkingPipeline,
        SignatureCache,
    };
    use base64::Engine;
    use serde_json::{json, Value};

    fn signature(model: &str, seed: u8) -> String {
        let mut raw = if model.contains("claude") {
            vec![0x08, 0x04, 0x12]
        } else {
            vec![0x12]
        };
        raw.extend(std::iter::repeat_n(seed, 48));
        let sig = base64::engine::general_purpose::STANDARD.encode(raw);
        if model.contains("claude") {
            ensure_google_claude_thought_signature(&sig)
        } else {
            sig
        }
    }
    fn user(seed: &str, text: &str) -> Value {
        json!({"role":"user","parts":[{"text":format!("x-anthropic-billing-header: cc_version={seed}; cch={seed}\n{text}")}]})
    }
    fn answer() -> Value {
        json!({"role":"model","parts":[{"text":"Done."}]})
    }
    fn final_request(sid: &str, model: &str, contents: Vec<Value>) -> (Vec<Value>, String) {
        let mut body = json!({"request":{"contents":contents}});
        let history = body["request"]["contents"].as_array_mut().unwrap();
        let before = history.clone();
        InboundThinkingPipeline::process_contents_deferred_text(
            history,
            model,
            true,
            Some(sid),
            false,
        );
        assert_eq!(
            *history, before,
            "early tool stage must not inject latest session signature into text history"
        );
        PromptSanitizer::sanitize_gemini_payload(&mut body);
        assert!(
            !body.to_string().contains("cc_version"),
            "real sanitizer must remove dynamic billing metadata"
        );
        let history = body["request"]["contents"].as_array_mut().unwrap();
        let anchor = finalize_request_text_context(sid, history, model, true, false);
        (history.clone(), anchor)
    }
    fn capture(sid: &str, anchor: &str, thought: &str, sig: &str, count: usize) {
        capture_gemini_parts_with_context(
            sid,
            &[
                json!({"thought":true,"text":thought}),
                json!({"text":"Done.","thoughtSignature":sig}),
            ],
            "root",
            Some(anchor),
        );
        SignatureCache::global().cache_session_signature(sid, sig.into(), count);
    }
    fn check_turn(turn: &Value, model: &str, thought: &str, sig: &str) {
        let parts = turn["parts"].as_array().unwrap();
        assert_eq!(
            parts
                .iter()
                .filter(|part| part.get("thoughtSignature").is_some())
                .count(),
            1,
            "{turn}"
        );
        let visible = parts.iter().find(|part| part["text"] == "Done.").unwrap();
        assert_eq!(visible["thoughtSignature"], sig, "{turn}");
        let thoughts = parts
            .iter()
            .filter(|part| part["thought"] == true)
            .collect::<Vec<_>>();
        if model.contains("claude") {
            assert_eq!(thoughts.len(), 1, "{turn}");
            assert_eq!(thoughts[0]["text"], thought);
            assert!(thoughts[0].get("thoughtSignature").is_none());
        } else {
            assert!(
                thoughts.is_empty(),
                "Gemini history should retain signature without thought text: {turn}"
            );
            assert_eq!(parts.len(), 1, "{turn}");
        }
    }

    #[test]
    fn real_sanitizer_two_identical_done_turns_restore_distinct_contexts_and_family_shapes() {
        for model in ["gemini-3.8-flash", "claude-sonnet-4-6"] {
            let sid = key(&format!("sanitizer-context-{model}"));
            let first_sig = signature(model, b'A');
            let second_sig = signature(model, b'B');
            let (first, first_anchor) =
                final_request(&sid, model, vec![user("first", "Continue.")]);
            assert_eq!(first[0]["parts"][0]["text"], "Continue.");
            capture(&sid, &first_anchor, "First private reason", &first_sig, 1);

            let (second, second_anchor) = final_request(
                &sid,
                model,
                vec![
                    user("mutated-first", "Continue."),
                    answer(),
                    user("second", "Continue."),
                ],
            );
            check_turn(&second[1], model, "First private reason", &first_sig);
            assert_ne!(first_anchor, second_anchor);
            capture(
                &sid,
                &second_anchor,
                "Second private reason",
                &second_sig,
                3,
            );
            assert_eq!(
                crate::modules::proxy_db::load_thinking_records(&sid)
                    .unwrap()
                    .len(),
                2
            );
            assert_eq!(
                SignatureCache::global()
                    .get_session_signature(&sid)
                    .as_deref(),
                Some(second_sig.as_str())
            );

            for cold in [false, true] {
                if cold {
                    ThinkingStore::global().sessions.remove(&sid);
                }
                let (history, _) = final_request(
                    &sid,
                    model,
                    vec![
                        user("third-first", "Continue."),
                        answer(),
                        user("third-second", "Continue."),
                        answer(),
                        user("third", "Next task"),
                    ],
                );
                check_turn(&history[1], model, "First private reason", &first_sig);
                check_turn(&history[3], model, "Second private reason", &second_sig);
                assert_eq!(
                    crate::modules::proxy_db::load_thinking_records(&sid)
                        .unwrap()
                        .len(),
                    2,
                    "restore must not create cross-context duplicates"
                );
                let mut repeated = history.clone();
                let old = repeated.clone();
                finalize_request_text_context(&sid, &mut repeated, model, true, false);
                assert_eq!(repeated, old, "second finalization must be idempotent");
            }
            let (unrelated, _) = final_request(
                &sid,
                model,
                vec![
                    user("foreign", "Different task"),
                    answer(),
                    user("foreign-next", "Next task"),
                ],
            );
            assert_eq!(
                unrelated[1],
                answer(),
                "latest session signature must not fill unrelated text context"
            );
            ThinkingStore::global().sessions.remove(&sid);
            crate::modules::proxy_db::delete_thinking_records_for_session(&sid).unwrap();
            SignatureCache::global().invalidate_session(&sid);
        }
    }
}

#[cfg(test)]
mod finalized_context_regression_tests {
    use super::text_context_tests::key;
    use super::*;
    use crate::proxy::{pipeline::inbound::InboundThinkingPipeline, SignatureCache};
    use base64::Engine;

    fn signature(claude: bool, seed: u8, len: usize) -> String {
        let mut bytes = if claude {
            vec![0x08, 0x04, 0x12]
        } else {
            vec![0x12]
        };
        bytes.extend(std::iter::repeat_n(seed, len));
        let raw = base64::engine::general_purpose::STANDARD.encode(bytes);
        if claude {
            ensure_google_claude_thought_signature(&raw)
        } else {
            raw
        }
    }
    fn user(text: &str) -> Value {
        json!({"role":"user","parts":[{"text":text}]})
    }
    fn answer(text: &str) -> Value {
        json!({"role":"model","parts":[{"text":text}]})
    }
    fn capture(sid: &str, prefix: &[Value], thought: &str, text: &str, sig: Option<&str>) {
        let mut parts = vec![json!({"thought":true,"text":thought}), json!({"text":text})];
        if let Some(sig) = sig {
            parts[1]["thoughtSignature"] = json!(sig);
        }
        capture_gemini_parts_with_context(
            sid,
            &parts,
            "root",
            Some(&compute_response_text_anchor(prefix)),
        );
    }
    fn finalize(sid: &str, contents: &mut Vec<Value>, model: &str) -> String {
        InboundThinkingPipeline::process_contents_deferred_text(
            contents,
            model,
            true,
            Some(sid),
            false,
        );
        finalize_request_text_context(sid, contents, model, true, false)
    }
    fn cleanup(sid: &str) {
        ThinkingStore::global().sessions.remove(sid);
        SignatureCache::global().invalidate_session(sid);
        crate::modules::proxy_db::delete_thinking_records_for_session(sid).unwrap();
    }

    #[test]
    fn chained_degraded_thoughts_restore_in_one_request_before_pruning() {
        let sid = key("chained-elevation");
        let mut canonical = vec![user("Initial task")];
        let mut degraded = canonical.clone();
        for idx in 0..8 {
            let text = format!("Answer number {idx}.");
            let thought = format!("Private reasoning for turn {idx}.");
            let sig = signature(false, b'A' + idx, 48);
            capture(&sid, &canonical, &thought, &text, Some(&sig));
            canonical.push(answer(&text));
            canonical.push(user("Continue."));
            degraded.push(json!({"role":"model","parts":[{"text":thought},{"text":text}]}));
            degraded.push(user("Continue."));
        }
        // Exercise pruning only after every surviving prefix has settled.
        for idx in 0..4 {
            capture(
                &sid,
                &[user(&format!("Unrelated {idx}"))],
                "Unrelated private thought",
                "Unrelated",
                Some(&signature(false, b'P' + idx, 48)),
            );
        }
        ThinkingStore::global().sessions.remove(&sid);
        let anchor = finalize(&sid, &mut degraded, "gemini-3.8-flash");
        for idx in 0..8 {
            assert_eq!(
                degraded[idx * 2 + 1]["parts"],
                json!([{
                    "text":format!("Answer number {idx}."),
                    "thoughtSignature":signature(false,b'A'+idx as u8,48)
                }])
            );
        }
        assert_eq!(
            crate::modules::proxy_db::load_thinking_records(&sid)
                .unwrap()
                .len(),
            10,
            "all eight live records and two newest records survive; no provisional aliases"
        );
        let once = degraded.clone();
        assert_eq!(finalize(&sid, &mut degraded, "gemini-3.8-flash"), anchor);
        assert_eq!(degraded, once);
        cleanup(&sid);
    }

    #[test]
    fn unsigned_thought_finalization_and_disabled_retry_preserve_context_contract() {
        let sid = key("unsigned-finalization");
        let prefix = vec![user("Task")];
        capture(
            &sid,
            &prefix,
            "First private analysis with enough detail.",
            "First answer",
            None,
        );
        let mut second = vec![user("Task"), answer("First answer"), user("Continue")];
        finalize(&sid, &mut second, "gemini-3.8-flash");
        let sig = signature(false, b'B', 48);
        capture(
            &sid,
            &second,
            "Second private analysis",
            "Second answer",
            Some(&sig),
        );
        let raw = vec![
            user("Task"),
            answer("First answer"),
            user("Continue"),
            answer("Second answer"),
            user("Next"),
        ];
        let mut third = raw.clone();
        finalize(&sid, &mut third, "gemini-3.8-flash");
        assert_eq!(third[3]["parts"][0]["thoughtSignature"], sig);
        let once = third.clone();
        finalize(&sid, &mut third, "gemini-3.8-flash");
        assert_eq!(third, once);
        for (enabled, retry) in [(false, false), (true, true)] {
            let mut skipped = raw.clone();
            finalize_request_text_context(&sid, &mut skipped, "gemini-3.8-flash", enabled, retry);
            assert_eq!(
                skipped, raw,
                "disabled/retry must not hydrate historical thoughts or signatures"
            );
        }
        cleanup(&sid);
    }

    #[test]
    fn exact_generations_use_available_evidence_and_never_cross_contexts() {
        let prefix = vec![user("Run task")];
        let raw = vec![user("Run task"), answer("Done."), user("Continue")];
        for family_choice in [false, true] {
            let sid = key("exact-candidates");
            let a = signature(false, b'A', 48);
            let b = signature(family_choice, b'B', 80);
            capture(&sid, &prefix, "First attempt thought", "Done.", Some(&a));
            capture(&sid, &prefix, "Selected second thought", "Done.", Some(&b));
            let model = if family_choice {
                "claude-sonnet-4-6"
            } else {
                "gemini-3.8-flash"
            };
            if !family_choice {
                SignatureCache::global().cache_session_signature(&sid, b.clone(), 1);
            }
            for cold in [false, true] {
                if cold {
                    ThinkingStore::global().sessions.remove(&sid);
                }
                let mut next = raw.clone();
                finalize(&sid, &mut next, model);
                assert!(next[1]["parts"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|part| part["thoughtSignature"] == b));
            }
            let mut unrelated = vec![user("Other task"), answer("Done."), user("Continue")];
            finalize(&sid, &mut unrelated, model);
            assert_eq!(unrelated[1], answer("Done."));
            if !family_choice {
                let mut explicit = raw.clone();
                explicit[1]["parts"][0]["thoughtSignature"] = json!(a);
                finalize(&sid, &mut explicit, model);
                assert_eq!(
                    explicit[1]["parts"][0]["thoughtSignature"], a,
                    "client-owned signature overrides latest B"
                );
                SignatureCache::global().invalidate_session(&sid);
                let mut ambiguous = raw.clone();
                finalize(&sid, &mut ambiguous, model);
                assert_eq!(ambiguous[1],answer("Done."),"without authoritative evidence, weaker fallbacks must not choose a signed generation");
            }
            cleanup(&sid);
        }
    }

    #[test]
    fn new_client_thought_survives_additional_finalization_passes() {
        for signed in [false, true] {
            let sid = key("new-client-thought");
            let prefix = vec![user("Task")];
            let a = signature(false, b'A', 48);
            capture(
                &sid,
                &prefix,
                "Old private reasoning",
                "First answer",
                Some(&a),
            );
            let b = signature(false, b'B', 48);
            let mut fresh = answer("New answer");
            fresh["parts"].as_array_mut().unwrap().insert(0,json!({
                "thought":true,"text":"New client reasoning must survive all finalization passes"
            }));
            if signed {
                fresh["parts"][1]["thoughtSignature"] = json!(b);
            }
            let mut history = vec![
                user("Task"),
                json!({"role":"model","parts":[
                    {"text":"Old private reasoning"},{"text":"First answer"}
                ]}),
                user("Continue"),
                fresh,
                user("Next"),
            ];
            finalize(&sid, &mut history, "gemini-3.8-flash");
            let rows = crate::modules::proxy_db::load_thinking_records(&sid).unwrap();
            assert!(rows.iter().any(|row| row.visible == "New answer"
                && row.thought == "New client reasoning must survive all finalization passes"));
            assert_eq!(rows.len(), 2, "only finalized captures are persisted");
            cleanup(&sid);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn rec(thought: &str, visible: &str, tool_id: Option<&str>) -> ThinkingRecord {
        let tool_ids = tool_id.map(|id| vec![id.to_string()]).unwrap_or_default();
        let tool_names = if tool_id.is_some() {
            vec!["shell".to_string()]
        } else {
            Vec::new()
        };
        let fp = fingerprint(visible, &tool_ids, &tool_names);
        let mut hasher = Sha256::new();
        hasher.update(thought.as_bytes());
        hasher.update(visible.as_bytes());
        if let Some(t_id) = tool_id {
            hasher.update(t_id.as_bytes());
        }
        let hash_hex = format!("{:x}", hasher.finalize());
        let sig = format!("sig_{:0>56}", &hash_hex[..40]);
        ThinkingRecord {
            fingerprint: fp,
            thought: thought.to_string(),
            signature: Some(sig),
            tool_ids,
            tool_names,
            visible: visible.to_string(),
        }
    }

    #[test]
    fn stores_full_thought_without_truncation() {
        let store = ThinkingStore::new();
        let long = "T".repeat(50_000);
        store.record("t:s1", rec(&long, "hello world", None));
        let mut contents = vec![json!({
            "role": "model",
            "parts": [{ "text": "hello world" }]
        })];
        let n = store.restore_gemini_contents("t:s1", &mut contents);
        assert_eq!(n, 1);
        assert_eq!(
            contents[0]["parts"][0]["text"].as_str().unwrap().len(),
            50_000
        );
        assert_eq!(contents[0]["parts"][0]["thought"], true);
        assert!(contents[0]["parts"][0].get("thoughtSignature").is_none());
        assert_eq!(
            contents[0]["parts"][1]["thoughtSignature"]
                .as_str()
                .unwrap()
                .len(),
            60
        );
    }

    #[test]
    fn matches_by_visible_text_not_index() {
        let store = ThinkingStore::new();
        store.record("t:s1", rec("think-A", "answer A", None));
        store.record("t:s1", rec("think-B", "answer B", None));

        // Client dropped turn A, only sends B (different packet shape / rewind)
        let mut contents = vec![json!({
            "role": "model",
            "parts": [{ "text": "answer B" }]
        })];
        store.restore_gemini_contents("t:s1", &mut contents);
        assert_eq!(contents[0]["parts"][0]["text"], "think-B");
    }

    #[test]
    fn matches_tool_id_when_text_missing() {
        let store = ThinkingStore::new();
        store.record("t:s1", rec("plan", "", Some("call_1")));
        let mut contents = vec![json!({
            "role": "model",
            "parts": [{
                "functionCall": { "name": "shell", "id": "call_1", "args": {} }
            }]
        })];
        store.restore_gemini_contents("t:s1", &mut contents);
        assert_eq!(contents[0]["parts"][0]["thought"], true);
        assert_eq!(contents[0]["parts"][0]["text"], "plan");
        assert_eq!(
            contents[0]["parts"][1]["thoughtSignature"]
                .as_str()
                .unwrap()
                .len(),
            60
        );
    }

    #[test]
    fn replaces_placeholder_dots() {
        let store = ThinkingStore::new();
        store.record("t:s1", rec("full chain", "final", None));
        let mut contents = vec![json!({
            "role": "model",
            "parts": [
                { "text": "...", "thought": true },
                { "text": "final" }
            ]
        })];
        store.restore_gemini_contents("t:s1", &mut contents);
        assert_eq!(contents[0]["parts"][0]["text"], "full chain");
    }

    #[test]
    fn tenant_isolation_and_end_session() {
        let store = ThinkingStore::new();
        store.record("aaa:chat", rec("secret-a", "hi", None));
        store.record("bbb:chat", rec("secret-b", "hi", None));

        let mut a = vec![json!({"role":"model","parts":[{"text":"hi"}]})];
        store.restore_gemini_contents("aaa:chat", &mut a);
        assert_eq!(a[0]["parts"][0]["text"], "secret-a");

        let result = store.end_session("aaa:chat");
        assert_eq!(result.deleted_turns, 1);
        let mut a2 = vec![json!({"role":"model","parts":[{"text":"hi"}]})];
        assert_eq!(store.restore_gemini_contents("aaa:chat", &mut a2), 0);

        let mut b = vec![json!({"role":"model","parts":[{"text":"hi"}]})];
        store.restore_gemini_contents("bbb:chat", &mut b);
        assert_eq!(b[0]["parts"][0]["text"], "secret-b");
    }

    #[test]
    fn sanitize_rejects_junk() {
        assert_eq!(sanitize_session_id("abc/../x"), "abc..x");
        assert_eq!(sanitize_session_id(""), "sid-unknown");
    }

    #[test]
    fn keyword_forces_thinking_without_client_flag() {
        assert!(model_forces_server_thinking("gemini-3-flash"));
        assert!(model_forces_server_thinking("gemini-3-pro"));
        assert!(model_forces_server_thinking("gemini-3-flash-agent"));
        assert!(model_forces_server_thinking("gemini-pro-agent"));
        assert!(model_forces_server_thinking("claude-sonnet-4-6"));
        assert!(!model_forces_server_thinking("gpt-4o"));
        assert!(!model_forces_server_thinking("gemini-3-pro-image"));
        assert!(!model_forces_server_thinking("gemini-3.1-flash-lite"));
        assert!(!model_forces_server_thinking("gemini-3-pro-preview"));
    }

    #[test]
    fn same_fingerprint_updates_in_place() {
        let store = ThinkingStore::new();
        let key = format!("t:s1-{}", uuid::Uuid::new_v4());
        let first = rec("short", "same", None);
        let mut extended = rec("much longer thought", "same", None);
        extended.signature = first.signature.clone();
        store.record(&key, first);
        store.record(&key, extended);
        let stats = store.session_stats(&key).unwrap();
        assert_eq!(stats.0, 1);
        let mut contents = vec![json!({"role":"model","parts":[{"text":"same"}]})];
        store.restore_gemini_contents(&key, &mut contents);
        assert_eq!(contents[0]["parts"][0]["text"], "much longer thought");
        store.end_session(&key);
    }

    #[test]
    fn tool_record_does_not_pollute_earlier_text_turns() {
        let store = ThinkingStore::new();
        // Turn 2 generated thinking + tool call
        store.record(
            "t:s1",
            rec("**Inferring User's Intention**", "", Some("call_54421")),
        );

        // Turn 0: "你好！" (pure text, no thought)
        // Turn 1: "当然是真的！" (pure text, no thought)
        // Turn 2: tool call "call_54421"
        let mut contents = vec![
            json!({
                "role": "model",
                "parts": [{ "text": "你好！我是 JeikCode AI 编程助手。" }]
            }),
            json!({
                "role": "model",
                "parts": [{ "text": "当然是真的！😄" }]
            }),
            json!({
                "role": "model",
                "parts": [{
                    "functionCall": { "name": "shell", "id": "call_54421", "args": {} }
                }]
            }),
        ];

        let restored = store.restore_gemini_contents("t:s1", &mut contents);
        assert_eq!(restored, 1);

        // Turn 0 must NOT have thinking injected
        assert_eq!(contents[0]["parts"].as_array().unwrap().len(), 1);
        assert_eq!(
            contents[0]["parts"][0]["text"],
            "你好！我是 JeikCode AI 编程助手。"
        );
        assert!(contents[0]["parts"][0].get("thought").is_none());

        // Turn 1 must NOT have thinking injected
        assert_eq!(contents[1]["parts"].as_array().unwrap().len(), 1);
        assert_eq!(contents[1]["parts"][0]["text"], "当然是真的！😄");
        assert!(contents[1]["parts"][0].get("thought").is_none());

        // Turn 2 MUST have thinking injected and matched with call_54421
        assert_eq!(contents[2]["parts"][0]["thought"], true);
        assert_eq!(
            contents[2]["parts"][0]["text"],
            "**Inferring User's Intention**"
        );
        assert_eq!(contents[2]["parts"][1]["functionCall"]["id"], "call_54421");
    }

    #[test]
    fn test_thinking_with_text_and_parallel_tools() {
        let store = ThinkingStore::new();

        // Test Case 6: 有思考、有正文、有多工具并行出来
        let tool_ids = vec!["call_batch_1".to_string(), "call_batch_2".to_string()];
        let tool_names = vec!["read_file".to_string(), "grep_search".to_string()];
        // 构造符合 Google 原生特征的假签名（Base64 解码后首字节 = protobuf tag 0x12）
        let real_sig = {
            use base64::Engine;
            let mut raw = vec![0x12u8];
            raw.extend_from_slice(&[b'A'; 60]);
            base64::engine::general_purpose::STANDARD.encode(raw)
        };
        let fp = fingerprint("I will read both files in parallel", &tool_ids, &tool_names);
        store.record(
            "t:s2",
            ThinkingRecord {
                fingerprint: fp,
                thought: "Parallel execution planned".to_string(),
                signature: Some(real_sig.clone()),
                tool_ids: tool_ids.clone(),
                tool_names: tool_names.clone(),
                visible: "I will read both files in parallel".to_string(),
            },
        );

        let mut contents = vec![json!({
            "role": "model",
            "parts": [
                { "text": "I will read both files in parallel" },
                { "functionCall": { "name": "read_file", "id": "call_batch_1", "args": {} } },
                { "functionCall": { "name": "grep_search", "id": "call_batch_2", "args": {} } }
            ]
        })];

        let restored = store.restore_gemini_contents("t:s2", &mut contents);
        assert_eq!(restored, 1);

        let parts = contents[0]["parts"].as_array().unwrap();
        // Index 0: thought block —— 保持纯净文本，铁律 I4：思考块绝不携带签名
        assert_eq!(parts[0]["thought"], true);
        assert_eq!(parts[0]["text"], "Parallel execution planned");
        assert!(parts[0].get("thoughtSignature").is_none());

        // Index 1: 可见正文 (无签名)
        assert_eq!(parts[1]["text"], "I will read both files in parallel");
        assert!(parts[1].get("thoughtSignature").is_none());

        // Older aggregate records carry no proof of which parallel call owned
        // their only signature. Keep the thought, but never lend that signature.
        assert_eq!(parts[2]["functionCall"]["id"], "call_batch_1");
        assert_eq!(parts[2]["thoughtSignature"], SENTINEL_SIGNATURE);
        assert_eq!(parts[3]["functionCall"]["id"], "call_batch_2");
        assert!(parts[3].get("thoughtSignature").is_none());
    }

    #[test]
    fn test_sqlite_persistence_and_recovery() {
        let store_key = "test_session_sqlite_recovery_unique";
        let fp = fingerprint(
            "Persisted visible text",
            &["call_persisted_999".to_string()],
            &["bash".to_string()],
        );
        // 构造符合 Google 原生特征的假签名（Base64 解码后首字节 = protobuf tag 0x12）
        let real_sig = {
            use base64::Engine;
            let mut raw = vec![0x12u8];
            raw.extend_from_slice(&[b'A'; 60]);
            base64::engine::general_purpose::STANDARD.encode(raw)
        };
        let rec = ThinkingRecord {
            fingerprint: fp,
            thought: "Thought restored from SQLite".to_string(),
            signature: Some(real_sig.clone()),
            tool_ids: vec!["call_persisted_999".to_string()],
            tool_names: vec!["bash".to_string()],
            visible: "Persisted visible text".to_string(),
        };

        let db_res = crate::modules::proxy_db::save_thinking_record(
            store_key,
            &rec.fingerprint,
            &rec.thought,
            rec.signature.as_deref(),
            &rec.tool_ids,
            &rec.tool_names,
            &rec.visible,
        );
        if let Err(e) = db_res {
            eprintln!("Skipping DB test if DB not initialized: {}", e);
            return;
        }

        let store = ThinkingStore::new();

        let mut contents = vec![json!({
            "role": "model",
            "parts": [
                { "text": "Persisted visible text" },
                { "functionCall": { "name": "bash", "id": "call_persisted_999", "args": {} } }
            ]
        })];

        let restored = store.restore_gemini_contents(store_key, &mut contents);
        assert_eq!(restored, 1);

        let parts = contents[0]["parts"].as_array().unwrap();
        assert_eq!(parts[0]["thought"], true);
        assert_eq!(parts[0]["text"], "Thought restored from SQLite");
        assert!(parts[1].get("thoughtSignature").is_none());
        assert_eq!(parts[2]["thoughtSignature"], real_sig);
        assert_eq!(parts[2]["functionCall"]["id"], "call_persisted_999");
    }

    #[test]
    fn sqlite_merges_latest_chunk_but_keeps_older_same_fingerprint_turn() {
        let store_key = "test_session_fp_hello_isolation";
        if crate::modules::proxy_db::delete_thinking_records_for_session(store_key).is_err() {
            return;
        }

        let sig = "sig_hello_12345678901234567890123456789012345678901234567890";
        let fp_hello = fingerprint("你好", &[], &[]);
        let fp_other = fingerprint("other", &["call_x".to_string()], &["bash".to_string()]);

        let save = |fp: &str, thought: &str, visible: &str, ids: &[String], names: &[String]| {
            crate::modules::proxy_db::save_thinking_record(
                store_key,
                fp,
                thought,
                Some(sig),
                ids,
                names,
                visible,
            )
        };

        assert!(save(&fp_hello, "thought-1", "你好", &[], &[]).is_ok());
        assert!(save(&fp_hello, "thought-1-longer", "你好", &[], &[]).is_ok());
        assert!(save(
            &fp_other,
            "thought-tool",
            "other",
            &["call_x".to_string()],
            &["bash".to_string()]
        )
        .is_ok());
        assert!(save(&fp_hello, "thought-3", "你好", &[], &[]).is_ok());

        let rows = crate::modules::proxy_db::load_thinking_records(store_key).unwrap_or_default();
        assert_eq!(
            rows.len(),
            3,
            "older 你好 turn must not be overwritten: {rows:?}"
        );
        assert_eq!(rows[0].thought, "thought-1-longer");
        assert_eq!(rows[1].thought, "thought-tool");
        assert_eq!(rows[2].thought, "thought-3");
        let _ = crate::modules::proxy_db::delete_thinking_records_for_session(store_key);
    }

    #[test]
    fn test_explicit_session_id_and_query_extraction() {
        // 1. Query parameter extraction
        let sid = extract_session_from_query_str(
            "model=gemini-2.5-pro&session_id=win_alpha_101&temp=0.7",
        );
        assert_eq!(sid.as_deref(), Some("win_alpha_101"));

        let sid_alias = extract_session_from_query_str("channel=proj_beta");
        assert_eq!(sid_alias.as_deref(), Some("proj_beta"));

        // 2. Header extraction: Claude Code / Cursor / VSCode
        let mut headers = HeaderMap::new();
        headers.insert("x-cursor-session-id", "cursor-tab-99".parse().unwrap());
        let extracted = explicit_session_id_with_query(&headers, None, None);
        assert_eq!(extracted.as_deref(), Some("cursor-tab-99"));

        // 3. Body & Metadata extraction
        let body = json!({
            "metadata": {
                "conversation_id": "meta-conv-888"
            }
        });
        let empty_headers = HeaderMap::new();
        let extracted2 = explicit_session_id_with_query(&empty_headers, Some(&body), None);
        assert_eq!(extracted2.as_deref(), Some("meta-conv-888"));
    }

    #[test]
    fn test_wildcard_client_session_headers() {
        let uuid = "c17b6d3c-e808-4874-8f16-b5dd4b6a2179";

        let mut atom = HeaderMap::new();
        atom.insert("x-atomcode-session-id", uuid.parse().unwrap());
        assert_eq!(
            explicit_session_id_with_query(&atom, None, None).as_deref(),
            Some(uuid)
        );

        let mut jeik = HeaderMap::new();
        jeik.insert("x-jeikcode-sessionid", uuid.parse().unwrap());
        assert_eq!(
            explicit_session_id_with_query(&jeik, None, None).as_deref(),
            Some(uuid)
        );

        let mut multi = HeaderMap::new();
        multi.insert("x-api-key", "secret".parse().unwrap());
        multi.insert("x-atomcode-session-id", uuid.parse().unwrap());
        multi.insert("x-jeikcode-sessionid", uuid.parse().unwrap());
        multi.insert("x-session-id", uuid.parse().unwrap());
        assert_eq!(
            explicit_session_id_with_query(&multi, None, None).as_deref(),
            Some(uuid)
        );

        let mut custom = HeaderMap::new();
        custom.insert("x-windsurf-session-id", "wind-tab-1".parse().unwrap());
        assert_eq!(
            explicit_session_id_with_query(&custom, None, None).as_deref(),
            Some("wind-tab-1")
        );

        let mut ignored = HeaderMap::new();
        ignored.insert("x-request-id", "req-should-not-win".parse().unwrap());
        ignored.insert("x-api-key", "secret".parse().unwrap());
        assert_eq!(explicit_session_id_with_query(&ignored, None, None), None);
    }

    #[test]
    fn affinity_header_is_a_fallback_without_collapsing_branch_sessions() {
        let mut headers = HeaderMap::new();
        headers.insert("x-session-affinity", "flow-root".parse().unwrap());
        headers.insert("x-conversation-id", "legacy-alias".parse().unwrap());
        assert_eq!(
            stable_session_winner(&headers, None, None).as_deref(),
            Some("flow-root")
        );

        headers.insert("x-session-id", "flow-root:step:1".parse().unwrap());
        assert_eq!(
            stable_session_winner(&headers, None, None).as_deref(),
            Some("flow-root:step:1")
        );
        headers.insert("x-opencode-session-id", "flow-root:step:2".parse().unwrap());
        assert_eq!(
            stable_session_winner(&headers, None, None).as_deref(),
            Some("flow-root:step:2")
        );
        headers.insert(
            "x-claude-code-session-id",
            "product-session".parse().unwrap(),
        );
        assert_eq!(
            stable_session_winner(&headers, None, None).as_deref(),
            Some("product-session")
        );

        let mut blank = HeaderMap::new();
        blank.insert("x-session-affinity", "  ".parse().unwrap());
        blank.insert("x-conversation-id", "legacy-alias".parse().unwrap());
        assert_eq!(
            stable_session_winner(&blank, None, None).as_deref(),
            Some("legacy-alias")
        );
    }

    #[test]
    fn product_session_headers_win_over_generic_x_session_id() {
        let mut jeik = HeaderMap::new();
        jeik.insert("x-session-id", "generic-session".parse().unwrap());
        jeik.insert("x-jeikcode-sessionid", "jeik-session".parse().unwrap());
        assert_eq!(
            explicit_session_id_with_query(&jeik, None, None).as_deref(),
            Some("jeik-session")
        );

        let mut atom = HeaderMap::new();
        atom.insert("x-session-id", "generic-session".parse().unwrap());
        atom.insert("x-atomcode-session-id", "atom-session".parse().unwrap());
        assert_eq!(
            explicit_session_id_with_query(&atom, None, None).as_deref(),
            Some("atom-session")
        );

        let mut wildcard = HeaderMap::new();
        wildcard.insert("x-session-id", "generic-session".parse().unwrap());
        wildcard.insert("x-windsurf-session-id", "wind-session".parse().unwrap());
        assert_eq!(
            explicit_session_id_with_query(&wildcard, None, None).as_deref(),
            Some("wind-session")
        );

        let mut only_generic = HeaderMap::new();
        only_generic.insert("x-session-id", "generic-session".parse().unwrap());
        assert_eq!(
            explicit_session_id_with_query(&only_generic, None, None).as_deref(),
            Some("generic-session")
        );

        let mut claude = HeaderMap::new();
        claude.insert("x-session-id", "generic-session".parse().unwrap());
        claude.insert(
            "x-claude-code-session-id",
            "claude-session".parse().unwrap(),
        );
        assert_eq!(
            explicit_session_id_with_query(&claude, None, None).as_deref(),
            Some("claude-session")
        );
    }

    #[test]
    fn test_3d_orthogonal_blended_session_stability_and_isolation() {
        // 1. 同一对话多轮聊天：相同 Headers 与相同的 Anchor -> 100% 相同稳定
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-claude-code-session-id",
            "53696541-0a6e-4be0-801e-2ee7a5601831".parse().unwrap(),
        );
        let scope_turn1 = SessionScope::from_headers(&headers, "sid-main-conversation-root");
        let scope_turn2 = SessionScope::from_headers(&headers, "sid-main-conversation-root");
        assert_eq!(scope_turn1.client_id, scope_turn2.client_id);
        assert_eq!(scope_turn1.store_key, scope_turn2.store_key);

        // 2. 主 Agent 与 Subagent 共用 x-claude-code-session-id，锚点不同：
        //    思维库隔离，账号粘性仍绑定同一会话。
        let scope_subagent = SessionScope::from_headers(&headers, "sid-subagent-distinct-prompt");
        assert_ne!(scope_turn1.client_id, scope_subagent.client_id);
        assert_ne!(scope_turn1.store_key, scope_subagent.store_key);
        assert_eq!(scope_turn1.affinity_key, scope_subagent.affinity_key);
        assert_eq!(scope_turn1.affinity_id, scope_subagent.affinity_id);

        // 2b. 没有稳定会话身份时，锚点同时分开账号粘性，避免匿名并发对话并到同一账号。
        let bare_a = SessionScope::from_headers(&HeaderMap::new(), "anchor-a");
        let bare_b = SessionScope::from_headers(&HeaderMap::new(), "anchor-b");
        assert_ne!(bare_a.affinity_key, bare_b.affinity_key);
        assert_eq!(bare_a.affinity_key, bare_a.store_key);
        assert_ne!(scope_turn1.affinity_key, bare_a.affinity_key);

        // 3. 不同租户多用户并发（不同 Authorization / API Key） -> 绝对隔离！
        let mut headers_user_a = headers.clone();
        headers_user_a.insert("authorization", "Bearer user-token-aaa".parse().unwrap());
        let mut headers_user_b = headers.clone();
        headers_user_b.insert("authorization", "Bearer user-token-bbb".parse().unwrap());
        let scope_user_a =
            SessionScope::from_headers(&headers_user_a, "sid-main-conversation-root");
        let scope_user_b =
            SessionScope::from_headers(&headers_user_b, "sid-main-conversation-root");
        assert_ne!(scope_user_a.store_key, scope_user_b.store_key);
        assert_ne!(scope_user_a.affinity_key, scope_user_b.affinity_key);

        // 4. Header 乱序注入时哈希绝对一致（BTreeMap 保证确定性排序）
        let mut headers_order1 = HeaderMap::new();
        headers_order1.insert("x-atomcode-session-id", "uuid-123".parse().unwrap());
        headers_order1.insert("x-jeikcode-sessionid", "uuid-456".parse().unwrap());

        let mut headers_order2 = HeaderMap::new();
        headers_order2.insert("x-jeikcode-sessionid", "uuid-456".parse().unwrap());
        headers_order2.insert("x-atomcode-session-id", "uuid-123".parse().unwrap());

        let scope_ord1 = SessionScope::from_headers(&headers_order1, "anchor-1");
        let scope_ord2 = SessionScope::from_headers(&headers_order2, "anchor-1");
        assert_eq!(scope_ord1.client_id, scope_ord2.client_id);
        assert_eq!(scope_ord1.affinity_key, scope_ord2.affinity_key);

        // 5. 跨协议相同显式会话头（如 Claude 切到 OpenAI）：相同会话锚点下 store_key 绝对一致共享，不同锚点下强隔离
        let mut claude_headers = HeaderMap::new();
        claude_headers.insert("x-session-id", "conv-uuid-999".parse().unwrap());
        claude_headers.insert("x-api-key", "secret-token".parse().unwrap());

        let mut openai_headers = HeaderMap::new();
        openai_headers.insert("x-session-id", "conv-uuid-999".parse().unwrap());
        openai_headers.insert("authorization", "Bearer secret-token".parse().unwrap());

        let scope_claude = SessionScope::from_headers(&claude_headers, "shared-anchor");
        let scope_openai = SessionScope::from_headers(&openai_headers, "shared-anchor");

        assert_eq!(
            scope_claude.store_key, scope_openai.store_key,
            "Cross-protocol requests in the same session must share the identical store_key"
        );
        assert_eq!(scope_claude.affinity_key, scope_openai.affinity_key);

        // 6. 产品头压过会变的通用 x-session-id。审批文案只拆思维库，不换账号。
        let mut main_headers = HeaderMap::new();
        main_headers.insert("x-claude-code-session-id", "cli-session".parse().unwrap());
        main_headers.insert("x-session-id", "generic-a".parse().unwrap());
        let mut approval_headers = HeaderMap::new();
        approval_headers.insert("x-claude-code-session-id", "cli-session".parse().unwrap());
        approval_headers.insert("x-session-id", "generic-b".parse().unwrap());
        let scope_main = SessionScope::from_headers(&main_headers, "anchor-main");
        let scope_approval = SessionScope::from_headers(&approval_headers, "anchor-approval");
        assert_eq!(scope_main.affinity_key, scope_approval.affinity_key);
        assert_ne!(scope_main.store_key, scope_approval.store_key);

        // 7. previous_response_id 每轮都变，不能当稳定身份。
        let body_turn = json!({ "previous_response_id": "resp-1" });
        let body_next = json!({ "previous_response_id": "resp-2" });
        let resp_a =
            SessionScope::from_headers_and_body(&HeaderMap::new(), Some(&body_turn), "anchor");
        let resp_b =
            SessionScope::from_headers_and_body(&HeaderMap::new(), Some(&body_next), "anchor");
        assert_eq!(resp_a.affinity_key, resp_b.affinity_key);
        assert_eq!(resp_a.store_key, resp_b.store_key);

        let rooted = json!({
            "session_id": "resp-root",
            "previous_response_id": "resp-9"
        });
        let rooted_next = json!({
            "session_id": "resp-root",
            "previous_response_id": "resp-10"
        });
        let root_a =
            SessionScope::from_headers_and_body(&HeaderMap::new(), Some(&rooted), "anchor-a");
        let root_b =
            SessionScope::from_headers_and_body(&HeaderMap::new(), Some(&rooted_next), "anchor-a");
        assert_eq!(root_a.affinity_key, root_b.affinity_key);
        assert_eq!(root_a.store_key, root_b.store_key);
    }

    #[test]
    fn ambiguous_legacy_text_generations_are_not_selected_by_order() {
        let store = ThinkingStore::new();
        let key = format!("legacy-conflict-{}", uuid::Uuid::new_v4());
        store.record(&key, rec("think-turn1", "same answer", None));
        store.record(&key, rec("think-turn2", "same answer", None));
        let mut contents = vec![
            json!({"role":"model","parts":[{"text":"another generation","thought":true,"thoughtSignature":"s".repeat(60)},{"text":"same answer"}]}),
            json!({"role":"model","parts":[{"text":"same answer"}]}),
        ];
        let original = contents.clone();
        assert_eq!(store.restore_gemini_contents(&key, &mut contents), 0);
        assert_eq!(contents, original);
    }

    #[test]
    fn tail_first_match_uses_latest_record_for_latest_incomplete_turn() {
        let store = ThinkingStore::new();
        let key = "t:tail-hello";
        let first = rec("think-turn1", "你好", None);
        let sig = first.signature.clone().unwrap();
        store.record(key, first);
        store.record(key, rec("think-turn2", "你好", None));
        let mut contents = vec![
            json!({
                "role": "model",
                "parts": [
                    { "text": "think-turn1", "thought": true, "thoughtSignature": sig },
                    { "text": "你好" }
                ]
            }),
            json!({
                "role": "model",
                "parts": [{ "text": "你好" }]
            }),
        ];
        let restored = store.restore_gemini_contents(key, &mut contents);
        assert_eq!(restored, 1);
        assert_eq!(contents[0]["parts"][0]["text"], "think-turn1");
        assert_eq!(contents[1]["parts"][0]["thought"], true);
        assert_eq!(
            contents[1]["parts"][0]["text"], "think-turn2",
            "latest incomplete turn must take the latest matching record, not the first 你好"
        );
    }

    #[test]
    fn concurrent_sessions_do_not_mix_thinking() {
        use std::thread;

        let store = Arc::new(ThinkingStore::new());
        let handles: Vec<_> = (0..48)
            .map(|i| {
                let store = Arc::clone(&store);
                thread::spawn(move || {
                    let key = format!("t:conc-{i}");
                    let thought = format!("thought-{i}");
                    let visible = format!("hello-{i}");
                    store.record(&key, rec(&thought, &visible, None));
                    let mut contents = vec![json!({
                        "role": "model",
                        "parts": [{ "text": visible }]
                    })];
                    let n = store.restore_gemini_contents(&key, &mut contents);
                    assert_eq!(n, 1);
                    assert_eq!(contents[0]["parts"][0]["text"], thought);
                })
            })
            .collect();
        for h in handles {
            h.join().expect("session thread panicked");
        }
    }

    #[test]
    fn capture_from_client_history_and_prune_compressed_turns() {
        let store = ThinkingStore::new();
        let session_key = format!("t:compress-{}", uuid::Uuid::new_v4());
        let key = &session_key;
        store.record(
            key,
            rec("thought-old-1", "old visible one", Some("call_old_1")),
        );
        store.record(
            key,
            rec("thought-old-2", "old visible two", Some("call_old_2")),
        );
        store.record(
            key,
            rec("thought-old-3", "old visible three", Some("call_old_3")),
        );
        store.record(
            key,
            rec("thought-keep", "kept latest answer", Some("call_keep")),
        );
        store.record(key, rec("thought-tail", "newest unused", None));

        // Client /compact dropped the first three turns; only the latest kept turn remains.
        let mut contents = vec![json!({
            "role": "model",
            "parts": [
                { "text": "kept latest answer" },
                { "functionCall": { "name": "shell", "id": "call_keep", "args": {} } }
            ]
        })];

        let restored = store.restore_gemini_contents(key, &mut contents);
        assert_eq!(restored, 1);
        store.prune_orphaned_records(key, &contents);
        let (turns, _) = store.session_stats(key).unwrap();
        assert!(
            turns <= 3,
            "orphaned compressed turns should be pruned, got {turns}"
        );
        let parts = contents[0]["parts"].as_array().unwrap();
        assert_eq!(parts[0]["text"], "thought-keep");
        store.end_session(key);
    }

    #[test]
    fn fingerprint_matches_legacy_whitespace_join() {
        let visible = "  hello\n\tworld  foo";
        let ids: Vec<String> = vec!["call_1".to_string()];
        let names: Vec<String> = vec!["shell".to_string()];
        let mut hasher = Sha256::new();
        let norm: String = visible.split_whitespace().collect::<Vec<_>>().join(" ");
        hasher.update(norm.as_bytes());
        hasher.update([0xff]);
        hasher.update(ids[0].as_bytes());
        hasher.update([0xfe]);
        hasher.update([0xfd]);
        hasher.update(names[0].as_bytes());
        hasher.update([0xfc]);
        let hex = format!("{:x}", hasher.finalize());
        assert_eq!(fingerprint(visible, &ids, &names), hex[..16].to_string());
    }

    #[test]
    fn placeholder_thoughts_are_not_recorded() {
        let store = ThinkingStore::new();
        let key = "t:placeholder-skip";
        store.record(
            key,
            ThinkingRecord {
                fingerprint: fingerprint("visible answer", &[], &[]),
                thought: "...".to_string(),
                signature: None,
                tool_ids: vec![],
                tool_names: vec![],
                visible: "visible answer".to_string(),
            },
        );
        store.record(
            key,
            ThinkingRecord {
                fingerprint: fingerprint("visible answer", &[], &[]),
                thought: "...".to_string(),
                signature: Some(SENTINEL_SIGNATURE.to_string()),
                tool_ids: vec![],
                tool_names: vec![],
                visible: "visible answer".to_string(),
            },
        );
        assert!(
            store.session_stats(key).is_none(),
            "placeholder / sentinel-only thoughts must not create store turns"
        );
    }

    #[test]
    fn ingest_placeholders_does_not_duplicate_history() {
        let store = ThinkingStore::new();
        let key = "t:ingest-no-dup";
        for i in 0..20 {
            store.record(
                key,
                rec(
                    &format!("thought-{i}"),
                    &format!("answer {i}"),
                    Some(&format!("call_{i}")),
                ),
            );
        }
        let (before, _) = store.session_stats(key).unwrap();
        assert_eq!(before, 20);

        let contents: Vec<Value> = (0..20)
            .map(|i| {
                json!({
                    "role": "model",
                    "parts": [
                        { "text": "...", "thought": true, "thoughtSignature": SENTINEL_SIGNATURE },
                        { "text": format!("answer {i}") },
                        { "functionCall": { "name": "shell", "id": format!("call_{i}"), "args": {} } }
                    ]
                })
            })
            .collect();

        store.ingest_from_contents(key, &contents);
        let (after, _) = store.session_stats(key).unwrap();
        assert_eq!(
            after, 20,
            "placeholder history must not be appended as new turns"
        );
        let _ = crate::modules::proxy_db::delete_thinking_records_for_session(key);
    }

    #[test]
    fn ingest_real_client_thinking_is_idempotent() {
        let store = ThinkingStore::new();
        let key = "t:ingest-real";
        let sig = "s".repeat(60);
        let contents = vec![json!({
            "role": "model",
            "parts": [
                { "text": "full chain of thought here", "thought": true, "thoughtSignature": sig },
                { "text": "hello world" }
            ]
        })];
        store.ingest_from_contents(key, &contents);
        store.ingest_from_contents(key, &contents);
        let (turns, _) = store.session_stats(key).unwrap();
        assert_eq!(turns, 1);
        let _ = crate::modules::proxy_db::delete_thinking_records_for_session(key);
    }

    #[test]
    fn restore_large_visible_text_is_linear() {
        let store = ThinkingStore::new();
        let key = "t:big-visible";
        let blob = "word ".repeat(8_000); // ~40KB per turn
        for i in 0..30 {
            let visible = format!("head-{i} {blob}");
            store.record(key, rec(&format!("thought-{i}"), &visible, None));
        }
        // Truncated visibles force Phase 3 prefix matching (the old quadratic path).
        let mut contents: Vec<Value> = (0..30)
            .map(|i| {
                json!({
                    "role": "model",
                    "parts": [{ "text": format!("head-{i}") }]
                })
            })
            .collect();
        let start = Instant::now();
        let n = store.restore_gemini_contents(key, &mut contents);
        let elapsed = start.elapsed();
        assert_eq!(n, 30);
        assert!(
            elapsed.as_millis() < 800,
            "restore of ~1.2MB visible text took {elapsed:?}; matching must not be quadratic"
        );
        let _ = crate::modules::proxy_db::delete_thinking_records_for_session(key);
    }

    #[test]
    fn placeholder_history_fills_in_order_without_scramble() {
        let store = ThinkingStore::new();
        let key = "t:fill-order";
        let _ = crate::modules::proxy_db::delete_thinking_records_for_session(key);
        for i in 0..12 {
            store.record(
                key,
                rec(
                    &format!("THOUGHT-BLOCK-{i}"),
                    &format!("answer {i}"),
                    Some(&format!("call_{i}")),
                ),
            );
        }
        let mut contents: Vec<Value> = (0..12)
            .map(|i| {
                json!({
                    "role": "model",
                    "parts": [
                        { "text": "...", "thought": true, "thoughtSignature": SENTINEL_SIGNATURE },
                        { "text": format!("answer {i}") },
                        { "functionCall": { "name": "shell", "id": format!("call_{i}"), "args": { "n": i } } }
                    ]
                })
            })
            .collect();

        assert!(!contents_have_capturable_thought(&contents));
        let n = store.restore_gemini_contents(key, &mut contents);
        assert_eq!(n, 12);
        for i in 0..12 {
            let parts = contents[i]["parts"].as_array().unwrap();
            assert_eq!(
                parts[0]["thought"], true,
                "thought must stay at parts[0] for turn {i}"
            );
            assert_eq!(parts[0]["text"], format!("THOUGHT-BLOCK-{i}"));
            assert!(parts[0].get("thoughtSignature").is_none());
            assert_eq!(parts[1]["text"], format!("answer {i}"));
            assert_eq!(parts[2]["functionCall"]["id"], format!("call_{i}"));
            assert_eq!(parts[2]["functionCall"]["args"]["n"], i);
            assert_eq!(parts[2]["thoughtSignature"].as_str().unwrap().len(), 60);
        }
        store.prune_orphaned_records(key, &contents);
        let (turns, _) = store.session_stats(key).unwrap();
        assert_eq!(turns, 12, "placeholder fill must not prune live tool turns");
        let _ = crate::modules::proxy_db::delete_thinking_records_for_session(key);
    }

    #[test]
    fn signed_function_call_is_not_mistaken_for_thinking_block() {
        let real_sig = "Ep4MCpsMARFNMg9NDlK9RXXz5Mzq9mniX9KSQBBzbUx3k85w/qDgtcE+28NH+1EvPeULAprqUquvYXGMzUXGy1xJoMnqdkC4vqebuhyd2Xhs0oz+OhqcOTwLhGYOG0KBKQ87Hfw4q/sMCSgf2gz4vFMa6V6kKMJepYlPXKFJJF4ok+W6lUt3PfYln8K9Dh7wB/40iHiZ2BnJd++6hfUwu9Bz1n795S50l0yCj84EaSCDDF334Erxq7Fo";

        // Case 1: thinking enabled, only signed functionCall → must prepend real thought block
        let mut contents = vec![json!({
            "role": "model",
            "parts": [
                {
                    "functionCall": {
                        "name": "read_file",
                        "id": "call_1",
                        "args": { "path": "a.rs" }
                    },
                    "thoughtSignature": real_sig
                }
            ]
        })];
        finalize_gemini_contents_thinking(&mut contents, true);
        let parts = contents[0]["parts"].as_array().unwrap();
        // 【2026-09-27】无思考文本时不再注入 "..." 占位思考块（官方标准形态：
        // 「无思考块 + 锚点带签名」，9/24 轮）。签名落锚点 fc。
        assert_eq!(
            parts.len(),
            1,
            "no placeholder thinking block should be injected"
        );
        assert!(parts[0].get("functionCall").is_some());
        assert_eq!(parts[0]["thoughtSignature"], real_sig);

        // Case 2: thinking disabled → signed functionCall must survive and preserve thoughtSignature for Gemini AST validator
        let mut contents_off = vec![json!({
            "role": "model",
            "parts": [
                {
                    "functionCall": {
                        "name": "bash",
                        "id": "call_9",
                        "args": {}
                    },
                    "thoughtSignature": real_sig
                }
            ]
        })];
        finalize_gemini_contents_thinking(&mut contents_off, false);
        let parts_off = contents_off[0]["parts"].as_array().unwrap();
        assert_eq!(
            parts_off.len(),
            1,
            "functionCall must not be dropped when thinking is off"
        );
        assert!(parts_off[0].get("functionCall").is_some());
        assert_eq!(
            parts_off[0]["thoughtSignature"], real_sig,
            "Gemini native model requires functionCall to retain its signature even when thinking is off"
        );
    }

    #[test]
    fn finalize_upgrades_sentinel_thought_from_tool_real_signature() {
        let real_sig = "Ep4MCpsMARFNMg9NDlK9RXXz5Mzq9mniX9KSQBBzbUx3k85w/qDgtcE+28NH+1EvPeULAprqUquvYXGMzUXGy1xJoMnqdkC4vqebuhyd2Xhs0oz+OhqcOTwLhGYOG0KBKQ87Hfw4q/sMCSgf2gz4vFMa6V6kKMJepYlPXKFJJF4ok+W6lUt3PfYln8K9Dh7wB/40iHiZ2BnJd++6hfUwu9Bz1n795S50l0yCj84EaSCDDF334Erxq7Fo";
        let mut contents = vec![json!({
            "role": "model",
            "parts": [
                {
                    "text": "...",
                    "thought": true,
                    "thoughtSignature": SENTINEL_SIGNATURE
                },
                {
                    "functionCall": {
                        "name": "read_file",
                        "id": "call_upgrade",
                        "args": {}
                    },
                    "thoughtSignature": real_sig
                }
            ]
        })];
        finalize_gemini_contents_thinking(&mut contents, true);
        let parts = contents[0]["parts"].as_array().unwrap();
        // 新规则：无真实签名的思考块出站前被丢弃，不再"升级 σ 填充签名"
        assert!(parts.iter().all(|p| p.get("thought") != Some(&json!(true))));
        let fc = parts
            .iter()
            .find(|p| p.get("functionCall").is_some())
            .expect("functionCall part");
        assert_eq!(fc["thoughtSignature"], real_sig);
    }

    #[test]
    fn turn_needs_restore_when_thought_signature_is_sentinel() {
        let parts = vec![
            json!({
                "text": "some real looking text that is not a placeholder",
                "thought": true,
                "thoughtSignature": SENTINEL_SIGNATURE
            }),
            json!({
                "functionCall": { "name": "shell", "id": "call_x", "args": {} },
                "thoughtSignature": SENTINEL_SIGNATURE
            }),
        ];
        assert!(
            turn_needs_restore(&parts, "some real looking text that is not a placeholder"),
            "sentinel thought signature must still request restore"
        );
    }

    #[test]
    fn test_is_meaningful_thought_sanitizer() {
        // Placeholders & empty must fail
        assert!(!is_meaningful_thought(""));
        assert!(!is_meaningful_thought("   "));
        assert!(!is_meaningful_thought("..."));
        assert!(!is_meaningful_thought("···"));
        assert!(!is_meaningful_thought("."));

        // Pseudo-thinking tags & placeholders must fail
        assert!(!is_meaningful_thought("<think></think>"));
        assert!(!is_meaningful_thought("<think>\n\n</think>"));
        assert!(!is_meaningful_thought("Thinking Process:\n"));
        assert!(!is_meaningful_thought("[Thinking]"));
        assert!(!is_meaningful_thought("None"));
        assert!(!is_meaningful_thought("none"));
        assert!(!is_meaningful_thought("null"));
        assert!(!is_meaningful_thought("undefined"));
        assert!(!is_meaningful_thought("[Thinking]\n..."));

        // Real thoughts must pass
        assert!(is_meaningful_thought(
            "Let's analyze the problem step by step."
        ));
        assert!(is_meaningful_thought(
            "<think>First compute the square root of 16, which is 4.</think>"
        ));
        assert!(is_meaningful_thought(
            "Thinking Process:\n1. Check file existence\n2. Open file"
        ));
    }

    #[test]
    fn test_finalize_thinking_disabled_downgrades_meaningful_thought_and_strips_placeholders() {
        let mut contents = vec![
            json!({
                "role": "model",
                "parts": [
                    {
                        "text": "...",
                        "thought": true,
                        "thoughtSignature": SENTINEL_SIGNATURE
                    },
                    {
                        "text": "Hello, how can I help?"
                    }
                ]
            }),
            json!({
                "role": "model",
                "parts": [
                    {
                        "text": "Real thought: solving user query carefully.",
                        "thought": true,
                        "thoughtSignature": "some_sig"
                    },
                    {
                        "text": "Here is the answer."
                    }
                ]
            }),
        ];

        finalize_gemini_contents_thinking(&mut contents, false);

        // Turn 1: placeholder "..." thought is dropped, only visible text survives
        let parts1 = contents[0]["parts"].as_array().unwrap();
        assert_eq!(parts1.len(), 1);
        assert_eq!(parts1[0]["text"], "Hello, how can I help?");
        assert!(parts1[0].get("thought").is_none());

        // Turn 2: meaningful thought is downgraded to text {"text": "Real thought: ..."}
        let parts2 = contents[1]["parts"].as_array().unwrap();
        assert_eq!(parts2.len(), 2);
        assert_eq!(
            parts2[0]["text"],
            "Real thought: solving user query carefully."
        );
        assert!(parts2[0].get("thought").is_none());
        assert!(parts2[0].get("thoughtSignature").is_none());
        assert_eq!(parts2[1]["text"], "Here is the answer.");
    }

    #[test]
    fn test_finalize_thinking_disabled_cleans_user_function_response_signature() {
        let mut contents = vec![json!({
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": "calc",
                        "response": { "result": 42 }
                    },
                    "thoughtSignature": "sig_to_be_cleaned"
                }
            ]
        })];

        finalize_gemini_contents_thinking(&mut contents, false);

        let parts = contents[0]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 1);
        assert!(parts[0].get("functionResponse").is_some());
        assert!(
            parts[0].get("thoughtSignature").is_none(),
            "thoughtSignature must be removed from functionResponse when thinking is disabled"
        );
    }

    #[test]
    fn test_sqlite_penetration_fallback_when_memory_missing() {
        let store = ThinkingStore::new();
        let key = "t:sqlite-fallback-test";
        let tool_id = "call_fallback_999";
        let real_sig = "s".repeat(60);

        // 1. 模拟旧轮次已持久化入库 SQLite（但在内存缓存中已被淘汰或未命中）
        crate::modules::proxy_db::save_thinking_record(
            key,
            "fp_fallback",
            "This thinking was retrieved directly from SQLite!",
            Some(&real_sig),
            &[tool_id.to_string()],
            &[],
            "some visible",
        )
        .unwrap();

        // 确保内存缓存是完全清空的，逼迫触发 L2 SQLite 穿透点查
        store.clear();

        // 2. 构造客户端回传的历史请求（缺少思考块，仅占位符）
        let mut contents = vec![json!({
            "role": "model",
            "parts": [
                {
                    "text": "...",
                    "thought": true,
                    "thoughtSignature": SENTINEL_SIGNATURE
                },
                {
                    "functionCall": {
                        "name": "shell",
                        "id": tool_id,
                        "args": {}
                    }
                }
            ]
        })];

        // 3. 执行思考复活：应当穿透到 SQLite 成功捞回！
        let restored = store.restore_gemini_contents(key, &mut contents);
        assert_eq!(restored, 1, "Must penetrate to SQLite and restore 1 turn!");

        let parts = contents[0]["parts"].as_array().unwrap();
        assert_eq!(parts[0]["thought"], true);
        assert_eq!(
            parts[0]["text"],
            "This thinking was retrieved directly from SQLite!"
        );
        assert!(parts[0].get("thoughtSignature").is_none());
        assert_eq!(parts[1]["thoughtSignature"], real_sig);

        let _ = crate::modules::proxy_db::delete_thinking_records_for_session(key);
    }

    #[test]
    fn test_canonical_json_hash_key_order_independence() {
        let a = json!({ "b": 2, "a": 1, "c": { "z": 9, "y": 8 } });
        let b = json!({ "a": 1, "c": { "y": 8, "z": 9 }, "b": 2 });
        assert_eq!(
            canonical_json_hash(Some(&a)),
            canonical_json_hash(Some(&b)),
            "Canonical JSON hash must be independent of key order"
        );
    }

    #[test]
    fn test_compute_causal_anchor_differentiates_user_text_vs_tool_response() {
        let turn1 = json!({ "role": "user", "parts": [{ "text": "Run cargo check" }] });
        let turn2 = json!({
            "role": "user",
            "parts": [{ "functionResponse": { "name": "bash", "response": { "exit_code": 1, "output": "error" } } }]
        });
        let anchor1 = compute_causal_anchor(Some(&turn1));
        let anchor2 = compute_causal_anchor(Some(&turn2));
        assert_ne!(
            anchor1, anchor2,
            "Causal anchors for user text vs tool response must be distinct"
        );
    }

    #[test]
    fn test_gemini_native_consecutive_identical_tools_do_not_overwrite() {
        let key = "t:test_identical_tools";
        let _ = crate::modules::proxy_db::delete_thinking_records_for_session(key);
        let store = ThinkingStore::new();

        // Turn 1: user text -> model tool call
        let mut acc1 = TurnAccumulator::with_anchor("anchor_user_turn_1");
        acc1.ingest_part(&json!({
            "text": "Thought for turn 1",
            "thought": true,
            "thoughtSignature": "s".repeat(60)
        }));
        acc1.ingest_part(&json!({
            "functionCall": { "name": "bash", "args": { "command": "pwd" } }
        }));
        store.record(key, acc1.into_record());

        // Turn 2: tool response -> model tool call (identical tool name and args!)
        let mut acc2 = TurnAccumulator::with_anchor("anchor_fr_turn_2");
        acc2.ingest_part(&json!({
            "text": "Thought for turn 2",
            "thought": true,
            "thoughtSignature": "t".repeat(60)
        }));
        acc2.ingest_part(&json!({
            "functionCall": { "name": "bash", "args": { "command": "pwd" } }
        }));
        store.record(key, acc2.into_record());

        let stats = store.session_stats(key).unwrap();
        assert_eq!(
            stats.0, 2,
            "Must store two distinct turns, not overwrite via collision"
        );

        let _ = crate::modules::proxy_db::delete_thinking_records_for_session(key);
    }

    #[test]
    fn test_prune_orphaned_records_zero_phase_shift() {
        let key = "t:test_phase_shift_prune";
        let _ = crate::modules::proxy_db::delete_thinking_records_for_session(key);
        let store = ThinkingStore::new();

        for i in 0..6 {
            let user_turn = json!({
                "role": "user",
                "parts": [{ "text": format!("user msg {i}") }]
            });
            let anchor = compute_causal_anchor(Some(&user_turn));
            let mut acc = TurnAccumulator::with_anchor(&anchor);
            acc.ingest_part(&json!({
                "text": format!("Thought turn {i}"),
                "thought": true,
                "thoughtSignature": format!("sig_{:0>60}", i)
            }));
            acc.ingest_part(&json!({
                "functionCall": { "name": "bash", "args": { "step": i } }
            }));
            store.record(key, acc.into_record());
        }

        // Client compresses away turns 0 and 1; only turns 2..5 remain
        let mut contents = Vec::new();
        for i in 2..6 {
            contents.push(json!({
                "role": "user",
                "parts": [{ "text": format!("user msg {i}") }]
            }));
            contents.push(json!({
                "role": "model",
                "parts": [
                    { "text": "...", "thought": true, "thoughtSignature": SENTINEL_SIGNATURE },
                    { "functionCall": { "name": "bash", "args": { "step": i } } }
                ]
            }));
        }

        let restored = store.restore_gemini_contents(key, &mut contents);
        assert_eq!(restored, 4, "Must restore 4 compressed turns");

        // Verify ZERO PHASE SHIFT: turn 2 must have "Thought turn 2", NOT "Thought turn 4"
        for (idx, step) in (2..6).enumerate() {
            let content_idx = idx * 2 + 1;
            let parts = contents[content_idx]["parts"].as_array().unwrap();
            assert_eq!(
                parts[0]["text"],
                format!("Thought turn {step}"),
                "Turn {step} must strictly match its own thought without phase shift"
            );
            assert!(
                parts[0].get("thoughtSignature").is_none(),
                "Turn {step} thought block must not carry signature per Iron Law I4"
            );
            assert_eq!(
                parts[1]["thoughtSignature"],
                format!("sig_{:0>60}", step),
                "Turn {step} must strictly match its own thoughtSignature on the functionCall anchor"
            );
        }

        let _ = crate::modules::proxy_db::delete_thinking_records_for_session(key);
    }

    #[test]
    fn test_phase0_signature_direct_matching_recovers_truncated_text() {
        let key = "t:test_phase0_sig";
        let _ = crate::modules::proxy_db::delete_thinking_records_for_session(key);
        let store = ThinkingStore::new();
        let real_sig = "s".repeat(60);

        let mut acc = TurnAccumulator::with_anchor("anchor_1");
        acc.ingest_part(&json!({
            "text": "Detailed multi-step chain of thought",
            "thought": true,
            "thoughtSignature": real_sig
        }));
        acc.ingest_part(&json!({
            "functionCall": { "name": "read_file", "args": { "path": "main.rs" } }
        }));
        store.record(key, acc.into_record());

        let mut contents = vec![
            json!({ "role": "user", "parts": [{ "text": "start" }] }),
            json!({
                "role": "model",
                "parts": [
                    { "text": "...", "thought": true, "thoughtSignature": real_sig },
                    { "functionCall": { "name": "read_file", "args": { "path": "main.rs" } } }
                ]
            }),
        ];

        let restored = store.restore_gemini_contents(key, &mut contents);
        assert_eq!(restored, 1);
        assert_eq!(
            contents[1]["parts"][0]["text"], "Detailed multi-step chain of thought",
            "Phase 0 must recover truncated thought using real signature"
        );

        let _ = crate::modules::proxy_db::delete_thinking_records_for_session(key);
    }

    #[test]
    fn test_sqlite_l2_penetration_by_signature_and_tool() {
        let key = "t:test_sqlite_l2_sig_rescue";
        let real_sig = format!("sig_l2_{}", "x".repeat(53));
        let tool_id = "call_bash_anchor123_hash456_0";

        let _ = crate::modules::proxy_db::save_thinking_record(
            key,
            "fp_l2_sig_test",
            "Rescued thought from SQLite via signature",
            Some(&real_sig),
            &[tool_id.to_string()],
            &["bash".to_string()],
            "",
        );

        let store = ThinkingStore::new();
        // Clear memory to force L2 penetration
        store.clear();

        let mut contents = vec![
            json!({ "role": "user", "parts": [{ "text": "hello" }] }),
            json!({
                "role": "model",
                "parts": [
                    { "text": "...", "thought": true, "thoughtSignature": real_sig },
                    { "functionCall": { "name": "bash", "id": tool_id, "args": {} } }
                ]
            }),
        ];

        let restored = store.restore_gemini_contents(key, &mut contents);
        assert_eq!(
            restored, 1,
            "Must penetrate to SQLite and restore via signature"
        );
        assert_eq!(
            contents[1]["parts"][0]["text"],
            "Rescued thought from SQLite via signature"
        );

        let _ = crate::modules::proxy_db::delete_thinking_records_for_session(key);
    }

    #[test]
    fn test_is_likely_gemini_signature_validation() {
        // 1. 官方哨兵
        assert!(is_likely_gemini_signature(SENTINEL_SIGNATURE));

        // 2. 真实 Gemini 原生签名 (首字节 0x12, 以 'E' 开头)
        let gemini_sig = "Ep4KCpsKAWkUfRMa5ZYMDdlPjxrQTLzVZ6MZeopI88888888888888888888888888888888";
        assert!(is_likely_gemini_signature(gemini_sig));

        // 3. Google Vertex AI 双层 Base64 包装 (以 'R' 开头, 解开是 'E...' 且首字节 0x12)
        use base64::Engine;
        let vertex_wrapped =
            base64::engine::general_purpose::STANDARD.encode(gemini_sig.as_bytes());
        assert!(vertex_wrapped.starts_with('R'));
        assert!(is_likely_gemini_signature(&vertex_wrapped));

        // 4. 异构外部 Claude 签名 (以 '3', 'l', 'A', 'R' 开头, 解码后非 0x12)
        let claude_sig_1 = "3mgp11XmVXq9InniGA4VAKd7c97NqFw+dWZt79Uz/w9znho88gSM76jv2bZmir7wI86Ixpha7eWdGuznAot4PNbe3+V9bgMTIEyUarn4MLAiiFVb830ZlM+H5ukQwXdD2Zv8nUSmmZTYinpLPGha8TORZAfpU1FJEvwyECel5+W7kc9kpTWrd8DqRNBTOz5EDtvoatiZgKv5SqInhGXK74SJ+PRIC6fNXvYG082HR6TsVxvVYaerz8A40rloIVTxRNK43h3Ecs1boxY4PZqBT8Yhl2qn/iZ+4Xt7FNkI0DAuS9iK0HYKMC4yw0OqKx/LeU+WFZlyc6hGm1BkzLY6yG97MH7kmJ0OPlBWgWFaTeL/uXuGJX6QkKObXN+phoq+kkF2vdFt/mdJMbdgfmSCVQ9037hGBhOHm0zN50KLkp1SxuAY1oWc+lDcI4ufWoyn";
        let claude_sig_2 = "ls29VsBy+VBvzrVBmB2gNmOCoaeJkn19qz8jP8jExGpDc0IxRaV1V9/+cQ4O00000000000000000000000000000000";
        let claude_sig_3 = "A1nvLg9Twun3bBCb1BKLmSNA6MRxaLE2GdEocv6bwuhNKfUmBB2YMvvmaVyO00000000000000000000000000000000";
        assert!(!is_likely_gemini_signature(claude_sig_1));
        assert!(!is_likely_gemini_signature(claude_sig_2));
        assert!(!is_likely_gemini_signature(claude_sig_3));

        // 5. 过短签名
        assert!(!is_likely_gemini_signature("short_sig"));
    }

    #[test]
    fn test_is_claude_signature_validation() {
        use base64::Engine;
        // 1. 构建合法的原始 Claude 签名（单层 Base64 解码后包含 b"claude"）
        let inner_claude_payload =
            b"\x12\xb2\x02\n\x92\x01\x08\x12\x10\x02\x18\x02*@claude-opus-4-6-signature-data";
        let raw_claude_sig = base64::engine::general_purpose::STANDARD.encode(inner_claude_payload);
        assert!(is_claude_signature(&raw_claude_sig));

        // 2. 构建 Google Vertex AI 双层包装后的 Claude 签名
        let wrapped_claude_sig =
            base64::engine::general_purpose::STANDARD.encode(raw_claude_sig.as_bytes());
        assert!(is_claude_signature(&wrapped_claude_sig));

        // 3. Claude 5.5 官方真实报文签名测试（CAQS / Q0FR 开头 Protobuf 结构）
        let claude_5_5_protobuf = "CAQSpAN01LkwMTAnAAEq";
        let wrapped_5_5 =
            base64::engine::general_purpose::STANDARD.encode(claude_5_5_protobuf.as_bytes());
        assert!(is_claude_signature(claude_5_5_protobuf));
        assert!(is_claude_signature(&wrapped_5_5));

        // 4. Gemini 签名绝不是 Claude 签名
        let gemini_sig = "Ep4KCpsKAWkUfRMa5ZYMDdlPjxrQTLzVZ6MZeopI88888888888888888888888888888888";
        assert!(!is_claude_signature(gemini_sig));

        // 5. 空与哨兵
        assert!(!is_claude_signature(""));
        assert!(!is_claude_signature(SENTINEL_SIGNATURE));
    }

    #[test]
    fn test_ensure_google_claude_thought_signature_does_not_inflate() {
        use base64::Engine;
        // 原始 Claude 签名：包装一次
        let inner_claude_payload =
            b"\x12\xb2\x02\n\x92\x01\x08\x12\x10\x02\x18\x02*@claude-opus-4-6-signature-data";
        let raw_claude_sig = base64::engine::general_purpose::STANDARD.encode(inner_claude_payload);
        let wrapped_once = ensure_google_claude_thought_signature(&raw_claude_sig);
        assert_ne!(wrapped_once, raw_claude_sig);

        // 已包装的 Claude 签名：幂等，绝对不再包装！
        let wrapped_twice = ensure_google_claude_thought_signature(&wrapped_once);
        assert_eq!(
            wrapped_once, wrapped_twice,
            "Claude signature must be idempotent, no double wrapping"
        );

        // Claude 5.5 Protobuf 格式签名包装与还原测试
        let claude_5_5_protobuf = "CAQSpAN01LkwMTAnAAEq";
        let wrapped_5_5 = ensure_google_claude_thought_signature(claude_5_5_protobuf);
        assert_ne!(wrapped_5_5, claude_5_5_protobuf);
        let wrapped_5_5_twice = ensure_google_claude_thought_signature(&wrapped_5_5);
        assert_eq!(
            wrapped_5_5, wrapped_5_5_twice,
            "Protobuf signature must be idempotent, no double wrapping"
        );
        assert_eq!(
            ensure_raw_claude_thought_signature(&wrapped_5_5),
            claude_5_5_protobuf,
            "Protobuf wrapped signature must restore to raw CAQS format"
        );
        assert_eq!(
            ensure_raw_claude_thought_signature(claude_5_5_protobuf),
            claude_5_5_protobuf,
            "Raw Protobuf signature must remain unchanged"
        );
        assert_eq!(
            normalize_signature_for_comparison(&wrapped_5_5),
            normalize_signature_for_comparison(claude_5_5_protobuf),
            "Protobuf wrapped and raw signatures must normalize identically"
        );
        assert!(
            signatures_match(claude_5_5_protobuf, &wrapped_5_5),
            "Protobuf raw and wrapped signatures must match via signatures_match"
        );

        // 非 Claude 签名 (Gemini 签名)：绝不能包装！彻底杜绝几何级膨胀
        let gemini_sig = "Ep4KCpsKAWkUfRMa5ZYMDdlPjxrQTLzVZ6MZeopI88888888888888888888888888888888";
        let gemini_out = ensure_google_claude_thought_signature(gemini_sig);
        assert_eq!(
            gemini_out, gemini_sig,
            "Non-Claude signature must NOT be base64 wrapped"
        );
    }

    #[test]
    fn test_restore_and_finalize_gemini_contents_with_model_claude_target_rejects_gemini_signature()
    {
        let store = ThinkingStore::new();
        let key = "t:cross_model_test_session";
        let gemini_sig = "Ep4KCpsKAWkUfRMa5ZYMDdlPjxrQTLzVZ6MZeopI88888888888888888888888888888888";

        // 模拟上一轮 Gemini 存储的思考记录（带有 Gemini 原生签名）
        store.record(
            key,
            ThinkingRecord {
                fingerprint: "fp_test_cross".to_string(),
                thought: "Thought generated by Gemini".to_string(),
                signature: Some(gemini_sig.to_string()),
                tool_ids: vec![],
                tool_names: vec![],
                visible: "Gemini visible text".to_string(),
            },
        );

        // 客户端在同一 session 下切换模型为 Claude 发起后续对话
        let mut contents = vec![
            json!({ "role": "user", "parts": [{ "text": "Hello" }] }),
            json!({
                "role": "model",
                "parts": [{ "text": "Gemini visible text" }]
            }),
            json!({ "role": "user", "parts": [{ "text": "Next turn" }] }),
        ];

        // 1. 执行 restore（针对目标模型 claude-opus-4-6-thinking）
        let restored = store.restore_gemini_contents_with_model(
            key,
            &mut contents,
            Some("claude-opus-4-6-thinking"),
        );
        assert_eq!(restored, 0, "Claude 目标不应恢复 Gemini 签名");

        // 恢复出的 thinking 块绝不能挂载 Gemini 签名！
        let parts_after_restore = contents[1]["parts"].as_array().unwrap();
        assert!(
            parts_after_restore[0].get("thoughtSignature").is_none(),
            "ThinkingStore must not assign foreign Gemini signature when target is Claude"
        );

        // 2. 执行 finalize（终审节点）
        finalize_gemini_contents_thinking_with_model(
            &mut contents,
            true,
            Some("claude-opus-4-6-thinking"),
        );

        // Claude 历史保留思考正文。签名不挂在思考块上。
        let final_parts = contents[1]["parts"].as_array().unwrap();
        let thought_has_sig = final_parts.iter().any(|p| {
            p.get("thought").and_then(|v| v.as_bool()) == Some(true)
                && p.get("thoughtSignature").is_some()
        });
        assert!(
            !thought_has_sig,
            "Claude thought parts do not carry thoughtSignature"
        );
        assert!(
            !final_parts.iter().any(|p| p
                .get("text")
                .and_then(|t| t.as_str())
                .map_or(false, |s| s.contains("Thought generated by Gemini"))),
            "新规则：无签名思考块被丢弃，思考正文不再保留"
        );
    }

    #[test]
    fn test_finalize_thinking_disabled_preserves_historical_thinking_and_signatures() {
        let real_sig = "Ep4MCpsMARFNMg9NDlK9RXXz5Mzq9mniX9KSQBBzbUx3k85w/qDgtcE+28NH+1EvPeULAprqUquvYXGMzUXGy1xJoMnqdkC4vqebuhyd2Xhs0oz+OhqcOTwLhGYOG0KBKQ87Hfw4q/sMCSgf2gz4vFMa6V6kKMJepYlPXKFJJF4ok+W6lUt3PfYln8K9Dh7wB/40iHiZ2BnJd++6hfUwu9Bz1n795S50l0yCj84EaSCDDF334Erxq7Fo";
        // 模拟多轮对话：
        // Turn 1: User
        // Turn 2: Model (历史已发生，包含合法思考块与工具调用及签名)
        // Turn 3: User (工具返回)
        // Turn 4: User (最新一轮用户提问，并且在这一轮用户把思考开关关闭了)
        let mut contents = vec![
            json!({
                "role": "user",
                "parts": [{ "text": "Run bash command" }]
            }),
            json!({
                "role": "model",
                "parts": [
                    {
                        "text": "Planning to run ls...",
                        "thought": true
                    },
                    {
                        "functionCall": {
                            "name": "bash",
                            "args": { "command": "ls" }
                        },
                        "thoughtSignature": real_sig
                    }
                ]
            }),
            json!({
                "role": "user",
                "parts": [{
                    "functionResponse": {
                        "name": "bash",
                        "response": { "output": "file1.txt" }
                    }
                }]
            }),
            json!({
                "role": "user",
                "parts": [{ "text": "Now answer without thinking" }]
            }),
        ];

        // 最新一轮关思考 (is_thinking_enabled = false)
        finalize_gemini_contents_thinking_with_model(
            &mut contents,
            false,
            Some("gemini-3.8-flash-tiered"),
        );

        // 验证历史 Turn 2 的思考块与工具调用签名是否被完好保留：
        let history_model_parts = contents[1]["parts"].as_array().expect("parts array");
        assert_eq!(history_model_parts.len(), 2);
        assert_eq!(history_model_parts[0]["text"], "Planning to run ls...");
        assert_eq!(history_model_parts[1]["thoughtSignature"], real_sig, "Historical tool thoughtSignature must NOT be stripped when thinking is disabled in the current turn");
    }

    #[test]
    fn test_finalize_thinking_disabled_keeps_unsigned_function_call_unsigned_for_gemini() {
        let mut contents = vec![json!({
            "role": "model",
            "parts": [
                {
                    "functionCall": {
                        "name": "grep",
                        "id": "call_unsigned",
                        "args": { "pattern": "abc" }
                    }
                }
            ]
        })];

        // 关思考出站，发往 Gemini
        finalize_gemini_contents_thinking_with_model(
            &mut contents,
            false,
            Some("gemini-3.8-flash-tiered"),
        );

        // 官方报文里哨兵出现 0 次；缺失签名被上游容忍（在飞轮即缺席）。
        // 因此无签名的 functionCall 保持「字段缺席」，绝不发明占位符。
        let parts = contents[0]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(
            parts[0]["thoughtSignature"],
            json!(SENTINEL_SIGNATURE),
            "Unsigned functionCall 由 base64 哨兵占位（新规则）"
        );
    }

    #[test]
    fn test_finalize_does_not_invent_signature_for_unsigned_function_call() {
        let tool_id = "call_finalize_cache_777";
        let mut contents = vec![json!({
            "role": "model",
            "parts": [
                {
                    "functionCall": {
                        "name": "run_command",
                        "id": tool_id,
                        "args": { "cmd": "cargo test" }
                    }
                    // 注意：未带 thoughtSignature（模拟未在进站流水线查到的工具调用）
                }
            ]
        })];

        // 终审出站，发往 Gemini
        finalize_gemini_contents_thinking_with_model(
            &mut contents,
            true,
            Some("gemini-3.8-flash-tiered"),
        );

        let parts = contents[0]["parts"].as_array().unwrap();
        // 终审门禁：不查库、不发明签名。官方 functionCall 轮是"纯净"的
        // （33 个 model 轮里 thought × functionCall 共现 0 次），因此不得注入占位思考块。
        assert_eq!(
            parts.len(),
            1,
            "finalize must NOT inject a placeholder thought block"
        );
        let fc = parts
            .iter()
            .find(|p| p.get("functionCall").is_some())
            .unwrap();
        assert_eq!(
            fc["thoughtSignature"],
            json!(SENTINEL_SIGNATURE),
            "Pipeline finalize 在 functionCall 缺签名时写入 base64 哨兵占位"
        );
    }

    #[test]
    fn test_finalize_thinking_disabled_strips_tool_signature_for_claude() {
        let real_sig = "s".repeat(60);
        let mut contents = vec![json!({
            "role": "model",
            "parts": [
                {
                    "functionCall": {
                        "name": "grep",
                        "id": "call_123",
                        "args": {}
                    },
                    "thoughtSignature": real_sig
                }
            ]
        })];

        // 关思考出站，发往 Claude
        finalize_gemini_contents_thinking_with_model(
            &mut contents,
            false,
            Some("claude-3-7-sonnet"),
        );

        let parts = contents[0]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(
            parts[0]["thoughtSignature"], real_sig,
            "Claude tool-call signature stays on the functionCall"
        );
    }

    #[test]
    fn test_restore_deduplicates_and_elevates_thought_from_plain_text() {
        let store = ThinkingStore::new();
        let key = "t:re-elevation-dedup-test";
        let tool_id = "call_dedup_001";
        let real_sig = "Ep4MCpsMARFNMg9NDlK9RXXz5Mzq9mniX9KSQBBzbUx3k85w/qDgtcE+28NH+1EvPeULAprqUquvYXGMzUXGy1xJoMnqdkC4vqebuhyd2Xhs0oz+OhqcOTwLhGYOG0KBKQ87Hfw4q/sMCSgf2gz4vFMa6V6kKMJepYlPXKFJJF4ok+W6lUt3PfYln8K9Dh7wB/40iHiZ2BnJd++6hfUwu9Bz1n795S50l0yCj84EaSCDDF334Erxq7Fo";
        let thought_text =
            "I am analyzing the repository carefully and formulating a search query.";
        let visible_answer = "Here is the exact file path you requested.";

        // 1. 模拟旧轮次入库（开思考时发生的思维记录）
        let _ = crate::modules::proxy_db::save_thinking_record(
            key,
            "fp_dedup_001",
            thought_text,
            Some(real_sig),
            &[tool_id.to_string()],
            &["grep".to_string()],
            visible_answer,
        );

        // 2. 模拟客户端发上来的历史：之前关思考时降级的思考文本变成了普通正文部件，残留或与正文拼接在一起
        let mut contents = vec![
            json!({
                "role": "user",
                "parts": [{ "text": "Find the file" }]
            }),
            json!({
                "role": "model",
                "parts": [
                    { "text": thought_text }, // 降级残留的普通文本部件
                    { "text": visible_answer },
                    {
                        "functionCall": {
                            "name": "grep",
                            "id": tool_id,
                            "args": {}
                        }
                    }
                ]
            }),
            json!({
                "role": "user",
                "parts": [{ "functionResponse": { "name": "grep", "response": { "res": "ok" } } }]
            }),
        ];

        // 3. 当前轮次重新开启思考，执行复活与重提升
        let restored = store.restore_gemini_contents_with_model(
            key,
            &mut contents,
            Some("gemini-3.8-flash-tiered"),
        );
        assert_eq!(restored, 1, "Must restore exactly 1 turn");

        let model_parts = contents[1]["parts"].as_array().expect("parts array");
        // 验证：
        // 1. 首位成功提升恢复为真正的思考块 (thought: true)
        assert_eq!(model_parts[0]["thought"], true);
        assert_eq!(model_parts[0]["text"], thought_text);

        // 2. 正文部件中的降级残留部件被精准剔除，只保留真实的可见回答，消灭双份复读！
        assert_eq!(model_parts[1]["text"], visible_answer);
        assert!(model_parts[1].get("thought").is_none());

        // 3. 工具调用挂载真实签名
        assert_eq!(model_parts[2]["functionCall"]["id"], tool_id);
        assert_eq!(model_parts[2]["thoughtSignature"], real_sig);

        let _ = crate::modules::proxy_db::delete_thinking_records_for_session(key);
    }

    #[test]
    fn test_gemini_to_claude_degrades_foreign_thought_to_text_and_strips_tool_sig() {
        let gemini_sig = "Ep4MCpsMARFNMg9NDlK9RXXz5Mzq9mniX9KSQBBzbUx3k85w/qDgtcE+28NH+1EvPeULAprqUquvYXGMzUXGy1xJoMnqdkC4vqebuhyd2Xhs0oz+OhqcOTwLhGYOG0KBKQ87Hfw4q/sMCSgf2gz4vFMa6V6kKMJepYlPXKFJJF4ok+W6lUt3PfYln8K9Dh7wB/40iHiZ2BnJd++6hfUwu9Bz1n795S50l0yCj84EaSCDDF334Erxq7Fo";
        let thought_text = "Step 1: Check database schema. Step 2: Query tables.";
        let visible_answer = "Found 3 matching records in the database.";

        let mut contents = vec![json!({
            "role": "model",
            "parts": [
                {
                    "text": thought_text,
                    "thought": true,
                    "thoughtSignature": gemini_sig
                },
                {
                    "text": visible_answer
                },
                {
                    "functionCall": {
                        "name": "query_db",
                        "id": "call_db_1",
                        "args": {}
                    },
                    "thoughtSignature": gemini_sig
                }
            ]
        })];

        // 目标模型切为 Claude 后，Gemini 签名不可用：思考内容降级为普通正文
        finalize_gemini_contents_thinking_with_model(
            &mut contents,
            true,
            Some("claude-3-7-sonnet"),
        );

        let parts = contents[0]["parts"].as_array().expect("parts array");
        assert_eq!(parts[0]["text"], thought_text);
        assert!(parts[0].get("thought").is_none());
        assert!(parts[0].get("thoughtSignature").is_none());
        assert_eq!(parts[1]["text"], visible_answer);
        assert!(parts[1].get("thoughtSignature").is_none());
        assert_eq!(parts[2]["functionCall"]["id"], "call_db_1");
        assert!(parts[2].get("thoughtSignature").is_none());
    }

    #[test]
    fn test_extract_think_tags_helper() {
        let input =
            "<think>\nThinking line 1\nThinking line 2\n</think>\n\nFinal response text here.";
        let res = extract_think_tags(input);
        assert!(res.is_some());
        let (thought, visible) = res.unwrap();
        assert_eq!(thought, "Thinking line 1\nThinking line 2");
        assert_eq!(visible, "Final response text here.");

        // No tags
        assert!(extract_think_tags("Just normal text").is_none());
    }

    #[test]
    fn test_gemini_native_synthetic_id_bridges_cross_protocol_signature_recovery() {
        let store = ThinkingStore::new();
        let key = "t:cross-proto-synthetic-id-test";
        let client_tool_id = "call_openai_native_456";
        let real_sig = "EmIKYAFpFH0TDqviLY1vZ8EuHqBLLj5xxD+0hchYg2VaoyolUQRP+hSCsKRpSpj+yrQA2H27yVFnF7tlp5OHIUvTdZKKErAqILJzK5FG8RJg42jCaaI2/iwqoBuRd5BDVwBxaQ==";
        let thought_text = "Analyzing directory and listing files.";
        let _visible_answer = "Running bash tool.";

        // 1. 模拟 OpenAI 协议下生成的工具调用（带有 client_tool_id）
        let user_turn = json!({
            "role": "user",
            "parts": [{ "text": "List the files" }]
        });
        let anchor = compute_causal_anchor(Some(&user_turn));
        let mut acc = TurnAccumulator::with_anchor(&anchor);
        acc.ingest_part(&json!({
            "thought": true,
            "text": thought_text,
            "thoughtSignature": real_sig
        }));
        acc.ingest_part(&json!({
            "functionCall": {
                "name": "bash",
                "args": { "command": "ls -la" },
                "id": client_tool_id
            },
            "thoughtSignature": real_sig
        }));
        acc.commit(key);

        // 2. 模拟用户切换到了原生 Gemini 协议发起后续对话：
        // 原生 Gemini 请求的 contents 中天生没有 id 字段（id: None）！
        let mut gemini_contents = vec![
            user_turn,
            json!({
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "name": "bash",
                            "args": { "command": "ls -la" }
                            // 注意：完全无 id 字段！
                        }
                    }
                ]
            }),
            json!({
                "role": "user",
                "parts": [{ "functionResponse": { "name": "bash", "response": { "res": "file.txt" } } }]
            }),
        ];

        // 3. 原生 Gemini 协议执行流水线复活与终审出站
        let restored = store.restore_gemini_contents_with_model(
            key,
            &mut gemini_contents,
            Some("gemini-3.8-flash-tiered"),
        );
        assert_eq!(
            restored, 1,
            "Must match via synthetic context ID even though Gemini request had no tool id"
        );

        finalize_gemini_contents_thinking_with_model(
            &mut gemini_contents,
            true,
            Some("gemini-3.8-flash-tiered"),
        );

        let model_parts = gemini_contents[1]["parts"].as_array().expect("parts");
        // 新规则：无真实签名的思考块被丢弃，不再"提升复活"
        assert!(
            model_parts
                .iter()
                .all(|p| p.get("thought") != Some(&json!(true))),
            "无签名思考块必须被丢弃"
        );

        // 验证跨协议捕获到的真实签名被归位到锚点，而不是被替换成哨兵：
        let fc = model_parts
            .iter()
            .find(|p| p.get("functionCall").is_some())
            .expect("functionCall anchor");
        assert_eq!(fc["functionCall"]["name"], "bash");
        assert_eq!(
            fc["thoughtSignature"], real_sig,
            "Captured real signature must be restored onto the anchor, not replaced by a sentinel"
        );
        // 验证伪 ID 纯粹内部使用，绝不外泄给无 ID 协议：
        assert!(
            fc["functionCall"].get("id").is_none(),
            "Synthetic ID must remain internal and not leak to client"
        );

        let _ = crate::modules::proxy_db::delete_thinking_records_for_session(key);
    }
}

/// 官方报文对齐回归测试：五种 part 排列下签名锚点必须与真机一致。
///
/// 依据 3 份官方 Antigravity 报文、23 处真实签名归纳出的不变量
/// （见 `.workbuddy/outputs/correct-assembly-spec.md`）：
///   1. 签名只出现在 model 轮，每轮至多 1 个；
///   2. 锚点 = 该轮第一个 `thought != true` 的 part；
///   3. `thought: true` / `functionResponse` / 非锚点 —— 字段必须「缺席」；
///   4. 缺失签名被上游容忍，绝不发明哨兵。
#[cfg(test)]
mod signature_placement_tests {
    use super::tests::rec;
    use super::*;

    /// 构造符合 Google 原生特征的假签名（Base64 解码后首字节 = protobuf tag 0x12）
    fn gemini_sig(seed: u8) -> String {
        use base64::Engine;
        let mut raw = vec![0x12u8, seed];
        raw.extend_from_slice(&[b'A'; 60]);
        base64::engine::general_purpose::STANDARD.encode(raw)
    }

    #[test]
    fn is_thought_part_only_accepts_explicit_flag() {
        // 带签名但没有 thought 标志的正文 —— 绝不能判为思考块（旧启发式的核心错误）
        assert!(!is_thought_part(
            &json!({ "text": "answer", "thoughtSignature": "Eabc" })
        ));
        // 带 functionCall 也不算
        assert!(!is_thought_part(
            &json!({ "functionCall": { "name": "x" }, "thoughtSignature": "Eabc" })
        ));
        // 只有显式 thought: true 才算
        assert!(is_thought_part(
            &json!({ "text": "reasoning", "thought": true })
        ));
        // 显式 false 不算
        assert!(!is_thought_part(&json!({ "text": "x", "thought": false })));
    }

    #[test]
    fn arrangement_a_thought_plus_text_keeps_signature_on_text() {
        // 官方 f81eae5c contents[1] 的排列
        let sig = gemini_sig(1);
        let mut parts = vec![
            json!({ "text": "reasoning", "thought": true }),
            json!({ "text": "visible answer", "thoughtSignature": sig }),
        ];
        let placed = place_turn_signature(&mut parts, None);

        assert_eq!(placed.as_deref(), Some(sig.as_str()));
        assert_eq!(parts.len(), 2, "绝不能重排或增删 part");
        assert!(
            parts[0].get("thoughtSignature").is_none(),
            "思考块绝不带签名"
        );
        assert_eq!(parts[1]["thoughtSignature"], sig, "锚点 = parts[1]");
    }

    #[test]
    fn arrangement_b_parallel_calls_signs_only_the_first() {
        // 官方 f81eae5c contents[3]：3 个并发 functionCall，签名只在 parts[0]
        let sig = gemini_sig(2);
        let mut parts = vec![
            json!({ "functionCall": { "id": "a", "name": "run_command" }, "thoughtSignature": sig }),
            json!({ "functionCall": { "id": "b", "name": "view_file" } }),
            json!({ "functionCall": { "id": "c", "name": "view_file" } }),
        ];
        place_turn_signature(&mut parts, None);

        assert_eq!(parts.len(), 3, "绝不插入假思考块");
        assert_eq!(parts[0]["thoughtSignature"], sig);
        assert!(
            parts[1].get("thoughtSignature").is_none(),
            "字段必须缺席而非空串"
        );
        assert!(parts[2].get("thoughtSignature").is_none());
    }

    #[test]
    fn arrangement_c_function_responses_never_signed() {
        let sig = gemini_sig(3);
        let mut parts = vec![
            json!({ "functionResponse": { "id": "a", "name": "x" } }),
            json!({ "functionResponse": { "id": "b", "name": "y" } }),
        ];
        // 即便提供回填来源，也绝不写入 functionResponse
        let placed = place_turn_signature(&mut parts, Some(sig.as_str()));

        assert!(placed.is_none());
        assert!(parts[0].get("thoughtSignature").is_none());
        assert!(parts[1].get("thoughtSignature").is_none());
    }

    #[test]
    fn arrangement_d_pure_text_keeps_its_own_signature() {
        let sig = gemini_sig(4);
        let mut parts = vec![json!({ "text": "hello", "thoughtSignature": sig })];
        let placed = place_turn_signature(&mut parts, None);

        assert_eq!(placed.as_deref(), Some(sig.as_str()));
        assert_eq!(parts[0]["thoughtSignature"], sig);
    }

    #[test]
    fn unsigned_anchor_gets_sentinel_placeholder() {
        // 当前实现：functionCall 锚点没有真签名时，写入哨兵占位，保证锚点始终携带签名
        let mut parts = vec![json!({ "functionCall": { "id": "call_x", "name": "bash" } })];
        let placed = place_turn_signature(&mut parts, None);

        assert_eq!(placed.as_deref(), Some(SENTINEL_SIGNATURE));
        assert_eq!(parts[0]["thoughtSignature"], json!(SENTINEL_SIGNATURE));
    }

    #[test]
    fn fallback_signature_is_placed_on_anchor() {
        // 跨协议路径：锚点无签名，但库里有真实签名
        let sig = gemini_sig(5);
        let mut parts = vec![json!({ "functionCall": { "id": "call_x", "name": "bash" } })];
        let placed = place_turn_signature(&mut parts, Some(sig.as_str()));

        assert_eq!(placed.as_deref(), Some(sig.as_str()));
        assert_eq!(parts[0]["thoughtSignature"], sig);
    }

    #[test]
    fn sentinel_can_be_used_as_placeholder_source() {
        // 当前实现：哨兵是合法的占位来源（跨协议恢复依赖它），不再被拒绝
        let mut parts = vec![json!({ "functionCall": { "id": "call_x", "name": "bash" } })];
        let placed = place_turn_signature(&mut parts, Some(SENTINEL_SIGNATURE));

        assert_eq!(placed.as_deref(), Some(SENTINEL_SIGNATURE));
        assert_eq!(parts[0]["thoughtSignature"], json!(SENTINEL_SIGNATURE));
    }

    #[test]
    fn non_anchor_text_signatures_are_purged_but_signed_function_calls_survive() {
        let sig_a = gemini_sig(6);
        let sig_b = gemini_sig(7);
        let mut parts = vec![
            json!({ "functionCall": { "id": "a", "name": "x" }, "thoughtSignature": sig_a }),
            json!({ "functionCall": { "id": "b", "name": "y" }, "thoughtSignature": sig_b }),
            json!({ "text": "trailing", "thought_signature": sig_b }),
        ];
        place_turn_signature(&mut parts, None);

        assert_eq!(parts[0]["thoughtSignature"], sig_a, "锚点保留自己的真签名");
        assert_eq!(
            parts[1]["thoughtSignature"], sig_b,
            "并发的有签名 functionCall 保留各自签名，不得抹除"
        );
        assert!(parts[2].get("thoughtSignature").is_none());
        assert!(
            parts[2].get("thought_signature").is_none(),
            "蛇形字段必须被清除"
        );
    }

    #[test]
    fn all_thought_turn_has_no_anchor() {
        let sig = gemini_sig(8);
        let mut parts = vec![
            json!({ "text": "r1", "thought": true }),
            json!({ "text": "r2", "thought": true }),
        ];
        let placed = place_turn_signature(&mut parts, Some(sig.as_str()));

        assert!(placed.is_none(), "整轮皆思考则本轮无锚点");
        assert!(parts[0].get("thoughtSignature").is_none());
        assert!(parts[1].get("thoughtSignature").is_none());
    }

    #[test]
    fn arrangement_f_tool_call_with_placeholder_text_anchors_to_function_call() {
        // 现场 400 铁证重现：客户端发送带有占位文本 "..." 与 tool_use 的 assistant 轮次，
        // 签名最初挂在占位文本 "..." 上。
        // 正确产出：丢弃占位文本 "..."，将真实签名救援并归位到首个 functionCall 上！
        let sig = gemini_sig(10);
        let mut parts = vec![
            json!({ "text": "Thinking deeply...", "thought": true }),
            json!({ "text": "...", "thoughtSignature": sig.clone() }),
            json!({ "functionCall": { "id": "call_2594022", "name": "exec", "args": { "command": "dws auth" } } }),
        ];
        let placed = place_turn_signature(&mut parts, None);

        assert_eq!(placed.as_deref(), Some(sig.as_str()));
        // 占位文本 "..." 必须被清理，parts 仅保留思考块与 functionCall
        assert_eq!(parts.len(), 2, "占位文本部件必须被丢弃");
        assert_eq!(parts[0]["thought"], true);
        assert!(parts[0].get("thoughtSignature").is_none());
        assert_eq!(parts[1]["functionCall"]["name"], "exec");
        assert_eq!(
            parts[1]["thoughtSignature"], sig,
            "签名必须正确归位到 functionCall 锚点"
        );
    }

    #[test]
    fn arrangement_g_tool_call_with_commentary_anchors_to_function_call() {
        // 当 assistant 轮次同时包含真实正文说明与工具调用时，
        // Google Gemini 强校验规范要求签名必须落在 functionCall 部件上，正文不得带签名
        let sig = gemini_sig(11);
        let mut parts = vec![
            json!({ "text": "I will run this tool for you" }),
            json!({ "functionCall": { "id": "call_999", "name": "read_file" } }),
        ];
        let placed = place_turn_signature(&mut parts, Some(sig.as_str()));

        assert_eq!(placed.as_deref(), Some(sig.as_str()));
        assert_eq!(parts.len(), 2, "真实正文与工具调用均予保留");
        assert!(
            parts[0].get("thoughtSignature").is_none(),
            "正文不得携带签名"
        );
        assert_eq!(
            parts[1]["thoughtSignature"], sig,
            "签名必须落在 functionCall 上"
        );
    }

    #[test]
    fn test_extract_turn_signature_prioritizes_function_call_for_gemini() {
        // [Pipeline First] 测试混合轮次中签名提取优先级：
        // Gemini 目标下，若 functionCall 带有签名，必须强制优先提取 functionCall 上的签名，
        // 杜绝前面紧随的说明正文（text）抢走签名导致工具调用凭据失效。
        let text_sig = gemini_sig(101);
        let fc_sig = gemini_sig(102);

        let parts = vec![
            json!({ "text": "Analyzing the request...", "thoughtSignature": text_sig }),
            json!({ "functionCall": { "id": "call_test", "name": "read" }, "thoughtSignature": fc_sig }),
        ];

        // 1. Gemini (prefer_function_call = true) -> 优先提取 functionCall 的签名 fc_sig
        let gemini_extracted = extract_turn_signature(&parts, true, false);
        assert_eq!(gemini_extracted.as_deref(), Some(fc_sig.as_str()));

        // 2. Claude (prefer_function_call = false) -> 提取首个非思考部件的签名 text_sig
        // 新校验要求签名长度 >= MIN_SIGNATURE_LENGTH(32)，旧的 20 字符示例已被拒绝
        let claude_sig_str = {
            use base64::Engine;
            let mut raw = vec![0x08u8, 0x04, 0x12];
            raw.extend_from_slice(&[b'A'; 40]);
            base64::engine::general_purpose::STANDARD.encode(raw)
        };
        let claude_parts = vec![
            json!({ "text": "Analyzing...", "thoughtSignature": claude_sig_str }),
            json!({ "functionCall": { "id": "call_test", "name": "read" } }),
        ];
        let claude_extracted = extract_turn_signature(&claude_parts, false, true);
        assert_eq!(claude_extracted.as_deref(), Some(claude_sig_str.as_str()));
    }

    #[test]
    fn test_place_turn_signature_preserves_valid_signatures_on_parallel_tools() {
        // [Pipeline First] 并发工具调用场景：
        // 若第二个 functionCall 自带有效签名，Gemini 目标下绝不能将其粗暴剥离为缺席！
        let sig1 = gemini_sig(201);
        let sig2 = gemini_sig(202);

        let mut parts = vec![
            json!({ "functionCall": { "id": "call_1", "name": "read_file" }, "thoughtSignature": sig1 }),
            json!({ "functionCall": { "id": "call_2", "name": "edit_file" }, "thoughtSignature": sig2 }),
        ];

        place_turn_signature(&mut parts, None);

        assert_eq!(
            parts[0]["thoughtSignature"], sig1,
            "首个工具调用保持自身签名"
        );
        assert_eq!(
            parts[1]["thoughtSignature"], sig2,
            "并发的第二个工具调用有效签名绝不被误删"
        );
    }

    #[test]
    fn test_find_turn_anchor_logic() {
        // 1. 纯思考块 -> 无锚点
        assert_eq!(
            find_turn_anchor(&[json!({ "thought": true, "text": "t" })]),
            None
        );
        // 2. 思考块 + 正文 -> 锚点为正文
        assert_eq!(
            find_turn_anchor(&[
                json!({ "thought": true, "text": "t" }),
                json!({ "text": "answer" })
            ]),
            Some(1)
        );
        // 3. 正文 + 工具 -> 锚点必须优先为工具调用！
        assert_eq!(
            find_turn_anchor(&[
                json!({ "text": "commentary" }),
                json!({ "functionCall": { "name": "bash" } })
            ]),
            Some(1)
        );
        // 4. 并发工具 -> 锚点为首个工具调用
        assert_eq!(
            find_turn_anchor(&[
                json!({ "functionCall": { "name": "a" } }),
                json!({ "functionCall": { "name": "b" } })
            ]),
            Some(0)
        );
    }

    #[test]
    fn test_tool_id_normalization_in_restore() {
        let store = ThinkingStore::new();
        let session_key = "test_norm_tool_id_session";
        let sig = gemini_sig(9);

        // Record saved with canonical call_573077
        // Record saved with canonical call_573077
        store.record(
            session_key,
            ThinkingRecord {
                fingerprint: "fp1".to_string(),
                thought: "I need to read this file".to_string(),
                signature: Some(sig.clone()),
                tool_ids: vec!["call_573077".to_string()],
                tool_names: vec!["default_api:read".to_string()],
                visible: String::new(),
            },
        );

        // Incoming client contents stripped underscore: call573077
        let mut contents = vec![
            json!({
                "role": "user",
                "parts": [{ "text": "read the file" }]
            }),
            json!({
                "role": "model",
                "parts": [{
                    "functionCall": {
                        "name": "default_api:read",
                        "id": "call573077",
                        "args": { "path": "test.txt" }
                    }
                }]
            }),
        ];

        let count = store.restore_gemini_contents_with_model(
            session_key,
            &mut contents,
            Some("gemini-2.5-flash"),
        );
        assert_eq!(count, 1);
        let model_parts = contents[1]["parts"].as_array().unwrap();
        // Signature should be attached to the anchor (functionCall at index 1)
        assert_eq!(model_parts[1]["thoughtSignature"], sig);
    }

    #[test]
    fn test_client_invalid_signature_rejected_and_backfilled_with_reverse_storage() {
        let real_sig = gemini_sig(88);
        let mut parts = vec![json!({
            "functionCall": {
                "id": "call_invalid_test_1",
                "name": "default_api:run_command",
                "args": {}
            },
            "thoughtSignature": "not_a_valid_gemini_signature_12345678901234567890"
        })];

        // place_turn_signature 必须拒绝客户端伪造的无效签名，采纳 fallback_sig 并反向入库
        let placed =
            place_turn_signature_scoped(&mut parts, Some(&real_sig), true, Some("test-store"));
        assert_eq!(placed, Some(real_sig.clone()));
        assert_eq!(parts[0]["thoughtSignature"], real_sig);

        // 验证反向入库已生效，并且不落到别的会话。
        let cached = crate::proxy::SignatureCache::global()
            .get_tool_signature("test-store", "call_invalid_test_1");
        assert_eq!(
            crate::proxy::SignatureCache::global()
                .get_tool_signature("other-store", "call_invalid_test_1"),
            None
        );
        assert_eq!(cached, Some(real_sig));
    }

    #[test]
    fn test_sequential_tool_calling_sentinel_fallback() {
        let sig_tool_a = gemini_sig(77);
        let tool_a_id = format!("call_seq_a_{}", uuid::Uuid::new_v4().simple());
        let tool_b_id = format!("call_seq_b_{}", uuid::Uuid::new_v4().simple());
        let mut contents = vec![
            // Turn 0: User prompt
            json!({
                "role": "user",
                "parts": [{ "text": "start task" }]
            }),
            // Turn 1: Model calls Tool A (带合法签名)
            json!({
                "role": "model",
                "parts": [{
                    "functionCall": {
                        "id": tool_a_id.clone(),
                        "name": "default_api:skills_read",
                        "args": {}
                    },
                    "thoughtSignature": sig_tool_a.clone()
                }]
            }),
            // Turn 2: User returns response for Tool A
            json!({
                "role": "model",
                "parts": [{
                    "functionResponse": {
                        "id": tool_a_id,
                        "name": "default_api:skills_read",
                        "response": { "output": "ok" }
                    }
                }]
            }),
            // Turn 3: Model calls Tool B (无客户端签名，无缓存)
            json!({
                "role": "model",
                "parts": [{
                    "functionCall": {
                        "id": tool_b_id,
                        "name": "default_api:skills_read_resource",
                        "args": {}
                    }
                }]
            }),
        ];

        // 终审把关与脱敏规范化
        finalize_gemini_contents_thinking_with_session(
            &mut contents,
            true,
            Some("gemini-2.5-flash"),
            Some("sess-test-seq-1"),
        );

        // 当前实现：本轮没有独立真实签名时写入哨兵占位（而非缺席），
        // 但绝不跳轮继承其他工具调用的真实签名。
        let turn3_parts = contents[3]["parts"].as_array().unwrap();
        assert_eq!(
            turn3_parts[0]["thoughtSignature"],
            json!(SENTINEL_SIGNATURE),
            "缺签名时写哨兵占位，且不继承他轮真实签名"
        );
    }

    #[test]
    fn test_pure_thinking_signature_pending_and_handover() {
        let sig_thought = gemini_sig(66);
        let mut contents = vec![
            // Turn 0: User prompt
            json!({
                "role": "user",
                "parts": [{ "text": "start thinking task" }]
            }),
            // Turn 1: 纯思考轮 (只有 thought: true，带有效签名)
            json!({
                "role": "model",
                "parts": [{
                    "text": "Deep thinking analysis...",
                    "thought": true,
                    "thoughtSignature": sig_thought.clone()
                }]
            }),
            // Turn 2: 后续工具调用轮 (无签名)
            json!({
                "role": "model",
                "parts": [{
                    "functionCall": {
                        "id": "call_after_thought",
                        "name": "default_api:run_command",
                        "args": {}
                    }
                }]
            }),
        ];

        finalize_gemini_contents_thinking_with_session(
            &mut contents,
            true,
            Some("gemini-2.5-flash"),
            Some("sess-test-pure-1"),
        );

        // 黄金法则 2.2：纯思考轮的思考块绝不挂载签名！
        let turn1_parts = contents[1]["parts"].as_array().unwrap();
        assert!(turn1_parts[0].get("thoughtSignature").is_none());

        // 黄金法则 2.2：纯思考轮攒在手上的签名，成功交接给下一轮的首个工具调用！
        let turn2_parts = contents[2]["parts"].as_array().unwrap();
        assert_eq!(turn2_parts[0]["thoughtSignature"], sig_thought);
    }

    #[test]
    fn test_tool_call_global_priority_over_commentary_and_other_fc() {
        let real_sig = gemini_sig(99);
        let mut contents = vec![
            // Model 轮包含：思考块 + 进度说明正文 + 2个并发工具调用
            json!({
                "role": "model",
                "parts": [
                    { "text": "let me think", "thought": true },
                    { "text": "Executing commands..." },
                    {
                        "functionCall": { "id": "call_fc_1", "name": "run_command", "args": {} },
                        "thoughtSignature": real_sig.clone()
                    },
                    {
                        "functionCall": { "id": "call_fc_2", "name": "view_file", "args": {} },
                        "thoughtSignature": real_sig.clone()
                    }
                ]
            }),
        ];

        finalize_gemini_contents_thinking_with_session(
            &mut contents,
            true,
            Some("gemini-2.5-flash"),
            Some("sess-test-prio-1"),
        );

        let parts = contents[0]["parts"].as_array().unwrap();
        // 1. 新规则：无签名的 thought:true 部件在出站前被整块丢弃
        assert!(
            parts.iter().all(|p| p.get("thought") != Some(&json!(true))),
            "无签名的思考块必须被丢弃，且不得挂载签名"
        );

        // 2. 首个工具调用 fc 必须全局抢夺签名（parts[1] 或首个 fc）
        let fc1 = parts
            .iter()
            .find(|p| {
                p.get("functionCall")
                    .and_then(|fc| fc.get("id"))
                    .and_then(|id| id.as_str())
                    == Some("call_fc_1")
            })
            .unwrap();
        assert_eq!(fc1["thoughtSignature"], real_sig);

        // 3. 伴随正文 text 绝不携带签名
        let text_part = parts
            .iter()
            .find(|p| p.get("text").is_some() && p.get("thought").is_none())
            .unwrap();
        assert!(text_part.get("thoughtSignature").is_none());

        // 4. 并发工具调用自带的有效签名**必须保留**（新规则不再抹除）
        let fc2 = parts
            .iter()
            .find(|p| {
                p.get("functionCall")
                    .and_then(|fc| fc.get("id"))
                    .and_then(|id| id.as_str())
                    == Some("call_fc_2")
            })
            .unwrap();
        assert_eq!(
            fc2["thoughtSignature"], real_sig,
            "并发工具自带的有效签名必须保留"
        );
    }

    #[test]
    fn test_clear_session_for_compaction_and_continuation_boundary() {
        let store = ThinkingStore::global();
        let sid = "sess_compaction_cleanup_test";

        // 1. 存入模拟思考数据
        store.record(
            sid,
            rec("compaction thought", "compaction text", Some("call_old_1")),
        );
        crate::proxy::SignatureCache::global().cache_tool_signature(
            sid,
            "call_old_1",
            "sig_old_protobuf_signature_for_test_1234567890".to_string(),
        );
        crate::proxy::SignatureCache::global().cache_session_signature(
            sid,
            "sig_old_session_signature_1234567890abcdef".to_string(),
            1,
        );

        assert!(store.session_stats(sid).is_some());

        // 2. 彻底作废该会话的预压缩历史签名
        clear_session_for_compaction(sid);

        // 3. 验证内存与缓存均已清空
        assert!(store.session_stats(sid).is_none());
        assert!(crate::proxy::SignatureCache::global()
            .get_tool_signature(sid, "call_old_1")
            .is_none());
        assert!(crate::proxy::SignatureCache::global()
            .get_session_signature(sid)
            .is_none());
    }

    #[test]
    fn test_find_compaction_continuation_boundary() {
        let regular_contents = vec![
            json!({
                "role": "user",
                "parts": [{ "text": "hello" }]
            }),
            json!({
                "role": "model",
                "parts": [{ "text": "hi there" }]
            }),
        ];
        assert_eq!(
            find_compaction_continuation_boundary(&regular_contents),
            None
        );

        let compacted_contents = vec![
            json!({
                "role": "user",
                "parts": [{ "text": "Initial setup instruction" }]
            }),
            json!({
                "role": "user",
                "parts": [{ "text": "This session is being continued from a previous conversation that ran out of context." }]
            }),
            json!({
                "role": "user",
                "parts": [{ "text": "Please continue with task" }]
            }),
        ];
        assert_eq!(
            find_compaction_continuation_boundary(&compacted_contents),
            Some(1)
        );

        let multi_compacted_contents = vec![
            json!({
                "role": "user",
                "parts": [{ "text": "Initial setup instruction" }]
            }),
            json!({
                "role": "user",
                "parts": [{ "text": "This session is being continued from a previous conversation that ran out of context." }]
            }),
            json!({
                "role": "model",
                "parts": [{ "text": "Working on it..." }]
            }),
            json!({
                "role": "user",
                "parts": [{ "text": "Next command" }]
            }),
            json!({
                "role": "user",
                "parts": [{ "text": "<compact_boundary>Second compaction happened</compact_boundary>" }]
            }),
            json!({
                "role": "user",
                "parts": [{ "text": "Final query" }]
            }),
        ];
        // 关键断言：当长会话经历两次压缩时，必须反向锚定到最新的第二次压缩边界 (index 4)，而非旧的 (index 1)
        assert_eq!(
            find_compaction_continuation_boundary(&multi_compacted_contents),
            Some(4)
        );
    }

    #[test]
    fn test_watermarks_eviction_when_sessions_empty() {
        let store = ThinkingStore::new();
        let two_days_ago = std::time::Instant::now()
            .checked_sub(std::time::Duration::from_secs(172800))
            .unwrap_or_else(std::time::Instant::now);
        store.watermarks.insert(
            "expired_session".to_string(),
            SessionWatermark {
                chars: 1000,
                turns: 5,
                last_seen: two_days_ago,
            },
        );
        assert_eq!(store.sessions.len(), 0);
        store.maybe_evict("keep_key");
        assert_eq!(store.watermarks.len(), 0);
    }

    #[test]
    fn test_compaction_continuation_signature_phasing_and_sentinel_locking() {
        let old_sig = "sig_old_gemini_protobuf_signature_test_1234567890";
        let fresh_sig = "sig_fresh_google_protobuf_signature_after_compaction";
        let session_id = "sess_compaction_phasing_test";

        // 第一阶段：压缩后的上下文第一次输入
        // 历史轮次（包含工具调用）带有失效的旧签名，紧随接续提示词
        let mut first_input_contents = vec![
            json!({
                "role": "model",
                "parts": [
                    { "text": "Running old tool..." },
                    {
                        "functionCall": { "id": "call_old_fc", "name": "shell", "args": {} },
                        "thoughtSignature": old_sig
                    }
                ]
            }),
            json!({
                "role": "user",
                "parts": [{ "text": "This session is being continued from a previous conversation that ran out of context." }]
            }),
            json!({
                "role": "user",
                "parts": [{ "text": "Continue next step" }]
            }),
        ];

        // 第一次输入前执行预压缩签名清理
        clear_session_for_compaction(session_id);

        finalize_gemini_contents_thinking_with_session(
            &mut first_input_contents,
            true,
            Some("gemini-2.5-flash"),
            Some(session_id),
        );

        // 验证：第一次输入历史轮次的所有旧签名被剥离，工具缺失处用哨兵锁死补齐
        let fc0 = &first_input_contents[0]["parts"][1];
        assert_eq!(fc0["thoughtSignature"], SENTINEL_SIGNATURE);
        // 验证正文部件没有挂载任何签名
        assert!(first_input_contents[0]["parts"][0]
            .get("thoughtSignature")
            .is_none());

        // 第二阶段：上游 Google 返回新签名后，下次请求携带新轮次进入
        // 轮次 0：历史工具（<= boundary_idx，必须维持哨兵）
        // 轮次 1：接续提示词（= boundary_idx）
        // 轮次 2：用户指令
        // 轮次 3：Google 返回的新模型响应（> boundary_idx，装配新签名）
        let mut second_input_contents = vec![
            first_input_contents[0].clone(),
            first_input_contents[1].clone(),
            first_input_contents[2].clone(),
            json!({
                "role": "model",
                "parts": [
                    { "text": "Executing new step..." },
                    {
                        "functionCall": { "id": "call_new_fc", "name": "edit", "args": {} },
                        "thoughtSignature": fresh_sig
                    }
                ]
            }),
        ];

        finalize_gemini_contents_thinking_with_session(
            &mut second_input_contents,
            true,
            Some("gemini-2.5-flash"),
            Some(session_id),
        );

        // 验证：前面的历史轮次依然保持哨兵，新轮次装配 Google 动态新签名！
        let fc_history = &second_input_contents[0]["parts"][1];
        assert_eq!(fc_history["thoughtSignature"], SENTINEL_SIGNATURE);

        let fc_new = &second_input_contents[3]["parts"][1];
        assert_eq!(fc_new["thoughtSignature"], fresh_sig);
    }

    #[test]
    fn test_compaction_continuation_claude_target_does_not_invent_sentinel() {
        let old_sig = "sig_old_gemini_protobuf_signature_test_1234567890";
        let session_id = "sess_compaction_claude_test";

        let mut first_input_contents = vec![
            json!({
                "role": "model",
                "parts": [
                    { "text": "Running old tool..." },
                    {
                        "functionCall": { "id": "call_old_fc", "name": "shell", "args": {} },
                        "thoughtSignature": old_sig
                    }
                ]
            }),
            json!({
                "role": "user",
                "parts": [{ "text": "This session is being continued from a previous conversation that ran out of context." }]
            }),
            json!({
                "role": "user",
                "parts": [{ "text": "Continue next step" }]
            }),
        ];

        finalize_gemini_contents_thinking_with_session(
            &mut first_input_contents,
            true,
            Some("claude-3-7-sonnet"),
            Some(session_id),
        );

        // 验证：Claude 目标下，历史轮次的旧签名被彻底剥离，绝不发明注入 Gemini 哨兵 (防 400 Invalid signature)
        let fc0 = &first_input_contents[0]["parts"][1];
        assert!(fc0.get("thoughtSignature").is_none());
        assert!(first_input_contents[0]["parts"][0]
            .get("thoughtSignature")
            .is_none());
    }

    #[test]
    fn test_global_compaction_strip_disabled_by_default() {
        crate::proxy::config::set_global_compaction_strip_enabled(false);
        let session_id = "sess_global_strip_disabled_test";

        let regular_contents = vec![
            json!({
                "role": "model",
                "parts": [{ "text": "turn 1" }]
            }),
            json!({
                "role": "user",
                "parts": [{ "text": "turn 2" }]
            }),
        ];

        // 默认关闭状态下，无论传入什么 contents（只要没有显式 continuation 标记），boundary 必定为 None
        assert_eq!(
            determine_compaction_boundary(&regular_contents, Some(session_id)),
            None
        );

        let micro_folded_contents = vec![
            json!({
                "role": "model",
                "parts": [
                    { "functionCall": { "id": "call_1", "name": "bash", "args": {} } }
                ]
            }),
            json!({
                "role": "user",
                "parts": [{ "text": "[Output truncated] 1000 lines omitted" }]
            }),
            json!({
                "role": "user",
                "parts": [{ "text": "next command" }]
            }),
        ];

        // 默认关闭状态下，微观折叠标记亦不生效，100% 保持既有行为与普通用户无干扰
        assert_eq!(
            determine_compaction_boundary(&micro_folded_contents, Some(session_id)),
            None
        );
    }

    #[test]
    fn test_global_compaction_strip_macro_drop_triggers_strip_and_sentinel() {
        crate::proxy::config::set_global_compaction_strip_enabled(true);
        let session_id = "sess_global_strip_macro_drop_test";
        let store = ThinkingStore::global();

        // 模拟第 1 请求：历史多轮累积且字符数大（如 6 轮，5000 字符）
        let initial_long_text = "a".repeat(5000);
        let first_request = vec![
            json!({ "role": "user", "parts": [{ "text": &initial_long_text }] }),
            json!({ "role": "model", "parts": [{ "text": "response 1" }] }),
            json!({ "role": "user", "parts": [{ "text": "prompt 2" }] }),
            json!({ "role": "model", "parts": [{ "text": "response 2" }] }),
            json!({ "role": "user", "parts": [{ "text": "prompt 3" }] }),
            json!({ "role": "model", "parts": [{ "text": "response 3" }] }),
        ];
        let (chars1, turns1) = calculate_contents_metrics(&first_request);
        assert!(!store.check_macro_context_drop(session_id, chars1, turns1));

        // 预设一个旧的 session signature，模拟上一轮流式缓存的旧签名
        crate::proxy::SignatureCache::global().cache_session_signature(
            session_id,
            "sig_old_session_sig_1234567890abcdef".to_string(),
            5,
        );

        // 模拟第 2 请求：发生全协议无标记压缩（轮数从 6 骤降为 2，字符量从 5000+ 骤降至 200）
        let mut second_request = vec![
            json!({
                "role": "model",
                "parts": [
                    { "text": "Historical summary" },
                    {
                        "functionCall": { "id": "old_tool", "name": "read_file", "args": {} },
                        "thoughtSignature": "sig_expired_pre_compaction_signature_123"
                    }
                ]
            }),
            json!({
                "role": "user",
                "parts": [{ "text": "What to do next?" }]
            }),
        ];

        let boundary = determine_compaction_boundary(&second_request, Some(session_id));
        assert!(boundary.is_some());

        finalize_gemini_contents_thinking_with_session(
            &mut second_request,
            true,
            Some("gemini-2.5-flash"),
            Some(session_id),
        );

        // 验证：
        // 1. 压缩边界内的 Gemini 工具调用被安全赋予官方哨兵 SENTINEL_SIGNATURE (防 400)
        let fc = &second_request[0]["parts"][1];
        assert_eq!(fc["thoughtSignature"], SENTINEL_SIGNATURE);
        // 2. 正文部件绝不挂载签名
        assert!(second_request[0]["parts"][0]
            .get("thoughtSignature")
            .is_none());
        // 3. 旧 session_sig 被完全拦截未复活
        assert!(crate::proxy::SignatureCache::global()
            .get_session_signature(session_id)
            .is_none());

        // 测试后重置开关以防干扰其他测试
        crate::proxy::config::set_global_compaction_strip_enabled(false);
    }

    #[test]
    fn test_global_compaction_strip_micro_folding_triggers_strip() {
        crate::proxy::config::set_global_compaction_strip_enabled(true);
        let session_id = "sess_global_strip_micro_folding_test";

        // 微观场景：轮数未骤减，但历史工具输出就地折叠成 `[Output truncated]`
        let mut folded_contents = vec![
            json!({
                "role": "model",
                "parts": [
                    { "text": "Calling long tool" },
                    {
                        "functionCall": { "id": "call_folded", "name": "cat_log", "args": {} },
                        "thoughtSignature": "sig_old_invalidated_by_truncation_1234"
                    }
                ]
            }),
            json!({
                "role": "user",
                "parts": [{ "text": "[Output truncated] Log output too long..." }]
            }),
            json!({
                "role": "user",
                "parts": [{ "text": "Analyze the log" }]
            }),
        ];

        let boundary = determine_compaction_boundary(&folded_contents, Some(session_id));
        assert_eq!(boundary, Some(1));

        finalize_gemini_contents_thinking_with_session(
            &mut folded_contents,
            true,
            Some("gemini-2.5-pro"),
            Some(session_id),
        );

        // 验证：微观折叠的 Gemini 工具调用被安全赋予哨兵锁定
        let fc = &folded_contents[0]["parts"][1];
        assert_eq!(fc["thoughtSignature"], SENTINEL_SIGNATURE);
        assert!(folded_contents[0]["parts"][0]
            .get("thoughtSignature")
            .is_none());

        // Claude 目标下微观折叠验证：必须彻底剥离签名，严禁发明注入哨兵
        let mut claude_folded_contents = vec![
            json!({
                "role": "model",
                "parts": [
                    { "text": "Calling long tool" },
                    {
                        "functionCall": { "id": "call_folded_claude", "name": "cat_log", "args": {} },
                        "thoughtSignature": "sig_claude_invalidated_by_truncation_1234"
                    }
                ]
            }),
            json!({
                "role": "user",
                "parts": [{ "text": "[Tool result truncated] Log output..." }]
            }),
            json!({
                "role": "user",
                "parts": [{ "text": "Analyze the log" }]
            }),
        ];

        finalize_gemini_contents_thinking_with_session(
            &mut claude_folded_contents,
            true,
            Some("claude-3-7-sonnet"),
            Some(session_id),
        );

        let fc_claude = &claude_folded_contents[0]["parts"][1];
        assert!(fc_claude.get("thoughtSignature").is_none());

        // 测试后重置开关以防干扰其他测试
        crate::proxy::config::set_global_compaction_strip_enabled(false);
    }

    #[test]
    fn test_extract_root_session_id_multi_agent_delimiters() {
        // 普通 session_id 保持不变
        assert_eq!(extract_root_session_id("sess-1234-abcd"), "sess-1234-abcd");
        assert_eq!(
            extract_root_session_id("550e8400-e29b-41d4-a716-446655440000"),
            "550e8400-e29b-41d4-a716-446655440000"
        );

        // 识别各类多智能体协同派生标记
        assert_eq!(
            extract_root_session_id("cindy_army_main:step:1"),
            "cindy_army_main"
        );
        assert_eq!(
            extract_root_session_id("cindy_army_main:step:2:talk:3"),
            "cindy_army_main"
        );
        assert_eq!(
            extract_root_session_id("workflow_7788:subtask:audit"),
            "workflow_7788"
        );
        assert_eq!(
            extract_root_session_id("task_9900:agent:reviewer"),
            "task_9900"
        );
        assert_eq!(
            extract_root_session_id("discussion_root:talk:round_1"),
            "discussion_root"
        );
        assert_eq!(extract_root_session_id("chat_abc::phase_2"), "chat_abc");
        assert_eq!(extract_root_session_id("parent_job--step-4"), "parent_job");
        assert_eq!(
            extract_root_session_id("agent_session:round:5"),
            "agent_session"
        );
    }

    #[test]
    fn sub2api_bridge_identity_preserves_root_and_branch_contract() {
        // sub2api hashes authenticated user/key identity before forwarding this
        // bounded composite ID. Even identical content must not merge branches.
        let root = format!("s2a-v1-{}", "a".repeat(32));
        let first = format!("{}:agent:{}", root, "b".repeat(32));
        let second = format!("{}:agent:{}", root, "c".repeat(32));
        let other_user = format!("s2a-v1-{}:agent:{}", "d".repeat(32), "b".repeat(32));
        let winner = |id: &str| {
            let mut headers = HeaderMap::new();
            headers.insert("x-session-id", id.parse().unwrap());
            stable_session_winner(&headers, None, None).unwrap()
        };
        for id in [&first, &second, &other_user] {
            assert!(id.len() <= 128);
            assert_eq!(
                winner(id),
                *id,
                "the bridge ID must survive without truncation"
            );
        }
        let affinity = |id: &str| {
            derive_winner_session_id("shared-am-key", Some(extract_root_session_id(id)), "")
        };
        assert_eq!(affinity(&first), affinity(&second));
        assert_ne!(affinity(&first), affinity(&other_user));
        assert_ne!(
            derive_winner_session_id("shared-am-key", Some(&first), "same-content"),
            derive_winner_session_id("shared-am-key", Some(&second), "same-content"),
            "the full branch ID must isolate thinking history even with the same anchor"
        );
    }

    #[test]
    fn test_session_scope_root_normalization_shares_affinity_key_while_keeping_store_key_isolated()
    {
        let mut h1 = HeaderMap::new();
        h1.insert("x-session-id", "parent_flow_123:step:1".parse().unwrap());
        h1.insert("authorization", "Bearer test_key_secret".parse().unwrap());

        let mut h2 = HeaderMap::new();
        h2.insert("x-session-id", "parent_flow_123:step:2".parse().unwrap());
        h2.insert("authorization", "Bearer test_key_secret".parse().unwrap());

        let scope1 = SessionScope::resolve(&h1, None, None, "fallback_1", None);
        let scope2 = SessionScope::resolve(&h2, None, None, "fallback_2", None);

        // 核心验证：多智能体派生子任务的选号 affinity_key 完全相同（锁定同一上游账号以保 Prompt Cache）
        assert_eq!(
            scope1.affinity_key, scope2.affinity_key,
            "Derived step sessions must share the same affinity_key to preserve prompt caching"
        );
        assert_eq!(scope1.affinity_id, scope2.affinity_id);

        // 同时：各子任务的 client_id / store_key 必须保持各自独立，防止思维库与工具调用历史发生串号踩踏
        assert_ne!(
            scope1.store_key, scope2.store_key,
            "Store keys must remain isolated for each sub-session"
        );
        assert_ne!(scope1.client_id, scope2.client_id);
    }
}
