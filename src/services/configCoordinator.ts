import type { AppConfig } from '../types/config';

const equal = (a: unknown, b: unknown): boolean => JSON.stringify(a) === JSON.stringify(b);
const object = (value: unknown): value is Record<string, unknown> =>
    value !== null && typeof value === 'object' && !Array.isArray(value);

// Apply only the intended changes. Concurrent edits to the same field are a
// conflict, never permission to silently replay an obsolete whole snapshot.
export function mergeConfigChanges(base: unknown, desired: unknown, current: unknown, path = ''): unknown {
    if (equal(base, desired)) return current;
    if (object(base) && object(desired) && object(current)) {
        const result = { ...current };
        for (const key of new Set([...Object.keys(base), ...Object.keys(desired)])) {
            if (key === 'config_revision') continue;
            const value = mergeConfigChanges(base[key], desired[key], current[key], `${path}/${key}`);
            if (value === undefined) delete result[key];
            else result[key] = value;
        }
        return result;
    }
    if (!equal(current, base) && !equal(current, desired)) {
        throw new Error(`CONFIG_CONFLICT: ${path || '/'} changed; reload before saving`);
    }
    return desired;
}

type Transport = {
    load: () => Promise<AppConfig>;
    save: (config: AppConfig) => Promise<AppConfig>;
};

export class ConfigCoordinator {
    current: AppConfig | null = null;
    private confirmed: AppConfig | null = null;
    private tail: Promise<unknown> = Promise.resolve();
    private generation = 0;
    private sequence = 0;
    private pending = 0;
    private recovering = false;

    constructor(private transport: Transport, private publish: (config: AppConfig) => void) {}

    private set(config: AppConfig) {
        this.current = config;
        this.publish(config);
    }

    async load(): Promise<void> {
        // A read must not replace queued edits or publish a snapshot obtained
        // before a subsequent write started.
        await this.tail;
        const sequence = ++this.sequence;
        const config = await this.transport.load();
        if (this.pending === 0 && sequence === this.sequence) {
            this.confirmed = config;
            this.set(config);
        }
    }

    save(next: AppConfig): Promise<void> {
        if (this.recovering) return Promise.reject(new Error('Configuration recovery in progress; retry after reload'));
        const base = this.current;
        if (!base || !this.confirmed || next.config_revision !== base.config_revision) {
            return Promise.reject(new Error('CONFIG_CONFLICT: Reload configuration before saving'));
        }
        this.set(next);
        return this.enqueue(async () => {
            const desired = mergeConfigChanges(base, next, this.confirmed) as AppConfig;
            return this.transport.save(desired);
        }, next);
    }

    update(update: (current: AppConfig) => AppConfig): Promise<void> {
        if (!this.current) return Promise.reject(new Error('Configuration has not loaded'));
        return this.save(update(this.current));
    }

    // Independent commands (binding, starting/stopping) share the write queue
    // and refresh the same source of truth before later edits are persisted.
    mutate(action: () => Promise<unknown>): Promise<void> {
        if (this.recovering) return Promise.reject(new Error('Configuration recovery in progress; retry after reload'));
        return this.enqueue(async () => {
            await action();
            return this.transport.load();
        });
    }

    private enqueue(action: () => Promise<AppConfig>, optimistic?: AppConfig): Promise<void> {
        const generation = this.generation;
        ++this.sequence;
        ++this.pending;
        const operation = this.tail.then(async () => {
            if (generation !== this.generation) throw new Error('Configuration update cancelled after a failed save');
            const before = optimistic ?? this.confirmed;
            try {
                const saved = await action();
                this.confirmed = saved;
                const rebased = mergeConfigChanges(before, this.current, saved) as AppConfig;
                this.set(rebased);
            } catch (error) {
                this.recovering = true;
                ++this.generation;
                // An ambiguous network failure may have committed. Read back
                // rather than retrying an unsafe complete snapshot.
                try {
                    this.confirmed = await this.transport.load();
                } catch { /* Retain the last confirmed snapshot if offline. */ }
                if (this.confirmed) this.set(this.confirmed);
                this.recovering = false;
                throw error;
            }
        }).finally(() => { --this.pending; });
        this.tail = operation.catch(() => {});
        return operation;
    }
}

export type ConfigDraft<T> = { base: T | null; draft: T; conflict: string | null };

// Advance the baseline only when all local edits can be preserved. A conflict
// retains both the draft and its original baseline until explicitly discarded.
export function rebaseConfigDraft<T>(state: ConfigDraft<T>, source: T): ConfigDraft<T> {
    if (state.base === null) return { base: source, draft: source, conflict: null };
    try {
        const draft = mergeConfigChanges(state.base, state.draft, source) as T;
        return { base: source, draft, conflict: null };
    } catch (error) {
        return { ...state, conflict: String(error) };
    }
}
