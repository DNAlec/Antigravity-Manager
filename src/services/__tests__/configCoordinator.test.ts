// Run: node --experimental-transform-types src/services/__tests__/configCoordinator.test.ts
import { ConfigCoordinator } from '../configCoordinator.ts';
import type { AppConfig } from '../../types/config.ts';

function assert(value: unknown, message: string): asserts value {
    if (!value) throw new Error(message);
}
function deferred<T>() {
    let resolve!: (value: T) => void;
    let reject!: (error: Error) => void;
    const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
    return { promise, resolve, reject };
}
const fixture = () => ({
    config_revision: 1, theme: 'light', language: 'en',
    proxy: { api_key: 'synthetic-key', port: 8045, custom_mapping: {},
        proxy_pool: { proxies: [], account_bindings: {} },
        upload_timeouts: { extra_wait_ratio: 5 }, stream_timeouts: { initial_ttft_secs: 300 },
    },
} as unknown as AppConfig);

async function test(name: string, run: () => Promise<void>) {
    await run();
    console.log(`PASS ${name}`);
}

await test('rapid edits serialize and keep credentials and earlier fields', async () => {
    let disk = fixture();
    const first = deferred<void>();
    let writes = 0;
    const coordinator = new ConfigCoordinator({
        load: async () => structuredClone(disk),
        save: async config => {
            ++writes;
            if (writes === 1) await first.promise;
            assert(config.config_revision === disk.config_revision, 'queued save must use committed revision');
            disk = { ...structuredClone(config), config_revision: disk.config_revision! + 1 };
            return structuredClone(disk);
        },
    }, () => {});
    await coordinator.load();
    const one = coordinator.update(current => ({ ...current, theme: 'dark' }));
    const two = coordinator.update(current => ({ ...current, language: 'zh' }));
    await Promise.resolve();
    assert(writes === 1, 'only one request may be in flight');
    first.resolve();
    await Promise.all([one, two]);
    assert(disk.theme === 'dark' && disk.language === 'zh', 'both edits must survive');
    assert(disk.proxy.api_key === 'synthetic-key' && disk.proxy.port === 8045, 'unrelated fields changed');
    assert(coordinator.current?.config_revision === 3, 'UI revision must match disk');
});

await test('late GET cannot replace a newer edit', async () => {
    let disk = fixture();
    const stale = deferred<AppConfig>();
    let reads = 0;
    const coordinator = new ConfigCoordinator({
        load: async () => ++reads === 2 ? stale.promise : structuredClone(disk),
        save: async config => (disk = { ...config, config_revision: disk.config_revision! + 1 }),
    }, () => {});
    await coordinator.load();
    const old = structuredClone(disk);
    const loading = coordinator.load();
    await Promise.resolve();
    await coordinator.update(current => ({ ...current, theme: 'dark' }));
    stale.resolve(old);
    await loading;
    assert(coordinator.current?.theme === 'dark', 'old response rolled state back');
});

await test('binding command and a queued theme change preserve both changes', async () => {
    let disk = fixture();
    const binding = deferred<void>();
    const coordinator = new ConfigCoordinator({
        load: async () => structuredClone(disk),
        save: async config => (disk = { ...structuredClone(config), config_revision: disk.config_revision! + 1 }),
    }, () => {});
    await coordinator.load();
    const command = coordinator.mutate(async () => {
        await binding.promise;
        disk.proxy.proxy_pool!.account_bindings = { account: 'proxy' };
        disk.config_revision!++;
    });
    const theme = coordinator.update(current => ({ ...current, theme: 'dark' }));
    binding.resolve();
    await Promise.all([command, theme]);
    assert(disk.proxy.proxy_pool!.account_bindings?.account === 'proxy', 'theme reverted binding');
    assert(disk.theme === 'dark', 'queued theme was lost');
});

await test('a stale browser write fails and queued writes cannot silently retry it', async () => {
    let disk = fixture();
    let writes = 0;
    const coordinator = new ConfigCoordinator({
        load: async () => structuredClone(disk),
        save: async config => {
            ++writes;
            if (config.config_revision !== disk.config_revision) throw new Error('CONFIG_CONFLICT');
            return (disk = { ...config, config_revision: disk.config_revision! + 1 });
        },
    }, () => {});
    await coordinator.load();
    disk = { ...disk, config_revision: 2, proxy: { ...disk.proxy, api_key: 'other-client-key' } };
    const results = await Promise.allSettled([
        coordinator.update(current => ({ ...current, theme: 'dark' })),
        coordinator.update(current => ({ ...current, language: 'zh' })),
    ]);
    assert(results.every(result => result.status === 'rejected'), 'failed save reported success');
    assert(writes === 1, 'stale queued snapshot was retried');
    assert(coordinator.current?.proxy.api_key === 'other-client-key', 'failure did not refresh canonical config');
});

await test('failed disk write rolls back optimistic UI and rejects caller', async () => {
    const disk = fixture();
    const coordinator = new ConfigCoordinator({
        load: async () => structuredClone(disk),
        save: async () => { throw new Error('disk full'); },
    }, () => {});
    await coordinator.load();
    const [result] = await Promise.allSettled([coordinator.update(current => ({ ...current, theme: 'dark' }))]);
    assert(result.status === 'rejected', 'disk failure was swallowed');
    assert(coordinator.current?.theme === 'light', 'failed value remains in UI');
});

await test('unloaded default form and stale revision are rejected before transmission', async () => {
    let writes = 0;
    const coordinator = new ConfigCoordinator({
        load: async () => fixture(),
        save: async config => { ++writes; return config; },
    }, () => {});
    const [unloaded] = await Promise.allSettled([coordinator.save(fixture())]);
    await coordinator.load();
    const [stale] = await Promise.allSettled([coordinator.save({ ...fixture(), config_revision: 0 })]);
    assert(unloaded.status === 'rejected' && stale.status === 'rejected', 'unsafe form accepted');
    assert(writes === 0, 'unsafe form reached server');
});

await test('a combined timeout reset is one write with both groups', async () => {
    let disk = fixture();
    let writes = 0;
    const coordinator = new ConfigCoordinator({
        load: async () => structuredClone(disk),
        save: async config => { ++writes; return (disk = { ...config, config_revision: 2 }); },
    }, () => {});
    await coordinator.load();
    await coordinator.update(current => ({ ...current, proxy: { ...current.proxy,
        upload_timeouts: { extra_wait_ratio: 0.4 }, stream_timeouts: { initial_ttft_secs: 180 },
    } }));
    assert(writes === 1 && disk.proxy.upload_timeouts?.extra_wait_ratio === 0.4 &&
        disk.proxy.stream_timeouts?.initial_ttft_secs === 180, 'timeout reset was split');
});

await test('writes during delayed failure recovery are rejected without UI/disk divergence', async () => {
    let disk = fixture();
    const recovery = deferred<AppConfig>();
    const recovering = deferred<void>();
    let reads = 0;
    let writes = 0;
    const coordinator = new ConfigCoordinator({
        load: async () => {
            if (++reads === 2) { recovering.resolve(); return recovery.promise; }
            return structuredClone(disk);
        },
        save: async config => {
            if (++writes === 1) throw new Error('disk full');
            return (disk = { ...config, config_revision: disk.config_revision! + 1 });
        },
    }, () => {});
    await coordinator.load();
    const failed = coordinator.update(current => ({ ...current, theme: 'dark' })).catch(() => {});
    await recovering.promise;
    let ranCommand = false;
    const attempts = await Promise.allSettled([
        coordinator.update(current => ({ ...current, language: 'zh' })),
        coordinator.mutate(async () => { ranCommand = true; }),
    ]);
    assert(attempts.every(result => result.status === 'rejected') && !ranCommand, 'recovery accepted new work');
    recovery.resolve(structuredClone(disk));
    await failed;
    await coordinator.update(current => ({ ...current, language: 'zh' }));
    assert(JSON.stringify(coordinator.current) === JSON.stringify(disk), 'UI/disk diverged after recovery');
    assert(disk.language === 'zh' && disk.theme === 'light', 'retry did not preserve canonical state');
});

const { rebaseConfigDraft } = await import('../configCoordinator.ts');
await test('dirty draft survives disjoint publication and gets the canonical revision', async () => {
    const base = fixture();
    const draft = { ...base, language: 'zh' };
    const source = { ...base, config_revision: 2, proxy: { ...base.proxy,
        proxy_pool: { ...base.proxy.proxy_pool!, account_bindings: { account: 'proxy' } },
    } };
    const rebased = rebaseConfigDraft({ base, draft, conflict: null }, source);
    assert(!rebased.conflict && rebased.draft.language === 'zh', 'dirty field lost');
    assert(rebased.draft.config_revision === 2 && rebased.draft.proxy.proxy_pool!.account_bindings?.account === 'proxy', 'canonical update lost');
    const saved = { ...rebased.draft, config_revision: 3 };
    const confirmed = rebaseConfigDraft(rebased, saved);
    assert(!confirmed.conflict && confirmed.draft.config_revision === 3, 'own save left a stale draft');
});

await test('conflicting draft is retained until explicitly discarded', async () => {
    const base = fixture();
    const draft = { ...base, language: 'zh' };
    const state = rebaseConfigDraft({ base, draft, conflict: null }, { ...base, config_revision: 2, language: 'ja' });
    assert(!!state.conflict && state.draft.language === 'zh' && state.base === base, 'conflict erased local input');
});
