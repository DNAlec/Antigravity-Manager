import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { test } from 'node:test';

const script = readFileSync(new URL('./bump-version.mjs', import.meta.url), 'utf8');
function fixture(version, run) {
    const root = mkdtempSync(join(tmpdir(), 'am-fork-version-'));
    const files = {
        'scripts/bump-version.mjs': script,
        'package.json': JSON.stringify({ name: 'fixture', version }, null, 2),
        'package-lock.json': JSON.stringify({ name: 'fixture', version, packages: { '': { name: 'fixture', version }, 'node_modules/example': { version: '1.0.0' } } }, null, 2),
        'src-tauri/Cargo.toml': `[package]\nname = "antigravity-tools"\nversion = "${version}"\n`,
        'src-tauri/Cargo.lock': `[[package]]\nname = "antigravity-tools"\nversion = "${version}"\n`,
        'src-tauri/tauri.conf.json': JSON.stringify({ version }, null, 2),
        'Casks/antigravity-tools.rb': 'version "5.0.4"\n',
    };
    for (const [name, content] of Object.entries(files)) {
        mkdirSync(join(root, name, '..'), { recursive: true });
        writeFileSync(join(root, name), content);
    }
    const bump = (...args) => spawnSync(process.execPath, ['scripts/bump-version.mjs', ...args, '--skip-cargo-check'], { cwd: root, encoding: 'utf8' });
    try { run({ root, bump }); } finally { rmSync(root, { recursive: true, force: true }); }
}

test('legacy numbering migration is explicit and synchronizes manifests without dependency drift', () => {
    fixture('5.0.16', ({ root, bump }) => {
        const target = '5.0.4-beta.20-fork.1';
        assert.notEqual(bump(target).status, 0);
        assert.equal(JSON.parse(readFileSync(join(root, 'package.json'))).version, '5.0.16');
        const migrated = bump(target, '--adopt-upstream-version');
        assert.equal(migrated.status, 0, migrated.stderr);
        assert.equal(JSON.parse(readFileSync(join(root, 'package.json'))).version, target);
        const lock = JSON.parse(readFileSync(join(root, 'package-lock.json')));
        assert.equal(lock.version, target);
        assert.equal(lock.packages[''].version, target);
        assert.equal(lock.packages['node_modules/example'].version, '1.0.0');
        assert.equal(JSON.parse(readFileSync(join(root, 'src-tauri/tauri.conf.json'))).version, target);
        for (const file of ['src-tauri/Cargo.toml', 'src-tauri/Cargo.lock']) {
            assert.ok(readFileSync(join(root, file), 'utf8').includes(`version = "${target}"`));
        }
        assert.equal(readFileSync(join(root, 'Casks/antigravity-tools.rb'), 'utf8'), 'version "5.0.4"\n');
    });
});

test('fork increment retains upstream baseline and uses main in release guidance', () => {
    fixture('5.0.4-beta.20-fork.9', ({ root, bump }) => {
        const result = bump('fork');
        assert.equal(result.status, 0, result.stderr);
        assert.equal(JSON.parse(readFileSync(join(root, 'package.json'))).version, '5.0.4-beta.20-fork.10');
        assert.ok(result.stdout.includes('git push origin main'));
        assert.ok(!result.stdout.includes('git push origin beta'));
    });
});

test('upstream and fork revision comparisons are numeric and refuse rollback', () => {
    fixture('5.0.4-beta.20-fork.10', ({ bump }) => {
        for (const target of ['5.0.4-beta.9-fork.99', '5.0.4-beta.20-fork.9', '5.0.4-beta.20-fork.10']) {
            assert.notEqual(bump(target, '--dry-run').status, 0, target);
        }
        assert.equal(bump('5.0.4-beta.21-fork.1', '--dry-run').status, 0);
        assert.notEqual(bump('5.0.4-beta.9-fork.1', '--adopt-upstream-version', '--dry-run').status, 0);
        for (const shorthand of ['patch', 'minor', 'major', 'beta']) {
            assert.notEqual(bump(shorthand, '--dry-run').status, 0);
        }
    });
});

test('release gate accepts fork tags on main while isolating upstream beta artifacts', () => {
    const workflow = readFileSync(new URL('../.github/workflows/release.yml', import.meta.url), 'utf8');
    const validation = workflow.split('\n').find(line => line.includes('Invalid release tag'));
    const start = workflow.indexOf('          channel=main');
    const end = workflow.indexOf('          git merge-base', start);
    const shell = `${validation}\n${workflow.slice(start, end)}\nprintf '%s %s' "$channel" "$prerelease"`;
    for (const [tag, expected] of [
        ['v5.0.4-beta.20-fork.1', 'main true'],
        ['v5.0.4-fork.1', 'main false'],
        ['v5.0.16', 'main false'],
        ['v5.0.4-beta.20', 'beta true'],
    ]) {
        const result = spawnSync('bash', ['-c', shell], { env: { ...process.env, tag }, encoding: 'utf8' });
        assert.equal(result.status, 0, result.stderr);
        assert.equal(result.stdout, expected);
    }
    assert.notEqual(spawnSync('bash', ['-c', shell], { env: { ...process.env, tag: 'v5.0.4-fork.bad' } }).status, 0);
});

test('latest alias follows numbering migration without letting old rebuilds roll it back', () => {
    const workflow = readFileSync(new URL('../.github/workflows/release.yml', import.meta.url), 'utf8');
    const start = workflow.indexOf('            if [[ "$RELEASE_TAG" == *-fork.*');
    const end = workflow.indexOf('\n          fi', start);
    assert.ok(start >= 0 && end > start);
    const shell = `make_latest=false\n${workflow.slice(start, end)}\nprintf '%s' "$make_latest"`;
    for (const [latest, RELEASE_TAG, expected] of [
        ['v5.0.16', 'v5.0.4-fork.1', 'true'],
        ['v5.0.4-fork.1', 'v5.0.16', 'false'],
        ['v5.0.4-fork.10', 'v5.0.4-fork.9', 'false'],
        ['v5.0.4-fork.9', 'v5.0.4-fork.10', 'true'],
    ]) {
        const result = spawnSync('bash', ['-c', shell], { env: { ...process.env, latest, RELEASE_TAG }, encoding: 'utf8' });
        assert.equal(result.status, 0, result.stderr);
        assert.equal(result.stdout, expected);
    }
});
