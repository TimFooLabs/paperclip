/**
 * stream-json protocol tests for the hermes-local adapter.
 *
 * Covers TIM-68: stop screen-scraping the hermes terminal UI and parse the
 * `--format stream-json` JSONL protocol instead.
 *
 * Fixtures under ./__fixtures__ are real captures from the installed
 * hermes-agent (v0.21.5+5778.g0a374d1), not hand-written:
 *   - stream-json-ok.jsonl        — a minimal successful one-shot run.
 *   - stream-json-tool-run.jsonl  — a run with terminal/execute_code tool calls
 *                                   and ~90 interleaved `text` deltas.
 */

import { describe, expect, it, vi, beforeEach } from "vitest";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

vi.mock("@paperclipai/adapter-utils/server-utils", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@paperclipai/adapter-utils/server-utils")>();
  return {
    ...actual,
    runChildProcess: vi.fn(async () => ({
      exitCode: 0,
      signal: null,
      timedOut: false,
      stdout: "",
      stderr: "",
    })),
  };
});

vi.mock("node:fs/promises", () => ({
  readFile: vi.fn(async () => ""),
  writeFile: vi.fn(async () => undefined),
  mkdir: vi.fn(async () => undefined),
  rm: vi.fn(async () => undefined),
  access: vi.fn(async () => undefined),
  readdir: vi.fn(async () => []),
  stat: vi.fn(async () => ({ isFile: () => true, isDirectory: () => false })),
}));

import { execute, parseHermesOutput, parseHermesStreamJson } from "./execute.js";
import * as serverUtils from "@paperclipai/adapter-utils/server-utils";

const fixturesDir = resolve(dirname(fileURLToPath(import.meta.url)), "__fixtures__");

function fixture(name: string): string {
  return readFileSync(resolve(fixturesDir, name), "utf-8");
}

function makeCtx(overrides: Record<string, unknown> = {}) {
  return {
    runId: "test-run-1",
    agent: {
      id: "agent-1",
      companyId: "company-1",
      name: "Hermes",
      adapterType: "hermes_local",
      adapterConfig: {},
    },
    runtime: {
      sessionId: null,
      sessionParams: null,
      sessionDisplayId: null,
      taskKey: null,
    },
    config: {
      command: "/usr/bin/hermes",
      timeoutSec: 60,
      graceSec: 5,
      ...overrides,
    },
    context: {
      issueId: "issue-1",
      wakeReason: "manual",
      paperclipWake: null,
    },
    onLog: vi.fn(async () => undefined),
    onMeta: vi.fn(async () => undefined),
    onSpawn: vi.fn(async () => undefined),
  } satisfies Record<string, unknown>;
}

function lastArgs(): string[] {
  const call = vi.mocked(serverUtils.runChildProcess).mock.calls.at(-1)!;
  return call[2] as string[];
}

function runWith(ctx: any, stdout: string, stderr = "", exitCode = 0) {
  vi.mocked(serverUtils.runChildProcess).mockResolvedValueOnce({
    exitCode,
    signal: null,
    timedOut: false,
    stdout,
    stderr,
    pid: null,
    startedAt: null,
  });
  return execute(ctx);
}

describe("hermes-local adapter stream-json protocol", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("requests --format stream-json by default", async () => {
    await execute(makeCtx() as any);
    expect(lastArgs()).toEqual(expect.arrayContaining(["--format", "stream-json"]));
  });

  it("still passes -Q alongside stream-json (stream-json implies quiet; -Q keeps TIM-67 explicit)", async () => {
    await execute(makeCtx() as any);
    expect(lastArgs()).toContain("-Q");
  });

  it("omits --format when outputFormat is text", async () => {
    await execute(makeCtx({ outputFormat: "text" }) as any);
    const args = lastArgs();
    expect(args).not.toContain("--format");
    expect(args).toContain("-Q");
  });

  it("falls back to the default for an unknown outputFormat rather than failing the run", async () => {
    await execute(makeCtx({ outputFormat: "yaml" }) as any);
    expect(lastArgs()).toEqual(expect.arrayContaining(["--format", "stream-json"]));
  });

  it("extracts response, session_id and usage from a real captured stream", async () => {
    const result = await runWith(
      makeCtx(),
      fixture("stream-json-ok.jsonl"),
      "\nsession_id: 20261004_151357_601525\n",
    );

    expect(result.summary).toBe("ok");
    expect(result.resultJson).toMatchObject({
      result: "ok",
      session_id: "20261004_151357_601525",
    });
    expect(result.usage).toEqual({
      inputTokens: 13894,
      outputTokens: 32,
      cachedInputTokens: 0,
    });
    expect(result.sessionParams).toEqual({ sessionId: "20261004_151357_601525" });
    expect(result.errorMessage).toBeUndefined();
  });

  it("takes the response from result.text, not the interleaved text deltas", async () => {
    // The tool-run capture carries ~90 `text` deltas of pre-tool commentary plus
    // a terminal `result`. Concatenating deltas would misattribute that
    // commentary as the agent's answer — the exact failure mode of the
    // screen-scraping parser this issue replaces.
    const stdout = fixture("stream-json-tool-run.jsonl");
    const deltas = stdout
      .split("\n")
      .filter((line) => line.includes('"type": "text"'))
      .map((line) => JSON.parse(line).text as string)
      .join("");

    const parsed = parseHermesStreamJson(stdout);

    expect(parsed.recognized).toBe(true);
    expect(parsed.response).toBeTruthy();
    expect(parsed.response).not.toBe(deltas);
    expect(parsed.response).toBe(
      JSON.parse(stdout.split("\n").filter((l) => l.includes('"type": "result"'))[0]).text,
    );
    // The final answer is also reachable without trailing tool noise.
    expect(parsed.response!.startsWith("I was not able to run the command")).toBe(true);
    expect(parsed.usage?.cachedInputTokens).toBe(27776);
  });

  it("reports session_id from init when the run died before its result envelope", async () => {
    const result = await runWith(
      makeCtx(),
      '{"type": "system", "subtype": "init", "model": "m", "session_id": "sess-died", "timestamp": 1}\n',
    );

    expect(result.sessionParams).toEqual({ sessionId: "sess-died" });
  });

  it("surfaces the result envelope error", async () => {
    const result = await runWith(
      makeCtx(),
      [
        '{"type": "system", "subtype": "init", "model": "m", "session_id": "s1", "timestamp": 1}',
        '{"type": "result", "session_id": "s1", "exit_code": 1, "text": "", "tokens": {"input": 1, "output": 1}, "error": "provider unavailable"}',
      ].join("\n") + "\n",
      "",
      1,
    );

    expect(result.errorMessage).toBe("provider unavailable");
    expect(result.exitCode).toBe(1);
  });

  it("treats a failed tool_result as tool noise, not a failed run", async () => {
    const result = await runWith(
      makeCtx(),
      [
        '{"type": "system", "subtype": "init", "model": "m", "session_id": "s1", "timestamp": 1}',
        '{"type": "tool_use", "name": "terminal", "input": {"command": "ls"}, "timestamp": 2}',
        '{"type": "tool_result", "name": "terminal", "output": "boom", "is_error": true, "duration_ms": 5, "timestamp": 3}',
        '{"type": "result", "session_id": "s1", "exit_code": 0, "text": "recovered", "tokens": {"input": 2, "output": 2}, "timestamp": 4}',
      ].join("\n") + "\n",
    );

    expect(result.errorMessage).toBeUndefined();
    expect(result.summary).toBe("recovered");
  });

  it("logs which parse path was used", async () => {
    const ctx = makeCtx();
    await runWith(ctx, fixture("stream-json-ok.jsonl"));

    const logged = vi
      .mocked(ctx.onLog as ReturnType<typeof vi.fn>)
      .mock.calls.map((c) => String(c[1]))
      .join("");
    expect(logged).toContain("stream-json protocol");
    expect(logged).not.toContain("legacy text fallback");
  });

  it("falls back to the legacy text parser when stdout holds no stream events", async () => {
    const ctx = makeCtx();
    const result = await runWith(
      ctx,
      "The legacy answer\n\nsession_id: legacy-session-9\n",
      "\nsession_id: legacy-session-9\n",
    );

    expect(result.summary).toBe("The legacy answer");
    expect(result.sessionParams).toEqual({ sessionId: "legacy-session-9" });

    const logged = vi
      .mocked(ctx.onLog as ReturnType<typeof vi.fn>)
      .mock.calls.map((c) => String(c[1]))
      .join("");
    expect(logged).toContain("legacy text fallback");
  });

  it("keeps the legacy parser able to read quiet output directly", () => {
    const parsed = parseHermesOutput(
      "Plain response body\n\nsession_id: q-1\n",
      "\nsession_id: q-1\n",
    );

    expect(parsed.response).toBe("Plain response body");
    expect(parsed.sessionId).toBe("q-1");
  });

  it("ignores non-JSON noise lines instead of throwing", () => {
    const parsed = parseHermesStreamJson(
      [
        "not json at all",
        '{"type": "text", "text": "partial"',
        '{"type": "unrecognized_event", "text": "nope"}',
        '{"type": "text", "text": "kept"}',
        '{"type": "result", "session_id": "s", "exit_code": 0, "text": "kept", "timestamp": 1}',
      ].join("\n"),
    );

    expect(parsed.recognized).toBe(true);
    expect(parsed.response).toBe("kept");
  });

  it("reports recognized: false for plain text output", () => {
    expect(parseHermesStreamJson("some\nplain\nstdout").recognized).toBe(false);
    expect(parseHermesStreamJson("").recognized).toBe(false);
  });
});
