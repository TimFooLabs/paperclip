import vm from "node:vm";

import { afterEach, describe, expect, it, vi } from "vitest";

import { invalidateDynamicParser, loadDynamicParser } from "./dynamic-loader";
import { getWorkerBootstrapSource } from "./sandboxed-parser-worker";
import { buildTranscript, type RunLogChunk } from "./transcript";

/**
 * The dynamic loader talks to a real Worker, which does not exist in Node.
 * These tests substitute a fake worker that runs the actual worker bootstrap
 * source inside a vm context, so the async message protocol — init, parse,
 * result — is exercised exactly as the browser would drive it.
 */
let currentWorker: FakeSandboxWorker | null = null;

vi.mock("./sandboxed-parser-worker", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./sandboxed-parser-worker")>();
  return {
    getWorkerBootstrapSource: actual.getWorkerBootstrapSource,
    createSandboxedWorker: () => currentWorker as unknown as Worker,
  };
});

class FakeSandboxWorker {
  onmessage: ((e: { data: unknown }) => void) | null = null;
  onerror: ((e: { message?: string }) => void) | null = null;
  /** Parse requests received from the main thread. */
  parseRequests = 0;
  /** Replies posted back to the main thread. */
  replies = 0;
  terminated = false;

  private readonly self: Record<string, unknown>;

  constructor() {
    this.self = {
      navigator: {},
      // Worker → main: the bootstrap posts results back to worker.onmessage.
      postMessage: (msg: { type: string }) => {
        if (msg.type === "ready" || msg.type === "result") {
          this.replies += 1;
        }
        // A real worker reply arrives on a later tick.
        queueMicrotask(() => this.onmessage?.({ data: msg }));
      },
    };
    vm.runInNewContext(getWorkerBootstrapSource(), { self: this.self });
  }

  /** Main → worker: the loader posts requests to the bootstrap's handler. */
  postMessage(msg: unknown) {
    if ((msg as { type?: string }).type === "parse") this.parseRequests += 1;
    queueMicrotask(() => this.self.onmessage?.({ data: msg }));
  }

  terminate() {
    this.terminated = true;
  }
}

/**
 * A parser with cross-line state, shaped like the Hermes Reasoning-box parser:
 * a border pair brackets lines that must stay thinking, everything else is
 * assistant text. The borders are identical so every box looks the same to a
 * result cache.
 */
const STATEFUL_PARSER_SOURCE = `
function createParser() {
  let open = false;
  return {
    parseLine(line, ts) {
      if (line === "OPEN") { open = true; return []; }
      if (line === "CLOSE") { open = false; return []; }
      if (open) return [{ kind: "thinking", ts, text: line, delta: true }];
      return [{ kind: "assistant", ts, text: line }];
    },
    reset() { open = false; },
  };
}
const shared = createParser();
module.exports = { parseStdoutLine: shared.parseLine, createStdoutParser: createParser };
`;

const LINES = ["OPEN", " wrapped thought", "CLOSE", " visible answer"];

function lineChunks(lines: string[], ts: string): RunLogChunk[] {
  return lines.map((line, index) => ({ ts, stream: "stdout" as const, chunk: `${line}\n`, seq: index }));
}

async function loadStatefulParser(adapterType: string) {
  currentWorker = new FakeSandboxWorker();
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(STATEFUL_PARSER_SOURCE, { status: 200 })),
  );
  const parserModule = await loadDynamicParser(adapterType);
  expect(parserModule?.createStdoutParser).toBeTypeOf("function");
  return parserModule!;
}

/** Wait until every parse request sent so far has been answered. */
async function settle() {
  await vi.waitFor(() => {
    expect(currentWorker?.replies ?? 0).toBeGreaterThanOrEqual(currentWorker?.parseRequests ?? 0);
  });
}

describe("dynamic loader — stateful parsers across transcript rebuilds", () => {
  afterEach(() => {
    if (currentWorker) {
      for (const type of ["fake-adapter-settle", "fake-adapter-borders", "fake-adapter-transcript"]) {
        invalidateDynamicParser(type);
      }
      currentWorker = null;
    }
    vi.unstubAllGlobals();
  });

  it("returns parsed entries synchronously on the rebuild that follows the worker round-trip", async () => {
    const parserModule = await loadStatefulParser("fake-adapter-settle");

    // Build 1: nothing is cached yet, so lines come back unparsed while the
    // requests travel to the worker.
    const first = parserModule.createStdoutParser();
    expect(first.parseLine("OPEN", "t1")).toEqual([]);
    expect(first.parseLine(" wrapped thought", "t1")).toEqual([]);
    first.reset?.();

    await settle();

    // Build 2 replays the same lines. It must see the results that arrived,
    // otherwise the notification that triggered this rebuild triggers another
    // one forever and the transcript never renders parsed output.
    const second = parserModule.createStdoutParser();
    expect(second.parseLine("OPEN", "t1")).toEqual([]);
    expect(second.parseLine(" wrapped thought", "t1")).toEqual([
      { kind: "thinking", ts: "t1", text: " wrapped thought", delta: true },
    ]);
    second.reset?.();
  });

  it("feeds repeated identical border lines to the stateful parser", async () => {
    const parserModule = await loadStatefulParser("fake-adapter-borders");

    const build = parserModule.createStdoutParser();
    // Two Reasoning boxes in one stream: identical border text, so a cache
    // keyed on line content alone would skip the second OPEN and leave the
    // second box's body classified as assistant output.
    const seen = LINES.map((line) => build.parseLine(line, "t1"));
    expect(seen.every((entries) => entries.length === 0)).toBe(true);
    build.reset?.();

    await settle();

    const rebuilt = parserModule.createStdoutParser();
    expect(rebuilt.parseLine(LINES[0], "t1")).toEqual([]);
    expect(rebuilt.parseLine(LINES[1], "t1")).toEqual([
      { kind: "thinking", ts: "t1", text: LINES[1], delta: true },
    ]);
    expect(rebuilt.parseLine(LINES[2], "t1")).toEqual([]);
    expect(rebuilt.parseLine(LINES[3], "t1")).toEqual([
      { kind: "assistant", ts: "t1", text: LINES[3] },
    ]);
    rebuilt.reset?.();
  });

  it("buildTranscript over the dynamic module settles on a parsed transcript", async () => {
    const parserModule = await loadStatefulParser("fake-adapter-transcript");
    const chunks = lineChunks(LINES, "2026-06-29T12:00:00.000Z");

    const first = buildTranscript(chunks, { createStdoutParser: parserModule.createStdoutParser });
    // Nothing is cached on the first pass: every miss returns [], so the
    // transcript renders empty until the worker's results come back.
    expect(first).toEqual([]);

    await settle();

    // The rebuild the UI runs once results arrive must show the parsed kinds.
    const second = buildTranscript(chunks, { createStdoutParser: parserModule.createStdoutParser });
    expect(second.map((entry) => entry.kind)).toEqual(["thinking", "assistant"]);
  });
});
