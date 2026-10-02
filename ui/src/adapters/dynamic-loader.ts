/**
 * Dynamic UI parser loading for external adapters — sandboxed execution.
 *
 * When the Paperclip UI encounters an adapter type that doesn't have a
 * built-in parser (e.g., an external adapter loaded via the plugin system),
 * it fetches the parser JS from `/api/adapters/:type/ui-parser.js` and
 * executes it **inside a dedicated Web Worker** so it cannot access the
 * board UI's same-origin state (cookies, localStorage, DOM, authenticated
 * fetch, etc.).
 *
 * The worker communicates via a narrow postMessage protocol:
 *   Main → Worker:  { type: "init", source }
 *   Worker → Main:  { type: "ready" } | { type: "error", message }
 *   Main → Worker:  { type: "parse", id, line, ts }
 *   Worker → Main:  { type: "result", id, entries }
 *
 * Because the parse call is async (cross-thread postMessage), but the
 * existing `parseStdoutLine` contract is synchronous, we cache completed
 * worker results and ask the adapter registry to recompute transcripts when
 * a new result arrives.
 *
 * **Synchronous fast-path**: After init, parse requests are sent to the
 * worker which responds asynchronously.  The `parseStdoutLine` wrapper
 * returns cached results synchronously on the next transcript recomputation.
 * In practice this adds ~1 frame of latency which is imperceptible.
 *
 * Security: see `sandboxed-parser-worker.ts` for the full lockdown.
 */

import type { TranscriptEntry } from "@paperclipai/adapter-utils";
import type { StdoutLineParser, StdoutParserFactory } from "./types";
import { createSandboxedWorker } from "./sandboxed-parser-worker";
import type { SandboxRequest, SandboxResponse } from "./sandboxed-parser-worker";

// ── Types ───────────────────────────────────────────────────────────────────

interface DynamicParserModule {
  parseStdoutLine: StdoutLineParser;
  createStdoutParser?: StdoutParserFactory;
}

interface SandboxedParser {
  worker: Worker;
  ready: boolean;
  nextId: number;
  pendingResolves: Map<number, (entries: TranscriptEntry[]) => void>;
}

// ── State ───────────────────────────────────────────────────────────────────

/** Cache of fully initialised sandboxed parsers by adapter type. */
const sandboxedParsers = new Map<string, SandboxedParser>();

/** Cache of the public DynamicParserModule wrappers. */
const dynamicParserCache = new Map<string, DynamicParserModule>();

/** Track which types we've already attempted to load (to avoid repeat 404s). */
const failedLoads = new Set<string>();

/** In-flight init promises so concurrent callers share the same load. */
const loadPromises = new Map<string, Promise<DynamicParserModule | null>>();

let resultNotifier: (() => void) | null = null;

export function setDynamicParserResultNotifier(fn: (() => void) | null): void {
  resultNotifier = fn;
}

// ── Internal helpers ────────────────────────────────────────────────────────

function sendToWorker(sandbox: SandboxedParser, msg: SandboxRequest): void {
  sandbox.worker.postMessage(msg);
}

function nextRequestId(sandbox: SandboxedParser): number {
  return sandbox.nextId++;
}

function lineCacheKey(line: string, ts: string): string {
  return `${ts}\u0000${line}`;
}

function notifyResultReady(): void {
  resultNotifier?.();
}

/**
 * Parse a single line synchronously by delegating to the worker.
 * Returns a Promise that resolves with the TranscriptEntry[] from the worker.
 *
 * `buildId` tells the worker which transcript build the line belongs to, so a
 * stateful parser (Reasoning-box tracking) starts fresh on every build.
 */
function parseLineAsync(
  sandbox: SandboxedParser,
  line: string,
  ts: string,
  buildId?: number,
): Promise<TranscriptEntry[]> {
  return new Promise((resolve) => {
    const id = nextRequestId(sandbox);
    sandbox.pendingResolves.set(id, resolve);
    sendToWorker(sandbox, { type: "parse", id, line, ts, buildId });
  });
}

function drainPendingRequests(sandbox: SandboxedParser): void {
  for (const resolver of sandbox.pendingResolves.values()) {
    resolver([]);
  }
  sandbox.pendingResolves.clear();
}

/**
 * Create a sandboxed worker, send the parser source, and wait for init.
 */
function initSandboxedWorker(source: string): Promise<SandboxedParser> {
  return new Promise((resolve, reject) => {
    const worker = createSandboxedWorker();
    const sandbox: SandboxedParser = {
      worker,
      ready: false,
      nextId: 1,
      pendingResolves: new Map(),
    };

    // Timeout if the worker doesn't respond within 5s
    const timeout = setTimeout(() => {
      drainPendingRequests(sandbox);
      worker.terminate();
      reject(new Error("Parser worker init timed out"));
    }, 5000);

    worker.onmessage = (e: MessageEvent<SandboxResponse>) => {
      const msg = e.data;

      if (msg.type === "ready") {
        clearTimeout(timeout);
        sandbox.ready = true;

        // Switch to the steady-state message handler.
        worker.onmessage = (ev: MessageEvent<SandboxResponse>) => {
          const resp = ev.data;
          if (resp.type === "result") {
            const resolver = sandbox.pendingResolves.get(resp.id);
            if (resolver) {
              sandbox.pendingResolves.delete(resp.id);
              resolver(resp.entries as TranscriptEntry[]);
            }
          } else if (resp.type === "error") {
            console.error("[adapter-ui-loader] Worker reported error:", resp.message);
            drainPendingRequests(sandbox);
          }
        };

        resolve(sandbox);
        return;
      }

      if (msg.type === "error") {
        clearTimeout(timeout);
        drainPendingRequests(sandbox);
        worker.terminate();
        reject(new Error(msg.message));
        return;
      }
    };

    worker.onerror = (ev) => {
      clearTimeout(timeout);
      drainPendingRequests(sandbox);
      worker.terminate();
      reject(new Error(`Worker error: ${ev.message}`));
    };

    // Send the parser source to the worker for evaluation.
    sendToWorker(sandbox, { type: "init", source });
  });
}

/**
 * Build a DynamicParserModule that delegates all calls to the sandboxed worker.
 *
 * The parseStdoutLine wrapper is **synchronous** to match the existing contract.
 * Cache misses send a parse request to the worker and return `[]`; when the
 * worker responds, the registry notification path recomputes transcripts and
 * this wrapper returns the cached result synchronously.
 *
 * In practice, because the existing codebase already handles the "bridge"
 * pattern where parseStdoutLine returns [] until the dynamic parser loads,
 * the same UX applies here: the first render may show raw lines, and a
 * subsequent render shows the parsed entries.
 */
function buildParserModule(sandbox: SandboxedParser): DynamicParserModule {
  const parseCache = new Map<string, TranscriptEntry[]>();
  const pendingParseKeys = new Set<string>();

  /**
   * One past the identity of the last transcript build that went through this
   * module. Each {@link createStdoutParser} call is a new build, so it gets a
   * new id: the worker drops the previous build's stateful parser instance
   * when it sees the new id, so a stateful parser never carries Reasoning-box
   * (or similar) state across builds. Cache entries are keyed per line
   * occurrence without the build id — see {@link createStdoutParser}.
   */
  let lastBuildId = 0;

  /** Upper bound on cached results; {@link createStdoutParser} reset trims. */
  const MAX_PARSE_CACHE_ENTRIES = 8192;

  const requestParse = (key: string, buildId: number | undefined, line: string, ts: string, notify: boolean) => {
    if (pendingParseKeys.has(key)) return;
    pendingParseKeys.add(key);
    parseLineAsync(sandbox, line, ts, buildId).then((entries) => {
      pendingParseKeys.delete(key);
      parseCache.set(key, entries);
      if (notify) notifyResultReady();
    });
  };

  /**
   * Legacy stateless entry point. Results are cached per (ts, line), which is
   * only sound while the parser has no cross-line state.
   */
  const parseStdoutLine: StdoutLineParser = (line: string, ts: string) => {
    const key = lineCacheKey(line, ts);
    const cached = parseCache.get(key);
    if (cached) return cached.slice();

    requestParse(key, undefined, line, ts, true);
    return [];
  };

  /**
   * Stateful entry point: one worker parser instance per transcript build.
   *
   * The cache is keyed per line occurrence (position + ts + line) rather than
   * per (ts, line): identical lines are common in stream output (Reasoning box
   * borders, wrapped blanks), and a (ts, line) hit would silently skip feeding
   * the parser, desyncing its state. The build id is deliberately NOT part of
   * the key: a rebuild replays the same lines in the same order, so keys that
   * changed per build would miss forever, and every miss notifies, so each
   * notification would schedule another rebuild and the transcript would never
   * settle into parsed output.
   *
   * Cached lines are still forwarded to the worker — a stateful parser must
   * observe every line in order, so the cache only avoids recomputing the
   * *result*, never the feed. Only misses trigger a transcript recompute.
   */
  const createStdoutParser: StdoutParserFactory = () => {
    const buildId = ++lastBuildId;
    let ordinal = 0;

    return {
      parseLine: (line: string, ts: string) => {
        const key = `${ordinal++}\u0000${lineCacheKey(line, ts)}`;
        const cached = parseCache.get(key);
        // Feed the worker even on a hit; return the known result immediately.
        requestParse(key, buildId, line, ts, !cached);
        return cached ? cached.slice() : [];
      },
      reset: () => {
        // buildTranscript resets when a build ends. Entries must survive so
        // the next build — a fresh worker parser over the same lines — hits
        // them; only trim once the cache grows past its bound.
        if (parseCache.size > MAX_PARSE_CACHE_ENTRIES) parseCache.clear();
      },
    };
  };

  return { parseStdoutLine, createStdoutParser };
}

// ── Public API ──────────────────────────────────────────────────────────────

/**
 * Dynamically load a UI parser for an adapter type from the server API,
 * executing it inside a sandboxed Web Worker.
 *
 * @returns A DynamicParserModule, or null if unavailable.
 */
export async function loadDynamicParser(adapterType: string): Promise<DynamicParserModule | null> {
  // Return cached parser if already loaded.
  const cached = dynamicParserCache.get(adapterType);
  if (cached) return cached;

  // Don't retry types that previously failed.
  if (failedLoads.has(adapterType)) return null;

  // Coalesce concurrent loads.
  const inflight = loadPromises.get(adapterType);
  if (inflight) return inflight;

  const loadPromise = (async (): Promise<DynamicParserModule | null> => {
    try {
      const response = await fetch(`/api/adapters/${encodeURIComponent(adapterType)}/ui-parser.js`);
      if (!response.ok) {
        failedLoads.add(adapterType);
        return null;
      }

      const source = await response.text();

      // Initialise the sandboxed worker with the parser source.
      const sandbox = await initSandboxedWorker(source);
      sandboxedParsers.set(adapterType, sandbox);

      const parserModule = buildParserModule(sandbox);
      dynamicParserCache.set(adapterType, parserModule);

      console.info(`[adapter-ui-loader] Loaded sandboxed UI parser for "${adapterType}"`);
      return parserModule;
    } catch (err) {
      console.warn(`[adapter-ui-loader] Failed to load UI parser for "${adapterType}":`, err);
      failedLoads.add(adapterType);
      return null;
    } finally {
      loadPromises.delete(adapterType);
    }
  })();

  loadPromises.set(adapterType, loadPromise);
  return loadPromise;
}

/**
 * Invalidate a cached dynamic parser, removing it from both the parser cache
 * and the failed-loads set so that the next load attempt will try again.
 * Also terminates the sandboxed worker if one exists.
 */
export function invalidateDynamicParser(adapterType: string): boolean {
  const wasCached = dynamicParserCache.has(adapterType);
  dynamicParserCache.delete(adapterType);
  failedLoads.delete(adapterType);
  loadPromises.delete(adapterType);

  // Terminate the worker to free resources.
  const sandbox = sandboxedParsers.get(adapterType);
  if (sandbox) {
    drainPendingRequests(sandbox);
    sandbox.worker.terminate();
    sandboxedParsers.delete(adapterType);
  }

  if (wasCached) {
    console.info(`[adapter-ui-loader] Invalidated sandboxed UI parser for "${adapterType}"`);
  }
  return wasCached;
}
