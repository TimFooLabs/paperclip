/**
 * Server-side execution logic for the Hermes Agent adapter.
 *
 * Spawns `hermes chat -q "..." -Q --format stream-json` as a child process,
 * streams output, and returns structured results to Paperclip.
 *
 * Verified CLI flags (hermes chat):
 *   -q/--query         single query (non-interactive)
 *   -Q/--quiet         quiet mode (no banner/spinner, only response + session_id)
 *   --format           output protocol: "text" (human-oriented) or "stream-json"
 *                      (newline-delimited JSONL events; implies --quiet and
 *                      cannot be combined with --tui)
 *   -m/--model         model name (e.g. anthropic/claude-sonnet-4)
 *   -t/--toolsets      comma-separated toolsets to enable
 *   --provider         inference provider (auto, openrouter, nous, etc.)
 *   -r/--resume        resume session by ID
 *   -w/--worktree      isolated git worktree
 *   -v/--verbose       verbose output
 *   --checkpoints      filesystem checkpoints
 *   --yolo             bypass dangerous-command approval prompts (agents have no TTY)
 *   --source           session source tag for filtering
 *
 * stream-json is the default protocol. Verified against the StreamJsonEmitter in
 * hermes_cli/stream_json.py of the installed hermes-agent (v0.21.5+5778.g0a374d1):
 * it emits `system/init`, interleaved `text` deltas / `tool_use` / `tool_result`,
 * then one terminal `result` envelope, and finally re-prints `session_id:` on
 * stderr (same stderr contract as -Q). Because that protocol is machine-readable,
 * the adapter no longer has to guess at Rich-rendered console output — which is
 * what made prompt text like `[guide](/issues/TIM-66)` fatal.
 */

import fs from "node:fs/promises";
import path from "node:path";

import type {
  AdapterExecutionContext,
  AdapterExecutionResult,
  UsageSummary,
} from "@paperclipai/adapter-utils";

import {
  resolveHermesBilledCostUsd,
  resolveHermesBillingIdentity,
} from "./billing.js";

import {
  runChildProcess,
  buildPaperclipEnv,
  buildRuntimeToolsEnv,
  renderTemplate,
  ensureAbsoluteDirectory,
  DEFAULT_PAPERCLIP_AGENT_PROMPT_TEMPLATE,
  DEFAULT_PAPERCLIP_CONVERSATION_PROMPT_TEMPLATE,
  joinPromptSections,
  selectPaperclipPromptSections,
  stringifyPaperclipWakePayload,
  isPaperclipRecoveryWakePayload,
} from "@paperclipai/adapter-utils/server-utils";

import {
  HERMES_CLI,
  DEFAULT_TIMEOUT_SEC,
  DEFAULT_GRACE_SEC,
  DEFAULT_MODEL,
  DEFAULT_OUTPUT_FORMAT,
  HERMES_OUTPUT_FORMATS,
  type HermesOutputFormat,
} from "../shared/constants.js";

import {
  detectModel,
  resolveProvider,
} from "./detect-model.js";
import { reconcileHermesPaperclipSkills } from "./skills.js";

// ---------------------------------------------------------------------------
// Config helpers
// ---------------------------------------------------------------------------

function cfgString(v: unknown): string | undefined {
  return typeof v === "string" && v.length > 0 ? v : undefined;
}
function cfgNumber(v: unknown): number | undefined {
  return typeof v === "number" ? v : undefined;
}
function cfgBoolean(v: unknown): boolean | undefined {
  return typeof v === "boolean" ? v : undefined;
}
function cfgStringArray(v: unknown): string[] | undefined {
  return Array.isArray(v) && v.every((i) => typeof i === "string")
    ? (v as string[])
    : undefined;
}

export function resolveHermesCommand(config: Record<string, unknown>): string {
  return cfgString(config.hermesCommand) || cfgString(config.command) || HERMES_CLI;
}

// ---------------------------------------------------------------------------
// Wake-up prompt builder
// ---------------------------------------------------------------------------

const HERMES_DEFAULT_PROMPT_TEMPLATE = [
  'You are "{{agent.name}}", an AI agent employee in a Paperclip-managed company.',
  "",
  "Paperclip runtime identity:",
  "- Agent ID: {{agent.id}}",
  "- Company ID: {{agent.companyId}}",
  "- Run ID: {{run.id}}",
  "- API base: {{paperclipApiUrl}}",
  "",
  "Paperclip API guidance:",
  "- Use `curl` from the terminal for Paperclip API calls; browser/web extraction tools may not reach localhost.",
  "- Use `$PAPERCLIP_API_URL`, `$PAPERCLIP_API_KEY`, and `$PAPERCLIP_RUN_ID`; do not hard-code local ports or copy secrets into comments.",
  "- Displayed command logs may redact secrets; rely on environment variables instead of printed token values.",
  "- Include `-H \"Authorization: Bearer $PAPERCLIP_API_KEY\"` on API requests.",
  "- Include `-H \"X-Paperclip-Run-Id: $PAPERCLIP_RUN_ID\"` on mutating issue requests.",
  "- For multiline comments or status updates, preserve newlines with `jq --arg` or a heredoc-fed helper rather than hand-escaping JSON.",
  "",
  "Safe multiline update pattern:",
  "```bash",
  "api=\"${PAPERCLIP_API_URL%/}\"",
  "case \"$api\" in */api) ;; *) api=\"$api/api\" ;; esac",
  "",
  "body=$(cat <<'MD'",
  "Summary line",
  "",
  "- Detail one",
  "- Detail two",
  "MD",
  ")",
  "jq -n --arg status done --arg comment \"$body\" '{status:$status, comment:$comment}' | \\",
  "  curl -sS -X PATCH \"$api/issues/{{context.issueId}}\" \\",
  "    -H \"Authorization: Bearer $PAPERCLIP_API_KEY\" \\",
  "    -H \"X-Paperclip-Run-Id: $PAPERCLIP_RUN_ID\" \\",
  "    -H \"Content-Type: application/json\" \\",
  "    --data-binary @-",
  "```",
  "",
  DEFAULT_PAPERCLIP_AGENT_PROMPT_TEMPLATE,
].join("\n");

function renderConditionalSections(template: string, vars: Record<string, unknown>): string {
  const isTruthy = (key: string) => {
    if (key === "noTask") return !vars.taskId;
    const value = vars[key];
    if (Array.isArray(value)) return value.length > 0;
    return Boolean(value);
  };
  return template.replace(
    /\{\{#([a-zA-Z0-9_.-]+)\}\}([\s\S]*?)\{\{\/\1\}\}/g,
    (_match, key: string, body: string) => (isTruthy(key) ? body : ""),
  );
}

export function buildPrompt(
  ctx: AdapterExecutionContext,
  config: Record<string, unknown>,
  options: { resumedSession?: boolean } = {},
): string {
  const context = (ctx as any).context || {};
  const template = cfgString(config.promptTemplate) || (context.conversationMode === true
    ? DEFAULT_PAPERCLIP_CONVERSATION_PROMPT_TEMPLATE
    : HERMES_DEFAULT_PROMPT_TEMPLATE);
  const taskId = cfgString(context.taskId) || cfgString(context.issueId) || cfgString(ctx.config?.taskId);
  const taskTitle = cfgString(context.taskTitle) || cfgString(ctx.config?.taskTitle) || "";
  const taskBody = cfgString(context.taskBody) || cfgString(ctx.config?.taskBody) || "";
  const commentId = cfgString(context.commentId) || cfgString(context.wakeCommentId) || cfgString(ctx.config?.commentId) || "";
  const wakeReason = cfgString(context.wakeReason) || cfgString(ctx.config?.wakeReason) || "";
  const agentName = ctx.agent?.name || "Hermes Agent";
  const companyName = cfgString(context.companyName) || cfgString(ctx.config?.companyName) || "";
  const projectName = cfgString(context.projectName) || cfgString(ctx.config?.projectName) || "";

  // Build API URL — ensure it has the /api path
  let paperclipApiUrl =
    cfgString(config.paperclipApiUrl) ||
    process.env.PAPERCLIP_API_URL ||
    "http://127.0.0.1:3100/api";
  // Ensure /api suffix
  if (!paperclipApiUrl.endsWith("/api")) {
    paperclipApiUrl = paperclipApiUrl.replace(/\/+$/, "") + "/api";
  }

  const { taskContextNote: taskContextMarkdown, wakePrompt } = selectPaperclipPromptSections(context, {
    resumedSession: options.resumedSession === true,
    includeCommunicationGuidance: true,
  });
  // Keep the historical variable available to custom templates. Automatic
  // assembly uses the ownership-aware assignment variant below.
  const paperclipTaskMarkdown = cfgString(context.paperclipTaskMarkdown)?.trim() || "";
  const sessionHandoffMarkdown = cfgString(context.paperclipSessionHandoffMarkdown)?.trim() || "";
  const wakePayloadJson = stringifyPaperclipWakePayload(context.paperclipWake) || "";

  const vars: Record<string, unknown> = {
    agentId: ctx.agent?.id || "",
    agentName,
    companyId: ctx.agent?.companyId || "",
    companyName,
    runId: ctx.runId || "",
    agent: ctx.agent || {},
    company: { id: ctx.agent?.companyId || "", name: companyName },
    run: { id: ctx.runId || "", source: "on_demand" },
    context,
    taskId: taskId || "",
    taskTitle,
    taskBody,
    commentId,
    wakeReason,
    projectName,
    paperclipApiUrl,
    paperclipWakePrompt: wakePrompt,
    paperclipTaskMarkdown,
    taskContext: paperclipTaskMarkdown,
    taskContextMarkdown,
    paperclipWakeJson: wakePayloadJson,
    wakePayloadJson,
    paperclipApiKeyEnv: "PAPERCLIP_API_KEY",
    paperclipRunIdEnv: "PAPERCLIP_RUN_ID",
  };

  const rendered = isPaperclipRecoveryWakePayload(context.paperclipWake)
    ? ""
    : renderTemplate(renderConditionalSections(template, vars), vars);
  return joinPromptSections([
    wakePrompt,
    sessionHandoffMarkdown,
    taskContextMarkdown,
    rendered,
  ]);
}

// ---------------------------------------------------------------------------
// Output parsing
// ---------------------------------------------------------------------------

/** Regex to extract session ID from Hermes quiet-mode output: "session_id: <id>" */
const SESSION_ID_REGEX = /^session_id:\s*(\S+)/m;

/** Regex for legacy session output format */
const SESSION_ID_REGEX_LEGACY = /session[_ ](?:id|saved)[:\s]+([a-zA-Z0-9_-]+)/i;

/** Regex to extract token usage from Hermes output. */
const TOKEN_USAGE_REGEX =
  /tokens?[:\s]+(\d+)\s*(?:input|in)\b.*?(\d+)\s*(?:output|out)\b/i;

/** Regex to extract cost from Hermes output. */
const COST_REGEX = /(?:cost|spent)[:\s]*\$?([\d.]+)/i;

interface ParsedOutput {
  sessionId?: string;
  response?: string;
  usage?: UsageSummary;
  costUsd?: number;
  errorMessage?: string;
  /** Exit code reported by the stream-json `result` envelope, when nonzero. */
  exitCode?: number;
}

// ---------------------------------------------------------------------------
// Response cleaning
// ---------------------------------------------------------------------------

/** Strip noise lines from a Hermes response (tool output, system messages, etc.) */
function cleanResponse(raw: string): string {
  return raw
    .split("\n")
    .filter((line) => {
      const t = line.trim();
      if (!t) return true; // keep blank lines for paragraph separation
      if (t.startsWith("[tool]") || t.startsWith("[hermes]") || t.startsWith("[paperclip]")) return false;
      if (t.startsWith("session_id:")) return false;
      if (/^\[\d{4}-\d{2}-\d{2}T/.test(t)) return false;
      if (/^\[done\]\s*┊/.test(t)) return false;
      if (/^┊\s*[\p{Emoji_Presentation}]/u.test(t) && !/^┊\s*💬/.test(t)) return false;
      if (/^\p{Emoji_Presentation}\s*(Completed|Running|Error)?\s*$/u.test(t)) return false;
      return true;
    })
    .map((line) => {
      let t = line.replace(/^[\s]*┊\s*💬\s*/, "").trim();
      t = t.replace(/^\[done\]\s*/, "").trim();
      return t;
    })
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

// ---------------------------------------------------------------------------
// Output parsing
// ---------------------------------------------------------------------------

export function parseHermesOutput(stdout: string, stderr: string): ParsedOutput {
  const combined = stdout + "\n" + stderr;
  const result: ParsedOutput = {};

  // In quiet mode, Hermes outputs:
  //   <response text>
  //
  //   session_id: <id>
  const sessionMatch = stdout.match(SESSION_ID_REGEX);
  if (sessionMatch?.[1]) {
    result.sessionId = sessionMatch?.[1] ?? null;
    // The response is everything before the session_id line
    const sessionLineIdx = stdout.lastIndexOf("\nsession_id:");
    if (sessionLineIdx > 0) {
      result.response = cleanResponse(stdout.slice(0, sessionLineIdx));
    }
  } else {
    // Legacy format (non-quiet mode)
    const legacyMatch = combined.match(SESSION_ID_REGEX_LEGACY);
    if (legacyMatch?.[1]) {
      result.sessionId = legacyMatch?.[1] ?? null;
    }
    // In non-quiet mode, extract clean response from stdout by
    // filtering out tool lines, system messages, and noise
    const cleaned = cleanResponse(stdout);
    if (cleaned.length > 0) {
      result.response = cleaned;
    }
  }

  // Extract token usage
  const usageMatch = combined.match(TOKEN_USAGE_REGEX);
  if (usageMatch) {
    result.usage = {
      inputTokens: parseInt(usageMatch[1], 10) || 0,
      outputTokens: parseInt(usageMatch[2], 10) || 0,
    };
  }

  // Extract cost
  const costMatch = combined.match(COST_REGEX);
  if (costMatch?.[1]) {
    result.costUsd = parseFloat(costMatch[1]);
  }

  // Check for error patterns in stderr
  if (stderr.trim()) {
    const errorLines = stderr
      .split("\n")
      .filter((line) => /error|exception|traceback|failed/i.test(line))
      .filter((line) => !/INFO|DEBUG|warn/i.test(line)); // skip log-level noise
    if (errorLines.length > 0) {
      result.errorMessage = errorLines.slice(0, 5).join("\n");
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// stream-json protocol parsing (--format stream-json)
// ---------------------------------------------------------------------------

/**
 * One newline-delimited JSON record from `hermes chat --format stream-json`.
 *
 * Field surface mirrors hermes_cli/stream_json.py (StreamJsonEmitter) on the
 * installed hermes-agent; only the fields the adapter consumes are typed, and
 * every one of them is optional because older emitters may omit it.
 */
export interface HermesStreamEvent {
  type?: unknown;
  subtype?: unknown;
  session_id?: unknown;
  model?: unknown;
  /** `text` events: a streaming delta of the current turn. */
  text?: unknown;
  /** `tool_use` / `tool_result` events. */
  name?: unknown;
  is_error?: unknown;
  /** Terminal `result` envelope. */
  exit_code?: unknown;
  tokens?: {
    input?: unknown;
    output?: unknown;
    total?: unknown;
    cache_read?: unknown;
    cache_write?: unknown;
  };
  error?: unknown;
}

const STREAM_EVENT_TYPES = new Set(["system", "text", "tool_use", "tool_result", "result"]);

function asString(v: unknown): string | undefined {
  return typeof v === "string" ? v : undefined;
}

function asNumber(v: unknown): number | undefined {
  return typeof v === "number" && Number.isFinite(v) ? v : undefined;
}

/**
 * Parse a `--format stream-json` run.
 *
 * Returns `recognized: false` when stdout holds no stream-json events at all —
 * the signal that an older Hermes rejected the flag and fell back to text, or
 * that the process died before emitting anything. Callers then fall back to
 * `parseHermesOutput`.
 *
 * The response comes from the terminal `result.text`, NOT from concatenating
 * `text` deltas. Deltas are per-turn segments and include the pre-tool
 * commentary of every intermediate turn; `result.text` is the final answer
 * only. Verified against a real tool-using capture: the two disagree. Deltas
 * are used only when the run produced no `result` envelope (truncated stream),
 * where the last turn's text is still better than nothing.
 */
export function parseHermesStreamJson(
  stdout: string,
): ParsedOutput & { recognized: boolean } {
  const events: HermesStreamEvent[] = [];

  for (const line of stdout.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || !trimmed.startsWith("{")) continue;
    let parsed: unknown;
    try {
      parsed = JSON.parse(trimmed);
    } catch {
      continue; // not JSON — the legacy parser's problem, not ours
    }
    if (typeof parsed !== "object" || parsed === null) continue;
    const candidate = parsed as HermesStreamEvent;
    if (typeof candidate.type !== "string") continue;
    if (!STREAM_EVENT_TYPES.has(candidate.type)) continue;
    events.push(candidate);
  }

  const out: ParsedOutput & { recognized: boolean } = { recognized: events.length > 0 };
  if (!out.recognized) return out;

  const resultEvent = [...events].reverse().find((e) => e.type === "result");
  const initEvent = events.find((e) => e.type === "system" && e.subtype === "init");

  // `result.session_id` wins; `system/init.session_id` covers a run that died
  // before its result envelope was written.
  const sessionId =
    asString(resultEvent?.session_id) ?? asString(initEvent?.session_id);
  if (sessionId) out.sessionId = sessionId;

  const resultText = asString(resultEvent?.text);
  if (resultText !== undefined) {
    out.response = resultText;
  } else {
    const deltas = events
      .filter((e) => e.type === "text")
      .map((e) => asString(e.text) ?? "")
      .join("");
    if (deltas.length > 0) out.response = deltas;
  }

  const tokens = resultEvent?.tokens;
  if (tokens && typeof tokens === "object") {
    const inputTokens = asNumber(tokens.input);
    const outputTokens = asNumber(tokens.output);
    if (inputTokens !== undefined || outputTokens !== undefined) {
      out.usage = {
        inputTokens: inputTokens ?? 0,
        outputTokens: outputTokens ?? 0,
        ...(asNumber(tokens.cache_read) !== undefined
          ? { cachedInputTokens: asNumber(tokens.cache_read) }
          : {}),
      };
    }
  }

  const exitCode = asNumber(resultEvent?.exit_code);
  if (exitCode !== undefined && exitCode !== 0) {
    out.exitCode = exitCode;
  }

  const eventError = asString(resultEvent?.error);
  if (eventError) {
    out.errorMessage = eventError;
  }

  // stream-json carries no cost figure; leave costUsd unset rather than guess.
  return out;
}

/**
 * Resolve the requested output protocol. Unknown or unset values fall back to
 * the default rather than throwing — a bad adapterConfig override should not
 * stop an agent from running.
 */
export function resolveOutputFormat(config: Record<string, unknown>): HermesOutputFormat {
  const requested = cfgString(config.outputFormat);
  if (requested && (HERMES_OUTPUT_FORMATS as readonly string[]).includes(requested)) {
    return requested as HermesOutputFormat;
  }
  return DEFAULT_OUTPUT_FORMAT;
}

// ---------------------------------------------------------------------------
// Main execute
// ---------------------------------------------------------------------------

export async function execute(
  ctx: AdapterExecutionContext,
): Promise<AdapterExecutionResult> {
  const config = (ctx.config ?? ctx.agent?.adapterConfig ?? {}) as Record<string, unknown>;

  // ── Resolve configuration ──────────────────────────────────────────────
  const hermesCmd = resolveHermesCommand(config);
  const model = cfgString(config.model) || DEFAULT_MODEL;
  const timeoutSec = cfgNumber(config.timeoutSec) || DEFAULT_TIMEOUT_SEC;
  const graceSec = cfgNumber(config.graceSec) || DEFAULT_GRACE_SEC;
  const maxTurns = cfgNumber(config.maxTurnsPerRun);
  const toolsets = cfgString(config.toolsets) || cfgStringArray(config.enabledToolsets)?.join(",");
  const extraArgs = cfgStringArray(config.extraArgs);
  const persistSession = cfgBoolean(config.persistSession) !== false;
  const worktreeMode = cfgBoolean(config.worktreeMode) === true;
  const checkpoints = cfgBoolean(config.checkpoints) === true;
  const prevSessionId = cfgString(
    (ctx.runtime?.sessionParams as Record<string, unknown> | null)?.sessionId,
  );

  // The server adds this runtime inventory at the run boundary. Requiring the
  // marker avoids touching a developer's real Hermes home in direct unit or
  // library calls that did not opt into Paperclip runtime skills.
  if (Object.prototype.hasOwnProperty.call(config, "paperclipRuntimeSkills")) {
    try {
      const selectedSkills = await reconcileHermesPaperclipSkills(config);
      if (selectedSkills.length > 0) {
        await ctx.onLog(
          "stdout",
          `[hermes] Reconciled ${selectedSkills.length} Paperclip-managed skill(s) into the Hermes skills home.\n`,
        );
      }
    } catch (err) {
      const reason = err instanceof Error ? err.message : String(err);
      await ctx.onLog("stderr", `[hermes] Cannot start without the required Paperclip-managed skills: ${reason}\n`);
      throw err;
    }
  }

  // ── Resolve provider (defense in depth) ────────────────────────────────
  // Priority chain:
  //   1. Explicit provider in adapterConfig (user override)
  //   2. Provider from ~/.hermes/config.yaml (detected at runtime)
  //   3. Provider inferred from model name prefix
  //   4. "auto" (let Hermes decide)
  //
  // This ensures that even if the agent was created before provider tracking
  // was added, or if the model was changed without updating provider, the
  // correct provider is still used.
  let detectedConfig: Awaited<ReturnType<typeof detectModel>> | null = null;
  const explicitProvider = cfgString(config.provider);

  if (!explicitProvider) {
    try {
      detectedConfig = await detectModel();
    } catch {
      // Non-fatal — detection failure shouldn't block execution
    }
  }

  const { provider: resolvedProvider, resolvedFrom } = resolveProvider({
    explicitProvider,
    detectedProvider: detectedConfig?.provider,
    detectedModel: detectedConfig?.model,
    detectedBaseUrl: detectedConfig?.baseUrl,
    detectedHasApiKey: detectedConfig?.hasApiKey,
    detectedApiMode: detectedConfig?.apiMode,
    model,
  });

  // Z.ai subscription routes are plan-covered: the CLI still prints a dollar
  // estimate, but that figure is a pricing-table synthetic, not money owed.
  // Attribution only — this does not touch routing, retries, or fallbacks.
  const billingIdentity = resolveHermesBillingIdentity({
    baseUrl: detectedConfig?.baseUrl ?? cfgString(config.baseUrl),
    provider: resolvedProvider,
    model,
  });
  const { billingType, biller, zeroReportedCostUsd } = billingIdentity;

  // ── Load agent instructions file (Paperclip instruction bundles) ──────
  // Paperclip can materialize managed instructions into instructionsFilePath;
  // when present, inject that bundle into the Hermes prompt.
  const instructionsFilePath = cfgString(config.instructionsFilePath);
  let agentInstructions = "";
  if (instructionsFilePath) {
    try {
      agentInstructions = await fs.readFile(instructionsFilePath, "utf-8");
      const loadedInstructionsLength = agentInstructions.length;
      const instructionsFileDir = path.dirname(instructionsFilePath);
      agentInstructions += `\nThe above agent instructions were loaded from ${instructionsFilePath}. Resolve any relative file references from ${instructionsFileDir}/.`;
      await ctx.onLog(
        "stdout",
        `[hermes] Loaded agent instructions from ${instructionsFilePath} (${loadedInstructionsLength} chars)\n`,
      );
    } catch (err) {
      const reason = err instanceof Error ? err.message : String(err);
      // Non-fatal: log to stdout with an explicit "Warning:" prefix so the
      // Paperclip UI doesn't render this as a red error (stderr output is
      // surfaced as an error signal even when execution continues).
      await ctx.onLog(
        "stdout",
        `[hermes] Warning: could not read agent instructions file "${instructionsFilePath}": ${reason}\n`,
      );
    }
  }

  // ── Build prompt ───────────────────────────────────────────────────────
  let prompt = buildPrompt(ctx, config, { resumedSession: Boolean(prevSessionId) });
  if (agentInstructions) {
    prompt = agentInstructions + "\n\n---\n\n" + prompt;
  }

  // ── Build command args ─────────────────────────────────────────────────
  // Use -Q (quiet) to get clean output: just response + session_id line.
  // Quiet is the default: agents run non-interactively, and hermes' non-quiet
  // path echoes the prompt through Rich markup unescaped, so prompt content like
  // `[guide](/issues/TIM-57)` raises rich.errors.MarkupError and the run exits 1.
  // Opt out explicitly with quiet: false.
  const useQuiet = cfgBoolean(config.quiet) !== false; // default true
  const args: string[] = ["chat", "-q", prompt];
  if (useQuiet) args.push("-Q");

  // Prefer the machine-readable stream-json protocol over the human-oriented
  // text surface. Verified against the installed hermes-agent: `--format
  // stream-json` emits JSONL events, *implies --quiet*, requires -q/--query
  // (every adapter run passes -q), and rejects --tui (never passed here).
  // Opt out with outputFormat: "text".
  const outputFormat = resolveOutputFormat(config);
  if (outputFormat === "stream-json") {
    args.push("--format", "stream-json");
    // stream-json already implies --quiet inside Hermes, so -Q above is
    // redundant here — kept so the TIM-67 guarantee holds even if a future
    // Hermes drops that implication. `quiet: false` cannot un-quiet the run.
  }

  if (model) {
    args.push("-m", model);
  }

  // Always pass --provider when we have a resolved one (not "auto").
  // "auto" means Hermes will decide on its own — no need to pass it.
  if (resolvedProvider !== "auto") {
    args.push("--provider", resolvedProvider);
  }

  if (toolsets) {
    args.push("-t", toolsets);
  }

  if (maxTurns && maxTurns > 0) {
    args.push("--max-turns", String(maxTurns));
  }

  if (worktreeMode) args.push("-w");
  if (checkpoints) args.push("--checkpoints");
  if (cfgBoolean(config.verbose) === true) args.push("-v");

  // Tag sessions as "tool" source so they don't clutter the user's session history.
  // Requires hermes-agent >= PR #3255 (feat/session-source-tag).
  args.push("--source", "tool");

  // Bypass Hermes dangerous-command approval prompts.
  // Paperclip agents run as non-interactive subprocesses with no TTY,
  // so approval prompts would always timeout and deny legitimate commands
  // (curl, python3 -c, etc.). Agents operate in a sandbox — the approval
  // system is designed for human-attended interactive sessions.
  args.push("--yolo");

  if (persistSession && prevSessionId) {
    args.push("--resume", prevSessionId);
  }

  if (extraArgs?.length) {
    args.push(...extraArgs);
  }

  // ── Build environment ──────────────────────────────────────────────────
  const userEnv = config.env as Record<string, string> | undefined;
  const env: Record<string, string> = {
    ...(process.env as Record<string, string>),
    ...(userEnv && typeof userEnv === "object" ? userEnv : {}),
    ...buildPaperclipEnv(ctx.agent),
    ...buildRuntimeToolsEnv(ctx.runtimeTools),
  };

  if (ctx.runId) env.PAPERCLIP_RUN_ID = ctx.runId;

  // PAPERCLIP_API_KEY is never accepted from config — the harness-minted run
  // token is the only source of Paperclip API identity.
  delete env.PAPERCLIP_API_KEY;
  // Wake context travels in the prompt; drop both inherited and configured copies.
  delete env.PAPERCLIP_WAKE_PAYLOAD_JSON;
  if ((ctx as any).authToken) env.PAPERCLIP_API_KEY = (ctx as any).authToken;

  // BUG FIX: Read task context from ctx.context (wake context), not ctx.config (adapter config)
  const ctxContext = (ctx as any).context || {};
  const envTaskId = cfgString(ctxContext.taskId) || cfgString(ctxContext.issueId) || cfgString(ctx.config?.taskId);
  if (envTaskId) env.PAPERCLIP_TASK_ID = envTaskId;
  const envWakeReason = cfgString(ctxContext.wakeReason) || cfgString(ctx.config?.wakeReason);
  if (envWakeReason) env.PAPERCLIP_WAKE_REASON = envWakeReason;
  const envCommentId = cfgString(ctxContext.commentId) || cfgString(ctxContext.wakeCommentId) || cfgString(ctx.config?.commentId);
  if (envCommentId) env.PAPERCLIP_WAKE_COMMENT_ID = envCommentId;

  // ── Resolve working directory ──────────────────────────────────────────
  const cwd =
    cfgString(config.cwd) || cfgString(ctx.config?.workspaceDir) || ".";
  try {
    await ensureAbsoluteDirectory(cwd);
  } catch {
    // Non-fatal
  }

  // ── Log start ──────────────────────────────────────────────────────────
  await ctx.onLog(
    "stdout",
    `[hermes] Starting Hermes Agent (model=${model}, provider=${resolvedProvider} [${resolvedFrom}], timeout=${timeoutSec}s${maxTurns ? `, max_turns=${maxTurns}` : ""}${zeroReportedCostUsd ? ", billing=subscription(zai)" : ""})\n`,
  );
  if (prevSessionId) {
    await ctx.onLog(
      "stdout",
      `[hermes] Resuming session: ${prevSessionId}\n`,
    );
  }

  // ── Execute ────────────────────────────────────────────────────────────
  // Hermes writes non-error noise to stderr (MCP init, INFO logs, etc).
  // Paperclip renders all stderr as red/error in the UI.
  // Wrap onLog to reclassify benign stderr lines as stdout.
  const wrappedOnLog = async (stream: "stdout" | "stderr", chunk: string) => {
    if (stream === "stderr") {
      const trimmed = chunk.trimEnd();
      // Benign patterns that should NOT appear as errors:
      // - Structured log lines: [timestamp] INFO/DEBUG/WARN: ...
      // - MCP server registration messages
      // - Python import/site noise
      const isBenign = /^\[?\d{4}[-/]\d{2}[-/]\d{2}T/.test(trimmed) || // structured timestamps
        /^[A-Z]+:\s+(INFO|DEBUG|WARN|WARNING)\b/.test(trimmed) || // log levels
        /Successfully registered all tools/.test(trimmed) ||
        /MCP [Ss]erver/.test(trimmed) ||
        /tool registered successfully/.test(trimmed) ||
        /Application initialized/.test(trimmed);
      if (isBenign) {
        return ctx.onLog("stdout", chunk);
      }
    }
    return ctx.onLog(stream, chunk);
  };

  const result = await runChildProcess(ctx.runId, hermesCmd, args, {
    cwd,
    env,
    timeoutSec,
    graceSec,
    onLog: wrappedOnLog,
    onSpawn: ctx.onSpawn,
  });

  // ── Parse output ───────────────────────────────────────────────────────
  // stream-json first; fall back to the legacy text parser when Hermes did not
  // emit the protocol (older CLI that rejected --format, or a crash before any
  // event was written). The path taken is logged so protocol regressions are
  // visible in the run log instead of silently degrading to screen-scraping.
  const stream = parseHermesStreamJson(result.stdout || "");
  const parsed = stream.recognized
    ? stream
    : parseHermesOutput(result.stdout || "", result.stderr || "");

  await ctx.onLog(
    "stdout",
    `[hermes] Exit code: ${result.exitCode ?? "null"}, timed out: ${result.timedOut}\n`,
  );
  await ctx.onLog(
    "stdout",
    `[hermes] Parsed output via ${stream.recognized ? "stream-json protocol" : "legacy text fallback"}\n`,
  );
  if (stream.recognized && parsed.exitCode !== undefined && parsed.exitCode !== result.exitCode) {
    await ctx.onLog(
      "stdout",
      `[hermes] Note: result envelope reported exit_code=${parsed.exitCode}, process exited ${result.exitCode}\n`,
    );
  }
  if (parsed.sessionId) {
    await ctx.onLog("stdout", `[hermes] Session: ${parsed.sessionId}\n`);
  }

  // ── Build result ───────────────────────────────────────────────────────
  const executionResult: AdapterExecutionResult = {
    exitCode: result.exitCode,
    signal: result.signal,
    timedOut: result.timedOut,
    provider: resolvedProvider,
    model,
    billingType,
    ...(biller ? { biller } : {}),
  };

  if (parsed.errorMessage) {
    executionResult.errorMessage = parsed.errorMessage;
  } else if (!result.timedOut && typeof result.exitCode === "number" && result.exitCode !== 0) {
    executionResult.errorMessage = `Hermes exited with code ${result.exitCode}`;
  }

  if (parsed.usage) {
    executionResult.usage = parsed.usage;
  }

  if (parsed.costUsd !== undefined) {
    // The CLI's printed estimate is kept in resultJson for inspection; the
    // billed field is the authoritative one and owes $0 on subscription routes.
    executionResult.costUsd = resolveHermesBilledCostUsd(billingIdentity, parsed.costUsd);
  }

  // Summary from agent response
  if (parsed.response) {
    executionResult.summary = parsed.response.slice(0, 2000);
  }

  // Set resultJson so Paperclip can persist run metadata (used for UI display + auto-comments)
  executionResult.resultJson = {
    result: parsed.response || "",
    session_id: parsed.sessionId || null,
    usage: parsed.usage || null,
    cost_usd: parsed.costUsd ?? null,
    billing_type: billingType,
  };

  // Store session ID for next run
  if (persistSession && parsed.sessionId) {
    executionResult.sessionParams = { sessionId: parsed.sessionId };
    executionResult.sessionDisplayId = parsed.sessionId.slice(0, 16);
  }

  return executionResult;
}
