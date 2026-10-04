function readEnv(env: NodeJS.ProcessEnv, key: string): string | null {
  const value = env[key];
  return typeof value === "string" && value.trim().length > 0 ? value.trim() : null;
}

export function inferOpenAiCompatibleBiller(
  env: NodeJS.ProcessEnv,
  fallback: string | null = "openai",
): string | null {
  const explicitOpenRouterKey = readEnv(env, "OPENROUTER_API_KEY");
  if (explicitOpenRouterKey) return "openrouter";

  const baseUrl =
    readEnv(env, "OPENAI_BASE_URL") ??
    readEnv(env, "OPENAI_API_BASE") ??
    readEnv(env, "OPENAI_API_BASE_URL");
  if (baseUrl && /openrouter\.ai/i.test(baseUrl)) return "openrouter";

  return fallback;
}

/** Hostnames owned by Z.ai (api.z.ai, open.z.ai, ...). */
const ZAI_HOSTNAME_PATTERN = /(^|\.)z\.ai$/i;
/** GLM models are Z.ai's own model family. */
const GLM_MODEL_PREFIX_PATTERN = /^glm[-_]/i;

/**
 * Recognize a Z.ai API base URL. Accepts values with or without a scheme so
 * config like `ANTHROPIC_BASE_URL=api.z.ai/api/anthropic` still matches.
 */
export function isZaiApiBaseUrl(value: string | null | undefined): boolean {
  const raw = typeof value === "string" ? value.trim() : "";
  if (!raw) return false;
  const withScheme = /^[a-z][a-z0-9+.-]*:\/\//i.test(raw) ? raw : `https://${raw}`;
  try {
    return ZAI_HOSTNAME_PATTERN.test(new URL(withScheme).hostname);
  } catch {
    return false;
  }
}

/** GLM model ids name Z.ai's own model family (metered resellers excepted). */
export function looksLikeGlmModel(model: string | null | undefined): boolean {
  return typeof model === "string" && GLM_MODEL_PREFIX_PATTERN.test(model.trim());
}

export interface ZaiSubscriptionRouteInput {
  /** API base URL the route points at (e.g. ANTHROPIC_BASE_URL / Hermes config base_url). */
  baseUrl?: string | null;
  /** Provider the request is routed through ("anthropic", "zai", "auto", "openrouter", ...). */
  provider?: string | null;
  /** Requested model id. */
  model?: string | null;
}

/**
 * Providers that can resolve a GLM model onto a Z.ai subscription route.
 * Anthropic never serves GLM, so a GLM model on its API shape is a Z.ai
 * proxy; "auto" means the adapter deferred to the CLI, which maps GLM to Z.ai.
 */
const GLM_ZAI_RESOLVABLE_PROVIDERS = new Set(["", "auto", "anthropic"]);

/**
 * Decide whether a route is a Z.ai subscription route whose usage is covered by
 * the plan: provider-reported `costUsd` figures on these routes are synthetic
 * pricing-table estimates, never dollars the company owes.
 *
 * A route qualifies when the base URL is Z.ai, the resolved provider is Z.ai,
 * or a GLM model rides a provider that can only reach Z.ai for it. Metered
 * resellers such as OpenRouter keep their existing cost accounting even when
 * they resell a GLM model.
 */
export function resolveZaiSubscriptionRoute(input: ZaiSubscriptionRouteInput): boolean {
  if (isZaiApiBaseUrl(input.baseUrl)) return true;

  const provider = typeof input.provider === "string" ? input.provider.trim().toLowerCase() : "";
  if (provider === "zai" || provider === "z_ai" || provider === "z.ai") return true;
  if (provider === "openrouter") return false;

  return GLM_ZAI_RESOLVABLE_PROVIDERS.has(provider) && looksLikeGlmModel(input.model);
}
