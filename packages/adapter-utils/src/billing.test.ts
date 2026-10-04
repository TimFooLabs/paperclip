import { describe, expect, it } from "vitest";
import {
  inferOpenAiCompatibleBiller,
  isZaiApiBaseUrl,
  looksLikeGlmModel,
  resolveZaiSubscriptionRoute,
} from "./billing.js";

describe("inferOpenAiCompatibleBiller", () => {
  it("returns openrouter when OPENROUTER_API_KEY is present", () => {
    expect(
      inferOpenAiCompatibleBiller({ OPENROUTER_API_KEY: "sk-or-123" } as NodeJS.ProcessEnv, "openai"),
    ).toBe("openrouter");
  });

  it("returns openrouter when OPENAI_BASE_URL points at OpenRouter", () => {
    expect(
      inferOpenAiCompatibleBiller(
        { OPENAI_BASE_URL: "https://openrouter.ai/api/v1" } as NodeJS.ProcessEnv,
        "openai",
      ),
    ).toBe("openrouter");
  });

  it("returns fallback when no OpenRouter markers are present", () => {
    expect(
      inferOpenAiCompatibleBiller(
        { OPENAI_BASE_URL: "https://api.openai.com/v1" } as NodeJS.ProcessEnv,
        "openai",
      ),
    ).toBe("openai");
  });
});

describe("isZaiApiBaseUrl", () => {
  it("matches Z.ai hostnames with and without a scheme", () => {
    expect(isZaiApiBaseUrl("https://api.z.ai/api/anthropic")).toBe(true);
    expect(isZaiApiBaseUrl("api.z.ai/api/anthropic")).toBe(true);
    expect(isZaiApiBaseUrl("https://open.z.ai/v1")).toBe(true);
  });

  it("rejects lookalike hostnames and empty values", () => {
    expect(isZaiApiBaseUrl("https://notz.ai/api")).toBe(false);
    expect(isZaiApiBaseUrl("https://z.ai.example.com/v1")).toBe(false);
    expect(isZaiApiBaseUrl("https://api.anthropic.com")).toBe(false);
    expect(isZaiApiBaseUrl("")).toBe(false);
    expect(isZaiApiBaseUrl(null)).toBe(false);
  });
});

describe("looksLikeGlmModel", () => {
  it("matches GLM model ids", () => {
    expect(looksLikeGlmModel("glm-5.3-flash")).toBe(true);
    expect(looksLikeGlmModel("GLM-4.7")).toBe(true);
  });

  it("rejects prefixed and non-GLM model ids", () => {
    expect(looksLikeGlmModel("openrouter/z-ai/glm-4.6")).toBe(false);
    expect(looksLikeGlmModel("claude-sonnet-4-5")).toBe(false);
    expect(looksLikeGlmModel(null)).toBe(false);
  });
});

describe("resolveZaiSubscriptionRoute", () => {
  it("treats a Z.ai base URL as subscription-covered regardless of provider", () => {
    expect(
      resolveZaiSubscriptionRoute({ baseUrl: "https://api.z.ai/api/anthropic", provider: "anthropic" }),
    ).toBe(true);
  });

  it("treats a Z.ai provider as subscription-covered", () => {
    expect(resolveZaiSubscriptionRoute({ provider: "zai", model: "glm-5.3-flash" })).toBe(true);
  });

  it("treats a GLM model on the Anthropic route as subscription-covered", () => {
    expect(resolveZaiSubscriptionRoute({ provider: "anthropic", model: "glm-5.3-flash" })).toBe(true);
  });

  it("treats a GLM model with deferred provider resolution as subscription-covered", () => {
    expect(resolveZaiSubscriptionRoute({ provider: "auto", model: "glm-5.3-flash" })).toBe(true);
    expect(resolveZaiSubscriptionRoute({ provider: undefined, model: "glm-4.7" })).toBe(true);
  });

  it("keeps OpenRouter metered even when it resells a GLM model", () => {
    expect(
      resolveZaiSubscriptionRoute({ provider: "openrouter", model: "glm-5.3-flash" }),
    ).toBe(false);
    expect(
      resolveZaiSubscriptionRoute({ provider: "openrouter", model: "openrouter/z-ai/glm-4.6" }),
    ).toBe(false);
  });

  it("keeps non-GLM models on third-party routes metered", () => {
    expect(
      resolveZaiSubscriptionRoute({ provider: "anthropic", model: "claude-sonnet-4-5" }),
    ).toBe(false);
    expect(
      resolveZaiSubscriptionRoute({ provider: "nous", model: "glm-5.3-flash" }),
    ).toBe(false);
  });
});
