import { describe, expect, it } from "vitest";

import {
  resolveHermesBillingIdentity,
  resolveHermesBilledCostUsd,
} from "./billing.js";

describe("resolveHermesBillingIdentity", () => {
  it("classifies an explicit zai provider as subscription-covered", () => {
    expect(
      resolveHermesBillingIdentity({ provider: "zai", model: "glm-5.3-flash" }),
    ).toEqual({
      billingType: "subscription",
      biller: "zai",
      zeroReportedCostUsd: true,
    });
  });

  it("classifies a Z.ai base URL as subscription-covered even when resolution deferred to auto", () => {
    expect(
      resolveHermesBillingIdentity({
        baseUrl: "https://api.z.ai/api/anthropic",
        provider: "auto",
        model: "glm-5.3-flash",
      }),
    ).toEqual({
      billingType: "subscription",
      biller: "zai",
      zeroReportedCostUsd: true,
    });
  });

  it("classifies a GLM model with deferred provider resolution as subscription-covered", () => {
    expect(
      resolveHermesBillingIdentity({ provider: "auto", model: "glm-5.3-flash" }),
    ).toEqual({
      billingType: "subscription",
      biller: "zai",
      zeroReportedCostUsd: true,
    });
  });

  it("leaves metered routes alone", () => {
    expect(
      resolveHermesBillingIdentity({ provider: "openrouter", model: "anthropic/claude-sonnet-4" }),
    ).toEqual({
      billingType: "unknown",
      biller: null,
      zeroReportedCostUsd: false,
    });
  });

  it("keeps OpenRouter metered when it resells a GLM model", () => {
    expect(
      resolveHermesBillingIdentity({ provider: "openrouter", model: "glm-5.3-flash" }),
    ).toEqual({
      billingType: "unknown",
      biller: null,
      zeroReportedCostUsd: false,
    });
  });
});

describe("resolveHermesBilledCostUsd", () => {
  it("zeroes the CLI estimate on a subscription route", () => {
    const identity = resolveHermesBillingIdentity({ provider: "zai", model: "glm-5.3-flash" });
    expect(resolveHermesBilledCostUsd(identity, 10)).toBe(0);
  });

  it("keeps the reported estimate on a metered route", () => {
    const identity = resolveHermesBillingIdentity({
      provider: "openrouter",
      model: "anthropic/claude-sonnet-4",
    });
    expect(resolveHermesBilledCostUsd(identity, 0.42)).toBe(0.42);
  });

  it("passes through an absent estimate", () => {
    const identity = resolveHermesBillingIdentity({ provider: "zai", model: "glm-5.3-flash" });
    expect(resolveHermesBilledCostUsd(identity, undefined)).toBeUndefined();
  });
});
