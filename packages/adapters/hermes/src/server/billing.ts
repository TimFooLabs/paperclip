import { resolveZaiSubscriptionRoute } from "@paperclipai/adapter-utils";

export interface HermesBillingIdentity {
  /** Ledger billing type; "subscription" when the route is plan-covered. */
  billingType: "subscription" | "unknown";
  /** Biller attribution; Z.ai when the route is plan-covered. */
  biller: "zai" | null;
  /**
   * The CLI prints a dollar estimate even on plan-covered routes. When true,
   * that estimate is synthetic pricing-table output and must not reach the
   * ledger as billed cost.
   */
  zeroReportedCostUsd: boolean;
}

/**
 * Resolve billing attribution for a Hermes route. Attribution only — this does
 * not touch routing, retries, or fallbacks.
 */
export function resolveHermesBillingIdentity(input: {
  /** API base URL from Hermes config (e.g. base_url). */
  baseUrl?: string | null;
  /** Provider the request is routed through (auto, openrouter, zai, ...). */
  provider?: string | null;
  /** Requested model id. */
  model?: string | null;
}): HermesBillingIdentity {
  const zaiSubscriptionRoute = resolveZaiSubscriptionRoute(input);

  return {
    billingType: zaiSubscriptionRoute ? "subscription" : "unknown",
    biller: zaiSubscriptionRoute ? "zai" : null,
    zeroReportedCostUsd: zaiSubscriptionRoute,
  };
}

/** Billed cost for a route: $0 when plan-covered, else the reported figure. */
export function resolveHermesBilledCostUsd(
  identity: HermesBillingIdentity,
  reportedCostUsd: number | undefined,
): number | undefined {
  if (reportedCostUsd === undefined) return undefined;
  return identity.zeroReportedCostUsd ? 0 : reportedCostUsd;
}
