import { describe, expect, it } from "vitest";
import { createRequestGuard } from "../main.js";

describe("atlas request guard", () => {
  it("invalidates stale responses and aborts the previous request", () => {
    const guard = createRequestGuard();
    const first = guard.next();
    const second = guard.next();
    expect(first.signal.aborted).toBe(true);
    expect(guard.isCurrent(first.id)).toBe(false);
    expect(guard.isCurrent(second.id)).toBe(true);
  });
});
