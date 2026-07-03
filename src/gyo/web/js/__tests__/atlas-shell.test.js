import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const html = readFileSync(new URL("../../index.html", import.meta.url), "utf8");
const css = readFileSync(new URL("../../style.css", import.meta.url), "utf8");

describe("semantic atlas shell", () => {
  it("exposes the accessible root action and projection hook", () => {
    expect(html).toMatch(/class="brand"[^>]*aria-label="[^"]*root[^"]*"/i);
    expect(html).toMatch(/id="projectionStatus" class="projection-status"/);
  });

  it("provides the expected lazy image template structure", () => {
    expect(html).toMatch(/<template id="imageTemplate"><figure class="sample"><div class="skeleton"><\/div><img loading="lazy"><figcaption><\/figcaption><\/figure><\/template>/);
  });

  it("keeps aggregate styling off the selection ring and documents residual patterns", () => {
    expect(css).toContain(".territory.aggregate > circle:not(.selection-ring)");
    expect(css).toContain('[data-residual-band="low"] > circle:not(.selection-ring)');
    expect(html).toContain("line-pattern scale");
  });

  it("contains long projection status text without intercepting breadcrumbs", () => {
    expect(css).toContain("#projectionStatus");
    expect(css).toMatch(/#projectionStatus\s*\{[^}]*max-width:\s*52%/s);
    expect(css).toMatch(/#projectionStatus\s*\{[^}]*pointer-events:\s*none/s);
  });

  it("does not let selected state override residual-band stroke widths", () => {
    expect(css).not.toMatch(/\.territory\.selected\s+circle\s*\{/);
    expect(css).toContain('[data-residual-band="low"] > circle:not(.selection-ring)');
    expect(css).toContain('[data-residual-band="high"] > circle:not(.selection-ring)');
  });
});
