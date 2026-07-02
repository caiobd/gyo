import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const html = readFileSync(new URL("../../index.html", import.meta.url), "utf8");

describe("semantic atlas shell", () => {
  it("exposes the accessible root action and projection hook", () => {
    expect(html).toMatch(/class="brand"[^>]*aria-label="[^"]*root[^"]*"/i);
    expect(html).toMatch(/id="projectionStatus" class="projection-status"/);
  });

  it("provides the expected lazy image template structure", () => {
    expect(html).toMatch(/<template id="imageTemplate"><figure class="sample"><div class="skeleton"><\/div><img loading="lazy"><figcaption><\/figcaption><\/figure><\/template>/);
  });
});
