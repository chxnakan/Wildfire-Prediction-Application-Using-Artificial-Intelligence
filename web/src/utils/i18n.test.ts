import { describe, expect, it } from "vitest";
import { DICT } from "./i18n";

describe("operator-facing translation contract", () => {
  it("keeps English and Thai key coverage in sync", () => {
    expect(Object.keys(DICT.en).sort()).toEqual(Object.keys(DICT.th).sort());
  });

  it("does not reintroduce obsolete pseudo-day or unsupported stability claims", () => {
    const copy = Object.values(DICT)
      .flatMap((translations) => Object.values(translations))
      .join("\n");

    expect(copy).not.toMatch(/days\s*=\s*1/i);
    expect(copy).not.toMatch(/next 7 days/i);
    expect(copy).not.toMatch(/7 วันถัดไป/i);
    expect(copy).not.toMatch(/17 months/i);
    expect(copy).not.toMatch(/17 เดือน/i);
    expect(copy).not.toMatch(/trustworthy as a real %/i);
  });
});
