import { afterEach, describe, expect, it, vi } from "vitest";
import {
  calendarDayLag,
  computeFreshness,
  dateAdd,
  localIsoDate,
} from "./dates";

afterEach(() => {
  vi.useRealTimers();
});

describe("calendar date utilities", () => {
  it("adds days without depending on the browser timezone", () => {
    expect(dateAdd("2026-07-24", 1)).toBe("2026-07-25");
    expect(dateAdd("2024-02-28", 1)).toBe("2024-02-29");
    expect(dateAdd("2026-12-31", 1)).toBe("2027-01-01");
  });

  it("rejects rolled-over and malformed calendar dates", () => {
    expect(dateAdd("2026-02-30", 1)).toBe("—");
    expect(dateAdd("not-a-date", 1)).toBe("—");
  });

  it("computes lag by calendar day rather than elapsed local hours", () => {
    const localNow = new Date(2026, 6, 24, 23, 59);
    expect(calendarDayLag("2026-07-24", localNow)).toBe(0);
    expect(calendarDayLag("2026-07-23", localNow)).toBe(1);
  });

  it("formats the viewer's local date without UTC rollover", () => {
    expect(localIsoDate(new Date(2026, 6, 25, 0, 30))).toBe("2026-07-25");
  });

  it("does not label future or invalid snapshots as live", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(2026, 6, 24, 12));
    expect(computeFreshness("2026-07-25")?.cls).toBe("expired");
    expect(computeFreshness("2026-02-30")?.label).toBe("invalid date");
  });
});
