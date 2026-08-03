export type ThaiFireSeason = "in-season" | "off-season";

// Thailand's main fire season spans November through April. Parse the
// snapshot's ISO base_date directly rather than using the viewer's clock, so
// historical and future snapshots always receive the correct context.
export function thaiFireSeasonForDate(
  baseDate: string | null | undefined,
): ThaiFireSeason | null {
  const match = /^(\d{4})-(\d{2})(?:-\d{2})?$/.exec(baseDate ?? "");
  if (!match) return null;
  const month = Number(match[2]);
  if (!Number.isInteger(month) || month < 1 || month > 12) return null;
  return month >= 11 || month <= 4 ? "in-season" : "off-season";
}
