// Province lookup for Thailand-only hotspots using bundled geoBoundaries polygons.
const Geo = (() => {
  let provinces = null;
  let loading = null;
  const cache = new Map();

  function ringBox(ring, box) {
    for (const [x, y] of ring) {
      if (x < box[0]) box[0] = x;
      if (y < box[1]) box[1] = y;
      if (x > box[2]) box[2] = x;
      if (y > box[3]) box[3] = y;
    }
  }

  function load() {
    if (provinces) return Promise.resolve(provinces);
    if (loading) return loading;
    loading = fetch("data/thailand-provinces.geojson")
      .then(r => { if (!r.ok) throw new Error("provinces " + r.status); return r.json(); })
      .then(data => {
        provinces = data.features.map(f => {
          const polys = f.geometry.type === "Polygon" ? [f.geometry.coordinates] : f.geometry.coordinates;
          const box = [Infinity, Infinity, -Infinity, -Infinity];
          polys.forEach(p => ringBox(p[0], box));
          return { code: f.properties.code, name_en: f.properties.name_en, name_th: f.properties.name_th,
                   polys, box, feature: f };
        });
        return provinces;
      })
      .catch(err => { console.warn("Province boundaries unavailable:", err); provinces = []; return provinces; });
    return loading;
  }

  function inRing(x, y, ring) {
    let inside = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const [xi, yi] = ring[i], [xj, yj] = ring[j];
      if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
    }
    return inside;
  }

  function inPoly(x, y, poly) {
    if (!inRing(x, y, poly[0])) return false;
    for (let h = 1; h < poly.length; h++) if (inRing(x, y, poly[h])) return false;
    return true;
  }

  // Returns a province code, or null when outside Thailand.
  function locate(lat, lon) {
    if (!provinces || !provinces.length) return null;
    const key = lat + "," + lon;
    if (cache.has(key)) return cache.get(key);
    let found = null;
    for (const p of provinces) {
      const b = p.box;
      if (lon < b[0] || lon > b[2] || lat < b[1] || lat > b[3]) continue;
      if (p.polys.some(poly => inPoly(lon, lat, poly))) { found = p.code; break; }
    }
    cache.set(key, found);
    return found;
  }

  function byCode(code) { return provinces ? provinces.find(p => p.code === code) || null : null; }
  function name(code, lang) {
    const p = byCode(code);
    if (!p) return null;
    return lang === "th" ? p.name_th : p.name_en;
  }
  function list() { return provinces || []; }
  const bounds = [[5.6, 97.3], [20.5, 105.7]];
  return { load, locate, byCode, name, list, bounds };
})();
