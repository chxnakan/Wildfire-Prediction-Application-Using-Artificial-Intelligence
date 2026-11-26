// A road basemap with language-selectable labels, underneath the Leaflet overlays.
// OpenFreeMap supplies OpenStreetMap data; no Google Maps API key is required.
const RoadBasemap = (() => {
  const attribution = '<a href="https://openfreemap.org/">OpenFreeMap</a> · © <a href="https://openmaptiles.org/">OpenMapTiles</a> · © <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>';

  function labelExpression(lang) {
    const fields = {
      th: ["name:th", "name_th", "name", "name:en", "name_en"],
      zh: ["name:zh-Hant", "name:zh", "name_zh", "name:en", "name_en", "name:latin", "name"],
    }[lang] || ["name:en", "name_en", "name:latin", "name"];
    // Empty translations should fall through just like missing translations.
    return ["case", ...fields.flatMap(field => [
      ["all", ["has", field], ["!=", ["get", field], ""]],
      ["to-string", ["get", field]],
    ]), ""];
  }

  const Layer = L.Layer.extend({
    initialize(lang) { this.lang = lang; },
    onAdd(map) {
      this.host = map;
      if (typeof maplibregl === "undefined" || !L.maplibreGL) {
        this.fallback();
        return;
      }
      try {
        this.vector = L.maplibreGL({
          style: "https://tiles.openfreemap.org/styles/liberty",
          attributionControl: { customAttribution: attribution }, interactive: false,
        });
        // The adapter's normal cleanup assumes WebGL creation succeeded.
        const removeVector = this.vector.onRemove;
        this.vector.onRemove = function(host) {
          if (this.getMaplibreMap()) removeVector.call(this, host);
          else this.getContainer()?.remove();
        };
        this.vector.addTo(map);
        const gl = this.vector.getMaplibreMap();
        gl.on("style.load", () => {
          if (!this.host) return;
          this.ready = true;
          clearTimeout(this.timer);
          // A familiar light road palette in either dashboard theme.
          const colors = { background: ["background-color", "#f5f3ef"],
            water: ["fill-color", "#a8d5ed"], park: ["fill-color", "#c5e1b2"],
            landuse_residential: ["fill-color", "#eeece8"] };
          for (const [id, [property, value]] of Object.entries(colors)) {
            if (gl.getLayer(id)) gl.setPaintProperty(id, property, value);
          }
          if (gl.getLayer("natural_earth")) gl.setLayoutProperty("natural_earth", "visibility", "none");
          this.setLanguage(this.lang);
        });
        // Keep the dashboard usable if WebGL or the style service is unavailable.
        this.timer = setTimeout(() => { if (!this.ready) this.fallback(); }, 15000);
      } catch (error) {
        this.fallback();
      }
    },
    setLanguage(lang) {
      this.lang = ["th", "zh"].includes(lang) ? lang : "en";
      if (!this.ready || !this.vector) return;
      const gl = this.vector.getMaplibreMap();
      for (const layer of gl.getStyle().layers) {
        const field = layer.layout && layer.layout["text-field"];
        // Preserve route numbers, shield labels and other non-name text.
        if (layer.type === "symbol" && field && JSON.stringify(field).includes('"name')) {
          gl.setLayoutProperty(layer.id, "text-field", labelExpression(this.lang));
        }
      }
    },
    fallback() {
      if (!this.host || this.raster) return;
      clearTimeout(this.timer);
      if (this.vector) { this.host.removeLayer(this.vector); this.vector = null; }
      this.ready = false;
      this.raster = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 19, attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
      }).addTo(this.host);
      // Raster fallback labels cannot be translated; make that limitation visible.
      if (typeof App !== "undefined") App.toast(I18N.t("map.tileFallback"));
    },
    onRemove(map) {
      clearTimeout(this.timer);
      this.host = null;
      if (this.vector) map.removeLayer(this.vector);
      if (this.raster) map.removeLayer(this.raster);
      this.vector = this.raster = null;
      this.ready = false;
    },
  });

  return { create: lang => new Layer(lang) };
})();
