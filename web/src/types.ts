// Mirrors what risk_map.append_geojson writes. The backend is the authority;
// these types only document what the frontend reads.

export type UrgencyLevel = "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "NONE";

export interface UrgencyThresholds {
  CRITICAL: number;
  HIGH: number;
  MEDIUM: number;
  LOW: number;
}

export interface ValidationMetrics {
  // Active task discriminator. Undefined is accepted only for compatibility
  // with historical exports; consumers must inspect which metric fields are
  // actually present instead of assuming that missing means regression.
  task?: string;
  imminent_days?: number;
  // Deprecated date-regression metrics. They describe historical artifacts,
  // not the active 3-day binary-risk model.
  mae_days?: number;
  rmse_days?: number;
  r2?: number;
  accuracy_within_1day?: number;
  accuracy_exact?: number;
  // Binary classification metrics — meaningful when task = "binary_fire_in_3d".
  roc_auc?: number;
  average_precision?: number;
  binary_accuracy?: number;
  precision?: number;
  recall?: number;
  f1?: number;
  best_f1?: number;
  best_threshold?: number;
  precision_at_best_thr?: number;
  recall_at_best_thr?: number;
  precision_at_top_5pct?: number;
  precision_at_top_10pct?: number;
  precision_at_top_20pct?: number;
  // Baseline + uplift framing (test_positive_rate is what a random
  // ranker would achieve on precision@K).
  test_positive_rate?: number;
  uplift_at_top_5pct?: number;
  uplift_at_top_10pct?: number;
  uplift_at_top_20pct?: number;
  // Calibration metrics (added by Phase 0.5 post-calibration).
  // ECE = aggregate expected calibration error on test (lower = better).
  // It must be interpreted with reliability-bin sample sizes and prevalence
  // drift; no single ECE cutoff guarantees that every cell probability is
  // reliable in every season.
  ece?: number;
  ece_val_before_calibration?: number;
  ece_val_after_calibration?: number;
  calibration_method?: string;
  // "calib" when the deployment threshold was locked on the calibration
  // split (which also implies Platt calibration was fitted there).
  threshold_selected_on?: string;
  // Locked deployment threshold + its metrics. Headline dashboard numbers
  // should come from these (they're the ones an operator actually sees).
  deployment_threshold?: number;
  deployment_precision?: number;
  deployment_recall?: number;
  deployment_f1?: number;
  deployment_accuracy?: number;
  // WATCH (recall-oriented monitoring) operating point + full HIGH/WATCH
  // reports + base-rate drift diagnostic. Present only on models trained with
  // the two-threshold selector; older snapshots omit them (render gracefully).
  watch_threshold?: number;
  high_test_metrics?: ThresholdReport;
  watch_test_metrics?: ThresholdReport;
  prevalence_drift?: PrevalenceDrift;
  recall_captured_top_5pct?: number;
  recall_captured_top_10pct?: number;
  recall_captured_top_20pct?: number;
  // Reliability bins for the calibration curve plot.
  reliability_bins?: ReliabilityBin[];
  // Where the metrics were evaluated. "full_distribution_test_window" = real
  // class balance (positive ~3-5%); legacy runs evaluated on undersampled.
  evaluated_on?: string;
  // Legacy monthly diagnostic fields from rolling_eval.json. That script
  // applies one deployed model across its full timeline, including training
  // months, so these values are in-sample descriptive evidence—not genuine
  // walk-forward/out-of-sample stability.
  stability_months?: number;
  stability_valid_months?: number;
  stability_auc_mean?: number;
  stability_auc_std?: number;
  stability_auc_min?: number;
  stability_auc_max?: number;
  rolling_by_month?: RollingMonthPoint[];
  feature_importance_top?: { feature: string; importance: number }[];
  // ── Scientific statistics (from scripts/scientific_stats.py) ──
  scientific_stats?: ScientificStats;
  // ── Training summary (real persisted values from dataset_info.json) ──
  training_summary?: TrainingSummary;
}

export interface TrainingSummary {
  trained_at?: string;
  data_source?: string;
  date_range?: [string | null, string | null];
  total_days?: number;
  active_cells?: number;
  grid_size_deg?: number;
  training_rows?: number;
  feature_count?: number;
  weather_features_count?: number;
  prediction_type?: string;
  imminent_days?: number;
  training_time_seconds?: number;
  model_type?: string;
  search_method?: string;
  search_iterations?: number;
  cv_n_splits?: number;
  cv_gap_days?: number | string;
  ensemble_size?: number;
  early_stopping_rounds?: number;
  best_params?: Record<string, number | string | boolean | null>;
}

export interface BootstrapCI {
  point: number;
  lower: number;
  upper: number;
  std: number;
  n_boot: number;
  confidence: number;
}

export interface ScientificStats {
  samples: {
    total_densified: number;
    train: SampleSplit;
    /** Legacy reports used `val`; current reports publish `validation`. */
    val?: SampleSplit;
    validation?: SampleSplit;
    calibration?: SampleSplit;
    test: SampleSplit;
  };
  ci_95: {
    roc_auc: BootstrapCI;
    average_precision: BootstrapCI;
    f1_at_deploy: BootstrapCI;
    precision_at_deploy: BootstrapCI;
    recall_at_deploy: BootstrapCI;
    brier_score: BootstrapCI;
  };
  confusion_matrix: {
    tn: number; fp: number; fn: number; tp: number;
    matrix: number[][];
    row_labels: string[];
    col_labels: string[];
  };
  classification_stats: {
    sensitivity: number;
    specificity: number;
    ppv: number;
    npv: number;
    false_positive_rate: number;
    false_negative_rate: number;
    cohen_kappa: number;
    matthews_corr_coef: number;
    log_loss: number;
    brier_score: number;
    brier_skill_score: number;
    baseline_class_prior: number;
  };
  roc_curve: CurvePoint[];
  pr_curve: CurvePoint[];
}

export interface SampleSplit {
  n: number;
  positives: number;
  positive_rate: number | null;
  date_range: [string | null, string | null];
}

export interface CurvePoint {
  x: number;
  y: number;
  t: number;  // threshold at this point
}

export interface RollingMonthPoint {
  month: string;          // "2025-04"
  auc: number;
  positive_rate: number;
  n: number;
}

export interface ReliabilityBin {
  bin_lower: number;
  bin_upper: number;
  mean_predicted: number;
  actual_rate: number;
  count: number;
}

export interface SnapshotHitRate {
  hits?: number;
  misses?: number;
  future?: number;
}

export interface ThresholdReport {
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
  alert_rate: number;
  confusion_matrix?: { tp: number; fp: number; fn: number; tn: number };
  n?: number;
  positives?: number;
}

export interface PrevalenceDrift {
  calib_prevalence: number;
  recent_prevalence: number;
  ratio: number;
  ratio_threshold: number;
  drift_flag: boolean;
  direction?: string;
}

export interface RankingPolicy {
  type?: string;
  policy_version?: string;
  watch_top_pct?: number;
  extended_watch_top_pct?: number;
  absolute_high_threshold?: number;
  rank_method?: string;
  tie_breaking?: string[];
  relative_tier_labels?: Record<string, string>;
  note?: string;
  historical_coverage_note?: string;
  historical_coverage?: {
    top_5pct?: number;
    top_10pct?: number;
    top_20pct?: number;
  };
}

export interface GeoJsonMetadata {
  // base_date is the daily prediction origin; latest_data_time is the real
  // UTC instant of the freshest FIRMS hotspot (finer than the day); both
  // written by risk_map.py. generated_at = when the snapshot was built.
  base_date?: string;
  /** Version identifier of the model that produced metadata.base_date. */
  model_version?: string | null;
  /** Explicit date scope for the top-level thresholds and metrics. */
  thresholds_apply_to_base_date?: string;
  /** Human-readable warning that retained snapshots can use older models. */
  snapshot_contract_note?: string;
  horizon_days?: number;
  risk_window_start?: string;
  risk_window_end?: string;
  latest_data_time?: string;
  /** Newest raw cache timestamp, which may be later than the trusted snapshot. */
  source_cache_latest_time?: string;
  generated_at?: string;
  urgency_thresholds?: UrgencyThresholds;
  // Human-readable provenance of urgency_thresholds, written by risk_map.py:
  //   "absolute calibrated probability, …"  → cutoffs are probabilities (0..1)
  //   "PRIORITY RANKING (relative percentiles …)" → relative rank, not risk
  //   otherwise → legacy days-based cutoffs
  thresholds_source?: string;
  /** Background candidates omitted from the contextual map layer. */
  background_context_omitted?: number;
  /** @deprecated Alias of background_context_omitted. */
  suppressed_by_display_cap?: number;
  suppressed_by_display_cap_note?: string;
  metrics?: ValidationMetrics;
  ranking_policy?: RankingPolicy;
  selection_policy?: {
    total_scored_cells?: number;
    retained_map_candidate_cells?: number;
    selection_changes_probability?: boolean;
    filters?: Record<string, Record<string, unknown>>;
    note?: string;
  };
  observed_scope?: {
    scope?: string;
    country_filter_enabled?: boolean;
    positive_source_rows?: number;
    emitted_rows?: number;
    excluded_outside_scope?: number;
    note?: string;
  };
  validation_summary?: {
    evaluation?: string;
    match_rule?: string;
    horizon_days?: number;
    operational_tiers?: string[];
    per_snapshot?: Record<string, SnapshotHitRate>;
  };
}

export interface PredictionProperties {
  source: "predicted" | "observed";
  base_date?: string;
  /** Model version persisted with this snapshot row, when known. */
  model_version?: string | null;
  /** @deprecated Compatibility-only pseudo-date; not a per-cell forecast. */
  predicted_fire_date?: string;
  /** @deprecated Compatibility-only pseudo-day value; not model output time. */
  days_until_fire?: number;
  /** Legacy encoded pseudo-day value used only to recover old probabilities. */
  raw_prediction?: number;
  // Real probability (0..1) recovered from the binary classifier — only
  // written by newer risk_map.py runs. Older snapshots omit it and the
  // frontend falls back to inverting raw_prediction.
  probability?: number;
  urgency_level?: UrgencyLevel;
  /** Relative resource-allocation band; never an absolute alert tier. */
  ranking_tier?:
    | "RELATIVE_TOP_10"
    | "RELATIVE_TOP_20"
    | "UNRANKED"
    // Legacy values remain readable but are labelled as relative by the UI.
    | "WATCH"
    | "EXTENDED_WATCH"
    | "LOW"
    | "HIGH";
  /** 1-based competition rank: equal scores share the same rank. */
  rank?: number;
  percentile?: number;
  confidence?: number;
  province?: string;
  historical_fire_count_30d?: number;
  fire_days_per_year?: number;
  tree_cover_pct_2000?: number | null;
  tree_loss_pct_recent?: number | null;
  nearest_urban_area?: string;
  nearest_urban_distance_km?: number;
  // observed
  date?: string;
  fire_count?: number;
  // Retrospective validation (set by risk_map.py once base_date+1 … +3 has
  // fully elapsed):
  //   "hit"    — the same grid cell had a FIRMS hotspot in that risk window
  //   "miss"   — the same grid cell had none in that risk window
  //   "future" — the complete risk window is not yet observable
  validation_status?: "hit" | "miss" | "future";
}

export interface FireFeature {
  type: "Feature";
  geometry: { type: "Point"; coordinates: [number, number] };
  properties: PredictionProperties;
}

export interface FireGeoJson {
  type: "FeatureCollection";
  features: FireFeature[];
  metadata?: GeoJsonMetadata;
}

// ───────────── Notify / Alert Dispatch ─────────────

export type NotifyChannel = "sms" | "line" | "email" | "all";
export type NotifyPriority = "normal" | "urgent" | "emergency";

export interface NotifyCapabilities {
  mode: "stub";
  /** Always false until an audited external provider is implemented. */
  delivery_available: false;
  /** The public SPA can only write to the deliberately open local-dev stub. */
  browser_queue_available: boolean;
  browser_log_available: boolean;
  persistence: "process_memory";
}

export interface NotifyRequest {
  channel: NotifyChannel;
  recipients: string[];
  message: string;
  zone_ids: string[];
  priority: NotifyPriority;
  template?: string;
}

export interface NotifyResponse {
  status: "queued";
  id: string;
  timestamp: string;
}

export interface NotifyLogRecord {
  id: string;
  timestamp: string;
  channel: NotifyChannel;
  recipients_count: number;
  /** Legacy development logs only; production intentionally omits PII. */
  recipients_preview?: string[];
  zone_ids_count: number;
  zone_ids_preview?: string[];
  priority: NotifyPriority;
  template: string | null;
  /** Current privacy-preserving API stores length, not message content. */
  message_length?: number;
  message_preview?: string;
  status: "queued" | "sent" | "failed";
}

export type AlertPageRoute =
  | "dashboard"
  | "notify"
  | "live"
  | "reports"
  | "compare";

// Risk-tier filter for the operator screen. Replaced the former day-0..7
// pseudo-date buckets — the binary model has no per-day resolution.
// The default operator view shows every retained absolute tier. Background
// stays visually neutral and is excluded from the heat layer, while the
// Alerts filter remains available for CRITICAL/HIGH + MEDIUM/WATCH only.
export type DaySelection =
  | "Alerts"
  | "High"
  | "Watch"
  | "Background"
  | "all";

export interface DisplayOptions {
  showObserved: boolean;
  showLiveFires: boolean;
  showPredicted: boolean;
  showCellPins: boolean;
  hideVeryLowRisk: boolean;
  heatRadius: number;
}

export type LiveFireStatus = "idle" | "loading" | "ok" | "error";

export interface LiveFireMeta {
  status: LiveFireStatus;
  count: number;
  lastFetch: Date | null;
  error: string | null;
  /** Backend flag: the server-side poller hasn't succeeded within its
   *  staleness window — cached detections still render but may be old. */
  isStale?: boolean;
  /** Last time the SERVER successfully pulled from GISTDA (ISO string). */
  lastSuccessAt?: string | null;
  /** True when the API returned only the requested page, not the full feed. */
  truncated?: boolean;
  requestedLimit?: number;
  returnedCount?: number;
}

// GISTDA hotspot attributes as served by OUR backend (/live-hotspots) in the
// legacy ArcGIS-like shape — typed loosely because the upstream schema
// varies between the NPP and MODIS layers.
export interface GistdaFeature {
  attributes: {
    latitude?: number | string;
    longitude?: number | string;
    confident?: string | number;
    lu_name?: string;
    pv_tn?: string;
    ap_tn?: string;
    tb_tn?: string;
    date?: number;
    time?: string;
    satellite?: string;
    hotspot_id?: string;
    detected_by?: string[];
    detected_at_thailand?: string;
    source?: string;
  };
}
