# Model Performance & Validation infographic

For **Wildfire Prediction Web Application Using Artificial Intelligence**.

The single infographic follows the supplied written design: pastel blue
`#C6D8FA`, navy headings `#343B7C`, white rounded panels, a two-column grid,
three metric cards and a test-information strip. No attached poster or example
image was available for a direct visual match.

## Files

- `model-performance-validation.png`: 4800 × 5100 pixels, 300 DPI.
- `model-performance-validation.svg`: editable vector version, with text retained.
- `create_infographic.py`: complete Matplotlib / NumPy / Pandas / Scikit-learn code.
- `plot_data/`: the real reduced holdout evidence and its provenance.
- `verified_metrics.json`: independently recomputed metrics and actual counts.

Use SVG for scalable poster placement. The native size is 40.64 × 43.18 cm.
Keep the aspect ratio and inspect an actual-size print before submission.
Use DejaVu Sans when editing the SVG to preserve the original layout.

## Scientific scope

These are the current **72-hour** results selected by the user:

- Model: `98bfd2a97aab`.
- Test base dates: November 6, 2025 – January 14, 2026.
- 259,860 overlapping grid-day windows, including 32,655 positives.
- ROC-AUC: 0.7800356902.
- Recall at threshold 0.50: 38.4229%.
- Precision at threshold 0.50: 39.4126%.
- Average precision: 0.3624660998.
- Prevalence: 12.5664%.
- TP = 12,547; FN = 20,108; FP = 19,288; TN = 207,917.

The repeated brief's AUC 0.779, recall 58.7%, precision 44.2%, and November 3–
January 10 period describe an older seven-day evaluation. They were replaced
with the selected three-day results; the older metrics must not be labeled
as 72-hour performance.

Actual counts replace the illustrative confusion matrix. Both curves are
calculated from every held-out label and probability. No synthetic curves,
invented prevalence, or illustrative observations are used. ROC-AUC is a ranking
metric, not percent classification accuracy. Positives represent recorded
satellite hotspots, not confirmed wildfire incidents. Newly downloaded source
files have not been used in this evaluation.

## Reproduce

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python create_infographic.py
```

The script needs only the included plotting evidence, not the original project
or a network connection. It checks the evidence checksum and verifies the
calculated metrics against the saved evaluation before drawing.
The project dashboard, trained model, and original evaluation remain unchanged.
