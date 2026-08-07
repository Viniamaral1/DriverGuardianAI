# Guardian OS V7.2.2 — Evaluation Target Contract Fix

## Audit finding

The active model was trained with:

- target column: `fatigue_level`
- `Alert` = 0
- `Mild Fatigue` = 1
- `Moderate Fatigue` = 1

The V7.2 evaluator previously selected `state` first and measured `normal`
versus `drowsy`. That is a different research question and produced the
incorrect reproduced result of about 58.6% balanced accuracy.

## Correct reproduced evidence

Using the model's own saved target mapping and the supplied held-out test split:

- rows: 6,940
- threshold: 0.640
- accuracy: 95.288%
- balanced accuracy: 95.960%
- precision: 99.973%
- recall: 91.955%
- F1: 95.796%
- ROC-AUC: 96.071%
- confusion matrix: TN 2,887; FP 1; FN 326; TP 3,726

These reproduce the metrics stored inside the active model bundle.

Condition balanced accuracy:

- glasses: 95.060%
- none: 97.284%

## Fixed

- The evaluator reads the target mapping stored in the model bundle.
- `fatigue_level` is preferred over the collection-state column.
- The calibration split uses the same target contract as the test split.
- The page displays the selected target column.
- Reproduced metrics are compared with the saved historical metrics.
- Old V7.2 evaluations based on `state` are not automatically restored.
- The existing strict JSON/NaN compatibility fix remains included.

## Unchanged

The trained model, live camera, personal calibration, temporal decision logic,
profiles, alerts, Commander, reports, weather, environmental perception and
Edge Memory are unchanged.
