# BrCPT: conformal prediction after adaptive medical evidence acquisition


## Primary data and roles

MIMIC-CXR pleural-effusion detection is the primary task, with PA followed by a
lateral radiograph. One earliest locally complete study is selected per subject
before labels are inspected. The incomplete local mirror and every exclusion
are quantified in each manifest summary.

```text
train -> image classifier
tune  -> epoch and temperature selection
route -> class-conditional harmful-gap route calibration
cal   -> Hegazy effective-rank recalibration
test  -> one evaluation
```

Seed 7 selected the route; power-aware seeds 19/31/43/59 rejected it in
development. The allocation was `55/10/15/10/10` for
`train/tune/route/cal/test`. Confirmatory seeds 71, 83, 97, 109, and 127 remain
untouched by comparative evaluation and must not be run for this rejected
candidate.
See [docs/METHOD_FREEZE_CLASS_SAFE_ROUTE.md](docs/METHOD_FREEZE_CLASS_SAFE_ROUTE.md)
and [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md).
The active amendment is
[docs/PROTOCOL_AMENDMENT_POWER_ALLOCATION.md](docs/PROTOCOL_AMENDMENT_POWER_ALLOCATION.md).

## Reproduce one rejected development run

Create a persistent five-role manifest:

```bash
python prepare_mimic_cxr_manifest.py \
  --root /home/ubuntu/zmh/dataset/MIMIC-CXR \
  --output data_manifests/mimic_cxr_pleural_effusion_power_seed19.csv \
  --target "Pleural Effusion" --seed 19 \
  --train-ratio .55 --tune-ratio .10 --route-ratio .15 \
  --cal-ratio .10 --test-ratio .10
```

Train and extract frozen outputs:

```bash
python train_frozen_classifier.py \
  --manifest data_manifests/mimic_cxr_pleural_effusion_power_seed19.csv \
  --checkpoint revised_artifacts/mimic_cxr/pleural_effusion_power_seed19/efficientnet_b0_imagenet_anatomical/classifier.pt \
  --dataset mimic_cxr --normalization imagenet --backbone efficientnet_b0 \
  --seed 19 --image-size 320 --class-weight-power 0.5 \
  --patience 5 --max-epochs 40

python extract_feature_bundle.py \
  --manifest data_manifests/mimic_cxr_pleural_effusion_power_seed19.csv \
  --checkpoint revised_artifacts/mimic_cxr/pleural_effusion_power_seed19/efficientnet_b0_imagenet_anatomical/classifier.pt \
  --output revised_artifacts/mimic_cxr/pleural_effusion_power_seed19/efficientnet_b0_imagenet_anatomical/features.npz
```

Run only the declared published comparisons and the candidate:

```bash
python routed_evidence_experiment.py \
  --bundle revised_artifacts/mimic_cxr/pleural_effusion_power_seed19/efficientnet_b0_imagenet_anatomical/features.npz \
  --output-dir revised_results/mimic_cxr/development_power_classsafe/seed19 \
  --view-order PA LATERAL --critical-label 1 \
  --alpha 0.10 --set-score threshold --set-random-seed 2026 \
  --accuracy-gap 0.02 --class-accuracy-gap 0.02 \
  --cats-epsilon 0.01 --ltt-delta 0.05
```

The experiment writes three separate tables so unlike guarantees are not
ranked as though they were identical:

- `primary_classwise_set_comparison.csv`;
- `secondary_classwise_randomized_aps.csv`;
- `published_route_comparison.csv`.

The route table contains the published Jazbec UCB-WSR, Ringel marginal and
conditional, and CATs Shared/SM algorithms. The rejected route is labelled
`candidate`, never `baseline`; internal variants are excluded.

## Verification

```bash
python -m pytest -q tests
python -m compileall -q brcpt *.py
```

Use `python -m pytest`; the standalone executable in this environment may use a
different import path. Read [docs/METHOD_RESTART.md](docs/METHOD_RESTART.md)
before interpreting any historical result.
