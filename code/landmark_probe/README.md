# Landmark Probe

Frozen feature extraction, MLP landmark probing, and aggregation for within-dataset
LM-Celeb/LM-CFD tasks and Celeb-to-CFD transfer. Use the public scripts from the
repository root:

```bash
python scripts/prepare_landmark_dataset.py --cfg configs/landmarks/prepare_celeb_cfd.yaml
python scripts/run_landmark_probe.py --cfg configs/landmarks/probe_within_and_transfer.yaml
```

Preparation uses `configs/landmarks/prepare_celeb_cfd.yaml`. Probing uses
`configs/landmarks/probe_within_and_transfer.yaml` and `configs/landmarks/mlp_default.yaml`.
The probe entrypoint runs extraction, probing, and aggregation; its `--help` lists
flags for skipping stages. No separate development scripts are required.

See [the main README](../../README.md#landmark-probing) and
[the quickstart](../../QUICKSTART.md) for source data, environment variables,
checkpoint prerequisites, and output locations.
