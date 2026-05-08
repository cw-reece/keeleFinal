# QC Pack

Unzip into your repo root. It adds:
- docs/claims_and_sources.md
- docs/qc_plan.md
- scripts/eval_baseline_ckpt.py
- scripts/eval_fusion_run.py
- tools/qc_run_all.sh

## Install
unzip -o qc_pack.zip -d .

## Run QC
bash tools/qc_run_all.sh

Open:
- reports/qc_log.md

Then include `docs/claims_and_sources.md` + `reports/qc_log.md` as your “audit trail” when writing.
