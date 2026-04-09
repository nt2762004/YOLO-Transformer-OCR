# Data Layout

This project uses both the Vietnamese and English receipt datasets stored in `data/vn_receipt/` and `data/en_receipt/`.

## Expected folders

- `data/vn_receipt/images/train`, `data/vn_receipt/images/val`, `data/vn_receipt/images/test`
- `data/vn_receipt/labels/train`, `data/vn_receipt/labels/val`
- `data/vn_receipt/train_transcripts.json`, `data/vn_receipt/val_transcripts.json`
- `data/en_receipt/images/train`, `data/en_receipt/images/valid`
- `data/en_receipt/labels/train`, `data/en_receipt/labels/valid`
- `data/en_receipt/train_transcripts.json`, `data/en_receipt/valid_transcripts.json`

## Notes

- Keep raw data here.
- Generated caches should be written to `artifacts/cache/` by the preprocessing script.
- The preprocessing step also prepares a combined YOLO dataset under `artifacts/combined_receipt/`.
- If you add more splits or datasets, extend `src/config.py` and rerun preprocessing.
