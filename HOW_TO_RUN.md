# How To Run

Run all commands from the project root folder.

## Prerequisites

**Before starting, ensure:**

1. **Dataset structure** - Both datasets must exist:
   ```text
   data/en_receipt/
   ├── images/
   │   ├── train/      # English receipt training images
   │   └── valid/      # English receipt validation images
   ├── labels/
   │   ├── train/      # YOLO format labels (*.txt)
   │   └── valid/
   ├── train_transcripts.json   # {image_name: [{text, box/bbox}, ...]}
   └── valid_transcripts.json

   data/vn_receipt/
   ├── images/
   │   ├── train/      # Vietnamese receipt training images
   │   ├── val/        # Vietnamese receipt validation images
   │   └── test/       # (optional) Test images
   ├── labels/
   │   ├── train/      # YOLO format labels (*.txt)
   │   └── val/
   ├── train_transcripts.json
   └── val_transcripts.json
   ```

2. **Transcript JSON format** - Each file should contain:
   ```json
   {
     "image_name.jpg": [
       {"text": "recognized text", "box": [x1, y1, x2, y2]},
       {"text": "more text", "bbox": [x1, y1, x2, y2]}
     ]
   }
   ```
   Both `box` and `bbox` keys are supported.

3. **YOLO label format** - Each image must have a corresponding `.txt` file with:
   ```
   0 x_center y_center width height   # class_id=0 for text
   0 x_center y_center width height
   ```
   (Normalized coordinates 0-1, class_id always 0)

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Prepare data caches and combined YOLO dataset

This step builds recognition caches for both Vietnamese and English receipts, and prepares the combined detection dataset.

```bash
python main.py preprocess
```

Expected outputs:

```text
artifacts/cache/vn_receipt_train/
artifacts/cache/vn_receipt_val/
artifacts/cache/en_receipt_train/
artifacts/cache/en_receipt_val/
artifacts/combined_receipt/
data/combined_receipt.yaml
```

## 3. Train the character tokenizer

```bash
python main.py train-tokenizer
```

**What it does:** Builds a character-level tokenizer from all text in the transcripts.

**Output:**
```text
artifacts/tokenizer/char_tokenizer.json  # Character vocab and special tokens
```

**Note:** This tokenizer will be automatically loaded in the next step. You only need to run this once.

## 4. Train the OCR recognition model

```bash
python main.py train-recognition
```

Optional: specify custom epochs
```bash
python main.py train-recognition --epochs 50
```

**What it does:**
- Loads the tokenizer from step 3 (no need to retrain)
- Combines training data from both VN and EN receipts
- Trains ResNet18 encoder + Transformer decoder
- Saves best checkpoint based on validation loss

**Expected output:**
```text
artifacts/checkpoints/recognition_best.pt  # OCR model checkpoint
```

**Training will:** 
- Use early stopping (patience=5 epochs)
- Report CER (Character Error Rate) and WER (Word Error Rate) on validation set
- Combine both vn_receipt_train and en_receipt_train for training
- Use both vn_receipt_val and en_receipt_val for validation

## 5. Train the YOLO detector

```bash
python main.py train-detector --weights yolov8n.pt --epochs 50
```

**Parameters:**
- `--weights`: YOLOv8 model size (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)
- `--epochs`: Training epochs (default: 50)
- `--imgsz`: Input image size, default 640. Reduce to 416 or 320 if OOM
- `--batch`: Batch size, default 16. Reduce to 8 or 4 if OOM
- `--device`: Device ID (0=GPU, -1=CPU, default: 0)

**What it does:**
- Trains YOLO on combined VN + EN receipt detection dataset
- Optimizes for text region localization

**If you get Out of Memory (OOM) error:**

Option 1 - Reduce batch size to 8:
```bash
python main.py train-detector --batch 8
```

Option 2 - Reduce image size and batch:
```bash
python main.py train-detector --imgsz 416 --batch 8
```

Option 3 - Use CPU (slow but no GPU memory):
```bash
python main.py train-detector --device -1 --batch 4
```

Option 4 - All combined (most conservative):
```bash
python main.py train-detector --imgsz 320 --batch 4 --epochs 20
```

**Expected output:**
```text
artifacts/detector_runs/yolo_textdet/
├── weights/
│   ├── best.pt           # Best model (use this for inference!)
│   └── last.pt
├── results.csv
└── confusion_matrix.png
```

⚠️ **Important:** After training, the best weights are at:
```
artifacts/detector_runs/yolo_textdet/weights/best.pt
```

## 6. Run inference on a single receipt image

```bash
python main.py infer \
  --image path/to/receipt.jpg \
  --detector artifacts/detector_runs/yolo_textdet/weights/best.pt \
  --recognizer artifacts/checkpoints/recognition_best.pt
```

**Parameters:**
- `--image`: Path to receipt image (JPG, PNG, etc.)
- `--detector`: Path to trained YOLO weights (from step 5)
- `--recognizer`: Path to trained recognition checkpoint (from step 4)

**Output:**
For each detected text region:
```
{'text': 'recognized_text', 'confidence': 0.95, 'box': [x1, y1, x2, y2]}
```

## 7. Quick reference: Run complete pipeline

**From scratch (recommended):**

```bash
python main.py preprocess
python main.py train-tokenizer
python main.py train-recognition
python main.py train-detector --weights yolov8n.pt --epochs 50
python main.py infer --image path/to/receipt.jpg \
  --detector artifacts/detector_runs/yolo_textdet/weights/best.pt \
  --recognizer artifacts/checkpoints/recognition_best.pt
```

**If you have memory issues (OOM crash on Ubuntu):**

```bash
python main.py preprocess
python main.py train-tokenizer
python main.py train-recognition --epochs 20

# Use reduced settings for detector
python main.py train-detector --imgsz 416 --batch 8 --epochs 30

python main.py infer --image path/to/receipt.jpg \
  --detector artifacts/detector_runs/yolo_textdet/weights/best.pt \
  --recognizer artifacts/checkpoints/recognition_best.pt


python infer_with_viz.py --image data/vn_receipt/images/val/mcocr_public_145013cxtop.jpg --detector artifacts/detector_runs/yolo_textdet/weights/best.pt --recognizer artifacts/checkpoints/recognition_best.pt --output-dir results
```

**Resume training (if interrupted):**
- Preprocessing and tokenizer are idempotent (safe to rerun)
- Recognition/detector training will start from saved checkpoints if they exist
- Delete `artifacts/checkpoints/recognition_best.pt` to train from scratch

## Troubleshooting

### Preprocess fails
- **Check:** Do all transcript JSON files exist? (`data/*/train_transcripts.json`, etc.)
- **Check:** Do image directories exist? (`data/*/images/train/`, etc.)
- **Check:** Do YOLO label files exist in `data/*/labels/`?

### Recognition training is slow
- Reduce `batch_size` in [src/config.py](src/config.py) if out of memory
- Use GPU: ensure `torch.cuda.is_available()` returns True

### Detector training fails or crashes

**If you get Out of Memory error (OOM crash):**
```bash
# Option 1: Reduce batch size
python main.py train-detector --batch 8

# Option 2: Reduce image size + batch
python main.py train-detector --imgsz 416 --batch 8

# Option 3: Use CPU instead of GPU (slow but works)
python main.py train-detector --device -1 --batch 4

# Option 4: Conservative settings (slowest)
python main.py train-detector --imgsz 320 --batch 4 --epochs 20
```

Quality vs Memory tradeoff:
- **Full quality (default):** `imgsz=640 batch=16` - Best accuracy, needs GPU with 6GB+ VRAM
- **Medium:** `imgsz=416 batch=8` - Good accuracy, needs 4GB VRAM
- **Conservative:** `imgsz=320 batch=4` - Lower accuracy, needs 2GB VRAM
- **CPU mode:** `device=-1` - Works anywhere, very slow (~10x slower)

### Inference shows no results
- Lower confidence threshold in [src/detector.py](src/detector.py) line ~40 (`conf=0.25`)
- Check image path is valid and readable
- Verify both `--detector` and `--recognizer` paths exist

## 8. Run Streamlit Web App

After training, or whenever you have the detector and recognizer checkpoints locally, launch the interactive web interface:

**Option A: Local development**
```bash
conda activate ai_env
streamlit run streamlit_app.py
```

The app will:
1. Discover sample receipt images from `data/vn_receipt/` and `data/en_receipt/`
2. Use local model paths from the sidebar, with defaults pointing to `artifacts/detector_runs/yolo_textdet/weights/best.pt` and `artifacts/checkpoints/recognition_best.pt`
3. Open in browser at `http://localhost:8501`

**Features:**

- 📁 **Sample Library Tab**: 
  - Browse pre-loaded screenshots from VN validation set or EN test set
  - Dropdown to select receipt type
  - Click "Scan Receipt" to run OCR
  
- 📤 **Upload Receipt Tab**: 
  - Upload your own receipt image (JPG, PNG)
  - Click "Scan Receipt" to run OCR
  
- 📊 **Results**: 
  - Visualization with bounding boxes and detected text
  - Table with confidence scores
  - Download results as CSV, TXT, JSON, and annotated PNG

## 9. Deploy to Streamlit Community Cloud

For online deployment (free hosting on Streamlit Community Cloud):

1. **Prepare code**: See [DEPLOYMENT_STRUCTURE.md](DEPLOYMENT_STRUCTURE.md)
2. **Push to GitHub**: Create repo and push source code
3. **Deploy**: 
   - Go to https://share.streamlit.io
   - Connect GitHub repo
   - Select `streamlit_app.py` as entry point
   - App deploys automatically

⚠️ **Important**: Bundle the model files in the repository or provide external storage paths if you want the deployed app to run inference immediately.

For detailed deployment instructions, see [DEPLOYMENT_STRUCTURE.md](DEPLOYMENT_STRUCTURE.md)

## Project Details

- **Original notebook:** [notebooks/YOLO-Transformer-OCR.ipynb](notebooks/YOLO-Transformer-OCR.ipynb)
- **Config:** [src/config.py](src/config.py) - hyperparameters and paths
- **Data formats:** [data/README.md](data/README.md)
- **Combined training:** Both Vietnamese and English receipts are trained together
- **Architecture:** ResNet18 + Transformer (6-layer decoder)
- **Deployment guide:** [DEPLOYMENT_STRUCTURE.md](DEPLOYMENT_STRUCTURE.md)
