# OCR Receipt Explorer

OCR Receipt Explorer is an end-to-end optical character recognition (OCR) project for receipt images. It combines YOLOv8 text detection with a Transformer-based text recognition model to extract and digitize receipt content from Vietnamese and English receipts. The system features an interactive Streamlit app for browsing samples, uploading images, and viewing OCR results with visualization.

**Deployed Link**: https://your-streamlit-cloud-url.streamlit.app/ (Coming soon)

**Link Dataset & Models**: https://drive.google.com/file/d/1R-Fb23ysgbei2ienMpCcWlGX2KULF3Q8/view?usp=sharing

## Key Features

- **Text Detection**: YOLOv8-based text region detection for precise localization of receipt content.
- **Text Recognition**: Transformer decoder with ResNet18 encoder for accurate character recognition (8.53% CER).
- **Multi-language Support**: Trained on both Vietnamese and English receipt datasets.
- **Interactive Streamlit App**: Browse dataset samples, upload custom receipts, and visualize OCR results with bounding boxes.
- **Export Options**: Save results as CSV (coordinates + confidence), PNG (visualization), or TXT (text-only).
- **Unified CLI Pipeline**: Run preprocessing, training, evaluation, and inference from a single entry point.
- **Google Drive Deployment**: Automatically downloads models and datasets from Google Drive for cloud deployment.

## Language & Libraries

- Python 3.10+
- PyTorch (2.1.2)
- Ultralytics YOLOv8 (8.0.203)
- Streamlit (1.28.1)
- Transformers & Tokenizers
- OpenCV
- Pillow
- NumPy
- Matplotlib
- Albumentations (data augmentation)
- gdown (Google Drive downloads)
- tqdm

## Dataset

The project uses receipt image datasets with text detection labels and recognition transcripts:

```
data/
├── vn_receipt/                    # Vietnamese receipts
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   ├── train_transcripts.json     # Text annotations
│   └── val_transcripts.json
├── en_receipt/                    # English receipts
│   ├── images/
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   └── valid/
│   ├── train_transcripts.json
│   └── valid_transcripts.json
└── combined_receipt.yaml          # YOLO detection config
```

### Dataset Preprocessing

The preprocessing stage generates:
- **Numpy cache** of cropped text regions for fast training
- **Metadata CSV** with image paths and transcript information
- **Combined YOLO config** for detector training

## Methodology (Core AI)

The OCR pipeline is built in two stages:

### Stage 1: Text Detection
- **Model**: YOLOv8 nano/small variants
- **Task**: Locate bounding boxes around text regions in receipt images
- **Input**: Full receipt images
- **Output**: Text region coordinates (x1, y1, x2, y2)

### Stage 2: Text Recognition
- **Encoder**: ResNet18 (ImageNet pretrained, 18 layers)
  - Input: Cropped text regions (32×256 pixels, grayscale)
  - Output: Image features (512 dimensions)
- **Decoder**: Transformer (6 layers, 8 attention heads, 256-dim model)
  - Input: Image features + character sequence (autoregressive)
  - Output: Predicted character sequence
- **Vocabulary**: 219 characters (Vietnamese + English + special tokens)

### Training Pipeline

1. **Preprocessing**: Scan dataset, extract text regions, build vocabulary
2. **Tokenizer Training**: Create character tokenizer from training transcripts
3. **Recognition Training**: Train Transformer decoder with detected/labeled crops
4. **Detector Training**: Train YOLOv8 on combined dataset for text localization
5. **Inference**: Detect → Crop → Recognize → Aggregate results

### Inference Workflow

```
Input Receipt Image
        ↓
[YOLO Detector] → Text Bounding Boxes
        ↓
[Crop Regions] → Individual text patches
        ↓
[Transformer Recognizer] → Character sequences + Confidence scores
        ↓
[Aggregate Results] → Final OCR output with coordinates & confidence
```

## Evaluation & Results

The latest recognition model training was run over **28 epochs** on combined Vietnamese and English receipt datasets.

### Recognition Model Performance

| Metric | Value |
| --- | ---: |
| **Best Validation Loss** | 0.2598 (epoch 23) |
| **Final Training Loss** | 0.0759 |
| **Final Validation Loss** | 0.2635 |
| **Character Error Rate (CER)** | 8.53% |
| **Word Error Rate (WER)** | 21.23% |
| **Early Stopping** | Patience: 5 epochs |

### Performance Interpretation

- **CER 8.53%**: Approximately 1 character error per 12 characters recognized
- **WER 21.23%**: Approximately 1 word error per 5 words extracted
- **Training Convergence**: Best checkpoint at epoch 23, no severe overfitting detected
- **Model Size**: 64.8 MB (recognition_best.pt)

### Training Details

- **Model**: ResNet18 Encoder + Transformer Decoder (6 layers)
- **Optimizer**: Adam (lr=1e-4)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 32
- **Input Shape**: 32×256 grayscale images
- **Max Sequence Length**: 100 characters
- **Training Set**: vn_receipt_train + en_receipt_train (combined)
- **Validation Set**: vn_receipt_val + en_receipt_val (combined)

## Installation & Usage

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Local Development

**Option 1: Run Streamlit App Locally**

```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

**Option 2: Run Command-Line Inference**

```bash
python infer_with_viz.py --image data/vn_receipt/images/val/sample.jpg
```

Output will be saved to `results/sample_result.png`

### Training & Preprocessing Pipeline

For full pipeline execution, see [HOW_TO_RUN.md](HOW_TO_RUN.md):

```bash
# 1. Preprocess dataset (create cache, prepare detector config)
python main.py preprocess

# 2. Train character tokenizer
python main.py train-tokenizer

# 3. Train recognition model
python main.py train-recognition --epochs 30

# 4. Train text detector (YOLO)
python main.py train-detector --epochs 50

# 5. Run inference
python main.py infer --image path/to/receipt.jpg \
  --detector artifacts/detector_runs/yolo_textdet/weights/best.pt \
  --recognizer artifacts/checkpoints/recognition_best.pt
```

## Project Structure

```
OCR_Receipt/
├── data/                          # Dataset folders
│   ├── en_receipt/                # English receipt images & labels
│   ├── vn_receipt/                # Vietnamese receipt images & labels
│   ├── combined_receipt.yaml      # YOLO detection config
│   └── README.md                  # Data documentation
├── artifacts/                     # Runtime artifacts
│   ├── cache/                     # Numpy cache of cropped regions
│   ├── checkpoints/               # Model checkpoints
│   │   ├── recognition_best.pt    # Best recognition model
│   │   ├── training_history.json  # Epoch-by-epoch metrics
│   │   └── training_summary.txt   # Training summary
│   ├── detector_runs/             # YOLO training outputs
│   ├── tokenizer/                 # Character tokenizer
│   └── combined_receipt/          # Combined detection dataset
├── notebooks/                     # Jupyter tutorials
│   ├── YOLO-Transformer-OCR.ipynb
│   └── Inference-Visualization.ipynb
├── results/                       # Inference outputs
├── src/                           # Core source code
│   ├── __init__.py
│   ├── config.py                  # Paths & training config
│   ├── dataset.py                 # Recognition dataset loaders
│   ├── detector.py                # YOLO training/inference
│   ├── infer.py                   # End-to-end inference pipeline
│   ├── model.py                   # Encoder/Decoder architecture
│   ├── preprocess.py              # Data preprocessing & cache
│   ├── tokenizer.py               # Character tokenizer
│   ├── train.py                   # Recognition training loop
│   ├── utils.py                   # Shared helpers
│   └── visualize.py               # Visualization utilities
├── .streamlit/                    # Streamlit configuration
│   └── config.toml
├── main.py                        # CLI entry point
├── streamlit_app.py               # Interactive web app (local)
├── app_deployed.py                # Deployed app (uses Google Drive)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── HOW_TO_RUN.md                  # Detailed runbook
├── DEPLOY_WITH_GOOGLE_DRIVE.md    # Deployment guide
└── DEPLOYMENT_STRUCTURE.md        # Architecture notes
```

### Core Modules

- **main.py**: CLI entry point for preprocessing, training, evaluation, and prediction
- **streamlit_app.py**: Interactive Streamlit interface (local development)
- **app_deployed.py**: Cloud-ready app with Google Drive integration
- **src/config.py**: Project paths and configuration
- **src/preprocess.py**: Dataset scanning, metadata generation, and cache building
- **src/infer.py**: End-to-end inference pipeline
- **src/model.py**: ResNet18 Encoder + Transformer Decoder architecture
- **src/train.py**: Recognition model training routine
- **src/detector.py**: YOLOv8 training and inference utilities
- **src/tokenizer.py**: Character-level tokenizer
- **src/dataset.py**: PyTorch dataset and dataloader implementations

## Quick Start

1. **Clone repository and install**
   ```bash
   git clone <repo-url>
   cd OCR_Receipt
   pip install -r requirements.txt
   ```

2. **Download data and models** (or prepare your own)
   ```bash
   # Place your receipt dataset in data/vn_receipt/ and data/en_receipt/
   ```

3. **Run Streamlit app**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Upload a receipt image** and watch the OCR pipeline in action!

## Additional Resources

- [HOW_TO_RUN.md](HOW_TO_RUN.md) - Detailed command references
- `notebooks/` - Tutorial notebooks with step-by-step examples

## 🌐 Streamlit Web App

After training, run an interactive web app to test OCR on receipt images:

```bash
conda activate ai_env
streamlit run streamlit_app.py
```

Features:
- 📁 **Sample Library**: Test on pre-loaded Vietnamese or English receipts
- 📤 **Upload Custom**: Upload your own receipt image
- 📊 **Live Visualization**: See bounding boxes and recognized text


## Notes

- The original notebook is kept in `notebooks/` as a reference and tutorial version.
- Generated caches, checkpoints, and training logs should stay inside `artifacts/` and should not be pushed to GitHub.
- The project is designed so you can train again from zero without editing the notebook.
- The Streamlit app uses local checkpoints by default: `artifacts/detector_runs/yolo_textdet/weights/best.pt` and `artifacts/checkpoints/recognition_best.pt`.

