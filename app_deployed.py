"""
OCR Receipt Web App - Streamlit Interface
Supports Vietnamese and English receipt OCR with text detection and recognition.
"""

from __future__ import annotations

# Monkeypatch cv2 to avoid headless display errors
import sys
from unittest.mock import MagicMock

def mock_cv2_bootstrap():
    """Bypass OpenCV's bootstrap that requires display."""
    pass

# Pre-import cv2-headless and mock the bootstrap
try:
    import cv2 as _cv2_module
    _cv2_module.bootstrap = mock_cv2_bootstrap
except:
    pass

import io
import zipfile
from pathlib import Path

import gdown
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.config import get_project_paths, get_training_config
from src.detector import crop_boxes, detect_text_regions
from src.infer import load_recognition_checkpoint, recognize_text
from src.model import OCRModel
from src.tokenizer import CharTokenizer
from src.utils import ensure_dir, load_json
from ultralytics import YOLO


# ============================================================================
# Page Config
# ============================================================================
st.set_page_config(
    page_title="OCR Receipt Explorer",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-10px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 4px 12px rgba(31, 119, 180, 0.3); }
        50% { box-shadow: 0 4px 20px rgba(31, 119, 180, 0.5); }
    }
    
    .tab-active {
        background: linear-gradient(135deg, #1f77b4 0%, #0d47a1 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        font-size: 16px;
        box-shadow: 0 4px 15px rgba(31, 119, 180, 0.4);
        animation: slideIn 0.3s ease-out, pulse 2s ease-in-out infinite;
        transition: all 0.3s ease;
        border: 2px solid #1565c0;
    }
    
    .metric-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .result-text {
        font-family: monospace;
        background-color: #f9f9f9;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #0066cc;
    }
    .confidence-high {color: #00b386;}
    .confidence-medium {color: #ff9800;}
    .confidence-low {color: #d32f2f;}
    
    .app-header {
        text-align: center;
        padding: 20px 0;
        margin-bottom: 30px;
        border-bottom: 3px solid #1f77b4;
    }
    
    .app-header h1 {
        color: #1f77b4;
        font-size: 2.5em;
        margin: 0;
        font-weight: bold;
    }
    
    .app-header p {
        color: #666;
        font-size: 1.1em;
        margin: 10px 0 0 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# App Header
# ============================================================================
st.markdown(
    """
    <div class='app-header'>
        <h1>📋 OCR Receipt Explorer</h1>
        <p>Text Detection & Recognition for Vietnamese & English Receipts</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Google Drive Download Configuration
# ============================================================================
# Get File ID from Google Drive sharing link:
# https://drive.google.com/file/d/FILE_ID_HERE/view
#                              ^^^^^^^^^^^^^^
GDRIVE_CONFIG = {
    # ZIP file ID containing: best.pt, recognition_best.pt, char_tokenizer.json, en_receipt/, vn_receipt/
    "zip_file_id": "1hj8DNQXyarTqFHp0Pq12kex0XUtDQtd2",  # Replace with your Data_Models_OCR.zip file ID
}


@st.cache_resource
def download_from_gdrive():
    """Download models and datasets from Google Drive on first run."""
    import os
    import zipfile
    import shutil
    
    base_dir = Path("/tmp/ocr_receipt")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # File ID của Data_Models_OCR.zip (lấy từ Google Drive sharing link)
    zip_file_id = GDRIVE_CONFIG.get("zip_file_id", None)
    
    status_info = st.empty()
    
    try:
        if zip_file_id and zip_file_id != "YOUR_ZIP_FILE_ID":
            # Check if already extracted
            if not (base_dir / "vn_receipt").exists() or not (base_dir / "best.pt").exists():
                status_info.info("📦 Downloading from Google Drive (this may take 5-10 minutes)...")
                
                # Download zip file
                zip_path = base_dir / "data_models.zip"
                try:
                    import urllib.request
                    import socket
                    
                    socket.setdefaulttimeout(300)  # 5 minute timeout
                    
                    # Try gdown first
                    status_info.info("📥 Attempting download with gdown...")
                    result = gdown.download(
                        url=f"https://drive.google.com/uc?id={zip_file_id}",
                        output=str(zip_path),
                        quiet=False
                    )
                    
                    # If gdown fails or file too small, try urllib (direct method)
                    if result is None or not zip_path.exists() or zip_path.stat().st_size < 1_000_000:
                        if zip_path.exists():
                            zip_path.unlink()  # Remove incomplete file
                        
                        status_info.info("📥 gdown failed, trying direct urllib download...")
                        # Use confirm=t to bypass Google Drive's warning dialog
                        url = f"https://drive.google.com/uc?id={zip_file_id}&confirm=t"
                        
                        def download_with_retry(url, output_path, max_retries=3):
                            for attempt in range(max_retries):
                                try:
                                    status_info.info(f"📥 Download attempt {attempt+1}/{max_retries}...")
                                    urllib.request.urlretrieve(url, str(output_path))
                                    return True
                                except Exception as e:
                                    status_info.warning(f"⚠️ Attempt {attempt+1} failed: {str(e)}")
                                    if output_path.exists():
                                        output_path.unlink()
                                    if attempt == max_retries - 1:
                                        raise
                            return False
                        
                        download_with_retry(url, zip_path)
                    
                    # Validate downloaded file
                    if not zip_path.exists():
                        raise FileNotFoundError(f"Downloaded file not found at {zip_path}")
                    
                    file_size = zip_path.stat().st_size
                    status_info.info(f"📦 Downloaded {file_size / 1_000_000:.1f} MB")
                    
                    if file_size < 1_000_000:
                        raise ValueError(f"File too small ({file_size} bytes) - likely incomplete download")
                    
                    
                    status_info.info("📦 Extracting files...")
                    
                    # Extract zip with validation
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            # Validate zip file integrity
                            corrupt_file = zip_ref.testzip()
                            if corrupt_file:
                                raise zipfile.BadZipFile(f"Corrupted file in zip: {corrupt_file}")
                            
                            zip_ref.extractall(base_dir)
                            status_info.success("✅ Extraction complete")
                    except zipfile.BadZipFile as zip_error:
                        zip_path.unlink()  # Remove corrupted file
                        raise Exception(f"Corrupted zip file - download may be incomplete. Error: {str(zip_error)}")
                    
                    # Remove zip after extraction
                    zip_path.unlink()
                    
                    status_info.success("✅ All files ready (models + datasets)")
                
                except Exception as gdown_error:
                    status_info.error(f"❌ Google Drive download failed: {str(gdown_error)}")
                    st.error(f"""
                    **Download Error Details:**
                    - File ID: {zip_file_id}
                    - Error: {str(gdown_error)}
                    
                    **Solutions:**
                    1. Check if File ID is correct
                    2. Make sure the Google Drive file is publicly accessible
                    3. Try again in a few moments
                    """)
                    return None, None
        else:
            status_info.error("❌ Please configure GDRIVE_CONFIG with your zip file ID")
            status_info.info("Get ID from: https://drive.google.com/file/d/YOUR_FILE_ID/view")
            return None, None
        
        status_info.empty()
        return str(base_dir), str(base_dir)
    
    except Exception as e:
        status_info.error(f"❌ Unexpected error: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None


# ============================================================================
# Cache Resource Declaration
# ============================================================================
@st.cache_resource
def load_models(artifacts_dir: str = None, device: str = "cpu"):
    """Load detector and recognizer models at startup."""
    
    # Use downloaded artifacts or local paths
    if artifacts_dir:
        # Models are directly in base directory from Google Drive
        detector_path = Path(artifacts_dir) / "best.pt"
        recognizer_path = Path(artifacts_dir) / "recognition_best.pt"
    else:
        paths = get_project_paths()
        detector_path = paths.artifacts_dir / "detector_runs/yolo_textdet/weights/best.pt"
        recognizer_path = paths.checkpoints_dir / "recognition_best.pt"
    
    try:
        # Load detector
        if not detector_path.exists():
            st.warning(f"⚠️ Detector not found at {detector_path}")
            detector = None
        else:
            detector = YOLO(str(detector_path))
            detector.to(device)  # Move to device
            st.sidebar.success("✅ Detector loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading detector: {e}")
        detector = None
    
    try:
        # Load recognizer on the target device
        if not recognizer_path.exists():
            st.warning(f"⚠️ Recognizer not found at {recognizer_path}")
            recognizer = None
            tokenizer = None
        else:
            recognizer, tokenizer = load_recognition_checkpoint(recognizer_path, device=device)
            recognizer.to(device)  # Ensure on correct device
            st.sidebar.success("✅ Recognizer loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading recognizer: {e}")
        recognizer = None
        tokenizer = None
    
    return detector, recognizer, tokenizer


# ============================================================================
# Helper Functions
# ============================================================================
def get_device():
    """Detect available device (CPU/CUDA)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def export_results_to_csv(results: list[dict], image_name: str = "image") -> str:
    """Export detection results to CSV format."""
    csv_content = "text,x1,y1,x2,y2,confidence\n"
    for result in results:
        box = result["box"]
        text = result["text"].replace(",", ";")  # Escape commas
        conf = result["confidence"]
        csv_content += f'"{text}",{box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f},{conf:.4f}\n'
    return csv_content


def export_results_to_txt(results: list[dict]) -> str:
    """Export detected text to TXT format (text-only)."""
    texts = [result["text"] for result in results]
    return "\n".join(texts)


def visualize_results_matplotlib(image: Image.Image, results: list[dict]):
    """Visualize OCR results with bounding boxes using matplotlib.
    
    Args:
        image: PIL Image object
        results: List of OCR results from recognize_text()
    
    Returns:
        matplotlib figure object
    """
    image_rgb = image.convert('RGB')
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(image_rgb)
    
    # Draw bounding boxes and text
    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))
    
    for i, item in enumerate(results):
        box = item['box']
        text = item['text']
        confidence = item['confidence']
        
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw rectangle
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2,
            edgecolor=colors[i % len(colors)],
            facecolor='none',
            alpha=0.7
        )
        ax.add_patch(rect)
        
        # Add text label
        label = f"{text} ({confidence:.2f})"
        ax.text(
            x1, y1 - 5,
            label,
            fontsize=9,
            color=colors[i % len(colors)],
            bbox=dict(facecolor='white', alpha=0.7, pad=2),
            verticalalignment='bottom'
        )
    
    ax.set_title(f"OCR Results - {len(results)} text regions detected", fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    
    return fig


def get_confidence_color(confidence: float) -> str:
    """Return HTML class for confidence color."""
    if confidence >= 0.7:
        return "confidence-high"
    elif confidence >= 0.4:
        return "confidence-medium"
    else:
        return "confidence-low"


def get_sample_images(dataset_type: str = "vn", split: str = "train", limit: int = 9, data_dir: str = None):
    """Get sample images from dataset."""
    
    if data_dir:
        base = Path(data_dir)
        if dataset_type == "vn":
            if split == "train":
                image_dir = base / "vn_receipt/images/train"
            else:
                image_dir = base / "vn_receipt/images/val"
        else:  # en
            if split == "train":
                image_dir = base / "en_receipt/images/train"
            elif split == "valid":
                image_dir = base / "en_receipt/images/val"
            else:
                image_dir = base / "en_receipt/images/test"
    else:
        paths = get_project_paths()
        if dataset_type == "vn":
            if split == "train":
                image_dir = paths.vn_receipt_train_images
            else:
                image_dir = paths.vn_receipt_val_images
        else:
            if split == "train":
                image_dir = paths.en_receipt_train_images
            elif split == "valid":
                image_dir = paths.en_receipt_val_images
            else:
                image_dir = paths.en_receipt_test_images
    
    if not image_dir.exists():
        return []
    
    image_files = sorted([f for f in image_dir.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    return image_files[:limit]


def recognize_single_crop(crop: Image.Image, recognizer, tokenizer, device: str) -> tuple[str, float]:
    """Recognize text from a single crop."""
    from src.infer import prepare_crop_transform
    
    transform = prepare_crop_transform()
    crop_array = np.array(crop)
    if crop_array.ndim == 2:
        crop_array = np.repeat(crop_array[:, :, None], 3, axis=2)
    
    augmented = transform(image=crop_array)
    tensor = augmented["image"].unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        memory = recognizer.encoder(tensor)
        seq = torch.full((1, 1), tokenizer.bos_token_id, dtype=torch.long, device=device)
        
        confidences = []
        for _ in range(100):
            logits = recognizer.decoder(seq, memory)
            probs = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
            next_token = probs.argmax(dim=-1, keepdim=True)
            confidences.append(float(probs[0, next_token.item()].item()))
            seq = torch.cat([seq, next_token], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
        
        text = tokenizer.decode(seq.squeeze(0).tolist())
        conf = np.exp(np.mean(np.log(np.array(confidences) + 1e-12)))
        
        return text, conf


# ============================================================================
# Main App
# ============================================================================
def main():
    # Download models and data from Google Drive if configured
    artifacts_dir, data_dir = download_from_gdrive()
    
    # Store in session state for access in other functions
    st.session_state.artifacts_dir = artifacts_dir
    st.session_state.data_dir = data_dir
    
    paths = get_project_paths()
    config = get_training_config()
    device = get_device()
    
    # ========================================================================
    # Sidebar Configuration
    # ========================================================================
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Confidence threshold
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Only show detections above this threshold",
        )
        
        st.divider()
        st.markdown("### 📊 Models Info")
        
        # Detector model
        st.markdown("**🔍 Detector:** YOLOv8 (best.pt)")
        
        # Recognizer model
        st.markdown("**📝 Recognizer:** Transformer OCR (recognition_best.pt)")
    
    # ========================================================================
    # Load models
    # ========================================================================
    with st.spinner("Loading models..."):
        detector, recognizer, tokenizer = load_models(artifacts_dir=artifacts_dir, device=device)
    
    if detector is None or recognizer is None or tokenizer is None:
        st.error("⚠️ Failed to load one or more models. Please check model configuration.")
        return
    
    # Initialize session state for tab navigation
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = 0  # Default: Browse Samples

    # ========================================================================
    # Tab Navigation
    # ========================================================================
    col_tabs = st.columns(3)
    with col_tabs[0]:
        if st.session_state.active_tab == 0:
            st.markdown("<div class='tab-active'>🖼️ Browse Samples</div>", unsafe_allow_html=True)
        else:
            if st.button("🖼️ Browse Samples", use_container_width=True, key="tab_browse"):
                st.session_state.active_tab = 0
                st.rerun()
    with col_tabs[1]:
        if st.session_state.active_tab == 1:
            st.markdown("<div class='tab-active'>📤 Upload Image</div>", unsafe_allow_html=True)
        else:
            if st.button("📤 Upload Image", use_container_width=True, key="tab_upload"):
                st.session_state.active_tab = 1
                st.rerun()
    with col_tabs[2]:
        if st.session_state.active_tab == 2:
            st.markdown("<div class='tab-active'>📊 Results</div>", unsafe_allow_html=True)
        else:
            if st.button("📊 Results", use_container_width=True, key="tab_results"):
                st.session_state.active_tab = 2
                st.rerun()

    st.divider()

    # ========================================================================
    # TAB 2: Upload Image
    # ========================================================================
    if st.session_state.active_tab == 1:
            st.header("📤 Upload & Process Receipt")
        
            col1, col2 = st.columns([1.5, 1])
        
            with col1:
                uploaded_file = st.file_uploader(
                    "Choose a receipt image",
                    type=["jpg", "jpeg", "png"],
                )
                if uploaded_file is not None:
                    if st.button("▶️ Run OCR", type="primary", use_container_width=True):
                        with st.spinner("🔍 Running OCR..."):
                            try:
                                image = Image.open(uploaded_file).convert("RGB")
                            
                                # Detection
                                st.info("🔍 Detecting text regions...")
                                boxes = detect_text_regions(detector, image, device=device, conf=conf_threshold)
                            
                                # Recognition
                                st.info(f"📝 Recognizing {len(boxes)} regions...")
                                results = []
                                for box_idx, box in enumerate(boxes):
                                    crops = crop_boxes(image, [box])
                                    crop = crops[0]
                                
                                    text, conf = recognize_single_crop(crop, recognizer, tokenizer, device)
                                
                                    results.append({
                                        "box": box.tolist() if hasattr(box, "tolist") else list(box),
                                        "text": text,
                                        "confidence": conf,
                                    })
                                st.success(f"✅ OCR Complete! Found {len(results)} text regions")
                                st.session_state.active_tab = 2  # Switch to Results tab
                                st.session_state.last_image = image
                                st.session_state.last_results = results
                                st.rerun()
                            
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                                import traceback
                                st.error(traceback.format_exc())
        
            with col2:
                if uploaded_file is not None:
                    image = Image.open(uploaded_file).convert("RGB")
                    st.image(image, caption="Preview", width="stretch")

    # ========================================================================
    # TAB 1: Browse Samples
    # ========================================================================
    if st.session_state.active_tab == 0:
            st.header("🖼️ Browse Dataset Samples")
        
            col1, col2 = st.columns(2)
        
            with col1:
                dataset = st.radio(
                    "Select Dataset",
                    ["Vietnamese", "English"],
                    horizontal=True,
                )
        
            with col2:
                if dataset == "Vietnamese":
                    split = st.radio("Select Split", ["train", "valid"], horizontal=True)
                else:
                    split = st.radio("Select Split", ["train", "valid", "test"], horizontal=True)
        
            # Initialize session state for samples count
            if "samples_count" not in st.session_state:
                st.session_state.samples_count = 5
            if "current_dataset" not in st.session_state:
                st.session_state.current_dataset = dataset
            if "current_split" not in st.session_state:
                st.session_state.current_split = split
        
            # Reset samples_count if dataset or split changes
            if dataset != st.session_state.current_dataset or split != st.session_state.current_split:
                st.session_state.samples_count = 5
                st.session_state.current_dataset = dataset
                st.session_state.current_split = split
        
            # Get sample images
            dataset_type = "vn" if dataset == "Vietnamese" else "en"
            samples = get_sample_images(dataset_type=dataset_type, split=split, limit=st.session_state.samples_count, data_dir=st.session_state.data_dir)
        
            if not samples:
                st.warning(f"No images found in {dataset} {split} split")
            else:
                st.info(f"Showing {len(samples)} sample images")
            
                # Display gallery with 5 columns
                cols = st.columns(5)
                for idx, image_path in enumerate(samples):
                    with cols[idx % 5]:
                        try:
                            sample_image = Image.open(image_path).convert("RGB")
                            st.image(sample_image, caption=image_path.name, width="stretch")
                        
                            if st.button("🔍 Process", key=f"sample_{idx}", use_container_width=True):
                                with st.spinner(f"Processing {image_path.name}..."):
                                    try:
                                        boxes = detect_text_regions(detector, sample_image, device=device, conf=conf_threshold)
                                    
                                        results = []
                                        for box in boxes:
                                            crops = crop_boxes(sample_image, [box])
                                            crop = crops[0]
                                        
                                            text, conf = recognize_single_crop(crop, recognizer, tokenizer, device)
                                        
                                            results.append({
                                                "box": box.tolist() if hasattr(box, "tolist") else list(box),
                                                "text": text,
                                                "confidence": conf,
                                            })
                                    
                                        st.success(f"✅ Processed! Found {len(results)} regions")
                                        st.session_state.active_tab = 2  # Switch to Results tab
                                        st.session_state.last_image = sample_image
                                        st.session_state.last_results = results
                                        st.rerun()
                                
                                    except Exception as e:
                                        st.error(f"❌ Error: {str(e)}")
                    
                        except Exception as e:
                            st.warning(f"Error loading image: {e}")
            
                st.divider()
            
                # Show more button (only if there are more samples available)
                all_samples = get_sample_images(dataset_type=dataset_type, split=split, limit=999, data_dir=st.session_state.data_dir)
                if len(samples) < len(all_samples):
                    if st.button("Show More", use_container_width=True, type="secondary"):
                        st.session_state.samples_count += 10
                        st.rerun()
                else:
                    st.info("✅ All samples displayed")

    # ========================================================================
    # TAB 3: Results
    # ========================================================================
    if st.session_state.active_tab == 2:
            st.header("📊 OCR Results")
        
            if "last_results" not in st.session_state or not st.session_state.last_results:
                st.info("⏳ No results yet. Process an image in the Upload or Browse tabs.")
            else:
                results = st.session_state.last_results
                image = st.session_state.last_image
            
                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📍 Regions Detected", len(results))
                with col2:
                    avg_conf = np.mean([r["confidence"] for r in results]) if results else 0
                    st.metric("📊 Avg Confidence", f"{avg_conf:.2%}")
                with col3:
                    total_chars = sum(len(r["text"]) for r in results)
                    st.metric("🔤 Total Characters", total_chars)
                with col4:
                    st.metric("✅ Non-empty", sum(1 for r in results if r["text"].strip()))
            
                st.divider()
            
                # Layout: Visualization left, Details right
                left_col, right_col = st.columns([1, 1.2])
            
                # LEFT COLUMN: Visualization
                with left_col:
                    st.subheader("🖼️ Visualization")
                    if image is not None:
                        fig = visualize_results_matplotlib(image, results)
                        st.pyplot(fig, use_container_width=True)
            
                # RIGHT COLUMN: Details and Export
                with right_col:
                    st.subheader("📋 Detailed Results")
                
                    # Results table
                    result_data = []
                    for idx, result in enumerate(results, 1):
                        box = result["box"]
                        result_data.append({
                            "#": idx,
                            "Text": result["text"],
                            "Confidence": f"{result['confidence']:.2%}",
                            "X1": f"{box[0]:.1f}",
                            "Y1": f"{box[1]:.1f}",
                            "X2": f"{box[2]:.1f}",
                            "Y2": f"{box[3]:.1f}",
                        })
                
                    st.dataframe(result_data, width="stretch", hide_index=True, height=400)
                
                    st.divider()
                
                    # Export options
                    st.subheader("⬇️ Export")
                
                    col1, col2, col3 = st.columns(3)
                
                    with col1:
                        csv_data = export_results_to_csv(results)
                        st.download_button(
                            label="📥 CSV",
                            data=csv_data,
                            file_name="ocr_results.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                
                    with col2:
                        txt_data = export_results_to_txt(results)
                        st.download_button(
                            label="📝 TXT",
                            data=txt_data,
                            file_name="ocr_results.txt",
                            mime="text/plain",
                            use_container_width=True,
                        )
                
                    with col3:
                        # Export visualization
                        if image is not None:
                            fig = visualize_results_matplotlib(image, results)
                            buf = io.BytesIO()
                            fig.savefig(buf, format="PNG", dpi=150, bbox_inches='tight')
                            buf.seek(0)
                            plt.close(fig)  # Close figure to free memory
                            st.download_button(
                                label="🖼️ PNG",
                                data=buf,
                                file_name="ocr_visualization.png",
                                mime="image/png",
                                use_container_width=True,
                            )
    


if __name__ == "__main__":
    # Initialize session state
    if "last_results" not in st.session_state:
        st.session_state.last_results = []
    if "last_image" not in st.session_state:
        st.session_state.last_image = None
    
    main()
