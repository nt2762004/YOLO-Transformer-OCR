# Dự án OCR: Phát hiện và Nhận dạng chữ trong hóa đơn (YOLO + Transformer)

Dự án này bao gồm mã nguồn Python (Jupyter Notebook) để thực hiện quy trình nhận dạng ký tự quang học (OCR) cho hóa đơn, kết hợp giữa mô hình phát hiện đối tượng (Object Detection) và mô hình nhận dạng chuỗi (Sequence Recognition).

**NOTE**: Dataset đều được tổng hợp trên nhiều nguồn khác nhau (không thể chú thích nguồn do không nhớ :D). Tuy nhiên, trong dự án này chỉ dùng tới hóa đơn Việt Nam được lấy ở trên huggingface. Trong dataset gồm có:
- en_IDCard: Chứa các ảnh về thẻ định danh cá nhân tiếng Anh và thông tin từng ảnh.
- en_image: Chứa các ảnh về bảng hiểu tiếng Anh và thông tin từng ảnh.
- en_receipt: Chứa các ảnh về hóa đơn tiếng Anh và thông tin từng ảnh.
- vn_cccd: Chứa các ảnh về thẻ căn cước công dân Việt Nam được tổng hợp bằng cách lấy mẫu thẻ cccd và tạo script để điền thông tin random vào từng vùng dữ liệu trên thẻ (đã xử lý).
- vn_image: Chứa các ảnh về bảng hiệu tiếng Việt và thông tin từng ảnh.
- vn_receipt: Chứa các ảnh về hóa đơn tiếng Việt và thông tin từng ảnh. Đây là data được sử dụng cho dự án này (đã xử lý).

Dự án xử lý bài toán qua 2 giai đoạn chính:
1.  **Text Detection (Phát hiện vùng chữ):** Sử dụng mô hình **YOLOv8** để xác định vị trí các khung bao (bounding box) chứa văn bản trong ảnh hóa đơn.
2.  **Text Recognition (Nhận dạng nội dung):** Sử dụng kiến trúc **ResNet + Transformer** để chuyển đổi hình ảnh vùng chữ đã cắt thành văn bản (text).

## Cấu trúc Thư mục

```
├── Link Dataset.txt              # File chứa liên kết hoặc thông tin về bộ dữ liệu sử dụng
├── YOLO-Transformer-OCR.ipynb    # Notebook chính thực hiện toàn bộ quy trình từ Detection đến Recognition
└── README.md                     # File mô tả dự án
```

### 1. `YOLO-Transformer-OCR.ipynb` (End-to-End OCR Pipeline)
Notebook này chứa toàn bộ pipeline xử lý, từ cài đặt môi trường, tiền xử lý dữ liệu, huấn luyện mô hình đến chạy thử nghiệm (inference).

*   **Mục tiêu:** Xây dựng một hệ thống OCR hoàn chỉnh có khả năng đọc thông tin từ hóa đơn (tiếng Việt/Anh).
*   **Các bước chính:**
    *   **Cài đặt & Môi trường:**
        *   Cài đặt các thư viện cần thiết: `ultralytics` (cho YOLO), `jiwer` (để đánh giá lỗi), `transformers`, `albumentations`.
    *   **Text Detection (YOLOv8):**
        *   Sử dụng mô hình YOLOv8 để phát hiện vùng văn bản.
        *   Minh họa kết quả phát hiện (bounding boxes) trên ảnh mẫu.
        *   Hàm `visualize_bbox` giúp vẽ khung và điểm tin cậy lên ảnh.
    *   **Data Preprocessing (Chuẩn bị dữ liệu Recognition):**
        *   Cắt (Crop) các vùng văn bản từ ảnh gốc dựa trên nhãn hoặc kết quả detection.
        *   Lưu trữ dữ liệu dưới dạng file `.npy` để tối ưu tốc độ đọc ghi.
        *   Xây dựng `OCRRecognitionDataset` với kỹ thuật tăng cường dữ liệu (Augmentation) sử dụng thư viện `albumentations` (Resize, Pad, Noise, Blur...).
    *   **Mô hình Recognition (ResNet + Transformer):**
        *   **Encoder:** Sử dụng **ResNet18** (bỏ các lớp cuối) để trích xuất đặc trưng hình ảnh (Visual Features).
        *   **Decoder:** Sử dụng **Transformer Decoder** với Positional Encoding để sinh chuỗi ký tự từ đặc trưng hình ảnh.
        *   **Tokenizer:** Huấn luyện Tokenizer cấp độ ký tự (Character-level) trên tập dữ liệu text.
    *   **Huấn luyện (Training):**
        *   Định nghĩa hàm loss (`CrossEntropyLoss`) và optimizer (`Adam`).
        *   Vòng lặp huấn luyện với Teacher Forcing.
        *   Lưu checkpoint mô hình tốt nhất dựa trên loss của tập validation.
    *   **Đánh giá (Evaluation):**
        *   Sử dụng độ đo **CER (Character Error Rate)** và **WER (Word Error Rate)** để đánh giá độ chính xác của mô hình nhận dạng.
        *   Vẽ biểu đồ Loss qua các epoch.
    *   **Inference (Dự đoán thực tế):**
        *   Kết hợp 2 mô hình: Ảnh đầu vào -> YOLO (Detect) -> Crop -> Transformer (Recognize) -> Text kết quả.
        *   Hàm `recognize_text` thực hiện quy trình end-to-end.
        *   Trực quan hóa kết quả cuối cùng trực tiếp trên ảnh hóa đơn.

## Yêu cầu cài đặt

Để chạy notebook này, cần cài đặt các thư viện Python sau:

```bash
pip install torch torchvision ultralytics transformers albumentations jiwer opencv-python matplotlib tqdm
```
