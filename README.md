# DeepSeek-OCR Fine-tuning

## Giới thiệu

Đồ án fine-tuning mô hình **DeepSeek-OCR** cho tác vụ nhận dạng chữ viết tay tiếng Việt (Vietnamese Handwriting Recognition) sử dụng bộ dữ liệu **UIT-HWDB**.

Mô hình được fine-tune bằng kỹ thuật **LoRA (Low-Rank Adaptation)** với thư viện **Unsloth** để tối ưu hóa hiệu năng training trên GPU hạn chế.

## Mục tiêu

- Fine-tune mô hình DeepSeek-OCR trên dữ liệu chữ viết tay tiếng Việt
- Đánh giá và so sánh hiệu năng giữa mô hình gốc (Baseline) và mô hình Fine-tuned
- Phân tích chi tiết các loại lỗi OCR: Insertion, Deletion, Substitution

## Kết quả

### Cải thiện CER (Character Error Rate)

| Metric | Baseline | Fine-tuned | Cải thiện |
|--------|----------|------------|-----------|
| Mean CER | ~40% | ~12% | **~70% ↓** |
| Median CER | ~33% | ~8% | **~75% ↓** |

### Tỉ lệ khớp hoàn hảo (Perfect Match Rate)

- **Baseline**: ~0% mẫu có CER = 0%
- **Fine-tuned**: ~25% mẫu có CER = 0%

## 🗂️ Cấu trúc thư mục

```
src/
├── kaggle_notebook/
│   └── deepseek-ocr-fine-tuning.ipynb    # Notebook training trên Kaggle
├── results/
│   ├── baseline_evaluation.json           # Kết quả đánh giá mô hình gốc
│   ├── finetuned_evaluation.json          # Kết quả đánh giá mô hình fine-tuned
│   ├── deepseek-ocr_logs.txt              # Log quá trình training
│   └── outputs/
│       ├── checkpoint-400/                # Checkpoint tại step 400
│       └── checkpoint-534/                # Checkpoint cuối cùng (1 epoch)
│           ├── adapter_config.json        # Cấu hình LoRA adapter
│           ├── adapter_model.safetensors  # Trọng số LoRA adapter
│           ├── tokenizer.json             # Tokenizer
│           └── trainer_state.json         # Trạng thái training
├── visualizations/                        # Thư mục lưu biểu đồ phân tích
├── analyze_results.py                     # Script phân tích kết quả
└── README.md                              # File này
```

## Cấu hình Fine-tuning

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (r) | 16 |
| Alpha | 16 |
| Dropout | 0 |
| Bias | none |
| Task Type | CAUSAL_LM |

### Target Modules

```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```

### Training Configuration

- **Base Model**: `unsloth/DeepSeek-OCR`
- **Quantization**: 4-bit (QLoRA)
- **Epochs**: 1
- **Total Steps**: 534
- **Hardware**: Tesla T4 GPU (2x)

## Dataset

**UIT-HWDB** (Vietnamese Handwriting Database):
- `UIT_HWDB_line`: Dữ liệu dòng chữ
- `UIT_HWDB_paragraph`: Dữ liệu đoạn văn
- `UIT_HWDB_word`: Dữ liệu từ đơn

### Data Split
- **Training**: ~10,000+ samples từ nhiều writer khác nhau
- **Testing**: 200+ samples

##  Hướng dẫn sử dụng

### 1. Cài đặt Dependencies

```bash
pip install unsloth
pip install transformers==4.56.2
pip install trl==0.22.2
pip install jiwer
pip install einops addict easydict
```

### 2. Training trên Kaggle

Sử dụng notebook `kaggle_notebook/deepseek-ocr-fine-tuning.ipynb`:

1. Upload notebook lên Kaggle
2. Thêm dataset `uit-hwdb-dataset`
3. Bật GPU accelerator (Tesla T4)
4. Chạy toàn bộ notebook

### 3. Phân tích kết quả

```bash
cd src
python analyze_results.py
```

Script sẽ:
- In thống kê so sánh Baseline vs Fine-tuned
- Phân tích loại lỗi (Insertion/Deletion/Substitution)
- Tạo biểu đồ trực quan trong thư mục `visualizations/`

### 4. Sử dụng mô hình Fine-tuned

```python
from unsloth import FastVisionModel
from transformers import AutoModel
from peft import PeftModel

# Load base model
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/DeepSeek-OCR",
    load_in_4bit=True,
    auto_model=AutoModel,
    trust_remote_code=True,
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "results/outputs/checkpoint-534")

# Inference
FastVisionModel.for_inference(model)
result = model.infer(
    tokenizer,
    prompt="<image>\nFree OCR. ",
    image_file="path/to/image.jpg",
)
```

## 📈 Evaluation Metrics

- **CER (Character Error Rate)**: Tỷ lệ lỗi ký tự
- **Perfect Match Rate**: Tỷ lệ mẫu nhận dạng chính xác 100%
- **Insertion Rate**: Tỷ lệ lỗi chèn ký tự thừa
- **Deletion Rate**: Tỷ lệ lỗi bỏ sót ký tự
- **Substitution Rate**: Tỷ lệ lỗi thay thế ký tự sai

## 📋 Phân tích Script (analyze_results.py)

Script cung cấp:
- `compute_edit_operations()`: Tính số lỗi Insertion/Deletion/Substitution bằng Levenshtein
- `analyze_error_types()`: Phân tích chi tiết các loại lỗi
- `calculate_perfect_match_rate()`: Tính tỷ lệ khớp hoàn hảo
- `compare_models()`: So sánh hiệu năng 2 mô hình
- `create_visualizations()`: Tạo biểu đồ phân tích

## Công nghệ sử dụng

- **Framework**: [Unsloth](https://github.com/unslothai/unsloth) - Fast fine-tuning
- **Base Model**: [DeepSeek-OCR](https://huggingface.co/unsloth/DeepSeek-OCR)
- **Fine-tuning**: LoRA/QLoRA với PEFT
- **Training**: Hugging Face Transformers + TRL
- **Evaluation**: jiwer (WER/CER metrics)
- **Visualization**: Matplotlib

## 📚 Tài liệu tham khảo

- [DeepSeek-OCR Paper](https://arxiv.org/abs/2410.05655)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [UIT-HWDB Dataset](https://www.kaggle.com/datasets/nvhieu/uit-hwdb-dataset)
- [Unsloth Documentation](https://docs.unsloth.ai/)

## 👤 Tác giả

**Bang My Linh -- 23122009 -- FIT@HCMUS**

## 📄 License

MIT License

