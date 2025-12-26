---
base_model: unsloth/DeepSeek-OCR
library_name: peft
pipeline_tag: image-to-text
license: apache-2.0
language:
- vi
tags:
- base_model:adapter:unsloth/DeepSeek-OCR
- lora
- transformers
- unsloth
- ocr
- vietnamese
- handwriting-recognition
datasets:
- UIT-HWDB
---

# DeepSeek-OCR Fine-tuned for Vietnamese Handwriting Recognition

<!-- Provide a quick summary of what the model is/does. -->

This is a LoRA adapter fine-tuned from **DeepSeek-OCR** for Vietnamese handwriting recognition (OCR) task using the **UIT-HWDB** dataset.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model is a LoRA (Low-Rank Adaptation) fine-tuned version of DeepSeek-OCR, specifically optimized for recognizing Vietnamese handwritten text including words, lines, and paragraphs.

- **Developed by:** Grace
- **Model type:** Vision-Language Model (VLM) with LoRA adapter
- **Language(s):** Vietnamese
- **License:** Apache-2.0
- **Finetuned from model:** [unsloth/DeepSeek-OCR](https://huggingface.co/unsloth/DeepSeek-OCR)

### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** [GitHub - deepseek-ocr_fine-tuning](https://github.com/gracebml/deepseek-ocr_fine-tuning)
- **Base Model:** [DeepSeek-OCR](https://huggingface.co/unsloth/DeepSeek-OCR)
- **Dataset:** [UIT-HWDB](https://www.kaggle.com/datasets/nvhieu/uit-hwdb-dataset)

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

This model can be used directly for OCR tasks on Vietnamese handwritten documents, including:
- Handwritten word recognition
- Handwritten line recognition  
- Handwritten paragraph recognition

### Downstream Use

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

- Document digitization systems
- Handwriting-to-text applications
- Educational tools for Vietnamese language learning

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

- Printed text OCR (use dedicated printed OCR models)
- Non-Vietnamese handwriting recognition
- Real-time video OCR

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

- Performance may vary depending on handwriting quality and style
- Model was trained on UIT-HWDB dataset which may not cover all Vietnamese handwriting variations
- Complex layouts or mixed content (text + diagrams) may not be handled well

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

- Use clear, well-lit images for best results
- Pre-process images to remove noise and enhance contrast
- For production use, implement confidence thresholds and human review

## How to Get Started with the Model

Use the code below to get started with the model.

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
model = PeftModel.from_pretrained(model, "path/to/checkpoint-534")

# Set to inference mode
FastVisionModel.for_inference(model)

# Run inference
result = model.infer(
    tokenizer,
    prompt="<image>\nFree OCR. ",
    image_file="path/to/handwritten_image.jpg",
    base_size=1024,
    image_size=640,
    crop_mode=True,
)
print(result)
```

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

**UIT-HWDB (Vietnamese Handwriting Database)**
- `UIT_HWDB_word`: Single word images
- `UIT_HWDB_line`: Line-level text images
- `UIT_HWDB_paragraph`: Paragraph-level text images

Total training samples: ~10,000+ images from multiple writers

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing

- Images converted to RGB format
- Dynamic image sizing with base_size=1024, image_size=640
- Crop mode enabled for better handling of varied aspect ratios

#### Training Hyperparameters

- **Training regime:** QLoRA (4-bit quantization with LoRA)
- **LoRA rank (r):** 16
- **LoRA alpha:** 16
- **LoRA dropout:** 0
- **Bias:** none
- **Target modules:** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Epochs:** 1
- **Total steps:** 534
- **Learning rate:** 1e-4 with cosine scheduler

#### Speeds, Sizes, Times

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

- **Training time:** ~4.5 hours
- **Adapter size:** ~297 MB
- **GPU memory usage:** ~14 GB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

200 samples from UIT-HWDB test set (evenly distributed across word, line, paragraph types)

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

- **CER (Character Error Rate):** Measures character-level accuracy
- **Perfect Match Rate:** Percentage of samples with 0% CER

### Results

| Metric | Baseline | Fine-tuned | Improvement |
|--------|----------|------------|-------------|
| Mean CER | ~40% | ~12% | **70% ↓** |
| Median CER | ~33% | ~8% | **75% ↓** |
| Perfect Match Rate | ~0% | ~25% | **+25%** |

#### Error Analysis

| Error Type | Baseline | Fine-tuned | Reduction |
|------------|----------|------------|-----------|
| Insertion | High | Low | ~70% |
| Deletion | High | Low | ~65% |
| Substitution | High | Low | ~72% |

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** Tesla T4 GPU (x2)
- **Hours used:** ~4.5 hours
- **Cloud Provider:** Kaggle
- **Compute Region:** Unknown
- **Carbon Emitted:** Minimal (short training time)

## Technical Specifications

### Model Architecture and Objective

- **Architecture:** DeepSeek-OCR (Vision-Language Model based on DeepSeek-VL-v2)
- **Objective:** Causal Language Modeling for image-to-text generation
- **Adaptation:** LoRA applied to attention and MLP layers

### Compute Infrastructure

#### Hardware

- 2x Tesla T4 GPUs (15GB VRAM each)
- Kaggle Notebooks environment

#### Software

- Python 3.10
- Transformers 4.56.2
- PEFT 0.16.0
- Unsloth 2025.12.7
- PyTorch 2.6.0+cu124

## Citation

**BibTeX:**

```bibtex
@misc{deepseek-ocr-vietnamese-hwdb,
  author = {Grace},
  title = {DeepSeek-OCR Fine-tuned for Vietnamese Handwriting Recognition},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/gracebml/deepseek-ocr_fine-tuning}
}
```

## Model Card Authors

Grace

## Model Card Contact

GitHub: [@gracebml](https://github.com/gracebml)

### Framework versions

- PEFT 0.16.0
- Transformers 4.56.2
- Unsloth 2025.12.7
- PyTorch 2.6.0
