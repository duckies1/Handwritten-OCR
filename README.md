# TrOCR Handwritten OCR

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

This project implements a fine-tuned **TrOCR (Transformer-based Optical Character Recognition)** model for recognizing handwritten text. The model leverages Microsoft's pre-trained TrOCR architecture and has been optimized on handwritten text datasets to achieve high accuracy in converting handwritten images to digital text.

Built with PyTorch and Hugging Face Transformers, this solution demonstrates state-of-the-art performance in handwritten text recognition with low character error rates and high exactness in matching recognized text.

## Model Architecture

The model is based on **TrOCR** (Microsoft's Transformer-based Optical Character Recognition), which consists of:

- **Vision Encoder**: ResNet-based visual feature extractor that processes input images
- **Text Decoder**: Transformer decoder that generates text sequences from visual features
- **Architecture Details**:
  - Base model: `microsoft/trocr-base-handwritten`
  - Input image height: 384 pixels (aspect ratio preserved)
  - Maximum output sequence length: 128 characters
  - Decoding strategy: Beam search with 4 beams
  - Tokenization: BPE (Byte Pair Encoding)

## Evaluation Metrics

The fine-tuned model achieves the following performance on the test set:

| Metric                         | Score  |
| ------------------------------ | ------ |
| **Character Error Rate (CER)** | 5.37%  |
| **Word Error Rate (WER)**      | 16.20% |

## Quick Start

### Prerequisites

- Python 3.12.4
- CUDA 11.8+ (for GPU acceleration, optional)
- 2GB+ disk space for model and dataset

### Installation

1. **Clone the repository**

   ```bash
   git clone <https://github.com/duckies1/Handwritten-OCR>
   cd Handwritten-OCR
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download model and dataset**

   ```bash
   python download_assets.py
   ```

   For linux & mac:

   ```bash
   python3 download_assets.py
   ```

### Running Inference

```bash
python model_test.py
```

This will test the model on the test dataset and display:

- Recognized text vs. ground truth
- Character Error Rate (CER)
- Word Error Rate (WER)
- Accuracy metrics

## Configuration

Key parameters can be adjusted in the code files:

- **`model_test.py`**:
  - `MODEL_PATH`: Path to your fine-tuned model
  - `TEST_PARQUET`: Path to test dataset
  - `NUM_SAMPLES`: Number of samples to test
  - `MAX_TARGET_LENGTH`: Maximum text length

- **`main.py` / `main_colab.py`** (for retraining):
  - `MODEL_NAME`: Base model from Hugging Face Hub
  - `OUTPUT_DIR`: Where to save the fine-tuned model
  - `MAX_TARGET_LENGTH`: Maximum output sequence length

## Performance Notes

- **GPU Recommended**: Model runs much faster with CUDA. On CPU, inference takes several seconds per image.
- **Batch Processing**: The model supports batch inference for faster processing of multiple images.
- **Memory Requirements**: ~2GB VRAM for inference, ~6GB for training

## Troubleshooting

**Model not found error**

```bash
# Re-run setup to download the model
python download_assets.py
```

**Out of memory (CUDA)**

- Reduce batch size in `model_test.py`
- Use CPU instead: `export CUDA_VISIBLE_DEVICES=""`

**Image loading errors**

- Ensure image is in RGB format (JPEG, PNG)
- Try converting: `Image.open("img.png").convert("RGB")`

## Dataset

This project uses handwritten text datasets in Parquet format:

- **Train set**: Training examples for fine-tuning
- **Validation set**: Used for early stopping and evaluation
- **Test set**: Final evaluation metrics

Dataset format:

```python
{
    "image": Image bytes,
    "text": Transcription text
}
```

## Requirements

See [requirements.txt](requirements.txt) for all dependencies. Key packages:

- `torch >= 2.8.0` - Deep learning framework
- `transformers >= 4.47.1` - Hugging Face models and utilities
- `Pillow >= 10.4` - Image processing
- `pandas >= 2.2` - Data handling
- `accelerate >= 1.2.1` - Distributed training support

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this project, please cite:

```bibtex
@inproceedings{li2021trocr,
  title={{TrOCR}: Transformer-based Optical Character Recognition},
  author={Li, Minghao and Lv, Tengchao and Chen, Lei and Cui, Yusheng and Zhang, Yijuan and Zhu, Furu and Manmatha, R},
  booktitle={arXiv preprint arXiv:2109.10282},
  year={2021}
}
```

## References

- [TrOCR Paper](https://arxiv.org/abs/2109.10282)
- [Hugging Face TrOCR](https://huggingface.co/docs/transformers/model_doc/trocr)
- [Microsoft TrOCR Hub](https://huggingface.co/microsoft/trocr-base-handwritten)

---

**Created**: February 2026  
**Questions?** Open an issue or check the documentation files.
