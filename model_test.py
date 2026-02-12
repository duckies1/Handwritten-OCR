import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import evaluate

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "./final"  # Your final model in project root
TEST_PARQUET = "./Dataset/Test/test.parquet"  # Test dataset
MAX_TARGET_LENGTH = 128
NUM_SAMPLES = 20  # Number of samples to test (set to None for full test set)

# -----------------------------
# Load model and processor
# -----------------------------
print(f"Loading final model from: {MODEL_PATH}")
processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
print(f"✅ Model loaded on: {device}\n")

# -----------------------------
# Load test data
# -----------------------------
print(f"Loading test data from: {TEST_PARQUET}")
df = pd.read_parquet(TEST_PARQUET)

if NUM_SAMPLES:
    df = df.head(NUM_SAMPLES)
    
print(f"Testing on {len(df)} samples\n")

# -----------------------------
# Helper function to load images
# -----------------------------
def load_image(img):
    """Load image from various formats"""
    if isinstance(img, dict):
        if "bytes" in img and img["bytes"] is not None:
            image = Image.open(BytesIO(img["bytes"])).convert("RGB")
        elif "path" in img and img["path"] is not None and os.path.exists(img["path"]):
            image = Image.open(img["path"]).convert("RGB")
        else:
            raise ValueError(f"Cannot load image from dict: {img}")
    elif isinstance(img, str) and os.path.exists(img):
        image = Image.open(img).convert("RGB")
    elif isinstance(img, (bytes, bytearray)):
        image = Image.open(BytesIO(img)).convert("RGB")
    elif isinstance(img, np.ndarray):
        image = Image.fromarray(img).convert("RGB")
    elif hasattr(img, 'mode') and hasattr(img, 'size'):
        image = img.convert("RGB")
    else:
        raise ValueError(f"Unsupported image type: {type(img)}")
    
    # Resize to 384 height (same as training)
    target_height = 384
    w, h = image.size
    new_w = int(w * (target_height / h))
    image = image.resize((new_w, target_height), Image.BILINEAR)
    
    return image

# -----------------------------
# Run inference
# -----------------------------
predictions = []
references = []

print("=" * 100)
print("RUNNING INFERENCE")
print("=" * 100 + "\n")

for idx, row in df.iterrows():
    # Load and process image
    image = load_image(row["image"])
    ground_truth = row["text"]
    
    # Prepare input
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    
    # Generate prediction
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_length=MAX_TARGET_LENGTH,
            num_beams=4,
        )
    
    # Decode prediction
    prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    predictions.append(prediction)
    references.append(ground_truth)
    
    # Determine if match
    is_match = prediction.strip().lower() == ground_truth.strip().lower()
    match_symbol = "✓" if is_match else "✗"
    
    # Print results
    print(f"[{idx + 1}/{len(df)}] {match_symbol}")
    print(f"  Ground Truth: '{ground_truth}'")
    print(f"  Prediction:   '{prediction}'")
    if not is_match:
        print(f"  ⚠️  MISMATCH")
    print("-" * 100)

# -----------------------------
# Calculate metrics
# -----------------------------
print("\n" + "=" * 100)
print("EVALUATION METRICS")
print("=" * 100 + "\n")

# Character Error Rate (CER)
cer_metric = evaluate.load("cer")
cer = cer_metric.compute(predictions=predictions, references=references)
print(f"📊 Character Error Rate (CER):  {cer:.4f} ({cer*100:.2f}%)")

# Word Error Rate (WER)
wer_metric = evaluate.load("wer")
wer = wer_metric.compute(predictions=predictions, references=references)
print(f"📊 Word Error Rate (WER):       {wer:.4f} ({wer*100:.2f}%)")

# Exact match accuracy (case-insensitive)
exact_matches = sum([1 for p, r in zip(predictions, references) 
                     if p.strip().lower() == r.strip().lower()])
accuracy = exact_matches / len(predictions)
print(f"📊 Exact Match Accuracy:        {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"📊 Exact Matches:               {exact_matches}/{len(predictions)}")

# Character accuracy (inverse of CER)
char_accuracy = 1 - cer
print(f"📊 Character Accuracy:          {char_accuracy:.4f} ({char_accuracy*100:.2f}%)")

print("\n" + "=" * 100)
print("✅ TESTING COMPLETE!")
print("=" * 100)

# Optional: Show some error examples
print("\n" + "=" * 100)
print("SAMPLE ERRORS (if any)")
print("=" * 100 + "\n")

errors_shown = 0
for i, (pred, ref) in enumerate(zip(predictions, references)):
    if pred.strip().lower() != ref.strip().lower() and errors_shown < 5:
        print(f"Error #{errors_shown + 1}:")
        print(f"  Expected: '{ref}'")
        print(f"  Got:      '{pred}'")
        print()
        errors_shown += 1

if errors_shown == 0:
    print("🎉 No errors! Perfect predictions on all samples!")