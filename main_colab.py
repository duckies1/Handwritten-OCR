import os
import argparse
import shutil
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    default_data_collator,
)

import evaluate


# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "microsoft/trocr-base-handwritten"
OUTPUT_DIR = "./drive/MyDrive/trOCR/model"
MAX_TARGET_LENGTH = 128


# # -----------------------------
# # Argparse (weights-only resume)
# # -----------------------------
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--resume",
#     default=None,
#     help="Path to a HF Trainer checkpoint directory (e.g. model/phase1/checkpoint-1000)",
# )

# args = parser.parse_args()


# -----------------------------
# Custom Dataset Loader
# -----------------------------
class IAMParquetDataset(Dataset):
    def __init__(self, parquet_path):
        self.data = pd.read_parquet(parquet_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = row["image"]
        image = None

        # --- Robust image loading ---
        if isinstance(img, dict):
            # Handle dict format (with 'bytes' and/or 'path' keys)
            if "bytes" in img and img["bytes"] is not None and isinstance(img["bytes"], (bytes, bytearray)):
                from io import BytesIO
                image = Image.open(BytesIO(img["bytes"])).convert("RGB")
            elif "path" in img and img["path"] is not None and isinstance(img["path"], str) and os.path.exists(img["path"]):
                image = Image.open(img["path"]).convert("RGB")
            else:
                raise ValueError(f"Unsupported dict image format or missing data at index {idx}: {img}")
        
        elif isinstance(img, str):
            if os.path.exists(img):
                image = Image.open(img).convert("RGB")
            else:
                raise ValueError(f"Image path does not exist at index {idx}: {img}")

        elif isinstance(img, (bytes, bytearray)):
            from io import BytesIO
            image = Image.open(BytesIO(img)).convert("RGB")

        elif isinstance(img, np.ndarray):
            image = Image.fromarray(img).convert("RGB")
        
        elif hasattr(img, 'mode') and hasattr(img, 'size'):
            # Already a PIL Image
            image = img.convert("RGB")

        else:
            raise ValueError(f"Unsupported image type at index {idx}: {type(img)}, value: {img}")
        
        if image is None:
            raise ValueError(f"Failed to load image at index {idx}")

        # Aspect-ratio preserving resize (shorter side → 384px)
        target_height = 384
        w, h = image.size
        new_w = int(w * (target_height / h))
        image = image.resize((new_w, target_height), Image.BILINEAR)

        return {
            "image": image,
            "text": row["text"],
        }


# -----------------------------
# Paths
# -----------------------------
train_path = "./drive/MyDrive/trOCR/Dataset/Train/train.parquet"
val_path   = "./drive/MyDrive/trOCR/Dataset/Validation/validation.parquet"
test_path  = "./drive/MyDrive/trOCR/Dataset/Test/test.parquet"


# -----------------------------
# Load processor & model
# -----------------------------
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

# # Optional weights-only resume
# if args.resume and os.path.exists(args.resume):
#     print(f"Loading weights from: {args.resume}")
    # model = VisionEncoderDecoderModel.from_pretrained(args.resume)

# -----------------------------
# Correct TrOCR generation config
# -----------------------------
model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.max_length = MAX_TARGET_LENGTH


# -----------------------------
# Datasets
# -----------------------------
train_dataset = IAMParquetDataset(train_path)
val_dataset   = IAMParquetDataset(val_path)
test_dataset  = IAMParquetDataset(test_path)


# -----------------------------
# Freezing utilities
# -----------------------------
def freeze_vision_encoder(model):
    for p in model.encoder.parameters():
        p.requires_grad = False


def unfreeze_last_beit_blocks(model, num_blocks=2):
    encoder_layers = model.encoder.encoder.layer
    for layer in encoder_layers[-num_blocks:]:
        for p in layer.parameters():
            p.requires_grad = True


def unfreeze_all(model):
    for p in model.encoder.parameters():
        p.requires_grad = True


# -----------------------------
# Data Collator (for batching)
# -----------------------------
def collate_fn(batch):
    """
    Custom collate function to process PIL images and text into tensors
    """
    images = [item["image"] for item in batch]
    texts = [item["text"] for item in batch]
    
    # Process images
    pixel_values = processor(
        images=images,
        return_tensors="pt"
    ).pixel_values

    # Process text labels
    labels = processor.tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_TARGET_LENGTH,
    ).input_ids

    # Replace padding token id with -100 (ignored in loss)
    labels = [
        [(t if t != processor.tokenizer.pad_token_id else -100) for t in seq]
        for seq in labels
    ]

    return {
        "pixel_values": pixel_values,
        "labels": torch.tensor(labels),
    }


# -----------------------------
# CER Metric
# -----------------------------
cer_metric = evaluate.load("cer")


def compute_metrics(eval_pred):
    preds = eval_pred.predictions
    labels = eval_pred.label_ids

    labels = [
        [(l if l != -100 else processor.tokenizer.pad_token_id) for l in seq]
        for seq in labels
    ]

    pred_str = processor.batch_decode(preds, skip_special_tokens=True)
    label_str = processor.batch_decode(labels, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}


# -----------------------------
# Custom Trainer to handle version incompatibility
# -----------------------------
class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override to remove num_items_in_batch from inputs passed to model
        """
        # Call the model without num_items_in_batch
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        
        outputs = model(**inputs)
        
        if labels is not None:
            if self.label_smoother is not None:
                loss = self.label_smoother(outputs, labels)
            else:
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        else:
            if isinstance(outputs, dict):
                loss = outputs["loss"]
            else:
                loss = outputs[0]
        
        return (loss, outputs) if return_outputs else loss


# -----------------------------
# Trainer factory
# -----------------------------
def build_trainer(output_dir, lr, epochs):
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=lr,
        num_train_epochs=epochs,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=50,
        predict_with_generate=True,
        generation_num_beams=4,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
    )

    return CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,  # Use custom collate function instead of default
        processing_class=processor,  # Changed from 'tokenizer' to 'processing_class'
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

def get_latest_checkpoint(phase_dir):
    """
    Returns path to latest HF Trainer checkpoint in a phase directory,
    or None if no checkpoint exists.
    """
    if not os.path.isdir(phase_dir):
        return None

    checkpoints = [
        os.path.join(phase_dir, d)
        for d in os.listdir(phase_dir)
        if d.startswith("checkpoint-")
    ]

    if not checkpoints:
        return None

    # Sort by global step number
    checkpoints = sorted(
        checkpoints,
        key=lambda x: int(x.split("-")[-1])
    )
    return checkpoints[-1]


def get_most_recent_checkpoint_across_phases():
    """
    Find the most recent checkpoint across all phases.
    Returns (phase_num, checkpoint_path) or (0, None) if none exist.
    """
    phase_dirs = [
        (1, f"{OUTPUT_DIR}/phase1"),
        (2, f"{OUTPUT_DIR}/phase2"),
        (3, f"{OUTPUT_DIR}/phase3"),
    ]
    
    latest_phase = 0
    latest_checkpoint = None
    
    for phase_num, phase_dir in reversed(phase_dirs):  # Check from phase3 -> phase1
        checkpoint = get_latest_checkpoint(phase_dir)
        if checkpoint:
            latest_phase = phase_num
            latest_checkpoint = checkpoint
            break
    
    return latest_phase, latest_checkpoint


def should_skip_phase(current_phase, completed_phase):
    """
    Determine if we should skip the current phase based on what's already completed.
    """
    return current_phase < completed_phase


# =============================
# Check what's already trained
# =============================
completed_phase, latest_checkpoint = get_most_recent_checkpoint_across_phases()

if latest_checkpoint:
    print(f"\n🔍 Found checkpoint from Phase {completed_phase}: {latest_checkpoint}")
    print(f"Loading model weights from this checkpoint...")
    model = VisionEncoderDecoderModel.from_pretrained(latest_checkpoint)
    print(f"✅ Model loaded from Phase {completed_phase} checkpoint")
    
    # Apply the freezing state that matches the checkpoint's phase
    if completed_phase == 1:
        freeze_vision_encoder(model)
        print("🔒 Applied Phase 1 freezing (encoder frozen)")
    elif completed_phase == 2:
        freeze_vision_encoder(model)
        unfreeze_last_beit_blocks(model, num_blocks=2)
        print("🔓 Applied Phase 2 freezing (last 2 blocks unfrozen)")
    elif completed_phase == 3:
        unfreeze_all(model)
        print("🔓 Applied Phase 3 freezing (all unfrozen)")
else:
    print("\n🆕 No checkpoints found. Starting training from scratch.")


# =============================
# PHASE 1: Freeze vision encoder
# =============================
if should_skip_phase(1, completed_phase):
    print(f"\n⏭️  Skipping Phase 1 (already completed Phase {completed_phase})")
else:
    print("\n==== Phase 1: Decoder adaptation ====")
    
    # Only apply freezing if starting fresh (not resuming)
    if completed_phase == 0:
        freeze_vision_encoder(model)
        print("🔒 Applied Phase 1 freezing")

    trainer = build_trainer(
        output_dir=f"{OUTPUT_DIR}/phase1",
        lr=5e-5,
        epochs=7,
    )
    phase1_dir = f"{OUTPUT_DIR}/phase1"
    resume_ckpt = get_latest_checkpoint(phase1_dir)

    trainer.train(resume_from_checkpoint=resume_ckpt)
    completed_phase = 1



# =============================
# PHASE 2: Unfreeze last BEiT blocks
# =============================
if should_skip_phase(2, completed_phase):
    print(f"\n⏭️  Skipping Phase 2 (already completed Phase {completed_phase})")
else:
    print("\n==== Phase 2: Partial vision fine-tuning ====")
    
    # Only apply freezing if transitioning from Phase 1 (not resuming Phase 2)
    if completed_phase <= 1:
        freeze_vision_encoder(model)
        unfreeze_last_beit_blocks(model, num_blocks=2)
        print("🔓 Applied Phase 2 freezing")

    trainer = build_trainer(
        output_dir=f"{OUTPUT_DIR}/phase2",
        lr=1e-5,
        epochs=10,
    )
    phase2_dir = f"{OUTPUT_DIR}/phase2"
    resume_ckpt = get_latest_checkpoint(phase2_dir)

    trainer.train(resume_from_checkpoint=resume_ckpt)
    completed_phase = 2




# =============================
# PHASE 3: Optional full fine-tune
# =============================
# print("\n==== Phase 3: Full fine-tuning ====")
# unfreeze_all(model)
# trainer = build_trainer(
#     output_dir=f"{OUTPUT_DIR}/phase3",
#     lr=5e-6,
#     epochs=5,
# )
# phase3_dir = f"{OUTPUT_DIR}/phase3"
# resume_ckpt = get_latest_checkpoint(phase3_dir)

# trainer.train(resume_from_checkpoint=resume_ckpt)



# -----------------------------
# Save final model
# -----------------------------
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print("\n✅ Training complete. Final model saved.")