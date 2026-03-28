"""
STEP 5 - IMPLEMENT FIX AND COMPARE
Fix applied: Segment-level training (20-sec chunks instead of full recordings)
- Splits each recording into ~20-second chunks
- Retrains Whisper on these segments (more training samples)
- Evaluates on FLEURS test again
- Prints before/after comparison table
"""

import os
import re
import json
import torch
import dataclasses
import pandas as pd
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Any, Dict, List, Union
from sklearn.model_selection import train_test_split
from datasets import Dataset, Audio, load_dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    pipeline,
)
import evaluate

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
AUDIO_16K     = BASE_DIR / "audio_16k"
TRANS_DIR     = BASE_DIR / "transcriptions"
MANIFESTS_DIR = BASE_DIR / "manifests"
MODELS_V1     = BASE_DIR / "models" / "whisper-hindi-v1"
MODELS_V2     = BASE_DIR / "models" / "whisper-hindi-v2"
SEGMENTS_DIR  = BASE_DIR / "audio_segments"
OUTPUTS       = BASE_DIR / "outputs"

MODELS_V2.mkdir(parents=True, exist_ok=True)
SEGMENTS_DIR.mkdir(exist_ok=True)

wer_metric = evaluate.load("wer")
device     = 0 if torch.cuda.is_available() else -1

# ── Helper: clean text ────────────────────────────────────────────────────────
def clean_text(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\u0900-\u097F\s\u0964\u0965.,!?]', '', text)
    return text.strip()

# ── Build segment-level dataset ───────────────────────────────────────────────
print("Building segment-level dataset (20-sec chunks)...")
segment_records = []

for trans_file in TRANS_DIR.glob("*.json"):
    rec_id     = trans_file.stem
    audio_path = AUDIO_16K / f"{rec_id}.wav"

    if not audio_path.exists():
        continue

    audio, sr = sf.read(str(audio_path))  # sr = 16000

    with open(trans_file, encoding='utf-8') as f:
        segments = json.load(f)

    # Merge short segments into ~20-second chunks
    chunks     = []
    cur_text   = []
    cur_start  = None

    for seg in segments:
        if cur_start is None:
            cur_start = seg['start']
        cur_text.append(seg.get('text', '').strip())
        cur_end = seg['end']

        if (cur_end - cur_start) >= 20:
            chunks.append((cur_start, cur_end, " ".join(cur_text)))
            cur_text  = []
            cur_start = None

    # flush leftover
    if cur_text and cur_start is not None:
        chunks.append((cur_start, cur_end, " ".join(cur_text)))

    for j, (start, end, text) in enumerate(chunks):
        text = clean_text(text)
        if len(text) < 10:
            continue
        s_idx = int(start * sr)
        e_idx = int(end   * sr)
        chunk = audio[s_idx:e_idx]

        seg_path = SEGMENTS_DIR / f"{rec_id}_{j}.wav"
        if not seg_path.exists():
            sf.write(str(seg_path), chunk, sr)

        segment_records.append({
            "audio_path": str(seg_path),
            "text":       text,
        })

seg_df = pd.DataFrame(segment_records)
print(f"Segment samples: {len(seg_df)}  (vs whole-file approach)")

train_seg, val_seg = train_test_split(seg_df, test_size=0.1, random_state=42)
train_seg.to_csv(MANIFESTS_DIR / "train_segments.csv", index=False)
val_seg.to_csv(  MANIFESTS_DIR / "val_segments.csv",   index=False)

# ── Load processor ────────────────────────────────────────────────────────────
processor = WhisperProcessor.from_pretrained(str(MODELS_V1), language="Hindi", task="transcribe")
model     = WhisperForConditionalGeneration.from_pretrained(str(MODELS_V1))
model.config.forced_decoder_ids = None

# ── Dataset helpers (same as step 2) ─────────────────────────────────────────
def load_dataset_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    ds = Dataset.from_dict({
        "audio": df["audio_path"].tolist(),
        "text":  df["text"].tolist(),
    })
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds

def prepare_dataset(batch):
    audio  = batch["audio"]
    inputs = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"])
    batch["input_features"] = inputs.input_features[0]
    batch["labels"]         = processor.tokenizer(batch["text"]).input_ids
    return batch

@dataclasses.dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str  = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": round(wer_metric.compute(predictions=pred_str, references=label_str), 4)}

# ── Load + prepare datasets ───────────────────────────────────────────────────
print("Preparing segment datasets...")
train_ds = load_dataset_from_csv(MANIFESTS_DIR / "train_segments.csv")
val_ds   = load_dataset_from_csv(MANIFESTS_DIR / "val_segments.csv")
train_ds = train_ds.map(prepare_dataset, remove_columns=train_ds.column_names)
val_ds   = val_ds.map(  prepare_dataset, remove_columns=val_ds.column_names)

# ── Retrain ───────────────────────────────────────────────────────────────────
training_args = Seq2SeqTrainingArguments(
    output_dir                  = str(MODELS_V2),
    per_device_train_batch_size = 8,
    per_device_eval_batch_size  = 4,
    gradient_accumulation_steps = 2,
    learning_rate               = 1e-5,
    warmup_steps                = 100,
    max_steps                   = 1000,
    gradient_checkpointing      = True,
    fp16                        = True,
    evaluation_strategy         = "steps",
    eval_steps                  = 200,
    save_steps                  = 200,
    logging_steps               = 50,
    load_best_model_at_end      = True,
    metric_for_best_model       = "wer",
    greater_is_better           = False,
    predict_with_generate       = True,
    generation_max_length       = 225,
    push_to_hub                 = False,
)

trainer = Seq2SeqTrainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_ds,
    eval_dataset    = val_ds,
    data_collator   = DataCollatorSpeechSeq2SeqWithPadding(processor),
    compute_metrics = compute_metrics,
    tokenizer       = processor.feature_extractor,
)

print("\n🚀 Retraining with segment-level data...")
trainer.train()
model.save_pretrained(str(MODELS_V2))
processor.save_pretrained(str(MODELS_V2))
print(f"✅ V2 model saved to {MODELS_V2}")

# ── Final comparison ──────────────────────────────────────────────────────────
print("\nLoading FLEURS test for final comparison...")
fleurs_test = load_dataset("google/fleurs", "hi_in", split="test", trust_remote_code=True)

def eval_on_fleurs(model_path, label):
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        chunk_length_s=30,
        generate_kwargs={"language": "hi", "task": "transcribe"},
        device=device,
    )
    preds, refs = [], []
    for i, sample in enumerate(fleurs_test):
        if i % 50 == 0: print(f"  [{label}] {i}/{len(fleurs_test)}")
        preds.append(pipe(sample["audio"])["text"].strip())
        refs.append(sample["transcription"].strip())
    wer = wer_metric.compute(predictions=preds, references=refs)
    print(f"  ✅ {label} WER: {wer*100:.2f}%")
    return preds, refs, wer

_, _, wer_base = eval_on_fleurs("openai/whisper-small",  "Baseline")
_, _, wer_v1   = eval_on_fleurs(str(MODELS_V1),          "V1 (whole-file)")
_, v2_refs, wer_v2 = eval_on_fleurs(str(MODELS_V2),      "V2 (segments, Fix 1)")

# ── Before/After table ────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"{'Model':<40} {'WER (%)':>10}")
print("-"*60)
print(f"{'Baseline (no fine-tuning)':<40} {wer_base*100:>9.2f}%")
print(f"{'Fine-tuned V1 (whole recordings)':<40} {wer_v1*100:>9.2f}%")
print(f"{'Fine-tuned V2 (segments - Fix 1)':<40} {wer_v2*100:>9.2f}%")
print("="*60)

comparison_df = pd.DataFrame([
    {"model": "Baseline",                    "wer_pct": round(wer_base*100, 2)},
    {"model": "Fine-tuned V1 (whole files)", "wer_pct": round(wer_v1*100,   2)},
    {"model": "Fine-tuned V2 (segments)",    "wer_pct": round(wer_v2*100,   2)},
])
comparison_df.to_csv(OUTPUTS / "before_after_comparison.csv", index=False)
print(f"\n✅ Saved: outputs/before_after_comparison.csv")
print("\n=== STEP 5 COMPLETE ===")