
import os
import torch
import dataclasses
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Union
from datasets import Dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate

BASE_DIR      = Path(__file__).parent
MANIFESTS_DIR = BASE_DIR / "manifests"
MODELS_DIR    = BASE_DIR / "models" / "whisper-hindi-v1"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "openai/whisper-small"
print(f"Loading {MODEL_NAME}...")
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="Hindi", task="transcribe")
model     = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.config.forced_decoder_ids = None

def load_dataset_from_csv(csv_path):
    df  = pd.read_csv(csv_path)
    ds  = Dataset.from_dict({
        "audio": df["audio_path"].tolist(),
        "text":  df["text"].tolist(),
    })
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds

print("Loading datasets...")
train_ds = load_dataset_from_csv(MANIFESTS_DIR / "train_manifest.csv")
val_ds   = load_dataset_from_csv(MANIFESTS_DIR / "val_manifest.csv")
print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

def prepare_dataset(batch):
    audio  = batch["audio"]
    inputs = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    )
    batch["input_features"] = inputs.input_features[0]
    batch["labels"]         = processor.tokenizer(batch["text"]).input_ids
    return batch

print("Extracting features (this takes a few minutes)...")
train_ds = train_ds.map(prepare_dataset, remove_columns=train_ds.column_names, num_proc=1)
val_ds   = val_ds.map(  prepare_dataset, remove_columns=val_ds.column_names,   num_proc=1)
print("✅ Feature extraction done")

@dataclasses.dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str  = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": round(wer, 4)}

training_args = Seq2SeqTrainingArguments(
    output_dir                  = str(MODELS_DIR),
    per_device_train_batch_size = 8,
    per_device_eval_batch_size  = 4,
    gradient_accumulation_steps = 2,
    learning_rate               = 1e-5,
    warmup_steps                = 50,
    max_steps                   = 500,
    gradient_checkpointing      = True,
    fp16                        = True,        # needs GPU
    evaluation_strategy         = "steps",
    eval_steps                  = 100,
    save_steps                  = 100,
    logging_steps               = 25,
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
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
    tokenizer       = processor.feature_extractor,
)

print("\n🚀 Starting fine-tuning...")
trainer.train()

model.save_pretrained(str(MODELS_DIR))
processor.save_pretrained(str(MODELS_DIR))
print(f"\n✅ Model saved to {MODELS_DIR}")
print("\n=== STEP 2 COMPLETE ===")