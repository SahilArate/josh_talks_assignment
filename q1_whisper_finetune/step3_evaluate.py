"""
STEP 3 - EVALUATE BOTH MODELS ON FLEURS HINDI TEST SET
- Runs baseline Whisper-small (no fine-tuning)
- Runs your fine-tuned model (from step 2)
- Computes WER for both
- Saves wer_results.csv and all predictions
"""

import torch
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from transformers import pipeline
import evaluate

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models" / "whisper-hindi-v1"
OUTPUTS    = BASE_DIR / "outputs"
OUTPUTS.mkdir(exist_ok=True)

wer_metric = evaluate.load("wer")
device     = 0 if torch.cuda.is_available() else -1

# ── Load FLEURS Hindi test set ────────────────────────────────────────────────
print("Loading FLEURS Hindi test set (downloads ~500MB, one-time)...")
fleurs_test = load_dataset("google/fleurs", "hi_in", split="test", trust_remote_code=True)
print(f"✅ FLEURS test samples: {len(fleurs_test)}")

# ── Evaluate any model ────────────────────────────────────────────────────────
def evaluate_model(model_path, dataset, label):
    print(f"\n--- Evaluating: {label} ---")
    pipe = pipeline(
        "automatic-speech-recognition",
        model        = model_path,
        chunk_length_s = 30,
        generate_kwargs = {"language": "hi", "task": "transcribe"},
        device       = device,
    )
    preds, refs = [], []
    for i, sample in enumerate(dataset):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(dataset)}")
        pred = pipe(sample["audio"])["text"].strip()
        ref  = sample["transcription"].strip()
        preds.append(pred)
        refs.append(ref)

    wer = wer_metric.compute(predictions=preds, references=refs)
    print(f"  ✅ WER: {wer*100:.2f}%")
    return preds, refs, wer

# ── Run both models ───────────────────────────────────────────────────────────
baseline_preds, baseline_refs, baseline_wer = evaluate_model(
    "openai/whisper-small", fleurs_test, "Baseline (no fine-tuning)"
)

ft_preds, ft_refs, ft_wer = evaluate_model(
    str(MODELS_DIR), fleurs_test, "Fine-tuned (step 2)"
)

# ── WER Table ─────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print(f"{'Model':<35} {'WER (%)':>10}")
print("-"*55)
print(f"{'Whisper-small (baseline)':<35} {baseline_wer*100:>9.2f}%")
print(f"{'Whisper-small (fine-tuned)':<35} {ft_wer*100:>9.2f}%")
print("="*55)

# ── Save results ──────────────────────────────────────────────────────────────
wer_df = pd.DataFrame([
    {"model": "Whisper-small (baseline)",   "wer": round(baseline_wer*100, 2)},
    {"model": "Whisper-small (fine-tuned)", "wer": round(ft_wer*100, 2)},
])
wer_df.to_csv(OUTPUTS / "wer_results.csv", index=False)

# Save all predictions for error analysis
preds_df = pd.DataFrame({
    "reference":        ft_refs,
    "ft_prediction":    ft_preds,
    "base_prediction":  baseline_preds,
})
preds_df.to_csv(OUTPUTS / "all_predictions.csv", index=False)

print(f"\n✅ Saved: outputs/wer_results.csv")
print(f"✅ Saved: outputs/all_predictions.csv")
print("\n=== STEP 3 COMPLETE ===")