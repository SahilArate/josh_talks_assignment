"""
STEP 4 - ERROR ANALYSIS
- Loads all_predictions.csv from step 3
- Samples 25 errors using stratified strategy (not cherry-picked)
- Computes per-utterance WER
- Saves sampled_25_errors.csv for your report
"""

import pandas as pd
from pathlib import Path
import evaluate

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
OUTPUTS  = BASE_DIR / "outputs"

wer_metric = evaluate.load("wer")

# ── Load predictions ──────────────────────────────────────────────────────────
preds_df = pd.read_csv(OUTPUTS / "all_predictions.csv")
print(f"Total predictions loaded: {len(preds_df)}")

# ── Compute per-utterance WER ─────────────────────────────────────────────────
def utt_wer(ref, pred):
    try:
        return wer_metric.compute(predictions=[pred], references=[ref])
    except:
        return 1.0

preds_df['utt_wer'] = preds_df.apply(
    lambda r: utt_wer(r['reference'], r['ft_prediction']), axis=1
)

# ── Keep only utterances with errors ─────────────────────────────────────────
errors_df = preds_df[preds_df['utt_wer'] > 0].copy()
errors_df = errors_df.sort_values('utt_wer').reset_index(drop=True)
print(f"Utterances with errors: {len(errors_df)}")

# ── Stratified sampling: spread across severity ───────────────────────────────
low  = errors_df[errors_df['utt_wer'] <  0.30]
mid  = errors_df[(errors_df['utt_wer'] >= 0.30) & (errors_df['utt_wer'] < 0.70)]
high = errors_df[errors_df['utt_wer'] >= 0.70]

n_low  = min(8,  len(low))
n_mid  = min(9,  len(mid))
n_high = min(8,  len(high))

sampled = pd.concat([
    low.sample( n_low,  random_state=42),
    mid.sample( n_mid,  random_state=42),
    high.sample(n_high, random_state=42),
]).reset_index(drop=True)

print(f"\nSampled: {len(sampled)} errors")
print(f"  Low WER  (<0.30): {n_low}")
print(f"  Mid WER  (0.30–0.70): {n_mid}")
print(f"  High WER (>=0.70): {n_high}")

# ── Print all 25 errors for your inspection ───────────────────────────────────
print("\n" + "="*70)
print("25 SAMPLED ERRORS FOR TAXONOMY ANALYSIS")
print("="*70)
for i, row in sampled.iterrows():
    print(f"\n[{i+1}] WER: {row['utt_wer']:.2f}")
    print(f"  REF : {row['reference']}")
    print(f"  PRED: {row['ft_prediction']}")

# ── Save ──────────────────────────────────────────────────────────────────────
sampled.to_csv(OUTPUTS / "sampled_25_errors.csv", index=False)
print(f"\n✅ Saved: outputs/sampled_25_errors.csv")
print("\n=== STEP 4 COMPLETE ===")
print("\nNOW: Open sampled_25_errors.csv and manually assign each error")
print("to a category. Common categories:")
print("  1. Deletion       - model drops words")
print("  2. Substitution   - model uses wrong but similar-sounding word")
print("  3. Insertion      - model adds extra words")
print("  4. OOV/Rare words - tribal names, place names wrong")
print("  5. Code-switching - Hindi/English script mismatch")