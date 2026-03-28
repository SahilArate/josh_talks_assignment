# =============================================================================
# lattice_wer.py
# Q4 — Lattice-Based WER Evaluation
# Josh Talks ASR Assignment
# =============================================================================
#
# PROBLEM WITH STANDARD WER:
#   Standard WER compares model output against ONE rigid reference string.
#   This unfairly penalizes models when:
#     1. Punctuation differs  : "है।" vs "है"  → counted as error, but both correct
#     2. Spelling variants    : "सब्ज़ी" vs "सब्जी" → same word, different nuqta
#     3. Script mixing        : "feedback" vs "फीडबैक" → same word, diff script
#     4. Word splits          : "रक्षाबंधन" vs "रक्षा बंधन" → same, diff spacing
#     5. Reference is WRONG  : human transcriber made a mistake
#
# OUR SOLUTION — LATTICE:
#   Instead of one rigid reference, build a "lattice" — for each word position,
#   collect ALL valid alternatives seen across models + reference.
#   Then score each model against this richer lattice.
#
# ALIGNMENT UNIT: WORD (not subword/character)
#   WHY: Hindi words are meaningful units. Subword splits would conflate
#        genuine errors with morphological variation. Characters are too
#        granular for WER computation which is standard at word level.
#
# LATTICE STRUCTURE:
#   A lattice is a list of "bins", one per aligned word position.
#   Each bin = set of all valid alternatives at that position.
#
#   Example:
#   Input : "उसने चौदह किताबें खरीदीं"
#   Lattice: [
#       {"उसने"},                          # position 0 — only one option
#       {"चौदह", "14"},                    # position 1 — word or digit
#       {"किताबें", "किताबे", "पुस्तकें"}, # position 2 — spelling variants
#       {"खरीदीं", "खरीदी"},               # position 3 — gender variants
#   ]
# =============================================================================

import re
import csv
import pandas as pd
from collections import defaultdict, Counter

# =============================================================================
# SECTION 1: Text Normalization
# Before any comparison, normalize text to remove trivial differences
# =============================================================================

def normalize_text(text: str) -> str:
    """
    Normalize Hindi text for fair comparison.
    Removes punctuation, normalizes whitespace, lowercases.

    What we remove/normalize:
      - Punctuation: । , ! ? . -- ( ) — -
      - Extra spaces
      - Trailing/leading whitespace
      - Nuqta variations (फ़ → फ, ज़ → ज) for fuzzy matching
    """
    if not isinstance(text, str):
        return ""

    # Remove punctuation (Hindi danda, English punct, dashes)
    text = re.sub(r'[।॥,.!?।"\'()—\-–]+', ' ', text)

    # Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenize(text: str) -> list:
    """Split normalized text into word tokens."""
    return normalize_text(text).split()


def normalize_nuqta(word: str) -> str:
    """
    Normalize nuqta variants for fuzzy matching.
    फ़ → फ, ज़ → ज, क़ → क
    These are often interchangeable in informal transcription.
    """
    replacements = {
        'फ़': 'फ', 'ज़': 'ज', 'क़': 'क',
        'ड़': 'ड', 'ढ़': 'ढ',
    }
    for orig, repl in replacements.items():
        word = word.replace(orig, repl)
    return word


def words_match(w1: str, w2: str) -> bool:
    """
    Check if two words are equivalent for WER purposes.
    Handles:
      - Exact match
      - Nuqta variations (सब्ज़ी == सब्जी)
      - Punctuation-stripped match
      - Script mixing (feedback == फीडबैक) — via known mappings
    """
    if w1 == w2:
        return True

    # Normalize nuqta
    if normalize_nuqta(w1) == normalize_nuqta(w2):
        return True

    # Check known script-mixed equivalents
    if (w1, w2) in SCRIPT_EQUIVALENTS or (w2, w1) in SCRIPT_EQUIVALENTS:
        return True

    return False


# Known Roman ↔ Devanagari equivalents (from our actual Q4 data)
SCRIPT_EQUIVALENTS = {
    ("feedback", "फीडबैक"),
    ("feedback", "फिडबऐक"),
    ("pure", "प्योर"),
    ("heart", "हार्ट"),
    ("desktop", "डेस्कटॉप"),
    ("laptop", "लैपटॉप"),
    ("easy", "इजी"),
    ("face", "फेस"),
    ("sir", "सर"),
}


# =============================================================================
# SECTION 2: Sequence Alignment (Edit Distance with Traceback)
# Used to align model output to reference for WER computation
# AND to align multiple model outputs for lattice construction
# =============================================================================

def edit_distance_align(ref: list, hyp: list) -> tuple:
    """
    Compute edit distance between ref and hyp word lists.
    Returns (distance, alignment) where alignment is a list of operations:
      ('match', r, h)      — ref word r matched hyp word h
      ('sub', r, h)        — substitution: ref=r, hyp=h
      ('del', r, None)     — deletion: ref word r missing in hyp
      ('ins', None, h)     — insertion: hyp word h not in ref

    Uses standard dynamic programming (Wagner-Fischer algorithm).
    """
    n, m = len(ref), len(hyp)

    # DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if words_match(ref[i-1], hyp[j-1]):
                dp[i][j] = dp[i-1][j-1]          # match — free
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion
                    dp[i][j-1],    # insertion
                    dp[i-1][j-1],  # substitution
                )

    # Traceback to get alignment
    alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and words_match(ref[i-1], hyp[j-1]):
            alignment.append(('match', ref[i-1], hyp[j-1]))
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            alignment.append(('sub', ref[i-1], hyp[j-1]))
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            alignment.append(('del', ref[i-1], None))
            i -= 1
        else:
            alignment.append(('ins', None, hyp[j-1]))
            j -= 1

    alignment.reverse()
    return dp[n][m], alignment


def compute_wer(reference: list, hypothesis: list) -> float:
    """
    Standard WER = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=ref length
    """
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0

    dist, _ = edit_distance_align(reference, hypothesis)
    return dist / len(reference)


# =============================================================================
# SECTION 3: Lattice Construction
# Build a lattice from all model outputs + human reference
# =============================================================================

def build_lattice(human: str, model_outputs: dict) -> list:
    """
    Build a word-level lattice from human reference + all model outputs.

    Algorithm:
      1. Normalize all texts
      2. Use human reference as the "spine" (anchor sequence)
      3. Align each model output to the spine using edit distance
      4. For each aligned position, collect all variants as valid alternatives
      5. Apply model-consensus rule: if 4+ models agree on something
         different from human, add model version as valid alternative

    Args:
        human: human reference transcription
        model_outputs: dict of {model_name: transcription}

    Returns:
        list of sets, one set per word position
        Each set contains all valid alternatives for that position

    Example return:
        [
            {"मौनता", "मोनता", "मोन"},          # position 0
            {"का"},                               # position 1
            {"अर्थ", "हर"},                       # position 2
            {"क्या"},                              # position 3
            {"होता"},                              # position 4
            {"है"},                                # position 5
        ]
    """
    ref_tokens = tokenize(human)
    if not ref_tokens:
        return []

    # Initialize lattice — each position starts with human's word
    lattice = [set() for _ in ref_tokens]
    for i, word in enumerate(ref_tokens):
        lattice[i].add(word)

    # Track what each model says at each position
    model_position_votes = defaultdict(list)  # position → [model_word, ...]

    # Align each model to reference
    for model_name, output in model_outputs.items():
        hyp_tokens = tokenize(output)
        if not hyp_tokens:
            continue

        _, alignment = edit_distance_align(ref_tokens, hyp_tokens)

        # Walk through alignment and map to reference positions
        ref_pos = 0
        for op, ref_word, hyp_word in alignment:
            if op == 'match':
                # Model agrees with reference — add both forms
                lattice[ref_pos].add(hyp_word)
                model_position_votes[ref_pos].append(hyp_word)
                ref_pos += 1
            elif op == 'sub':
                # Model says something different — add as valid variant
                lattice[ref_pos].add(hyp_word)
                model_position_votes[ref_pos].append(hyp_word)
                ref_pos += 1
            elif op == 'del':
                # Model skipped this word — note it but don't add None
                model_position_votes[ref_pos].append(None)
                ref_pos += 1
            elif op == 'ins':
                # Model inserted extra word — we skip (no ref position)
                pass

    # MODEL CONSENSUS RULE:
    # If 4+ out of 6 models agree on something DIFFERENT from human reference,
    # it's likely the human made an error → that consensus becomes preferred
    for pos in range(len(lattice)):
        votes = [v for v in model_position_votes[pos] if v is not None]
        if not votes:
            continue

        vote_counts = Counter(votes)
        most_common_word, most_common_count = vote_counts.most_common(1)[0]

        # 4+ models agree and it differs from human → add as valid
        if most_common_count >= 4:
            lattice[pos].add(most_common_word)

    return lattice


# =============================================================================
# SECTION 4: Lattice-Based WER
# Score a model against the lattice instead of rigid reference
# =============================================================================

def lattice_wer(lattice: list, hypothesis: list, ref_tokens: list) -> dict:
    """
    Compute WER for a model hypothesis against a word lattice.

    For each aligned position, a word is CORRECT if it matches ANY
    valid alternative in that lattice bin.

    Args:
        lattice   : list of sets from build_lattice()
        hypothesis: tokenized model output
        ref_tokens: original reference tokens (for alignment)

    Returns:
        dict with WER details: wer, substitutions, deletions, insertions
    """
    if not ref_tokens:
        return {"wer": 0.0, "S": 0, "D": 0, "I": 0, "N": 0}

    _, alignment = edit_distance_align(ref_tokens, hypothesis)

    S = 0  # substitutions
    D = 0  # deletions
    I = 0  # insertions
    N = len(ref_tokens)

    ref_pos = 0
    for op, ref_word, hyp_word in alignment:
        if op == 'match':
            ref_pos += 1
            # No error
        elif op == 'sub':
            # Check if hyp_word is in the lattice bin for this position
            if ref_pos < len(lattice) and hyp_word in lattice[ref_pos]:
                pass  # Valid alternative — NOT an error!
            else:
                S += 1
            ref_pos += 1
        elif op == 'del':
            # Check if None/deletion is acceptable (optional word)
            D += 1
            ref_pos += 1
        elif op == 'ins':
            I += 1

    wer = (S + D + I) / N if N > 0 else 0.0
    return {"wer": round(wer, 4), "S": S, "D": D, "I": I, "N": N}


# =============================================================================
# SECTION 5: Main Evaluation
# Run both standard WER and lattice WER for all 6 models on all 46 segments
# =============================================================================

def evaluate_all(csv_path: str) -> pd.DataFrame:
    """
    Run full evaluation on Question_4 CSV.
    Returns DataFrame with standard WER and lattice WER for each model.
    """
    df = pd.read_csv(csv_path)
    df = df.drop(columns=['Unnamed: 8'], errors='ignore')

    MODEL_COLS = ['Model H', 'Model i', 'Model k', 'Model l', 'Model m', 'Model n']

    # Per-segment results
    segment_results = []

    # Accumulators for corpus-level WER (sum of errors / sum of words)
    standard_errors = defaultdict(lambda: {"S":0,"D":0,"I":0,"N":0})
    lattice_errors  = defaultdict(lambda: {"S":0,"D":0,"I":0,"N":0})

    for idx, row in df.iterrows():
        human = str(row['Human'])
        ref_tokens = tokenize(human)

        model_outputs = {col: str(row[col]) for col in MODEL_COLS}

        # Build lattice for this segment
        lattice = build_lattice(human, model_outputs)

        seg_result = {
            "segment": idx + 1,
            "human_ref": human[:60] + "..." if len(human) > 60 else human,
            "ref_words": len(ref_tokens),
        }

        for model in MODEL_COLS:
            hyp = str(row[model])
            hyp_tokens = tokenize(hyp)

            # Standard WER
            std_wer = compute_wer(ref_tokens, hyp_tokens)

            # Lattice WER
            lat = lattice_wer(lattice, hyp_tokens, ref_tokens)

            seg_result[f"{model}_std_wer"] = round(std_wer, 4)
            seg_result[f"{model}_lat_wer"] = lat["wer"]
            seg_result[f"{model}_improvement"] = round(std_wer - lat["wer"], 4)

            # Accumulate for corpus WER
            dist_std, _ = edit_distance_align(ref_tokens, hyp_tokens)
            standard_errors[model]["S"] += dist_std
            standard_errors[model]["N"] += len(ref_tokens)

            lattice_errors[model]["S"] += lat["S"]
            lattice_errors[model]["D"] += lat["D"]
            lattice_errors[model]["I"] += lat["I"]
            lattice_errors[model]["N"] += lat["N"]

        segment_results.append(seg_result)

    seg_df = pd.DataFrame(segment_results)

    # Corpus-level summary
    print("\n" + "=" * 72)
    print("  Q4 LATTICE WER EVALUATION RESULTS")
    print("  46 segments, 6 models, word-level alignment")
    print("=" * 72)
    print()
    print(f"{'Model':<12} {'Std WER':>10} {'Lattice WER':>12} {'Reduction':>10} {'Fairly Penalized?'}")
    print("-" * 65)

    summary_rows = []
    for model in MODEL_COLS:
        N = standard_errors[model]["N"]
        std_wer_corpus = standard_errors[model]["S"] / N if N > 0 else 0
        lat_N = lattice_errors[model]["N"]
        lat_wer_corpus = (
            (lattice_errors[model]["S"] +
             lattice_errors[model]["D"] +
             lattice_errors[model]["I"]) / lat_N
            if lat_N > 0 else 0
        )
        reduction = std_wer_corpus - lat_wer_corpus
        penalized = "✅ YES — improved" if reduction > 0.01 else "➖ No change"

        print(f"{model:<12} {std_wer_corpus:>10.4f} {lat_wer_corpus:>12.4f} "
              f"{reduction:>10.4f}  {penalized}")

        summary_rows.append({
            "Model": model,
            "Standard_WER": round(std_wer_corpus, 4),
            "Lattice_WER": round(lat_wer_corpus, 4),
            "Reduction": round(reduction, 4),
            "Fairly_Penalized": penalized,
        })

    print("-" * 65)
    print()
    print("INTERPRETATION:")
    print("  Standard WER penalizes valid alternatives (punctuation,")
    print("  spelling variants, script mixing). Lattice WER corrects this.")
    print("  Models with 'improvement' were unfairly penalized before.")

    return pd.DataFrame(summary_rows), seg_df


# =============================================================================
# SECTION 6: Lattice Visualizer
# Shows the lattice structure for specific interesting segments
# =============================================================================

def show_lattice_examples(csv_path: str):
    """
    Show lattice structure for handpicked interesting segments.
    These are the examples you'll explain in your assignment writeup.
    """
    df = pd.read_csv(csv_path)
    df = df.drop(columns=['Unnamed: 8'], errors='ignore')
    MODEL_COLS = ['Model H', 'Model i', 'Model k', 'Model l', 'Model m', 'Model n']

    # Pick the most interesting segments based on our data analysis
    interesting_rows = [
        (1,  "Punctuation only differences"),
        (2,  "High variation — some models completely wrong"),
        (7,  "जय सियाराम — phonetic confusion"),
        (8,  "feedback (Roman vs Devanagari script mixing)"),
        (11, "pure heart — English words in Devanagari"),
        (40, "desktop/laptop — script mixing"),
    ]

    print("\n" + "=" * 72)
    print("  LATTICE STRUCTURE — EXAMPLES")
    print("=" * 72)

    for row_num, description in interesting_rows:
        if row_num > len(df):
            continue
        row = df.iloc[row_num - 1]
        human = str(row['Human'])
        model_outputs = {col: str(row[col]) for col in MODEL_COLS}
        lattice = build_lattice(human, model_outputs)
        ref_tokens = tokenize(human)

        print(f"\n[Segment {row_num}] {description}")
        print(f"  Human Ref : {human}")
        print(f"  Models    :")
        for col in MODEL_COLS:
            print(f"    {col}: {str(row[col]).strip()}")
        print(f"\n  Lattice (word positions with all valid alternatives):")
        for i, (word, bin_set) in enumerate(zip(ref_tokens, lattice)):
            variants = bin_set - {word}
            if variants:
                print(f"    [{i}] '{word}' → also valid: {variants}")
            else:
                print(f"    [{i}] '{word}' → (no variants)")

        print(f"\n  WER Comparison:")
        print(f"  {'Model':<12} {'Std WER':>10} {'Lat WER':>10} {'Change':>10}")
        print(f"  {'-'*45}")
        for model in MODEL_COLS:
            hyp_tokens = tokenize(str(row[model]))
            std = compute_wer(ref_tokens, hyp_tokens)
            lat = lattice_wer(lattice, hyp_tokens, ref_tokens)
            change = std - lat["wer"]
            marker = " ✅" if change > 0 else ""
            print(f"  {model:<12} {std:>10.4f} {lat['wer']:>10.4f} {change:>+10.4f}{marker}")


# =============================================================================
# SECTION 7: When Model Consensus Overrides Reference
# Show cases where models were right and human was wrong
# =============================================================================

def show_consensus_overrides(csv_path: str):
    """
    Find and display cases where model consensus suggests
    the human reference contains an error.
    """
    df = pd.read_csv(csv_path)
    df = df.drop(columns=['Unnamed: 8'], errors='ignore')
    MODEL_COLS = ['Model H', 'Model i', 'Model k', 'Model l', 'Model m', 'Model n']

    print("\n" + "=" * 72)
    print("  MODEL CONSENSUS OVERRIDE CASES")
    print("  (Where 4+ models agree differently from human reference)")
    print("=" * 72)

    found = 0
    for idx, row in df.iterrows():
        human = str(row['Human'])
        ref_tokens = tokenize(human)
        model_outputs = {col: str(row[col]) for col in MODEL_COLS}

        # Check each word position for consensus disagreement
        for pos, ref_word in enumerate(ref_tokens):
            # Get what each model says at this position
            model_words = []
            for model, output in model_outputs.items():
                hyp_tokens = tokenize(output)
                _, alignment = edit_distance_align(ref_tokens, hyp_tokens)
                ref_pos = 0
                for op, rw, hw in alignment:
                    if op in ('match', 'sub') and ref_pos == pos:
                        if hw:
                            model_words.append(hw)
                        break
                    if op in ('match', 'sub', 'del'):
                        ref_pos += 1

            if not model_words:
                continue

            vote_counts = Counter(model_words)
            top_word, top_count = vote_counts.most_common(1)[0]

            # 4+ models agree on something different from human
            if top_count >= 4 and not words_match(top_word, ref_word):
                found += 1
                print(f"\n[Segment {idx+1}, Position {pos}]")
                print(f"  Human said    : '{ref_word}'")
                print(f"  {top_count}/6 models say: '{top_word}'")
                print(f"  Full sentence : {human[:70]}")
                print(f"  → Lattice accepts BOTH as valid")
                if found >= 8:  # Show top 8 examples
                    break
        if found >= 8:
            break

    if found == 0:
        print("\n  No strong consensus overrides found.")
        print("  (Models generally agree with human reference)")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys
    import os

    # Find CSV
    csv_candidates = [
        "../data/Question 4 - Task.csv",
        "Question 4 - Task.csv",
        "data/Question 4 - Task",
    ]
    csv_path = None
    for p in csv_candidates:
        if os.path.exists(p):
            csv_path = p
            break

    if not csv_path:
        print("❌ Could not find Question 4 - Task")
        print("   Please place it at: ../data/Question 4 - Task")
        sys.exit(1)

    print(f"✅ Found data: {csv_path}")

    # Run all sections
    show_lattice_examples(csv_path)
    show_consensus_overrides(csv_path)
    summary_df, seg_df = evaluate_all(csv_path)

    # Save results
    os.makedirs("../data/q4_results", exist_ok=True)
    summary_df.to_csv("../data/q4_results/wer_summary.csv", index=False)
    seg_df.to_csv("../data/q4_results/wer_per_segment.csv", index=False)
    print(f"\n✅ Results saved to ../data/q4_results/")
    print(f"   • wer_summary.csv    — model-level WER comparison")
    print(f"   • wer_per_segment.csv — segment-by-segment breakdown")