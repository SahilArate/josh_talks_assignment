import pandas as pd
import re
import os
from tqdm import tqdm
from wordfreq import zipf_frequency
import Levenshtein
from collections import defaultdict

# -----------------------------
# Load File (robust path)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, "data", "Unique Words Data - Sheet1.csv")

print("Loading file from:", file_path)
df = pd.read_csv(file_path)

WORDS_COLUMN = df.columns[0]
words = df[WORDS_COLUMN].dropna().astype(str).unique()

# -----------------------------
# Helper Functions
# -----------------------------

def is_valid_chars(word):
    return re.match(r'^[\u0900-\u097F]+$', word) is not None

def is_valid_hindi_word(word):
    return zipf_frequency(word, 'hi') > 2.5

def is_likely_transliteration(word):
    patterns = ['कं', 'ट्र', 'प्र', 'फ', 'ग', 'ड']
    return any(p in word for p in patterns)

# -----------------------------
# Build Vocabulary
# -----------------------------
print("Building vocabulary...")

vocab = set()
for w in words:
    if zipf_frequency(w, 'hi') > 3:
        vocab.add(w)

vocab = list(vocab)

# -----------------------------
# Length-based optimization
# -----------------------------
length_vocab = defaultdict(list)
for w in vocab:
    length_vocab[len(w)].append(w)

closest_cache = {}

def fast_closest_word(word):
    if word in closest_cache:
        return closest_cache[word]

    word_len = len(word)

    candidates = []
    for l in range(word_len - 1, word_len + 2):
        candidates.extend(length_vocab.get(l, []))

    min_dist = float('inf')
    best_match = None

    for v in candidates:
        dist = Levenshtein.distance(word, v)

        if dist == 0:
            closest_cache[word] = (v, 0)
            return v, 0

        if dist < min_dist:
            min_dist = dist
            best_match = v

        if min_dist == 1:
            break

    closest_cache[word] = (best_match, min_dist)
    return best_match, min_dist

# -----------------------------
# Classification
# -----------------------------
results = []

print("Processing words...")

for word in tqdm(words):

    word = word.strip()

    label = "incorrect"
    confidence = "low"
    reason = ""

    # Rule 0: skip empty
    if not word:
        continue

    # Rule 1: invalid chars
    if not is_valid_chars(word):
        label = "incorrect"
        confidence = "high"
        reason = "Invalid characters"

    # Rule 2: very short words
    elif len(word) <= 2:
        label = "correct"
        confidence = "low"
        reason = "Too short to judge"

    # Rule 3: valid Hindi word
    elif is_valid_hindi_word(word):
        label = "correct"
        confidence = "high"
        reason = "High frequency Hindi word"

    # Rule 4: transliteration
    elif is_likely_transliteration(word):
        label = "correct"
        confidence = "medium"
        reason = "Likely English transliteration"

    # Rule 5: edit distance check
    else:
        match, dist = fast_closest_word(word)

        if match is not None:
            if dist <= 1:
                label = "incorrect"
                confidence = "high"
                reason = f"Very close to '{match}'"

            elif dist == 2:
                label = "incorrect"
                confidence = "medium"
                reason = f"Close to '{match}'"

            else:
                label = "incorrect"
                confidence = "low"
                reason = "No close match"
        else:
            label = "incorrect"
            confidence = "low"
            reason = "No vocab match"

    results.append([word, label, confidence, reason])

# -----------------------------
# Save Output
# -----------------------------
output_df = pd.DataFrame(results, columns=[
    "word", "label", "confidence", "reason"
])

output_path = os.path.join(BASE_DIR, "classified_words.csv")
output_df.to_csv(output_path, index=False)

# -----------------------------
# Stats
# -----------------------------
correct_count = (output_df['label'] == 'correct').sum()
incorrect_count = (output_df['label'] == 'incorrect').sum()

print("\n✅ DONE")
print(f"Correct words: {correct_count}")
print(f"Incorrect words: {incorrect_count}")
print(f"Saved to: {output_path}")

# -----------------------------
# Low confidence sample (for part C)
# -----------------------------
low_conf = output_df[output_df['confidence'] == 'low']

if len(low_conf) >= 50:
    sample = low_conf.sample(50, random_state=42)
    sample_path = os.path.join(BASE_DIR, "low_conf_sample.csv")
    sample.to_csv(sample_path, index=False)
    print(f"Low confidence sample saved to: {sample_path}")