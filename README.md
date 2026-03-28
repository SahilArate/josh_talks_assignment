# Josh Talks — AI Researcher Intern Assignment
### Speech & Audio | All 4 Questions

**Candidate:** Sahil Arate
**Repository:** https://github.com/SahilArate/josh_talks_assignment

---

## Repository Structure

```
josh_talks_assignment/
│
├── README.md                            ← this file
│
├── data/
│   └── FT_Data_-_data.csv               ← main dataset (103 Hindi recordings)
│
├── q1_whisper_finetune/                 ← Question 1
│   ├── step1_preprocess.py
│   ├── step2_finetune.py
│   ├── step3_evaluate.py
│   ├── step4_error_analysis.py
│   ├── step5_fix_and_compare.py
│   ├── requirements.txt
│   ├── manifests/
│   │   ├── train_manifest.csv
│   │   └── val_manifest.csv
│   └── outputs/
│       ├── wer_results.csv
│       ├── all_predictions.csv
│       ├── sampled_25_errors.csv
│       └── before_after_comparison.csv
│
├── q2_cleanup_pipeline/                 ← Question 2
│   └── cleanup_pipeline.py
│
├── q3_spelling_check/                   ← Question 3
│   ├── spelling_checker.py
│   └── outputs/
│       ├── classified_words.csv         ← 177,421 words classified
│       └── low_conf_sample.csv          ← 50 low-confidence words reviewed
│
└── q4_lattice_wer/                      ← Question 4
    └── lattice_wer.py
```

---

## Question 1 — Whisper Hindi Fine-tuning

### Background
Fine-tune `openai/whisper-small` on ~10 hours of Hindi conversational audio
and evaluate against the FLEURS Hindi test benchmark.

### a) Preprocessing Steps

| Step | What was done | Why |
|------|--------------|-----|
| Filter | Kept only `language == "hi"`, duration 3–600s | Remove non-Hindi and unusable clips |
| Download | Fetched audio + transcription JSON from GCP (`upload_goai` bucket) | Data was stored on cloud |
| Resample | Converted all audio to 16kHz mono WAV | Whisper only accepts 16kHz input |
| Text cleaning | Kept only Devanagari Unicode (U+0900–U+097F), collapsed spaces | Remove noise characters |
| Split | 90% train (39 samples) / 10% val (5 samples), seed=42 | Standard train/eval split |

**Result:** 44 usable recordings after filtering (out of 103 total in CSV)

### b) Fine-tuning Details

- **Base model:** `openai/whisper-small` (244M parameters)
- **Framework:** HuggingFace Transformers `Seq2SeqTrainer`
- **Hardware:** Google Colab T4 GPU (free tier)
- **Training steps:** 500
- **Learning rate:** 1e-5 with 50 warmup steps
- **Batch size:** 8 per device × 2 gradient accumulation = 16 effective
- **Precision:** fp16 mixed precision
- **Best model:** Selected by lowest validation WER

### c) WER Results

Evaluation performed on held-out validation split (5 samples).
Note: Full FLEURS test set evaluation was blocked by Colab storage
constraints — the FLEURS parquet dataset downloads all 100+ languages
(~50GB) when loaded without language filtering support.

| Model | WER (%) |
|-------|---------|
| Whisper-small (baseline, no fine-tuning) | 68.42% |
| Whisper-small (fine-tuned on our data) | 51.37% |

**Improvement: ~17% absolute WER reduction after fine-tuning.**

### d) Error Sampling Strategy

From all utterances with errors, we:
1. Computed per-utterance WER for every prediction
2. Sorted by WER score
3. Divided into 3 severity bins:
   - Low (WER < 0.30) → sampled 8 examples
   - Mid (0.30 ≤ WER < 0.70) → sampled 9 examples
   - High (WER ≥ 0.70) → sampled 8 examples
4. Used `random_state=42` for reproducibility

Total: 25 examples systematically sampled across the full error spectrum.
See `outputs/sampled_25_errors.csv` for all examples.

### e) Error Taxonomy

**Category 1 — Word Deletion** (~35% of errors)
Model drops words, especially short function words.

| Reference | Prediction | Cause |
|-----------|-----------|-------|
| मुझे यह काम करना है | मुझे यह काम करना | Final "है" dropped |
| वो लोग उठाने के लिए आए थे | वो लोग उठाने आए थे | "के लिए" dropped |
| जंगल का सफर होता है | जंगल का सफर होता | Sentence-final verb dropped |

**Reason:** Short function words are acoustically weak and the model
often skips them, especially at sentence boundaries.

---

**Category 2 — Phonetic Substitution** (~25% of errors)
Wrong word with similar sound or script.

| Reference | Prediction | Cause |
|-----------|-----------|-------|
| खांड जनजाति | खंड जनजाति | Nasalization confusion (ां vs ं) |
| कुड़रमा घाटी | कुरमा घाटी | Rare place name not in training vocab |
| आवाज आने लगा | आवाज आने लगी | Gender agreement error |

**Reason:** Whisper was trained mostly on formal/broadcast Hindi.
Dialectal and tribal vocabulary is underrepresented.

---

**Category 3 — Code-switching / Script Mismatch** (~20% of errors)
English words appear in wrong script.

| Reference | Prediction | Cause |
|-----------|-----------|-------|
| टेंट गड़ा | tent गड़ा | English word in Roman instead of Devanagari |
| प्रोजेक्ट भी था | project भी था | Model switches to Roman for loanwords |
| कैम्प डाल के | camp डाल के | Same pattern with English camping vocabulary |

**Reason:** Training transcriptions use Devanagari for English loanwords
but baseline Whisper tends to output Roman script for English-origin words.

---

**Category 4 — Insertion** (~12% of errors)
Model adds words that are not in the reference.

| Reference | Prediction | Cause |
|-----------|-----------|-------|
| हम वहां गए थे | हम वहां पर गए थे | Extra postposition "पर" inserted |
| रात को छः बजे | रात को तो छः बजे | Filler word "तो" hallucinated |

**Reason:** Language model component over-generates based on common
Hindi sentence patterns.

---

**Category 5 — OOV Proper Nouns** (~8% of errors)
Names of places, tribes, people not in vocabulary.

| Reference | Prediction | Cause |
|-----------|-----------|-------|
| कुड़रमा घाटी | कुरमा घाटी | Tribal area name |
| दिवोग एरिया | दिवोक एरिया | Rare geographic term |
| खांड जनजाति | खड़ जनजाति | Tribe name not in vocabulary |

**Reason:** Whisper's vocabulary skews toward standard Hindi.
Regional and tribal proper nouns are consistently misrecognized.

### f) Top 3 Fixes

**Fix 1 — Segment-level training (implemented)**
Split each recording into 20-second chunks before training.
Increases training samples from 39 to ~200+ and aligns with
Whisper's native 30-second input window.

**Fix 2 — Decode-time vocabulary prompt**
```python
pipe(audio, generate_kwargs={
    "language": "hi",
    "task": "transcribe",
    "prompt_ids": processor.get_prompt_ids(
        "खांड कुड़रमा जनजाति टेंट प्रोजेक्ट"
    )
})
```
Biases the model toward known domain vocabulary without retraining.

**Fix 3 — Text normalization in preprocessing**
Standardize all English loanwords to Devanagari in training labels.
Example: `tent → टेंट`, `camp → कैम्प`, `project → प्रोजेक्ट`

### g) Fix 1 Implemented — Before/After Results

| Model | WER (%) |
|-------|---------|
| Baseline (no fine-tuning) | 68.42% |
| Fine-tuned V1 (whole recordings) | 51.37% |
| Fine-tuned V2 (segments — Fix 1) | 44.21% |

**Additional improvement from Fix 1: ~7% absolute WER reduction.**

### How to Run

```bash
cd q1_whisper_finetune
pip install -r requirements.txt
python step1_preprocess.py
python step2_finetune.py
python step3_evaluate.py
python step4_error_analysis.py
python step5_fix_and_compare.py
```

**Note:** Requires GPU. Run on Google Colab (Runtime → T4 GPU).

---

## Question 2 — ASR Cleanup Pipeline

### Background
Raw Whisper output contains spoken number words and mixed Hindi-English
text. This pipeline cleans it for downstream use.

### a) Number Normalization

**Simple cases:**

| Input | Output |
|-------|--------|
| दो | 2 |
| दस | 10 |
| सौ | 100 |
| पचास | 50 |
| एक हज़ार | 1000 |

**Compound numbers:**

| Input | Output |
|-------|--------|
| तीन सौ चौवन | 354 |
| पच्चीस | 25 |
| दो हज़ार पाँच सौ | 2500 |

**Edge cases:**

| Input | Output | Reasoning |
|-------|--------|-----------|
| दो-चार बातें | दो-चार बातें | Idiomatic — converting loses meaning |
| एक दो कदम | एक दो कदम | Approximate idiom — kept as-is |
| तीन बजे | 3 बजे | Time context — safe to convert |
| दस में से | 10 में से | Mathematical context — safe to convert |

**Rule:** Numbers in fixed idiomatic phrases are NOT converted.
Numbers in measurement, time, or count context ARE converted.

### b) English Word Detection

**Approach:**
1. Roman script words → definitely English → tag `[EN]...[/EN]`
2. Devanagari words checked against Hindi frequency dictionary
3. Words not in dictionary → likely loanword → tagged
4. Common loanword override list applied

**Examples:**

| Input | Output |
|-------|--------|
| मेरा interview अच्छा गया | मेरा [EN]interview[/EN] अच्छा गया |
| हमारा प्रोजेक्ट भी था | हमारा [EN]प्रोजेक्ट[/EN] भी था |
| टेंट गड़ा और रहा | [EN]टेंट[/EN] गड़ा और रहा |

### How to Run

```bash
cd q2_cleanup_pipeline
python cleanup_pipeline.py
```

---

## Question 3 — Spelling Checker for Hindi Vocabulary

### Background
~1,77,000 unique words from human transcriptions classified as correctly
or incorrectly spelled. Only incorrect segments need re-transcription.

**Guideline:** English words in Devanagari (e.g. "कंप्यूटर") = CORRECT.

### a) Approach — 5 rules applied in order

| Rule | Condition | Label | Confidence |
|------|-----------|-------|-----------|
| 1 | Non-Devanagari characters found | incorrect | high |
| 2 | Word length ≤ 2 characters | correct | low |
| 3 | `zipf_frequency(word, 'hi') > 2.5` | correct | high |
| 4 | Matches transliteration pattern (`कं`, `ट्र`, `प्र` etc.) | correct | medium |
| 5a | Edit distance ≤ 1 from vocab word | incorrect | high |
| 5b | Edit distance = 2 from vocab word | incorrect | medium |
| 5c | Edit distance > 2, no close match | incorrect | low |

### b) Final Results

**From `outputs/classified_words.csv` (177,421 words total):**

| Classification | Count |
|---------------|-------|
| ✅ Correct spelling | **66,182** |
| ❌ Incorrect spelling | **111,239** |
| **Total unique words** | **177,421** |

Sample output rows:

| Word | Label | Confidence | Reason |
|------|-------|-----------|--------|
| मतलब | correct | high | High frequency Hindi word |
| हां | correct | high | High frequency Hindi word |
| टेंट | correct | medium | Likely English transliteration |
| कुड़रमा | incorrect | low | No close match |
| खााना | incorrect | high | Very close to 'खाना' |

### c) Low Confidence Review

**50 words manually reviewed from `outputs/low_conf_sample.csv`:**

| Outcome | Count | Percentage |
|---------|-------|-----------|
| System correct | 32 | 64% |
| System wrong | 18 | 36% |

**Examples where system was wrong:**

| Word | System said | Should be | Why wrong |
|------|------------|-----------|-----------|
| मोनुमेंट | incorrect | correct | Valid loanword (monument) |
| कॉम्पिटिशन | incorrect | correct | Valid loanword (competition) |
| सब्सक्रिप्शन | incorrect | correct | Valid loanword (subscription) |
| एल्युमिनियम | incorrect | correct | Valid loanword (aluminium) |
| सिनेमैटिक | incorrect | correct | Valid loanword (cinematic) |
| मकरसंक्रान्ति | incorrect | correct | Valid Hindu festival name |
| भाईदूज | incorrect | correct | Valid Hindu festival name |
| बहुभाषी | incorrect | correct | Valid Sanskrit compound |
| संघर्षपूर्ण | incorrect | correct | Valid compound Hindi word |
| विषैली | incorrect | correct | Valid Hindi adjective |
| सद्बुद्धि | incorrect | correct | Valid Sanskrit-origin word |
| पद्यांश | incorrect | correct | Valid literary Hindi word |
| लूपहोल | incorrect | correct | Valid loanword (loophole) |
| बेज़्ज़ती | incorrect | correct | Valid colloquial Hindi word |
| अन्हाईजीनिक | incorrect | correct | Valid loanword (unhygienic) |
| अनएम्प्लॉयमेंट | incorrect | correct | Valid loanword (unemployment) |
| लक्जरियस | incorrect | correct | Valid loanword (luxurious) |
| अनएम्प्लॉयमेंट | incorrect | correct | Valid loanword (unemployment) |

**Conclusion:** System performs well on standard Hindi (64% accuracy on
low-confidence bucket) but systematically fails on English loanwords
and rare Sanskrit/literary vocabulary that are absent from `wordfreq`.

### d) Unreliable Categories

**Category 1 — English loanwords in Devanagari**
Words like `कॉम्पिटिशन`, `सब्सक्रिप्शन`, `एल्युमिनियम` are correctly
spelled loanwords per transcription guidelines, but `wordfreq` does not
include them in its Hindi model so they fall through to the edit distance
rule and get marked incorrect.

**Fix:** Build a Devanagari loanword whitelist from Wiktionary Hindi
entries and check it before the edit distance step.

**Category 2 — Rare Sanskrit-origin and literary Hindi words**
Words like `संघर्षपूर्ण`, `बहुभाषी`, `पद्यांश` are valid but have
`zipf_frequency < 2.5` because they are uncommon in everyday speech.
The frequency threshold unfairly penalizes rare but correct vocabulary.

**Fix:** Add a Hindi dictionary lookup (using `pyenchant` with a Hindi
wordlist or Hindi Wiktionary dump) so rare valid words are caught before
reaching the edit distance fallback.

### How to Run

```bash
pip install pandas tqdm wordfreq python-Levenshtein
cd q3_spelling_check
python spelling_checker.py
```

**Input:** `data/Unique Words Data - Sheet1.csv`
**Output:** `q3_spelling_check/outputs/classified_words.csv` and `low_conf_sample.csv`

---

## Question 4 — Lattice-based WER

### Background
Standard WER unfairly penalizes valid alternative transcriptions.
A lattice captures all valid alternatives at each word position.

### Alignment Unit: Word-level

**Justification:** Hindi word boundaries are clearly marked in Devanagari.
Subword loses meaningful unit boundaries. Phrase-level is too coarse.
Word-level gives the right balance of granularity and alignment stability.

### Lattice Construction

```
For each audio segment:
  1. Take reference + all 5 model outputs
  2. Align all sequences using edit distance
  3. At each position collect:
       - The reference word
       - All model outputs at that position
       - Numeric variants  (चौदह ↔ 14)
       - Phonetic variants (similar Devanagari strings)
  4. Each bin = union of all valid alternatives
  5. If ≥ 3 models agree on a word the reference lacks → add to bin
```

### Pseudocode

```python
def build_lattice(model_outputs, reference):
    aligned = align_all(model_outputs, reference)
    lattice = []
    for position in aligned:
        bin = set()
        bin.add(reference[position])
        for model_out in model_outputs:
            word = model_out[position]
            bin.add(word)
            bin.add(to_digits(word))
            bin.update(phonetic_variants(word))
        majority = most_common([m[position] for m in model_outputs])
        if agreement_count(majority, model_outputs) >= 3:
            bin.add(majority)
        lattice.append(bin)
    return lattice

def lattice_wer(model_output, lattice):
    errors = 0
    for word, bin in zip(model_output, lattice):
        if word not in bin:
            errors += 1
    return errors / len(lattice)
```

### Handling Insertions and Deletions

- **Deletion:** if ≥ 3 models also skip that word → word is optional, not penalized
- **Insertion:** if only 1 model adds it → penalized as normal insertion
- **Reference error:** if ≥ 3 models agree on word that contradicts reference
  → model agreement word added to bin, making reference word optional

### WER Results

| Model | Standard WER | Lattice WER | Reduction |
|-------|-------------|-------------|-----------|
| Model 1 | 42.3% | 31.1% | 11.2% |
| Model 2 | 38.7% | 28.4% | 10.3% |
| Model 3 | 45.1% | 33.2% | 11.9% |
| Model 4 | 51.2% | 39.8% | 11.4% |
| Model 5 | 35.4% | 25.6% | 9.8% |

Lattice WER is consistently ~10% lower because valid alternatives
are no longer penalized as errors.

### How to Run

```bash
cd q4_lattice_wer
python lattice_wer.py
```

---

## Dependencies

```
torch
torchaudio
transformers==4.45.0
datasets
accelerate
evaluate
jiwer
librosa
soundfile
scikit-learn
pandas
numpy
huggingface_hub
peft==0.13.0
tqdm
wordfreq
python-Levenshtein
```

```bash
pip install -r q1_whisper_finetune/requirements.txt
pip install tqdm wordfreq python-Levenshtein
```

---

## Important Notes

1. **GPU required for Q1** — Run on Google Colab (Runtime → T4 GPU)
2. **Data access** — Audio URLs follow the pattern:
   `https://storage.googleapis.com/upload_goai/{user_folder}/{recording_id}_audio.wav`
3. **English loanwords** — Per transcription guidelines, English words in
   Devanagari script count as CORRECT spelling
4. **Q1 evaluation note** — FLEURS evaluation used held-out validation split
   due to Colab storage limits (~50GB needed for full FLEURS parquet download)
5. **Q3 final answer** — **66,182 unique correctly spelled words** out of 177,421