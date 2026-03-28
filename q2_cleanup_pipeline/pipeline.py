# =============================================================================
# pipeline.py
# Q2 — Full ASR Cleanup Pipeline
# Connects: Number Normalizer + English Word Detector
# Runs on real Josh Talks dataset (fetches live from GCP public URLs)
# Josh Talks ASR Assignment
# =============================================================================
#
# WHAT THIS DOES:
#   1. Reads FT_Data_-_data.csv (104 recordings)
#   2. Converts private GCP URLs → public upload_goai URLs
#   3. Fetches each transcription JSON
#   4. Runs Number Normalizer on each segment
#   5. Runs English Detector on each segment
#   6. Saves full results to CSV + summary report
#
# HOW TO RUN:
#   python pipeline.py
#   python pipeline.py --limit 5        (test on first 5 recordings only)
#   python pipeline.py --limit 5 --out results/  (custom output folder)
# =============================================================================

import os
import json
import time
import argparse
import requests
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Import our two modules (must be in same folder)
from number_normalizer import normalize_numbers
from english_detector import detect_english_words

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

# Where your CSV lives (adjust path if needed)
CSV_PATH = "../data/FT Data -data.csv"

# Output folder
DEFAULT_OUT = "../data/pipeline_results/"

# Delay between HTTP requests (be polite to GCP)
REQUEST_DELAY = 0.3  # seconds

# Max retries per URL
MAX_RETRIES = 3


# -----------------------------------------------------------------------------
# SECTION 1: URL Converter
# Converts private GCP bucket URL → public upload_goai URL
# -----------------------------------------------------------------------------

def to_public_url(private_url: str) -> str:
    """
    Convert private joshtalks-data-collection URL to public upload_goai URL.

    Example:
        IN : https://storage.googleapis.com/joshtalks-data-collection/hq_data/hi/967179/825780_transcription.json
        OUT: https://storage.googleapis.com/upload_goai/967179/825780_transcription.json

    Pattern: keep only the last two path segments (user_id/filename)
    """
    parts = private_url.strip().split("/")
    # Last segment = filename (825780_transcription.json)
    # Second-to-last = user_id (967179)
    filename = parts[-1]
    user_id = parts[-2]
    return f"https://storage.googleapis.com/upload_goai/{user_id}/{filename}"


# -----------------------------------------------------------------------------
# SECTION 2: Fetcher
# Downloads a transcription JSON from GCP with retry logic
# -----------------------------------------------------------------------------

def fetch_transcription(url: str, retries: int = MAX_RETRIES) -> list:
    """
    Fetch transcription JSON from a public GCP URL.

    Args:
        url: public GCP URL to fetch
        retries: number of retry attempts on failure

    Returns:
        list of segment dicts like:
        [{"start": 0.11, "end": 14.42, "speaker_id": 245746, "text": "..."}]
        Returns empty list on failure.
    """
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                # File doesn't exist in public bucket — skip silently
                return []
            else:
                time.sleep(1)  # wait before retry
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print(f"    ⚠️  Failed after {retries} attempts: {url}")
    return []


# -----------------------------------------------------------------------------
# SECTION 3: Segment Processor
# Runs both pipeline stages on a single text segment
# -----------------------------------------------------------------------------

def process_segment(segment: dict) -> dict:
    """
    Run the full cleanup pipeline on one transcript segment.

    Input segment:
        {"start": 0.11, "end": 14.42, "speaker_id": 245746, "text": "..."}

    Output:
        {
          "start": 0.11,
          "end": 14.42,
          "speaker_id": 245746,
          "original_text": "...",           # raw ASR output
          "after_number_norm": "...",        # after number normalization
          "after_english_tag": "...",        # after English tagging
          "english_words": [...],            # list of detected English words
          "english_count": 2,               # how many English words found
          "numbers_changed": True/False,    # did number norm change anything?
        }
    """
    original = segment.get("text", "").strip()
    if not original:
        return None

    # Stage 1: Number Normalization
    after_numbers = normalize_numbers(original)
    numbers_changed = (after_numbers != original)

    # Stage 2: English Word Detection (run on number-normalized text)
    en_result = detect_english_words(after_numbers)
    after_english = en_result["tagged"]
    english_words = [d["word"] for d in en_result["en_words"]]
    english_count = en_result["en_count"]

    return {
        "start": segment.get("start", 0),
        "end": segment.get("end", 0),
        "speaker_id": segment.get("speaker_id", ""),
        "original_text": original,
        "after_number_norm": after_numbers,
        "after_english_tag": after_english,
        "english_words": "|".join(english_words),  # pipe-separated for CSV
        "english_count": english_count,
        "numbers_changed": numbers_changed,
    }


# -----------------------------------------------------------------------------
# SECTION 4: Recording Processor
# Processes all segments in one recording
# -----------------------------------------------------------------------------

def process_recording(row: pd.Series) -> dict:
    """
    Process one full recording (all its segments).

    Args:
        row: one row from FT_Data CSV

    Returns:
        dict with recording-level stats + list of processed segments
    """
    recording_id = row["recording_id"]
    user_id = str(row["transcription_url_gcp"]).split("/")[-2]
    private_url = row["transcription_url_gcp"]
    public_url = to_public_url(private_url)

    # Fetch transcription
    segments = fetch_transcription(public_url)

    if not segments:
        return {
            "recording_id": recording_id,
            "user_id": user_id,
            "duration": row["duration"],
            "status": "FAILED",
            "segments_processed": 0,
            "english_words_total": 0,
            "number_changes_total": 0,
            "processed_segments": [],
        }

    # Process each segment
    processed = []
    for seg in segments:
        result = process_segment(seg)
        if result:
            result["recording_id"] = recording_id
            processed.append(result)

    english_total = sum(s["english_count"] for s in processed)
    number_changes = sum(1 for s in processed if s["numbers_changed"])

    return {
        "recording_id": recording_id,
        "user_id": user_id,
        "duration": row["duration"],
        "status": "OK",
        "segments_processed": len(processed),
        "english_words_total": english_total,
        "number_changes_total": number_changes,
        "processed_segments": processed,
    }


# -----------------------------------------------------------------------------
# SECTION 5: Main Pipeline Runner
# -----------------------------------------------------------------------------

def run_pipeline(csv_path: str, output_dir: str, limit: int = None):
    """
    Main entry point — runs the full Q2 pipeline.

    Args:
        csv_path  : path to FT Data - data.csv
        output_dir: folder to save results
        limit     : process only first N recordings (None = all)
    """
    print("\n" + "=" * 65)
    print("  Q2 ASR CLEANUP PIPELINE")
    print("  Number Normalization + English Word Detection")
    print("=" * 65)

    # Load CSV
    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit)
        print(f"\n📋 Processing first {limit} recordings (of {len(pd.read_csv(csv_path))} total)")
    else:
        print(f"\n📋 Processing all {len(df)} recordings")

    print(f"📁 Output folder: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # ── Process all recordings ─────────────────────────────────────────────
    all_segments = []
    recording_summaries = []
    failed = 0

    print("\n🔄 Fetching and processing...\n")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Recordings"):
        result = process_recording(row)
        recording_summaries.append({
            "recording_id": result["recording_id"],
            "user_id": result["user_id"],
            "duration_s": result["duration"],
            "status": result["status"],
            "segments": result["segments_processed"],
            "english_words_found": result["english_words_total"],
            "number_normalizations": result["number_changes_total"],
        })
        if result["status"] == "OK":
            all_segments.extend(result["processed_segments"])
        else:
            failed += 1
        time.sleep(REQUEST_DELAY)

    # ── Save results ───────────────────────────────────────────────────────
    # 1. Full segment-level results
    segments_df = pd.DataFrame(all_segments)
    seg_path = os.path.join(output_dir, "pipeline_segments.csv")
    segments_df.to_csv(seg_path, index=False, encoding="utf-8-sig")

    # 2. Recording-level summary
    summary_df = pd.DataFrame(recording_summaries)
    sum_path = os.path.join(output_dir, "pipeline_summary.csv")
    summary_df.to_csv(sum_path, index=False, encoding="utf-8-sig")

    # 3. Examples report — interesting cases
    save_examples_report(segments_df, output_dir)

    # ── Print final stats ──────────────────────────────────────────────────
    print_stats(segments_df, summary_df, failed, len(df))

    print(f"\n✅ Results saved to: {output_dir}")
    print(f"   • pipeline_segments.csv  — all processed segments")
    print(f"   • pipeline_summary.csv   — per-recording stats")
    print(f"   • examples_report.txt    — interesting examples for assignment")


# -----------------------------------------------------------------------------
# SECTION 6: Examples Report
# Generates the before/after examples needed for the assignment writeup
# -----------------------------------------------------------------------------

def save_examples_report(df: pd.DataFrame, output_dir: str):
    """
    Save a human-readable report of interesting pipeline examples.
    This is what you'll include in your assignment submission.
    """
    report_path = os.path.join(output_dir, "examples_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("Q2 PIPELINE — EXAMPLES REPORT\n")
        f.write("Josh Talks ASR Assignment\n")
        f.write("=" * 70 + "\n\n")

        # ── Number normalization examples ──────────────────────────────────
        f.write("SECTION A: NUMBER NORMALIZATION EXAMPLES\n")
        f.write("-" * 50 + "\n\n")

        changed = df[df["numbers_changed"] == True].head(10)
        if len(changed) == 0:
            f.write("(No number changes found in processed data)\n\n")
        else:
            for i, (_, row) in enumerate(changed.iterrows(), 1):
                f.write(f"Example {i}:\n")
                f.write(f"  BEFORE : {row['original_text']}\n")
                f.write(f"  AFTER  : {row['after_number_norm']}\n")
                f.write(f"  Recording: {row['recording_id']}\n\n")

        # ── English word detection examples ────────────────────────────────
        f.write("\nSECTION B: ENGLISH WORD DETECTION EXAMPLES\n")
        f.write("-" * 50 + "\n\n")

        has_english = df[df["english_count"] > 0].head(10)
        if len(has_english) == 0:
            f.write("(No English words found in processed data)\n\n")
        else:
            for i, (_, row) in enumerate(has_english.iterrows(), 1):
                f.write(f"Example {i}:\n")
                f.write(f"  ORIGINAL : {row['original_text']}\n")
                f.write(f"  TAGGED   : {row['after_english_tag']}\n")
                f.write(f"  EN WORDS : {row['english_words']}\n")
                f.write(f"  Recording: {row['recording_id']}\n\n")

        # ── Edge cases ─────────────────────────────────────────────────────
        f.write("\nSECTION C: EDGE CASES & JUDGMENT CALLS\n")
        f.write("-" * 50 + "\n\n")
        f.write("""EDGE CASE 1 — दो (two vs give):
  Sentence : "मुझे दस रुपये दो"
  Output   : "मुझे 10 रुपये दो"
  Reasoning: "दो" at end of sentence after "रुपये" = verb (give), NOT number 2.
             Context check: prev word "रुपये" ∈ VERB_PRECEDING_WORDS → skip.

EDGE CASE 2 — Idiomatic number phrases:
  Sentence : "दो-चार बातें करो"
  Output   : "दो-चार बातें करो"  (unchanged)
  Reasoning: "दो-चार" is a fixed idiom meaning "a few things".
             Converting to "2-4" would destroy the idiomatic meaning.

EDGE CASE 3 — Indian number format (करोड़/लाख):
  Sentence : "पाँच करोड़ की सम्पत्ति"
  Output   : "5 करोड़ की सम्पत्ति"  (NOT 50000000)
  Reasoning: In India, large numbers keep the unit word (करोड़, लाख).
             "50000000" is unreadable and non-standard in Indian context.

EDGE CASE 4 — Devanagari English words (transcription guideline):
  Sentence : "हमारा प्रोजेक्ट भी था"
  Output   : "हमारा [EN]प्रोजेक्ट[/EN] भी था"
  Reasoning: Per transcription guidelines, English words spoken in Hindi
             conversations are written in Devanagari. These are CORRECT
             spellings (not errors) but still need English tagging for
             downstream processing.

EDGE CASE 5 — एक as article vs number:
  Sentence : "एक बात बताओ"
  Output   : "एक बात बताओ"  (unchanged)
  Reasoning: "एक" before a non-numeric noun = article (a/an in English).
             Converting to "1" would be grammatically wrong here.
""")

    print(f"\n📄 Examples report saved: {report_path}")


# -----------------------------------------------------------------------------
# SECTION 7: Stats Printer
# -----------------------------------------------------------------------------

def print_stats(seg_df: pd.DataFrame, sum_df: pd.DataFrame,
                failed: int, total: int):
    """Print a clean summary of pipeline results."""
    print("\n" + "=" * 65)
    print("  PIPELINE RESULTS SUMMARY")
    print("=" * 65)

    ok = total - failed
    print(f"\n📊 Recordings:")
    print(f"   Total   : {total}")
    print(f"   ✅ OK   : {ok}")
    print(f"   ❌ Failed: {failed}")

    if len(seg_df) > 0:
        total_segs = len(seg_df)
        num_changed = seg_df["numbers_changed"].sum()
        en_words_total = seg_df["english_count"].sum()
        segs_with_en = (seg_df["english_count"] > 0).sum()

        print(f"\n📊 Segments:")
        print(f"   Total processed     : {total_segs}")
        print(f"   With number changes : {num_changed} ({num_changed/total_segs*100:.1f}%)")
        print(f"   With English words  : {segs_with_en} ({segs_with_en/total_segs*100:.1f}%)")
        print(f"   Total English words : {int(en_words_total)}")

        # Top English words found
        if en_words_total > 0:
            from collections import Counter
            all_en = []
            for words in seg_df["english_words"].dropna():
                if words:
                    all_en.extend(words.split("|"))
            top = Counter(all_en).most_common(10)
            print(f"\n📊 Top English words found:")
            for word, count in top:
                print(f"   {word:20s} : {count} times")
    print()


# -----------------------------------------------------------------------------
# SECTION 8: Quick Demo (runs without fetching — uses hardcoded examples)
# -----------------------------------------------------------------------------

def run_demo():
    """
    Run a quick demo using hardcoded real examples from the dataset.
    Use this to test the pipeline without internet / CSV access.
    """
    print("\n" + "=" * 65)
    print("  Q2 PIPELINE — QUICK DEMO (Real Dataset Examples)")
    print("=" * 65)

    # Real segments from 825780_transcription.json
    demo_segments = [
        {"start": 0.11,  "end": 14.42, "speaker_id": 245746,
         "text": "हमारा प्रोजेक्ट भी था कि जो जन जाती पाई जाती है उधर की एरिया में"},
        {"start": 42.47, "end": 50.57, "speaker_id": 245746,
         "text": "पहली बारी था क्योंकि चलना नहीं आता न वहाँ का जो लैंड एरिया होता है"},
        {"start": 52.7,  "end": 66.83, "speaker_id": 245746,
         "text": "हमने टेंट गड़ा और रहा तो जब पता जैसी रात हुआ ना शाम मतलब छै सात में"},
        {"start": 103.19,"end": 117.29,"speaker_id": 245746,
         "text": "छः सात आठ किलोमीटर में नौ बजे है नौ उसके बाद लेकिन शांति बहुत मिला"},
        {"start": 257.66,"end": 264.08,"speaker_id": 245746,
         "text": "हम ने मिस्टेक किए कि हम लाइट नहीं ले गए थे"},
        {"start": 373.55,"end": 384.47,"speaker_id": 245746,
         "text": "अगर कहीं भी कैम्पिंग करने जाते हैं तो आसपास की एरिया में थोड़ा आग लहरा देना"},
        # Extra examples showing number normalization
        {"start": 1.0,   "end": 5.0,  "speaker_id": 111,
         "text": "मुझे दस रुपये दो और पच्चीस लोग बुलाओ"},
        {"start": 2.0,   "end": 6.0,  "speaker_id": 111,
         "text": "तीन सौ चौवन किताबें और पाँच करोड़ का बजट है"},
        {"start": 3.0,   "end": 7.0,  "speaker_id": 111,
         "text": "दो-चार बातें करो और इंटरव्यू की तैयारी करो"},
    ]

    print(f"\nProcessing {len(demo_segments)} real segments...\n")

    results = []
    for seg in demo_segments:
        r = process_segment(seg)
        if r:
            results.append(r)

    # Display results in a clean table
    print(f"{'#':<3} {'ORIGINAL':<45} {'AFTER NORM':<35} {'ENGLISH TAGS'}")
    print("-" * 130)
    for i, r in enumerate(results, 1):
        orig = r["original_text"][:43] + ".." if len(r["original_text"]) > 45 else r["original_text"]
        norm = r["after_number_norm"][:33] + ".." if len(r["after_number_norm"]) > 35 else r["after_number_norm"]
        en   = r["english_words"] if r["english_words"] else "-"
        changed = "🔢" if r["numbers_changed"] else "  "
        has_en  = "🔤" if r["english_count"] > 0 else "  "
        print(f"{i:<3} {orig:<45} {changed}{norm:<35} {has_en}{en}")

    print("\nLegend: 🔢 = number normalized  🔤 = English words found\n")

    # Detailed view
    print("\n" + "=" * 65)
    print("DETAILED OUTPUT")
    print("=" * 65)
    for i, r in enumerate(results, 1):
        if r["numbers_changed"] or r["english_count"] > 0:
            print(f"\n[Segment {i}]")
            print(f"  Original : {r['original_text']}")
            if r["numbers_changed"]:
                print(f"  Numbers  : {r['after_number_norm']}")
            if r["english_count"] > 0:
                print(f"  Tagged   : {r['after_english_tag']}")
                print(f"  EN words : {r['english_words']}")


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Q2 ASR Cleanup Pipeline — Number Norm + English Detection"
    )
    parser.add_argument(
        "--csv", default=CSV_PATH,
        help="Path to FT Data - data.csv"
    )
    parser.add_argument(
        "--out", default=DEFAULT_OUT,
        help="Output directory for results"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only first N recordings (for testing)"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run quick demo with hardcoded examples (no internet needed)"
    )
    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        run_pipeline(
            csv_path=args.csv,
            output_dir=args.out,
            limit=args.limit,
        )
