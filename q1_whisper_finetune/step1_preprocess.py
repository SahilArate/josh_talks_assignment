

import os
import re
import json
import pandas as pd
import requests
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from pathlib import Path

BASE_DIR      = Path(__file__).parent         
DATA_DIR      = BASE_DIR.parent / "data"     
AUDIO_RAW     = BASE_DIR / "audio_raw"
AUDIO_16K     = BASE_DIR / "audio_16k"
TRANS_DIR     = BASE_DIR / "transcriptions"
MANIFESTS_DIR = BASE_DIR / "manifests"

for d in [AUDIO_RAW, AUDIO_16K, TRANS_DIR, MANIFESTS_DIR]:
    d.mkdir(exist_ok=True)

CSV_PATH = DATA_DIR / "FT Data - data.csv"

df = pd.read_csv(CSV_PATH)
print(f"Total rows in CSV: {len(df)}")


df = df[df['language'] == 'hi']
df = df[(df['duration'] >= 3) & (df['duration'] <= 600)]
df = df.reset_index(drop=True)
print(f"After filtering: {len(df)} recordings")


def download_file(url, save_path):
    if Path(save_path).exists():
        return True   # skip if already downloaded
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(r.content)
        return True
    except Exception as e:
        print(f"  ❌ Download failed: {url}\n     Error: {e}")
        return False


def resample_to_16k(input_path, output_path):
    if Path(output_path).exists():
        return True
    try:
        audio, _ = librosa.load(input_path, sr=16000, mono=True)
        sf.write(output_path, audio, 16000)
        return True
    except Exception as e:
        print(f"  ❌ Resample failed: {input_path}\n     Error: {e}")
        return False


def clean_text(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)                          
    text = re.sub(r'[^\u0900-\u097F\s\u0964\u0965.,!?]', '', text) 
    return text.strip()

successful_ids = []

for i, row in df.iterrows():
    rec_id = row['recording_id']
    print(f"[{i+1}/{len(df)}] Recording {rec_id}...")

    audio_raw_path = AUDIO_RAW  / f"{rec_id}.wav"
    audio_16k_path = AUDIO_16K  / f"{rec_id}.wav"
    trans_path     = TRANS_DIR  / f"{rec_id}.json"


    if not download_file(row['rec_url_gcp'], audio_raw_path):
        continue

    if not download_file(row['transcription_url_gcp'], trans_path):
        continue

   
    if not resample_to_16k(audio_raw_path, audio_16k_path):
        continue

    successful_ids.append(rec_id)

print(f"\n✅ Successfully processed: {len(successful_ids)}/{len(df)} recordings")

records = []

for rec_id in successful_ids:
    trans_path     = TRANS_DIR / f"{rec_id}.json"
    audio_16k_path = AUDIO_16K / f"{rec_id}.wav"

    with open(trans_path, encoding='utf-8') as f:
        segments = json.load(f)

    full_text = " ".join(
        seg.get('text', '') for seg in segments if seg.get('text', '').strip()
    )
    full_text = clean_text(full_text)

    if len(full_text) < 10:
        print(f"  ⚠️  Skipping {rec_id} - transcript too short")
        continue

    records.append({
        "recording_id": rec_id,
        "audio_path":   str(audio_16k_path),
        "text":         full_text,
        "duration":     segments[-1]['end'] if segments else 0,
        "num_segments": len(segments)
    })

dataset_df = pd.DataFrame(records)
print(f"\nUsable samples: {len(dataset_df)}")

train_df, val_df = train_test_split(dataset_df, test_size=0.1, random_state=42)
train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)

train_df.to_csv(MANIFESTS_DIR / "train_manifest.csv", index=False)
val_df.to_csv(  MANIFESTS_DIR / "val_manifest.csv",   index=False)

print(f"Train: {len(train_df)} | Val: {len(val_df)}")
print(f"✅ Saved to {MANIFESTS_DIR}")
print("\n=== STEP 1 COMPLETE ===")