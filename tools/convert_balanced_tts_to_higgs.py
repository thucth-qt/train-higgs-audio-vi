import pandas as pd
import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import random

# --- CONFIG ---
CSV_PATH = '/root/data/higgs/balanced_tts_dataset/balanced_metadata_full.csv'
AUDIO_DIR = '/root/data/higgs/balanced_tts_dataset/wavs/'
OUTPUT_DIR = '/root/data/higgs/balanced_tts_dataset_higgs/'
MINI_OUTPUT_DIR = '/root/data/higgs/balanced_tts_dataset_higgs_mini/'
MINI_SPEAKER_COUNT = 3
MINI_SAMPLES_PER_SPEAKER = 100

# --- Ensure output dirs exist ---
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(MINI_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# --- Load CSV ---
df = pd.read_csv(CSV_PATH, low_memory=False)
if 'valid' in df.columns:
    df = df[df['valid'] == True]
if 'duration' in df.columns:
    df = df[(df['duration'] >= 0.5) & (df['duration'] <= 15.0)]

def clean_speaker_id(speaker):
    if pd.isna(speaker):
        return "unknown_speaker"
    cleaned = str(speaker).replace("@", "").replace("_male", "").replace("_female", "")
    return cleaned.lower().replace(" ", "_")

df['clean_speaker_id'] = df['speaker'].apply(clean_speaker_id)

# --- Limit to max 10,000 per speaker for main set (arbitrary large) ---
main_df = df.groupby('clean_speaker_id').head(10000)

# --- Check which columns exist for optional fields ---
has_emotion = 'emotion' in df.columns
has_topic = 'topic' in df.columns
has_dialect = 'dialect' in df.columns
has_sample_rate = 'sample_rate' in df.columns

# --- Prepare output ---
metadata_samples = []
speaker_counters = {}

for idx, row in tqdm(main_df.iterrows(), total=len(main_df), desc="Processing main dataset"):
    speaker_id = row['clean_speaker_id']
    if speaker_id not in speaker_counters:
        speaker_counters[speaker_id] = 0
    sample_id = f"{speaker_id}_{speaker_counters[speaker_id]:06d}"
    speaker_counters[speaker_id] += 1

    audio_filename = f"{sample_id}.wav"
    text_filename = f"{sample_id}.txt"
    input_audio_path = os.path.join(AUDIO_DIR, row['wav_filename'])
    output_audio_path = os.path.join(OUTPUT_DIR, audio_filename)
    output_text_path = os.path.join(OUTPUT_DIR, text_filename)

    # Copy audio
    if os.path.exists(input_audio_path):
        shutil.copy2(input_audio_path, output_audio_path)
    else:
        continue  # skip if audio missing

    # Write text
    text_content = str(row.get('normalized_text', row.get('text', '')))
    if not text_content or pd.isna(text_content):
        continue
    with open(output_text_path, 'w', encoding='utf-8') as f:
        f.write(text_content)

    # Metadata
    sample_entry = {
        "id": sample_id,
        "audio_file": audio_filename,
        "transcript_file": text_filename,
        "duration": float(row.get('duration', 0.0)),
        "speaker_id": speaker_id,
        "speaker_name": str(row.get('speaker', 'unknown')),
        "scene": "unknown",
        "emotion": str(row['emotion']) if has_emotion and not pd.isna(row['emotion']) else "neutral",
        "language": "vi",
        "gender": str(row.get('gender', 'unknown')),
        "quality_score": 1.0,
        "original_audio_path": input_audio_path,
        "user_instruction": "<audio> /translate",
        "task_type": "audio_generation"
    }
    if has_topic and not pd.isna(row['topic']):
        sample_entry["topic"] = str(row['topic'])
    if has_dialect and not pd.isna(row['dialect']):
        sample_entry["dialect"] = str(row['dialect'])
    if has_sample_rate and not pd.isna(row['sample_rate']):
        sample_entry["sample_rate"] = int(row['sample_rate'])
    metadata_samples.append(sample_entry)

# --- Save main metadata ---
metadata = {
    "dataset_info": {
        "total_samples": len(metadata_samples),
        "speakers": list(speaker_counters.keys()),
        "languages": ["vi"],
        "total_duration": round(sum([s['duration'] for s in metadata_samples]), 2),
        "avg_duration": round(sum([s['duration'] for s in metadata_samples])/len(metadata_samples), 2) if metadata_samples else 0,
        "created_from": [CSV_PATH],
        "dataset_type": "vietnamese_tts_csv",
        "sample_rate": 24000,
        "speaker_counts": speaker_counters
    },
    "samples": metadata_samples
}
with open(os.path.join(OUTPUT_DIR, "metadata.json"), 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"Main dataset written to {OUTPUT_DIR} with {len(metadata_samples)} samples.")

# --- Mini validation set: 3 speakers x 100 samples ---
mini_speakers = random.sample(list(speaker_counters.keys()), min(MINI_SPEAKER_COUNT, len(speaker_counters)))
mini_samples = []
for spk in mini_speakers:
    spk_samples = [s for s in metadata_samples if s['speaker_id'] == spk]
    mini_samples.extend(random.sample(spk_samples, min(MINI_SAMPLES_PER_SPEAKER, len(spk_samples))))

# Copy files and write mini metadata
for s in tqdm(mini_samples, desc="Copying mini validation set"):
    shutil.copy2(os.path.join(OUTPUT_DIR, s['audio_file']), os.path.join(MINI_OUTPUT_DIR, s['audio_file']))
    shutil.copy2(os.path.join(OUTPUT_DIR, s['transcript_file']), os.path.join(MINI_OUTPUT_DIR, s['transcript_file']))

mini_metadata = {
    "dataset_info": {
        "total_samples": len(mini_samples),
        "speakers": mini_speakers,
        "languages": ["vi"],
        "total_duration": round(sum([s['duration'] for s in mini_samples]), 2),
        "avg_duration": round(sum([s['duration'] for s in mini_samples])/len(mini_samples), 2) if mini_samples else 0,
        "created_from": [CSV_PATH],
        "dataset_type": "vietnamese_tts_csv_mini",
        "sample_rate": 24000,
        "speaker_counts": {spk:MINI_SAMPLES_PER_SPEAKER for spk in mini_speakers}
    },
    "samples": mini_samples
}
with open(os.path.join(MINI_OUTPUT_DIR, "metadata.json"), 'w', encoding='utf-8') as f:
    json.dump(mini_metadata, f, indent=2, ensure_ascii=False)

print(f"Mini validation set written to {MINI_OUTPUT_DIR} with {len(mini_samples)} samples.")
