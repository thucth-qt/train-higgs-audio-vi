import os
import json
from tqdm import tqdm

DATA_DIR = "/root/data/higgs/train-higgs-audio-vi/vietnamese_training_data"
META_PATH = os.path.join(DATA_DIR, "metadata.json")

with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

ids = set()
missing_audio = []
missing_text = []
bad_samples = []
for sample in tqdm(meta["samples"], desc="Checking samples"):
    sid = sample["id"]
    if sid in ids:
        print(f"Duplicate id: {sid}")
    ids.add(sid)
    audio_path = os.path.join(DATA_DIR, sample["audio_file"])
    text_path = os.path.join(DATA_DIR, sample["transcript_file"])
    if not os.path.isfile(audio_path):
        missing_audio.append(audio_path)
    if not os.path.isfile(text_path):
        missing_text.append(text_path)
    # Optionally, check for empty or corrupt files
    if os.path.isfile(audio_path) and os.path.getsize(audio_path) == 0:
        bad_samples.append(audio_path)
    if os.path.isfile(text_path):
        with open(text_path, "r", encoding="utf-8") as tf:
            if not tf.read().strip():
                bad_samples.append(text_path)

print(f"Total samples: {len(meta['samples'])}")
print(f"Missing audio files: {len(missing_audio)}")
print(f"Missing text files: {len(missing_text)}")
print(f"Empty/corrupt files: {len(bad_samples)}")

if missing_audio:
    print("Missing audio files:", missing_audio[:10])
if missing_text:
    print("Missing text files:", missing_text[:10])
if bad_samples:
    print("Empty/corrupt files:", bad_samples[:10])