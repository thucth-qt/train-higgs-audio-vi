import pandas as pd
import soundfile as sf
import os
import json
from pathlib import Path
import io
from tqdm import tqdm
import concurrent.futures
import traceback

def process_sample(sample_data):
    """
    Verarbeitet ein einzelnes Audiosample: dekodiert Audio, schreibt WAV- und TXT-Dateien
    und gibt den Metadateneintrag und die Dauer zurück.
    Diese Funktion ist so konzipiert, dass sie in einem separaten Thread ausgeführt wird.

    Args:
        sample_data (dict): Ein Wörterbuch mit den Daten für ein einzelnes Sample,
                            einschließlich 'index', 'row', 'output_path', 'dataset_name'.

    Returns:
        tuple: Ein Tupel, das den Metadateneintrag (dict) und die Dauer des Audios (float) enthält.
               Gibt (None, None) zurück, wenn ein Fehler auftritt.
    """
    try:
        index = sample_data['index']
        row = sample_data['row']
        output_path = sample_data['output_path']
        dataset_name = sample_data['dataset_name']
        file_path_name = sample_data['file_path_name'] # Originaler Parquet-Dateiname für besseres Debugging

        audio_info = row['audio']
        transcription = row['transcription']

        # Audio-Bytes dekodieren
        audio_bytes = audio_info['bytes']
        audio_data, sampling_rate = sf.read(io.BytesIO(audio_bytes), dtype='float32')
        
        duration = len(audio_data) / sampling_rate

        # Eindeutige Dateinamen generieren
        file_id = f"{dataset_name}_speaker_{index:06d}"
        wav_filename = f"{file_id}.wav"
        txt_filename = f"{file_id}.txt"
        
        wav_filepath = output_path / wav_filename
        txt_filepath = output_path / txt_filename

        # E/A-Operationen: Schreiben der WAV- und TXT-Dateien
        sf.write(wav_filepath, audio_data, sampling_rate)
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            f.write(transcription)

        # Metadateneintrag für dieses Sample erstellen
        sample_entry = {
            "id": file_id,
            "audio_file": str(wav_filename),
            "transcript_file": str(txt_filename),
            "duration": round(duration, 2),
            "speaker_id": f"{dataset_name}_speaker",
            "speaker_name": dataset_name.capitalize(),
            "scene": "unknown",
            "emotion": "neutral",
            "language": "vi",
            "gender": "unknown",
            "quality_score": 1.0,
            "original_audio_path": audio_info.get('path', f"from_{file_path_name}_row_{row.name}"),
            "user_instruction": "<audio> /translate",
            "task_type": "audio_generation"
        }
        
        return sample_entry, duration

    except Exception as e:
        # Fehlerinformationen für die spätere Überprüfung ausgeben
        print(f"\nFehler bei der Verarbeitung von Sample {index}: {e}")
        traceback.print_exc()
        return None, None

def process_parquet_files(input_dir, output_dir, dataset_name="huo", max_workers=None):
    """
    Verarbeitet Parquet-Dateien, extrahiert Audio und Transkriptionen und generiert 
    entsprechende WAV-, TXT-Dateien und eine metadata.json mithilfe von Multithreading
    für E/A-Operationen.
    """
    if max_workers is None:
        # Eine gute Standardeinstellung ist die Anzahl der CPUs + 4, was bei E/A-gebundenen Aufgaben gut funktioniert.
        # os.cpu_count() kann None zurückgeben, daher der Fallback auf 4.
        max_workers = (os.cpu_count() or 4) + 4

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata_samples = []
    total_duration = 0.0
    processed_count = 0
    
    parquet_files = sorted(list(input_path.glob("train-*.parquet")))
    if not parquet_files:
        print(f"Keine Parquet-Dateien im Verzeichnis gefunden: {input_path}")
        return

    # --- Schritt 1: Gesamtzahl der Samples für die Fortschrittsanzeige berechnen ---
    print("Berechne Gesamtzahl der Samples für die Fortschrittsanzeige...")
    total_samples = 0
    for file_path in tqdm(parquet_files, desc="Dateien scannen"):
        try:
            df = pd.read_parquet(file_path, columns=['audio'])
            total_samples += len(df)
        except Exception as e:
            print(f"Fehler beim Scannen der Datei {file_path}: {e}")
    
    if total_samples == 0:
        print("Keine Samples in den Parquet-Dateien gefunden.")
        return
        
    print(f"{total_samples} Samples gefunden, starte Verarbeitung mit bis zu {max_workers} Threads...")

    # --- Schritt 2: Verarbeitung der Dateien mit einem ThreadPoolExecutor ---
    global_sample_index = 0
    with tqdm(total=total_samples, desc="Samples verarbeiten", unit="sample") as pbar:
        for file_path in parquet_files:
            try:
                df = pd.read_parquet(file_path)
                
                tasks = []
                # Erstellen Sie einen ThreadPoolExecutor, um Aufgaben für die aktuelle Parquet-Datei zu verwalten
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Senden Sie eine Aufgabe für jede Zeile im DataFrame
                    for _, row in df.iterrows():
                        sample_data = {
                            'index': global_sample_index,
                            'row': row,
                            'output_path': output_path,
                            'dataset_name': dataset_name,
                            'file_path_name': file_path.name,
                        }
                        tasks.append(executor.submit(process_sample, sample_data))
                        global_sample_index += 1
                    
                    # Verarbeiten Sie die Ergebnisse, sobald sie eintreffen
                    for future in concurrent.futures.as_completed(tasks):
                        result, duration = future.result()
                        if result and duration is not None:
                            metadata_samples.append(result)
                            total_duration += duration
                            processed_count += 1
                        # Aktualisieren Sie die Fortschrittsanzeige für jedes abgeschlossene Sample
                        pbar.update(1)

            except Exception as e:
                pbar.write(f"\nFehler bei der Verarbeitung der Datei {file_path}: {e}")
                traceback.print_exc()

    # --- Schritt 3: Endgültige Metadaten erstellen und speichern ---
    print("\nErstelle endgültige Metadatendatei...")
    
    avg_duration = (total_duration / processed_count) if processed_count > 0 else 0.0

    metadata = {
        "dataset_info": {
            "total_samples": processed_count,
            "speakers": [f"{dataset_name}_speaker"],
            "languages": ["vi"],
            "total_duration": round(total_duration, 2),
            "avg_duration": round(avg_duration, 2),
            "created_from": [str(input_path.resolve())]
        },
        "samples": metadata_samples
    }

    metadata_filepath = output_path / "metadata.json"
    with open(metadata_filepath, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nVerarbeitung abgeschlossen! Dateien wurden gespeichert unter: {output_path}")
    print(f"Insgesamt {processed_count} von {total_samples} Samples erfolgreich verarbeitet.")


def process_csv_dataset(csv_path, audio_dir, output_dir, dataset_name="vn", max_samples_per_speaker=500, max_workers=None):
    """
    Process CSV-based Vietnamese TTS dataset (like your balanced dataset)
    with multithreading for better performance.
    """
    import shutil
    from datetime import datetime
    
    if max_workers is None:
        max_workers = (os.cpu_count() or 4) + 4
    
    print(f"Processing CSV dataset: {csv_path}")
    
    # Load CSV data
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Loaded {len(df)} samples from CSV")
    
    # Filter valid samples
    valid_df = df[df['valid'] == True].copy() if 'valid' in df.columns else df.copy()
    
    # Duration filtering (1-15 seconds)
    if 'duration' in valid_df.columns:
        valid_df = valid_df[(valid_df['duration'] >= 1.0) & (valid_df['duration'] <= 15.0)]
    
    # Clean speaker IDs and limit samples per speaker
    def clean_speaker_id(speaker):
        if pd.isna(speaker):
            return "unknown_speaker"
        cleaned = str(speaker).replace("@", "").replace("_male", "").replace("_female", "")
        return cleaned.lower().replace(" ", "_")
    
    valid_df['clean_speaker_id'] = valid_df['speaker'].apply(clean_speaker_id)
    balanced_df = valid_df.groupby('clean_speaker_id').head(max_samples_per_speaker)
    
    print(f"Filtered to {len(balanced_df)} samples from {len(balanced_df['clean_speaker_id'].unique())} speakers")
    
    # Prepare output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process samples with multithreading
    def process_csv_sample(sample_data):
        try:
            idx, row = sample_data['idx'], sample_data['row']
            speaker_id = row['clean_speaker_id']
            sample_id = f"{speaker_id}_{sample_data['speaker_counter']:06d}"
            
            # File paths
            audio_filename = f"{sample_id}.wav"
            text_filename = f"{sample_id}.txt"
            
            # Input audio path - use audio_dir parameter instead of raw_path from CSV
            input_audio_path = os.path.join(audio_dir, row.get('wav_filename', ''))
            if not os.path.exists(input_audio_path):
                print(f"Warning: Audio file not found: {input_audio_path}")
                return None, None
            
            # Output paths
            output_audio_path = output_path / audio_filename
            output_text_path = output_path / text_filename
            
            # Copy audio file
            shutil.copy2(input_audio_path, output_audio_path)
            
            # Write text file
            text_content = str(row.get('normalized_text', row.get('text', '')))
            if not text_content or pd.isna(text_content):
                return None, None
                
            with open(output_text_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            # Vietnamese emotion analysis
            def analyze_emotion(text):
                if not text or pd.isna(text):
                    return "neutral"
                text = str(text).lower()
                if any(word in text for word in ["?", "？", "gì", "nào", "sao"]):
                    return "questioning"
                elif any(word in text for word in ["!", "！", "tuyệt", "tốt", "đúng"]):
                    return "affirming"
                elif any(word in text for word in ["lỗi", "sai", "không", "nguy hiểm"]):
                    return "alerting"
                return "neutral"
            
            # Create sample metadata
            sample_entry = {
                "id": sample_id,
                "audio_file": audio_filename,
                "transcript_file": text_filename,
                "duration": float(row.get('duration', 0.0)),
                "speaker_id": speaker_id,
                "speaker_name": str(row.get('speaker', 'unknown')),
                "scene": "quiet_room",
                "emotion": analyze_emotion(text_content),
                "language": "vi",
                "gender": str(row.get('gender', 'unknown')),
                "quality_score": 1.0,
                "original_audio_path": input_audio_path,
                "user_instruction": "<audio>",
                "task_type": "audio_generation",
                "topic": str(row.get('topic', '')),
                "dialect": str(row.get('dialect', '')),
                "sample_rate": int(row.get('sample_rate', 24000))
            }
            
            return sample_entry, sample_entry['duration']
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            return None, None
    
    # Prepare tasks with speaker counters
    speaker_counters = {}
    tasks_data = []
    
    for idx, (_, row) in enumerate(balanced_df.iterrows()):
        speaker_id = row['clean_speaker_id']
        if speaker_id not in speaker_counters:
            speaker_counters[speaker_id] = 0
        
        tasks_data.append({
            'idx': idx,
            'row': row,
            'speaker_counter': speaker_counters[speaker_id]
        })
        speaker_counters[speaker_id] += 1
    
    # Process with multithreading
    metadata_samples = []
    total_duration = 0.0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(tasks_data), desc="Processing samples") as pbar:
            futures = [executor.submit(process_csv_sample, task) for task in tasks_data]
            
            for future in concurrent.futures.as_completed(futures):
                result, duration = future.result()
                if result and duration is not None:
                    metadata_samples.append(result)
                    total_duration += duration
                pbar.update(1)
    
    # Create metadata
    avg_duration = total_duration / len(metadata_samples) if metadata_samples else 0
    
    metadata = {
        "dataset_info": {
            "total_samples": len(metadata_samples),
            "speakers": list(speaker_counters.keys()),
            "languages": ["vi"],
            "total_duration": round(total_duration, 2),
            "avg_duration": round(avg_duration, 2),
            "created_from": [csv_path],
            "created_at": datetime.now().isoformat(),
            "dataset_type": "vietnamese_tts_csv",
            "sample_rate": 24000,
            "speaker_counts": speaker_counters
        },
        "samples": metadata_samples
    }
    
    # Save metadata
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessing completed! {len(metadata_samples)} samples saved to: {output_path}")
    return str(output_path)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Process Vietnamese TTS datasets')
    parser.add_argument('--mode', choices=['parquet', 'csv'], required=True, 
                       help='Dataset format: parquet (VLSP2020) or csv (balanced dataset)')
    parser.add_argument('--input', required=True, help='Input directory or CSV file path')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--audio_dir', help='Audio directory (for CSV mode)')
    parser.add_argument('--dataset_name', default='vn', help='Dataset name prefix')
    parser.add_argument('--max_workers', type=int, help='Max worker threads')
    parser.add_argument('--max_samples_per_speaker', type=int, default=500, 
                       help='Max samples per speaker (CSV mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'parquet':
        # Original VLSP2020 parquet processing
        input_dir = os.path.expanduser(args.input)
        output_dir = os.path.expanduser(args.output)
        process_parquet_files(input_dir, output_dir, args.dataset_name, args.max_workers)
        
    elif args.mode == 'csv':
        # New CSV processing for your balanced dataset
        if not args.audio_dir:
            print("Error: --audio_dir is required for CSV mode")
            exit(1)
        
        csv_path = os.path.expanduser(args.input)
        audio_dir = os.path.expanduser(args.audio_dir)
        output_dir = os.path.expanduser(args.output)
        
        process_csv_dataset(
            csv_path, audio_dir, output_dir, 
            args.dataset_name, args.max_samples_per_speaker, args.max_workers
        )
    
    # Example usage:
    # For your balanced CSV dataset:
    # python data_preprocess_vn.py --mode csv \
    #   --input /mnt/tsharp/thucth/voice/balanced_tts_dataset/balanced_metadata_full.csv \
    #   --audio_dir /mnt/tsharp/thucth/voice/balanced_tts_dataset/wavs/ \
    #   --output /home/thuc/thuc/voice/train-higgs-audio-vi/vietnamese_training_data_fast \
    #   --max_workers 16
    #
    # For VLSP2020 parquet dataset:
    # python data_preprocess_vn.py --mode parquet \
    #   --input /root/datasets/doof-ferb/vlsp2020_vinai_100h/data \
    #   --output /root/code/higgs-audio-main/higgs_training_data_mini_vn
