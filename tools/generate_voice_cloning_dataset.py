#!/usr/bin/env python3
"""
Voice Cloning Dataset Generator for Higgs Audio v2

This script converts existing datasets to the format required for voice cloning tasks:
- zero_shot_voice_cloning: Requires reference audio files for each target voice
- multi_speaker_voice_cloning: Requires multiple speaker references per sample

Usage:
    python generate_voice_cloning_dataset.py --task_type zero_shot_voice_cloning --input_dir ./input --output_dir ./voice_cloning_dataset
    python generate_voice_cloning_dataset.py --task_type multi_speaker_voice_cloning --input_dir ./input --output_dir ./voice_cloning_dataset
"""

import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VoiceCloningDatasetGenerator:
    """
    Generates dataset for voice cloning tasks from existing audio datasets
    """
    
    def __init__(self, task_type: str, input_dir: str, output_dir: str):
        self.task_type = task_type
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.ref_audio_ratio = 0.2  # 20% of samples will be used as reference audio
        
        # Validate task type
        if task_type not in ["zero_shot_voice_cloning", "multi_speaker_voice_cloning"]:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        logger.info(f"Initializing Voice Cloning Dataset Generator")
        logger.info(f"Task Type: {task_type}")
        logger.info(f"Input Directory: {input_dir}")
        logger.info(f"Output Directory: {output_dir}")

    def load_source_dataset(self) -> Dict[str, Any]:
        """Load the source dataset metadata"""
        metadata_file = self.input_dir / "metadata.json"
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded source dataset with {len(metadata.get('samples', []))} samples")
        return metadata

    def prepare_reference_audio_pool(self, samples: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Prepare a pool of reference audio samples grouped by speaker
        """
        speaker_samples = {}
        
        for sample in samples:
            speaker_id = sample.get('speaker_id', 'unknown')
            if speaker_id not in speaker_samples:
                speaker_samples[speaker_id] = []
            speaker_samples[speaker_id].append(sample)
        
        # Select reference samples for each speaker
        reference_pool = {}
        for speaker_id, speaker_samples_list in speaker_samples.items():
            # Use first 20% of samples as reference audio
            num_ref = max(1, int(len(speaker_samples_list) * self.ref_audio_ratio))
            reference_pool[speaker_id] = speaker_samples_list[:num_ref]
            logger.info(f"Speaker {speaker_id}: {num_ref} reference samples from {len(speaker_samples_list)} total")
        
        return reference_pool

    def generate_zero_shot_voice_cloning_dataset(self, source_metadata: Dict) -> Dict[str, Any]:
        """
        Generate dataset for zero-shot voice cloning
        
        Format required:
        - Each sample needs a 'ref_audio_file' pointing to reference audio
        - Each sample needs a 'ref_transcript' for the reference audio
        """
        samples = source_metadata['samples']
        reference_pool = self.prepare_reference_audio_pool(samples)
        
        new_samples = []
        
        for sample in samples:
            speaker_id = sample.get('speaker_id', 'unknown')
            
            # Skip if this sample is being used as reference
            ref_samples = reference_pool.get(speaker_id, [])
            if sample['id'] in [ref['id'] for ref in ref_samples]:
                continue
            
            # Select a random reference sample from the same speaker
            if ref_samples:
                ref_sample = random.choice(ref_samples)
                
                new_sample = {
                    "id": sample['id'],
                    "audio_file": sample['audio_file'],
                    "transcript_file": sample['transcript_file'],
                    "duration": sample.get('duration', 0),
                    "speaker_id": speaker_id,
                    "speaker_name": sample.get('speaker_name', ''),
                    "scene": sample.get('scene', 'voice_cloning'),
                    "emotion": sample.get('emotion', 'neutral'),
                    "language": sample.get('language', 'vi'),
                    "gender": sample.get('gender', 'unknown'),
                    "quality_score": sample.get('quality_score', 1.0),
                    
                    # Voice cloning specific fields
                    "ref_audio_file": ref_sample['audio_file'],
                    "ref_transcript": self._get_transcript_content(ref_sample),
                    "task_type": "zero_shot_voice_cloning"
                }
                
                new_samples.append(new_sample)
        
        # Update dataset info
        new_metadata = {
            "dataset_info": {
                "total_samples": len(new_samples),
                "speakers": source_metadata['dataset_info'].get('speakers', []),
                "languages": source_metadata['dataset_info'].get('languages', ['vi']),
                "task_type": "zero_shot_voice_cloning",
                "reference_audio_ratio": self.ref_audio_ratio,
                "created_from": str(self.input_dir)
            },
            "samples": new_samples
        }
        
        logger.info(f"Generated {len(new_samples)} zero-shot voice cloning samples")
        return new_metadata

    def generate_multi_speaker_voice_cloning_dataset(self, source_metadata: Dict) -> Dict[str, Any]:
        """
        Generate dataset for multi-speaker voice cloning
        
        Format required:
        - Each sample needs 'ref_speakers' list with multiple speaker references
        - Each ref_speaker needs: speaker_tag, ref_audio_file, ref_transcript
        """
        samples = source_metadata['samples']
        reference_pool = self.prepare_reference_audio_pool(samples)
        
        available_speakers = list(reference_pool.keys())
        if len(available_speakers) < 2:
            raise ValueError("Multi-speaker voice cloning requires at least 2 speakers in the dataset")
        
        new_samples = []
        
        for sample in samples:
            speaker_id = sample.get('speaker_id', 'unknown')
            
            # Skip if this sample is being used as reference
            ref_samples_for_speaker = reference_pool.get(speaker_id, [])
            if sample['id'] in [ref['id'] for ref in ref_samples_for_speaker]:
                continue
            
            # Create reference speakers list (2-4 speakers including the target)
            num_ref_speakers = random.randint(2, min(4, len(available_speakers)))
            selected_speakers = [speaker_id]  # Always include target speaker
            
            # Add random other speakers
            other_speakers = [s for s in available_speakers if s != speaker_id]
            selected_speakers.extend(random.sample(other_speakers, num_ref_speakers - 1))
            
            ref_speakers = []
            for i, spk_id in enumerate(selected_speakers):
                if spk_id in reference_pool and reference_pool[spk_id]:
                    ref_sample = random.choice(reference_pool[spk_id])
                    ref_speakers.append({
                        "speaker_tag": f"[SPEAKER{i}]",
                        "speaker_id": spk_id,
                        "ref_audio_file": ref_sample['audio_file'],
                        "ref_transcript": self._get_transcript_content(ref_sample)
                    })
            
            if len(ref_speakers) >= 2:  # Ensure we have at least 2 reference speakers
                new_sample = {
                    "id": sample['id'],
                    "audio_file": sample['audio_file'],
                    "transcript_file": sample['transcript_file'],
                    "duration": sample.get('duration', 0),
                    "speaker_id": speaker_id,
                    "speaker_name": sample.get('speaker_name', ''),
                    "scene": sample.get('scene', 'multi_speaker_cloning'),
                    "emotion": sample.get('emotion', 'neutral'),
                    "language": sample.get('language', 'vi'),
                    "gender": sample.get('gender', 'unknown'),
                    "quality_score": sample.get('quality_score', 1.0),
                    
                    # Multi-speaker voice cloning specific fields
                    "ref_speakers": ref_speakers,
                    "num_ref_speakers": len(ref_speakers),
                    "task_type": "multi_speaker_voice_cloning"
                }
                
                new_samples.append(new_sample)
        
        # Update dataset info
        new_metadata = {
            "dataset_info": {
                "total_samples": len(new_samples),
                "speakers": source_metadata['dataset_info'].get('speakers', []),
                "languages": source_metadata['dataset_info'].get('languages', ['vi']),
                "task_type": "multi_speaker_voice_cloning",
                "reference_audio_ratio": self.ref_audio_ratio,
                "avg_ref_speakers": sum(len(s['ref_speakers']) for s in new_samples) / len(new_samples) if new_samples else 0,
                "created_from": str(self.input_dir)
            },
            "samples": new_samples
        }
        
        logger.info(f"Generated {len(new_samples)} multi-speaker voice cloning samples")
        return new_metadata

    def _get_transcript_content(self, sample: Dict) -> str:
        """Get transcript content from transcript file"""
        try:
            transcript_file = self.input_dir / sample['transcript_file']
            if transcript_file.exists():
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            else:
                return "This is a voice sample for cloning."
        except Exception as e:
            logger.warning(f"Could not read transcript for {sample['id']}: {e}")
            return "This is a voice sample for cloning."

    def copy_audio_files(self, metadata: Dict):
        """Copy required audio files to output directory"""
        logger.info("Copying audio files...")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        audio_files_to_copy = set()
        
        for sample in metadata['samples']:
            # Add main audio file
            audio_files_to_copy.add(sample['audio_file'])
            
            # Add reference audio files
            if self.task_type == "zero_shot_voice_cloning":
                if 'ref_audio_file' in sample:
                    audio_files_to_copy.add(sample['ref_audio_file'])
            
            elif self.task_type == "multi_speaker_voice_cloning":
                if 'ref_speakers' in sample:
                    for ref_speaker in sample['ref_speakers']:
                        audio_files_to_copy.add(ref_speaker['ref_audio_file'])
        
        # Copy files
        copied_count = 0
        for audio_file in audio_files_to_copy:
            src_path = self.input_dir / audio_file
            dst_path = self.output_dir / audio_file
            
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
                copied_count += 1
            else:
                logger.warning(f"Audio file not found: {src_path}")
        
        logger.info(f"Copied {copied_count} audio files")

    def copy_transcript_files(self, metadata: Dict):
        """Copy transcript files to output directory"""
        logger.info("Copying transcript files...")
        
        transcript_files_to_copy = set()
        
        for sample in metadata['samples']:
            transcript_files_to_copy.add(sample['transcript_file'])
        
        # Copy files
        copied_count = 0
        for transcript_file in transcript_files_to_copy:
            src_path = self.input_dir / transcript_file
            dst_path = self.output_dir / transcript_file
            
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
                copied_count += 1
            else:
                logger.warning(f"Transcript file not found: {src_path}")
        
        logger.info(f"Copied {copied_count} transcript files")

    def generate_dataset(self):
        """Main method to generate voice cloning dataset"""
        logger.info(f"Starting dataset generation for {self.task_type}")
        
        # Load source dataset
        source_metadata = self.load_source_dataset()
        
        # Generate new dataset based on task type
        if self.task_type == "zero_shot_voice_cloning":
            new_metadata = self.generate_zero_shot_voice_cloning_dataset(source_metadata)
        elif self.task_type == "multi_speaker_voice_cloning":
            new_metadata = self.generate_multi_speaker_voice_cloning_dataset(source_metadata)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
        
        # Copy files
        self.copy_audio_files(new_metadata)
        self.copy_transcript_files(new_metadata)
        
        # Save new metadata
        output_metadata_file = self.output_dir / "metadata.json"
        with open(output_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(new_metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dataset generation completed!")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Generated {len(new_metadata['samples'])} samples")
        logger.info(f"Metadata saved to: {output_metadata_file}")
        
        return new_metadata


def main():
    parser = argparse.ArgumentParser(description="Generate Voice Cloning Dataset for Higgs Audio v2")
    parser.add_argument("--task_type", required=True, 
                       choices=["zero_shot_voice_cloning", "multi_speaker_voice_cloning"],
                       help="Type of voice cloning task")
    parser.add_argument("--input_dir", required=True, type=str,
                       help="Input directory containing source dataset")
    parser.add_argument("--output_dir", required=True, type=str, 
                       help="Output directory for generated dataset")
    parser.add_argument("--ref_audio_ratio", type=float, default=0.2,
                       help="Ratio of samples to use as reference audio (default: 0.2)")
    
    args = parser.parse_args()
    
    try:
        # Create generator
        generator = VoiceCloningDatasetGenerator(
            task_type=args.task_type,
            input_dir=args.input_dir,
            output_dir=args.output_dir
        )
        
        # Set reference audio ratio
        generator.ref_audio_ratio = args.ref_audio_ratio
        
        # Generate dataset
        metadata = generator.generate_dataset()
        
        print(f"\n🎉 Dataset Generation Successful!")
        print(f"📊 Task Type: {args.task_type}")
        print(f"📁 Output Directory: {args.output_dir}")
        print(f"🎵 Total Samples: {len(metadata['samples'])}")
        print(f"🗣️ Speakers: {len(metadata['dataset_info']['speakers'])}")
        
        if args.task_type == "multi_speaker_voice_cloning":
            avg_ref_speakers = metadata['dataset_info']['avg_ref_speakers']
            print(f"👥 Average Reference Speakers per Sample: {avg_ref_speakers:.1f}")
        
        print(f"\n🚀 Ready for voice cloning training!")
        print(f"Use: --task_type {args.task_type} --train_data_dir {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        raise


if __name__ == "__main__":
    main()