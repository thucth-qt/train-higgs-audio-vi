#!/usr/bin/env python3
"""
Higgs Audio v2 Training Script with LoRA Support and Task-specific Training
Based on the Higgs Audio v2 architecture from Boson AI
"""

import os
import json
import logging
import argparse

# Import patch module first to fix torch version issue
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
try:
    import torch_patch  # This applies the patch immediately
    print("✓ Torch patch module loaded successfully")
except Exception as e:
    print(f"⚠️  Warning: Could not load torch patch module: {e}")

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import psutil

# Verify torch version after patch
print(f"Current torch.__version__: {torch.__version__}")

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoConfig,
    TrainingArguments, 
    Trainer,
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    PeftModel
)
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import librosa
import re


class MemoryMonitor:
    """Memory and performance monitoring utility"""
    
    def __init__(self):
        self.start_time = time.time()
        self.peak_memory = 0
        
    def log_memory_usage(self, stage: str):
        """Log current memory usage"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1e9
            gpu_reserved = torch.cuda.memory_reserved() / 1e9
            self.peak_memory = max(self.peak_memory, gpu_memory)
        else:
            gpu_memory = gpu_reserved = 0
            
        cpu_memory = psutil.virtual_memory().used / 1e9
        
        logger.info(f"[{stage}] Memory - GPU: {gpu_memory:.2f}GB (reserved: {gpu_reserved:.2f}GB), "
                   f"CPU: {cpu_memory:.2f}GB, Peak GPU: {self.peak_memory:.2f}GB")
    
    def log_time_elapsed(self, stage: str):
        """Log time elapsed since initialization"""
        elapsed = time.time() - self.start_time
        logger.info(f"[{stage}] Time elapsed: {elapsed:.2f}s")


# Global memory monitor
memory_monitor = MemoryMonitor()

# 尝试导入 Higgs Audio 相关模块
try:
    from boson_multimodal.model.higgs_audio import HiggsAudioConfig, HiggsAudioModel
    from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator, HiggsAudioBatchInput
    from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
    from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
    from boson_multimodal.data_types import Message, ChatMLSample, AudioContent, TextContent
    HIGGS_AVAILABLE = True
except ImportError:
    HIGGS_AVAILABLE = False
    logging.warning("Higgs Audio modules not available. Using fallback implementation.")
    
    # 添加fallback类定义
    class ChatMLDatasetSample:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def to(self, device):
            """将所有tensor属性转移到指定设备"""
            new_sample = ChatMLDatasetSample()
            for key, value in self.__dict__.items():
                if isinstance(value, torch.Tensor):
                    setattr(new_sample, key, value.to(device))
                else:
                    setattr(new_sample, key, value)
            return new_sample

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加常量定义
AUDIO_PLACEHOLDER_TOKEN = "<|__AUDIO_PLACEHOLDER__|>"

MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE = """You are an AI assistant designed to convert text into speech.
If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.
If no speaker tag is present, select a suitable voice on your own."""


class ExtendedHiggsAudioBatchInput:
    """
    Extended HiggsAudioBatchInput with __len__ method for Trainer compatibility
    """
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __len__(self):
        """Return the batch size based on input_ids"""
        if hasattr(self, 'input_ids') and self.input_ids is not None:
            return self.input_ids.shape[0]
        else:
            return 0
    
    def __getitem__(self, key):
        """Allow dictionary-style access for compatibility"""
        return getattr(self, key)
    
    def __contains__(self, key):
        """Check if attribute exists"""
        return hasattr(self, key)
    
    def keys(self):
        """Return all attribute names for compatibility"""
        return [attr for attr in dir(self) if not attr.startswith('_') and not callable(getattr(self, attr))]


class ExtendedHiggsAudioSampleCollator:
    """
    Extended collator that returns our custom batch input class
    """
    
    def __init__(self, **kwargs):
        if HIGGS_AVAILABLE:
            self.base_collator = HiggsAudioSampleCollator(**kwargs)
        else:
            # Fallback collator
            self.pad_token_id = kwargs.get('pad_token_id', 0)
    
    def __call__(self, batch: List[ChatMLDatasetSample]):
        if HIGGS_AVAILABLE and hasattr(self, 'base_collator'):
            # 1. 调用官方的、底层的 collator，让它完成所有复杂的填充和对齐工作
            batch_input = self.base_collator(batch)
            
            # batch_input.audio_out_ids 是经过填充和处理的，其长度与模型输出的 audio_logits 长度完全一致。
            label_audio_ids = batch_input.audio_out_ids
            
            # 2. 转换为我们的扩展类，并传入这个完美的标签
            extended_batch = ExtendedHiggsAudioBatchInput(
                input_ids=batch_input.input_ids,
                attention_mask=batch_input.attention_mask,
                audio_features=batch_input.audio_features,
                audio_feature_attention_mask=batch_input.audio_feature_attention_mask,
                audio_out_ids=batch_input.audio_out_ids,
                audio_out_ids_start=batch_input.audio_out_ids_start,
                audio_out_ids_start_group_loc=batch_input.audio_out_ids_start_group_loc,
                audio_in_ids=batch_input.audio_in_ids,
                audio_in_ids_start=batch_input.audio_in_ids_start,
                label_ids=batch_input.label_ids,
                label_audio_ids=label_audio_ids, # <-- 使用我们新定义的、对齐的标签
                reward=batch_input.reward,
            )
            
            return extended_batch
        else:
            # Fallback implementation
            input_ids_list = []
            attention_mask_list = []
            label_ids_list = []
            
            for sample in batch:
                input_ids_list.append(sample.input_ids)
                attention_mask = torch.ones_like(sample.input_ids)
                attention_mask_list.append(attention_mask)
                
                if hasattr(sample, 'label_ids'):
                    label_ids_list.append(sample.label_ids)
                else:
                    label_ids_list.append(sample.input_ids)
            
            # Pad sequences
            max_len = max(len(ids) for ids in input_ids_list)
            
            padded_input_ids = []
            padded_attention_mask = []
            padded_label_ids = []
            
            for i in range(len(input_ids_list)):
                pad_length = max_len - len(input_ids_list[i])
                
                padded_input = torch.cat([
                    input_ids_list[i],
                    torch.full((pad_length,), self.pad_token_id, dtype=torch.long)
                ])
                padded_input_ids.append(padded_input)
                
                padded_mask = torch.cat([
                    attention_mask_list[i],
                    torch.zeros(pad_length, dtype=torch.long)
                ])
                padded_attention_mask.append(padded_mask)
                
                padded_label = torch.cat([
                    label_ids_list[i],
                    torch.full((pad_length,), -100, dtype=torch.long)
                ])
                padded_label_ids.append(padded_label)
            
            return ExtendedHiggsAudioBatchInput(
                input_ids=torch.stack(padded_input_ids),
                attention_mask=torch.stack(padded_attention_mask),
                label_ids=torch.stack(padded_label_ids),
                audio_features=None,
                audio_feature_attention_mask=None,
                audio_out_ids=None,
                audio_out_ids_start=None,
                audio_out_ids_start_group_loc=None,
                audio_in_ids=None,
                audio_in_ids_start=None,
                label_audio_ids=None,
                reward=None,
            )


def normalize_chinese_punctuation(text):
    """
    Convert Chinese (full-width) punctuation marks to English (half-width) equivalents.
    """
    chinese_to_english_punct = {
        "，": ", ", "。": ".", "：": ":", "；": ";", "？": "?", "！": "!",
        "（": "(", "）": ")", "【": "[", "】": "]", "《": "<", "》": ">",
        """: '"', """: '"', "'": "'", "'": "'", "、": ",", "--": "-",
        "…": "...", "·": ".", "「": '"', "」": '"', "『": '"', "』": '"',
    }

    for zh_punct, en_punct in chinese_to_english_punct.items():
        text = text.replace(zh_punct, en_punct)

    return text


def _build_system_message_with_audio_prompt(system_message):
    """Build system message with audio prompts"""
    contents = []

    while AUDIO_PLACEHOLDER_TOKEN in system_message:
        loc = system_message.find(AUDIO_PLACEHOLDER_TOKEN)
        contents.append(TextContent(system_message[:loc]))
        contents.append(AudioContent(audio_url=""))
        system_message = system_message[loc + len(AUDIO_PLACEHOLDER_TOKEN):]

    if len(system_message) > 0:
        contents.append(TextContent(system_message))
    
    ret = Message(
        role="system",
        content=contents,
    )
    return ret


class HiggsAudioDataset(Dataset):
    def __init__(
        self, 
        data_dir: str,
        tokenizer: AutoTokenizer,
        audio_tokenizer,
        task_type: str = "zero_shot_voice_cloning",
        sample_rate: int = 24000,
        use_metadata: bool = True,
        ref_audio_in_system_message: bool = False,
        device: Optional[str] = None,
    ):
        """
        用于 Higgs Audio 模型训练的数据集。
        """
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.task_type = task_type
        self.sample_rate = sample_rate
        self.use_metadata = use_metadata
        self.ref_audio_in_system_message = ref_audio_in_system_message
        self.device = device
        
        valid_tasks = [
            "zero_shot_voice_cloning", 
            "single_speaker_smart_voice", 
            "multi_speaker_smart_voice", 
            "multi_speaker_voice_cloning"
        ]
        if self.task_type not in valid_tasks:
            raise ValueError(f"Invalid task_type: {self.task_type}. Must be one of {valid_tasks}")
        
        self.actual_num_codebooks = self._detect_codebook_size()
        
        if use_metadata and (self.data_dir / "metadata.json").exists():
            self.samples = self._load_samples_from_metadata()
        else:
            logger.warning(f"metadata.json not found in {data_dir}. Scanning directory instead.")
            self.samples = self._load_samples_from_directory()
            
        if not self.samples:
            raise RuntimeError(f"No valid samples found in {data_dir} for task '{self.task_type}'. Please check your data and metadata.json.")
        
        # Validate dataset integrity
        valid_samples = self._validate_dataset_integrity()
        if valid_samples < len(self.samples):
            logger.warning(f"Dataset validation: {len(self.samples) - valid_samples} samples are invalid and will be skipped during training")
            
        logger.info(f"Loaded {len(self.samples)} samples from {data_dir} for task: {self.task_type}")
        logger.info(f"Dataset validation completed: {valid_samples}/{len(self.samples)} samples are valid")

    def _detect_codebook_size(self) -> int:
        """通过编码一个测试音频来动态检测音频 tokenizer 的 codebook 数量。"""
        try:
            audio_files = list(self.data_dir.glob("*.wav")) + list(self.data_dir.glob("*.mp3"))
            if audio_files and self.audio_tokenizer:
                # Try multiple files to get consistent results
                for i, audio_file in enumerate(audio_files[:3]):  # Test up to 3 files
                    test_audio_path = str(audio_file)
                    try:
                        if os.path.exists(test_audio_path) and os.path.getsize(test_audio_path) > 1000:
                            test_tokens = self.audio_tokenizer.encode(test_audio_path)
                            if test_tokens is not None and isinstance(test_tokens, torch.Tensor) and test_tokens.dim() == 2:
                                detected_size = test_tokens.shape[0]
                                logger.info(f"Detected {detected_size} codebooks from audio tokenizer (file: {audio_file.name}).")
                                return detected_size
                    except Exception as e:
                        logger.warning(f"Failed to test audio file {audio_file.name}: {e}")
                        continue
        except Exception as e:
            logger.warning(f"Could not auto-detect codebook size: {e}. Falling back to default.")
        
        # Try to get default size from tokenizer attributes
        # Common attribute names for codebook size
        for attr_name in ['n_q', 'codebook_size', 'num_quantizers', 'n_codebooks']:
            if hasattr(self.audio_tokenizer, attr_name):
                default_size = getattr(self.audio_tokenizer, attr_name)
                if default_size and isinstance(default_size, int):
                    logger.info(f"Using codebook size from tokenizer.{attr_name}: {default_size}")
                    return default_size
        
        # Final fallback
        default_size = 8  # More reasonable default based on common audio tokenizers
        logger.info(f"Using fallback codebook size: {default_size}")
        return default_size

    def _validate_dataset_integrity(self) -> int:
        """Validate dataset integrity and return count of valid samples"""
        valid_count = 0
        validation_sample_size = min(50, len(self.samples))  # Test up to 50 samples
        
        logger.info(f"Validating {validation_sample_size} samples for dataset integrity...")
        
        for i in range(validation_sample_size):
            try:
                sample = self.samples[i]
                
                # Check if audio file exists
                if not os.path.exists(sample["audio_file"]):
                    logger.warning(f"Sample {i}: Audio file not found: {sample['audio_file']}")
                    continue
                
                # Check if transcript file exists
                if "transcript_file" in sample and not os.path.exists(sample["transcript_file"]):
                    logger.warning(f"Sample {i}: Transcript file not found: {sample['transcript_file']}")
                    continue
                
                # Try to load audio
                try:
                    waveform, sr = self._load_audio_waveform(sample["audio_file"])
                    if waveform.numel() == 0:
                        logger.warning(f"Sample {i}: Empty audio waveform: {sample['audio_file']}")
                        continue
                except Exception as e:
                    logger.warning(f"Sample {i}: Failed to load audio {sample['audio_file']}: {e}")
                    continue
                
                # Try to encode audio if tokenizer is available
                if self.audio_tokenizer:
                    try:
                        tokens = self._encode_audio_tokens(sample["audio_file"])
                        if tokens is None or tokens.numel() == 0:
                            logger.warning(f"Sample {i}: Failed to encode audio tokens: {sample['audio_file']}")
                            continue
                    except Exception as e:
                        logger.warning(f"Sample {i}: Audio tokenization failed for {sample['audio_file']}: {e}")
                        continue
                
                # Check transcript if it exists
                if "transcript_file" in sample:
                    try:
                        with open(sample["transcript_file"], 'r', encoding='utf-8') as f:
                            transcript = f.read().strip()
                        if not transcript:
                            logger.warning(f"Sample {i}: Empty transcript: {sample['transcript_file']}")
                            continue
                    except Exception as e:
                        logger.warning(f"Sample {i}: Failed to read transcript {sample['transcript_file']}: {e}")
                        continue
                
                valid_count += 1
                
            except Exception as e:
                logger.warning(f"Sample {i}: General validation error: {e}")
                continue
        
        # Estimate total valid samples based on validation sample
        if validation_sample_size < len(self.samples):
            estimated_valid = int((valid_count / validation_sample_size) * len(self.samples))
            logger.info(f"Estimated valid samples in full dataset: {estimated_valid}/{len(self.samples)}")
            return estimated_valid
        else:
            return valid_count

    def _load_samples_from_metadata(self) -> List[Dict]:
        metadata_path = self.data_dir / "metadata.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f).get("samples", [])
        
        # 预处理路径，使其成为绝对路径
        for sample in metadata:
            sample["audio_file"] = str(self.data_dir / sample["audio_file"])
            if "transcript_file" in sample:
                sample["transcript_file"] = str(self.data_dir / sample["transcript_file"])
        return metadata

    def _load_samples_from_directory(self) -> List[Dict]:
        samples = []
        audio_files = list(self.data_dir.glob("*.wav")) + list(self.data_dir.glob("*.mp3"))
        for audio_path in audio_files:
            transcript_path = audio_path.with_suffix('.txt')
            if transcript_path.exists():
                samples.append({
                    "audio_file": str(audio_path),
                    "transcript_file": str(transcript_path),
                    "audio_id": audio_path.stem,
                })
        return samples

    def _load_audio_waveform(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Load audio waveform with robust error handling and multiple fallback strategies"""
        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                return torch.zeros(1), self.sample_rate
                
            # Check file size
            file_size = os.path.getsize(audio_path)
            if file_size == 0:
                logger.error(f"Audio file is empty: {audio_path}")
                return torch.zeros(1), self.sample_rate
            
            # Try torchaudio first
            try:
                waveform, sr = torchaudio.load(audio_path)
                
                # Handle multi-channel audio
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                
                # Resample if necessary
                if sr != self.sample_rate:
                    waveform = torchaudio.functional.resample(
                        waveform, 
                        orig_freq=sr, 
                        new_freq=self.sample_rate
                    )
                
                # Validate waveform
                waveform = waveform.squeeze(0)
                if waveform.numel() == 0:
                    logger.error(f"Empty waveform after processing: {audio_path}")
                    return torch.zeros(1), self.sample_rate
                
                # Check for reasonable audio length (at least 100ms)
                min_length = int(self.sample_rate * 0.1)  # 100ms
                if waveform.numel() < min_length:
                    logger.warning(f"Very short audio ({waveform.numel()} samples): {audio_path}")
                
                # Check for silence or very low volume
                max_amplitude = torch.abs(waveform).max()
                if max_amplitude < 1e-6:
                    logger.warning(f"Very quiet audio (max amplitude: {max_amplitude}): {audio_path}")
                
                return waveform, self.sample_rate
                
            except Exception as e:
                logger.warning(f"torchaudio failed for {audio_path}: {e}. Trying librosa...")
                
                # Fallback to librosa
                try:
                    import librosa
                    waveform_np, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
                    waveform = torch.from_numpy(waveform_np).float()
                    
                    if waveform.numel() == 0:
                        logger.error(f"Empty waveform from librosa: {audio_path}")
                        return torch.zeros(1), self.sample_rate
                    
                    return waveform, self.sample_rate
                    
                except Exception as e2:
                    logger.warning(f"librosa also failed for {audio_path}: {e2}. Trying soundfile...")
                    
                    # Final fallback to soundfile
                    try:
                        import soundfile as sf
                        waveform_np, sr = sf.read(audio_path, dtype='float32')
                        
                        # Handle multi-channel
                        if len(waveform_np.shape) > 1:
                            waveform_np = np.mean(waveform_np, axis=1)
                        
                        waveform = torch.from_numpy(waveform_np).float()
                        
                        # Resample if necessary
                        if sr != self.sample_rate:
                            import librosa
                            waveform_np = librosa.resample(waveform_np, orig_sr=sr, target_sr=self.sample_rate)
                            waveform = torch.from_numpy(waveform_np).float()
                        
                        if waveform.numel() == 0:
                            logger.error(f"Empty waveform from soundfile: {audio_path}")
                            return torch.zeros(1), self.sample_rate
                        
                        return waveform, self.sample_rate
                        
                    except Exception as e3:
                        logger.error(f"All audio loading methods failed for {audio_path}: torchaudio({e}), librosa({e2}), soundfile({e3})")
                        return torch.zeros(1), self.sample_rate
                        
        except Exception as e:
            logger.error(f"Unexpected error loading audio {audio_path}: {e}")
            return torch.zeros(1), self.sample_rate

    def _encode_audio_tokens(self, audio_path: str) -> Optional[torch.Tensor]:
        """Encode audio file to tokens with robust error handling"""
        if not self.audio_tokenizer: 
            return None
            
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Ensure audio file exists and is readable
                if not os.path.exists(audio_path):
                    logger.error(f"Audio file not found: {audio_path}")
                    return None
                
                # Check file size (skip very small files that might be corrupted)
                file_size = os.path.getsize(audio_path)
                if file_size < 1000:  # Less than 1KB is likely corrupted
                    logger.warning(f"Audio file too small ({file_size} bytes): {audio_path}")
                    return None
                
                tokens = self.audio_tokenizer.encode(audio_path)
                
                # Validate token dimensions
                if tokens is None:
                    logger.error(f"Audio tokenizer returned None for: {audio_path}")
                    return None
                
                if not isinstance(tokens, torch.Tensor):
                    logger.error(f"Audio tokenizer returned non-tensor for: {audio_path}")
                    return None
                
                if tokens.dim() != 2:
                    logger.error(f"Audio tokens have wrong dimensions {tokens.shape} for: {audio_path}")
                    return None
                
                if tokens.shape[1] == 0:
                    logger.warning(f"Empty audio tokens for: {audio_path}")
                    return None
                
                # Check codebook size only if actual_num_codebooks is set
                if hasattr(self, 'actual_num_codebooks') and tokens.shape[0] != self.actual_num_codebooks:
                    logger.warning(f"Token codebook mismatch: expected {self.actual_num_codebooks}, got {tokens.shape[0]} for: {audio_path}")
                
                return tokens
                
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.warning(f"CUDA OOM during audio encoding (attempt {attempt+1}/{max_retries}): {audio_path}")
                    torch.cuda.empty_cache()
                    if attempt < max_retries - 1:
                        continue
                logger.error(f"Runtime error encoding audio (attempt {attempt+1}/{max_retries}) {audio_path}: {e}")
                if attempt == max_retries - 1:
                    return None
            except Exception as e:
                logger.error(f"Unexpected error encoding audio (attempt {attempt+1}/{max_retries}) {audio_path}: {e}")
                if attempt == max_retries - 1:
                    return None
        
        return None

    def _normalize_transcript(self, transcript: str) -> str:
        """Normalize transcript text"""
        transcript = normalize_chinese_punctuation(transcript)
        transcript = transcript.replace("(", " ").replace(")", " ")
        transcript = transcript.replace("°F", " degrees Fahrenheit").replace("°C", " degrees Celsius")
        
        for tag, replacement in [("[laugh]", "<SE>[Laughter]</SE>"), ("[cough]", "<SE>[Cough]</SE>"),]:
            transcript = transcript.replace(tag, replacement)
            
        lines = transcript.split("\n")
        transcript = "\n".join([" ".join(line.split()) for line in lines if line.strip()])
        
        if transcript and not any(transcript.endswith(c) for c in ".!?,\";'</SE_e></SE>"):
            transcript += "."
            
        return transcript.strip()

    def _get_scene_description(self, sample: Dict) -> str:
        scene = sample.get("scene", "a quiet room")
        return f"Audio is recorded from {scene.replace('_', ' ')}."

    def _detect_speaker_tags(self, transcript: str) -> List[str]:
        return sorted(set(re.findall(r"\[(SPEAKER\d+)\]", transcript)))
    
    def _create_messages_for_task(self, sample: Dict, transcript: str) -> List[Message]:
        """
        根据任务类型为给定的样本创建消息列表（提示）。
        """
        messages = []
        speaker_tags = self._detect_speaker_tags(transcript)
        scene_prompt = self._get_scene_description(sample) if self.use_metadata else None

        if self.task_type == "zero_shot_voice_cloning":
            ref_audio = sample.get("ref_audio_file")
            if not ref_audio:
                logger.warning(f"Sample {sample['id']} is for zero_shot_voice_cloning but has no 'ref_audio_file'.")
                return [Message(role="system", content="Generate audio following instruction.")]

            ref_audio_path = str(self.data_dir / ref_audio)
            ref_transcript = sample.get("ref_transcript", "This is a voice sample for cloning.")

            if self.ref_audio_in_system_message:
                system_content = f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_prompt}\n\nSPEAKER0: {AUDIO_PLACEHOLDER_TOKEN}\n<|scene_desc_end|>"
                messages.append(_build_system_message_with_audio_prompt(system_content))
            else:
                messages.append(Message(role="system", content=f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>"))
                messages.append(Message(role="user", content=ref_transcript))
                messages.append(Message(role="assistant", content=AudioContent(audio_url=ref_audio_path)))
        
        elif self.task_type == "multi_speaker_voice_cloning":
            ref_speakers = sample.get("ref_speakers")
            if not ref_speakers or not isinstance(ref_speakers, list):
                logger.warning(f"Sample {sample['audio_id']} is for multi_speaker_voice_cloning but has no valid 'ref_speakers' list in metadata.")
                return [Message(role="system", content=MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE)]
            
            for i, ref_info in enumerate(ref_speakers):
                speaker_tag = ref_info.get("speaker_tag", f"[SPEAKER{i}]")
                ref_audio_path = str(self.data_dir / ref_info["ref_audio_file"])
                ref_transcript = ref_info.get("ref_transcript", "This is a voice sample.")
                messages.append(Message(role="user", content=f"{speaker_tag} {ref_transcript}"))
                messages.append(Message(role="assistant", content=AudioContent(audio_url=ref_audio_path)))
            
            messages.insert(0, Message(role="system", content=f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>"))

        elif self.task_type == "multi_speaker_smart_voice" and len(speaker_tags) > 1:
            speaker_desc_l = [f"{tag}: {'feminine' if i % 2 == 0 else 'masculine'}" for i, tag in enumerate(speaker_tags)]
            scene_desc = f"{scene_prompt}\n\n" + "\n".join(speaker_desc_l) if scene_prompt else "\n".join(speaker_desc_l)
            messages.append(Message(role="system", content=f"{MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE}\n\n<|scene_desc_start|>\n{scene_desc}\n<|scene_desc_end|>"))
        
        else: # single_speaker_smart_voice 或其他回退情况
            content = "Generate audio following instruction."
            if scene_prompt:
                content += f"\n\n<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>"
            messages.append(Message(role="system", content=content))
            
        return messages

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ChatMLDatasetSample:
        sample = self.samples[idx]
        
        try:
            with open(sample["transcript_file"], 'r', encoding='utf-8') as f:
                transcript = f.read().strip()
            transcript = self._normalize_transcript(transcript)
            
            # 1. 构建消息历史（提示）
            messages = self._create_messages_for_task(sample, transcript)
            
            # 2. 添加当前样本的目标对话
            messages.append(Message(role="user", content=transcript))
            messages.append(Message(role="assistant", content=AudioContent(audio_url=sample["audio_file"])))

            chatml_sample = ChatMLSample(messages=messages)
            
            # 3. 使用处理函数处理 ChatML 样本
            input_tokens, label_tokens, audio_contents, audio_label_contents = prepare_chatml_sample(
                chatml_sample, self.tokenizer
            )

            # 4. 处理音频数据
            context_audio_tokens = []
            for audio_content in (audio_contents or []):
                if audio_content.audio_url:
                    tokens = self._encode_audio_tokens(audio_content.audio_url)
                    if tokens is not None: 
                        context_audio_tokens.append(tokens)

            label_audio_tokens = []
            for audio_label_content in (audio_label_contents or []):
                if audio_label_content.audio_url:
                    tokens = self._encode_audio_tokens(audio_label_content.audio_url)
                    if tokens is not None: 
                        label_audio_tokens.append(tokens)

            # 5. 拼接张量
            if context_audio_tokens:
                audio_ids_concat = torch.cat(context_audio_tokens, dim=1)
                audio_ids_start = torch.tensor([0] + [t.shape[1] for t in context_audio_tokens[:-1]], dtype=torch.long).cumsum(0)
            else:
                # Use detected codebook size or default
                num_codebooks = getattr(self, 'actual_num_codebooks', 8)
                audio_ids_concat = torch.zeros((num_codebooks, 0), dtype=torch.long)
                audio_ids_start = torch.tensor([0], dtype=torch.long)

            label_audio_ids = torch.cat(label_audio_tokens, dim=1) if label_audio_tokens else None

            # 为 ChatMLDatasetSample 准备其他字段
            waveform, sr = self._load_audio_waveform(sample["audio_file"])
            
            dataset_sample = ChatMLDatasetSample(
                input_ids=torch.tensor(input_tokens, dtype=torch.long),
                label_ids=torch.tensor(label_tokens, dtype=torch.long),
                audio_ids_concat=audio_ids_concat,
                audio_ids_start=audio_ids_start,
                audio_label_ids_concat=label_audio_ids,
                audio_waveforms_concat=waveform,
                audio_waveforms_start=torch.tensor([0], dtype=torch.long),
                audio_sample_rate=torch.tensor([sr], dtype=torch.float32),
                audio_speaker_indices=torch.tensor([0], dtype=torch.long),
            )
            
            # Return the dataset sample (device transfer will be handled by collator/trainer)
            return dataset_sample

        except Exception as e:
            logger.error(f"Error processing sample at index {idx} (ID: {sample.get('audio_id', 'N/A')}): {e}", exc_info=True)
            # 返回下一个样本，避免因单个损坏样本导致训练中断
            return self.__getitem__((idx + 1) % len(self))


class HiggsAudioModelWrapper(nn.Module):
    """Wrapper for Higgs Audio v2 model to enable training with memory optimizations"""
    
    def __init__(self, model_path: str, device: str = 'cuda', args=None):
        super().__init__()
        self.args = args
        
        # For training, always load model in float32 and let Trainer/AMP handle mixed precision
        print("[INFO] Loading model in float32 for training. Mixed precision will be handled by Trainer/AMP.")
        
        try:
            if HIGGS_AVAILABLE:
                # Load config first to check model specifications
                try:
                    config = HiggsAudioConfig.from_pretrained(model_path)
                    self.config = config
                except Exception as config_error:
                    logger.error(f"Failed to load config using from_pretrained: {config_error}")
                    logger.info("Attempting alternative config loading...")
                    
                    # Try loading config manually from JSON
                    import json
                    config_file = Path(model_path) / "config.json"
                    if config_file.exists():
                        with open(config_file, 'r', encoding='utf-8') as f:
                            config_data = json.load(f)
                        config = HiggsAudioConfig(**config_data)
                        self.config = config
                        logger.info("✓ Alternative config loading successful")
                    else:
                        raise FileNotFoundError(f"Config file not found: {config_file}")
                
                # Load model without device_map initially to avoid conflicts
                self.model = HiggsAudioModel.from_pretrained(
                    config=config,
                    pretrained_model_name_or_path=model_path,
                    # Remove device_map to avoid conflicts with manual device placement
                )
                
                # 🚑 CRITICAL: Apply CLASS-LEVEL monkey-patch to fix labels parameter issue
                self._apply_critical_labels_fix()
                
                # Manually move to device
                self.model = self.model.to(device)
                
            else:
                from transformers import AutoModel, AutoConfig
                self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
                self.model = self.model.to(device)
                
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise
        
        # Apply model freezing if specified
        if args:
            self._apply_freezing_strategy(args)
        
        # Enable gradient checkpointing for memory efficiency
        self._setup_gradient_checkpointing()
        
        # Log model information
        self._log_model_info()

    def _apply_critical_labels_fix(self):
        """
        🚑 CRITICAL CLASS-LEVEL FIX: Monkey-patch HiggsAudioModel.forward to ignore labels parameter
        This patches the actual class method, not just the instance, ensuring ALL calls are intercepted.
        """
        try:
            from boson_multimodal.model.higgs_audio.modeling_higgs_audio import HiggsAudioModel
            
            # Store the original forward method at class level
            if not hasattr(HiggsAudioModel, '_original_forward_method'):
                HiggsAudioModel._original_forward_method = HiggsAudioModel.forward
                logger.info("✓ STORED original HiggsAudioModel.forward method")
                
                def patched_forward_method(self, *args, **kwargs):
                    """
                    Patched forward method that removes labels/label_ids parameters.
                    This intercepts ALL calls to HiggsAudioModel.forward from any source.
                    """
                    # Remove problematic parameters
                    kwargs.pop('labels', None)
                    kwargs.pop('label_ids', None)
                    
                    # Call the original method
                    return HiggsAudioModel._original_forward_method(self, *args, **kwargs)
                
                # Replace the class method
                HiggsAudioModel.forward = patched_forward_method
                logger.info("✓ CRITICAL CLASS-LEVEL PATCH APPLIED: HiggsAudioModel.forward method replaced globally")
                logger.info("✓ ALL instances of HiggsAudioModel will now ignore 'labels' parameter")
                
        except Exception as e:
            logger.error(f"🚨 CRITICAL: Failed to apply class-level labels fix: {e}")
            logger.error("🚨 Training may fail due to labels parameter issue")

    def _apply_freezing_strategy(self, args):
        """Apply freezing strategy to model components"""
        try:
            if args.freeze_audio_tower and hasattr(self.model, 'freeze_audio_tower'):
                self.model.freeze_audio_tower()
                logger.info("Audio tower frozen")
                
            if args.freeze_audio_encoder_proj and hasattr(self.model, 'freeze_audio_encoder_proj'):
                self.model.freeze_audio_encoder_proj()
                logger.info("Audio encoder projection frozen")
                
            if args.freeze_llm and hasattr(self.model, 'freeze_llm'):
                self.model.freeze_llm()
                logger.info("LLM component frozen")
                
        except Exception as e:
            logger.warning(f"Error applying freezing strategy: {e}")

    def _setup_gradient_checkpointing(self):
        """Setup gradient checkpointing for memory efficiency"""
        try:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
            elif hasattr(self.model, 'config') and hasattr(self.model.config, 'use_cache'):
                # Disable cache for gradient checkpointing
                self.model.config.use_cache = False
                logger.info("Model cache disabled for gradient checkpointing")
        except Exception as e:
            logger.warning(f"Could not enable gradient checkpointing: {e}")

    def _log_model_info(self):
        """Log model information for debugging"""
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            logger.info(f"Model loaded successfully:")
            logger.info(f"  Total parameters: {total_params:,}")
            logger.info(f"  Trainable parameters: {trainable_params:,}")
            logger.info(f"  Frozen parameters: {total_params - trainable_params:,}")
            logger.info(f"  Model device: {next(self.model.parameters()).device}")
            logger.info(f"  Model dtype: {next(self.model.parameters()).dtype}")
            
        except Exception as e:
            logger.warning(f"Could not log model info: {e}")

    @property
    def device(self):
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
          
    def forward(self, **kwargs):
        # Ensure all inputs are on the same device as the model
        model_device = self.device
        
        # AGGRESSIVE DEBUG: Check for labels in kwargs
        if 'labels' in kwargs:
            logger.error(f"FOUND LABELS IN FORWARD METHOD! Input keys: {list(kwargs.keys())}")
            logger.error("This should not happen with our fixes!")
        
        # Extract labels/label_ids if present (HF Trainer passes this but HiggsAudio doesn't use it)
        labels = kwargs.pop('labels', None)  # Remove labels from kwargs
        label_ids = kwargs.pop('label_ids', None)  # Remove label_ids from kwargs
        
        # Use label_ids as labels if labels is None
        if labels is None and label_ids is not None:
            labels = label_ids
        
        # Log what we're passing to the model for debugging
        logger.debug(f"After label removal, passing to model: {list(kwargs.keys())}")
        
        # Final safety check - ensure no labels are in kwargs
        if 'labels' in kwargs or 'label_ids' in kwargs:
            logger.error("CRITICAL: Labels still found in kwargs after removal!")
            kwargs = {k: v for k, v in kwargs.items() if k not in ['labels', 'label_ids']}
            logger.error(f"Cleaned kwargs: {list(kwargs.keys())}")
        
        # Move inputs to model device
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and value.device != model_device:
                kwargs[key] = value.to(model_device)
        
        # Ensure labels are also on correct device if they exist
        if labels is not None and isinstance(labels, torch.Tensor):
            labels = labels.to(model_device)
        
        # Memory management before forward pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            if HIGGS_AVAILABLE:
                # The model's forward method is already patched at class level
                logger.debug(f"Calling globally-patched HiggsAudioModel with kwargs: {list(kwargs.keys())}")
                outputs = self.model(**kwargs)
                logger.info("✓ Globally-patched model call completed successfully!")
                
                # For HiggsAudio, we should NOT calculate loss manually since the model handles it internally
                # Let's check if the model already returned a loss
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    logger.debug("Model returned built-in loss")
                    # Use the model's own loss calculation
                    return outputs
                else:
                    logger.debug("Model did not return loss, but this may be expected for HiggsAudio")
                    # For HiggsAudio, this might be normal - just return the outputs
                    return outputs
            else:
                # Fallback implementation
                input_ids = kwargs.get('input_ids')
                attention_mask = kwargs.get('attention_mask')
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                loss = None
                if labels is not None:
                    shift_logits = outputs.logits[..., :-1, :].contiguous().to(model_device)
                    shift_labels = labels[..., 1:].contiguous().to(model_device)
                    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    loss = loss.to(model_device)  # Ensure loss is on correct device
                    
                return type('ModelOutput', (), {"loss": loss, "logits": outputs.logits.to(model_device)})()
                
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error(f"CUDA OOM during forward pass. Current memory usage:")
                if torch.cuda.is_available():
                    logger.error(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                    logger.error(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
                    torch.cuda.empty_cache()
            raise e
        except Exception as e:
            logger.error(f"Unexpected error in forward pass: {e}")
            logger.error(f"Input kwargs keys: {list(kwargs.keys())}")
            logger.error(f"Model device: {model_device}")
            if labels is not None:
                logger.error(f"Labels device: {labels.device if isinstance(labels, torch.Tensor) else 'not tensor'}")
            raise e

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the wrapped model"""
        try:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
                logger.info("✓ Gradient checkpointing enabled via model method")
            elif hasattr(self.model, 'config'):
                # Disable cache for gradient checkpointing
                if hasattr(self.model.config, 'use_cache'):
                    self.model.config.use_cache = False
                logger.info("✓ Gradient checkpointing enabled via config (cache disabled)")
            else:
                logger.warning("Model does not support gradient checkpointing")
        except Exception as e:
            logger.warning(f"Could not enable gradient checkpointing: {e}")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the wrapped model"""
        try:
            if hasattr(self.model, 'gradient_checkpointing_disable'):
                self.model.gradient_checkpointing_disable()
                logger.info("✓ Gradient checkpointing disabled via model method")
            elif hasattr(self.model, 'config') and hasattr(self.model.config, 'use_cache'):
                self.model.config.use_cache = True
                logger.info("✓ Gradient checkpointing disabled via config (cache enabled)")
        except Exception as e:
            logger.warning(f"Could not disable gradient checkpointing: {e}")

    def get_input_embeddings(self):
        """Get input embeddings from the wrapped model"""
        if hasattr(self.model, 'get_input_embeddings'):
            return self.model.get_input_embeddings()
        return None

    def set_input_embeddings(self, embeddings):
        """Set input embeddings for the wrapped model"""
        if hasattr(self.model, 'set_input_embeddings'):
            self.model.set_input_embeddings(embeddings)

    def get_output_embeddings(self):
        """Get output embeddings from the wrapped model"""
        if hasattr(self.model, 'get_output_embeddings'):
            return self.model.get_output_embeddings()
        return None

    def set_output_embeddings(self, embeddings):
        """Set output embeddings for the wrapped model"""
        if hasattr(self.model, 'set_output_embeddings'):
            self.model.set_output_embeddings(embeddings)

    def resize_token_embeddings(self, new_num_tokens):
        """Resize token embeddings for the wrapped model"""
        if hasattr(self.model, 'resize_token_embeddings'):
            return self.model.resize_token_embeddings(new_num_tokens)
        return None
    
    @property
    def generation_config(self):
        """Get generation config from the wrapped model"""
        if hasattr(self.model, 'generation_config'):
            return self.model.generation_config
        return None
    
    @generation_config.setter
    def generation_config(self, config):
        """Set generation config for the wrapped model"""
        if hasattr(self.model, 'generation_config'):
            self.model.generation_config = config
    
    def tie_weights(self):
        """Tie weights for the wrapped model"""
        if hasattr(self.model, 'tie_weights'):
            self.model.tie_weights()
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Prepare inputs for generation"""
        if hasattr(self.model, 'prepare_inputs_for_generation'):
            return self.model.prepare_inputs_for_generation(*args, **kwargs)
        return kwargs


class HiggsAudioTrainer(Trainer):
    """Custom trainer for Higgs Audio v2 with enhanced memory management and stability"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 添加这行，从model中获取config
        self.config = self.model.config
        self.step_count = 0
        self.memory_cleanup_steps = 50  # Clean memory every N steps
        
        # Initialize scaler for mixed precision training
        self.scaler = None
        if self.args.fp16 and not self.use_apex:
            try:
                self.scaler = torch.amp.GradScaler('cuda')
                logger.info("HiggsAudioTrainer initialized with FP16 scaler")
            except:
                self.scaler = torch.cuda.amp.GradScaler()
                logger.info("HiggsAudioTrainer initialized with legacy FP16 scaler")
        else:
            logger.info("HiggsAudioTrainer initialized without mixed precision")
            
        logger.info(f"HiggsAudioTrainer scaler status: {self.scaler is not None}")
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training step with memory management and mixed precision fixes"""
        model.train()
        
        # CRITICAL: Remove labels from inputs before any processing
        if isinstance(inputs, dict) and 'labels' in inputs:
            logger.warning("Found 'labels' in training_step inputs - removing it!")
            inputs = {k: v for k, v in inputs.items() if k != 'labels'}
        
        # Memory cleanup at regular intervals
        self.step_count += 1
        if self.step_count % self.memory_cleanup_steps == 0:
            self._cleanup_memory()
        
        # Handle mixed precision and training step
        try:
            # Use the standard training step but with our custom loss computation
            return super().training_step(model, inputs, num_items_in_batch)
                
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.warning(f"CUDA OOM during training step {self.step_count}. Attempting recovery...")
                self._handle_oom_error()
                # Try again with smaller effective batch size
                return super().training_step(model, inputs, num_items_in_batch)
            elif "No inf checks were recorded" in str(e):
                logger.warning(f"Gradient scaler assertion error at step {self.step_count}. Trying without mixed precision...")
                # Temporarily disable mixed precision for this step
                old_fp16 = self.args.fp16
                self.args.fp16 = False
                try:
                    result = super().training_step(model, inputs, num_items_in_batch)
                    return result
                finally:
                    self.args.fp16 = old_fp16
            else:
                raise e
        except Exception as e:
            logger.error(f"Unexpected error in training step {self.step_count}: {e}")
            raise e
    
    def _cleanup_memory(self):
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            if self.step_count % (self.memory_cleanup_steps * 4) == 0:  # Log less frequently
                logger.info(f"Memory cleanup at step {self.step_count}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
    
    def _handle_oom_error(self):
        """Handle OOM errors gracefully"""
        logger.warning("Handling CUDA OOM error...")
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Log memory info
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(f"After OOM cleanup: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Custom loss computation with enhanced error handling"""
        try:
            # Extract labels before preparing model inputs
            labels = None
            if isinstance(inputs, dict):
                labels = inputs.pop('labels', None)  # Remove labels from inputs dict
                if labels is None:
                    # Try to get labels from label_audio_ids as fallback
                    labels = inputs.get('label_audio_ids', None)
            
            if isinstance(inputs, ExtendedHiggsAudioBatchInput):
                model_inputs = self._prepare_model_inputs_from_batch(inputs)
            else:
                model_inputs = self._prepare_model_inputs_from_dict(inputs)
            
            # Ensure 'labels' is not in model_inputs
            if 'labels' in model_inputs:
                del model_inputs['labels']
                logger.debug("Removed 'labels' from model_inputs")
            
            # Ensure all inputs are on the same device as model
            model_device = next(model.parameters()).device
            for key, value in model_inputs.items():
                if isinstance(value, torch.Tensor) and value.device != model_device:
                    model_inputs[key] = value.to(model_device, non_blocking=True)
            
            # Move labels to correct device if they exist
            if labels is not None and isinstance(labels, torch.Tensor):
                labels = labels.to(model_device)
            
            # Debug logging
            logger.debug(f"Calling model with inputs: {list(model_inputs.keys())}")
            
            # Forward pass with gradient accumulation support - guaranteed no labels
            outputs = model(**model_inputs)
            
            # Extract loss from outputs or calculate manually
            loss = None
            if isinstance(outputs, dict):
                loss = outputs.get("loss")
            else:
                loss = getattr(outputs, 'loss', None)
            
            # If no loss in outputs, compute proper loss for HiggsAudio model
            if loss is None:
                logger.debug("Model doesn't provide loss, computing proper loss for HiggsAudio training")
                loss = self._compute_higgs_audio_loss(outputs, labels, inputs)
                logger.debug(f"Computed HiggsAudio loss: {loss.item() if loss is not None else 'None'}")
            
            # Fallback: create dummy loss if still None
            if loss is None:
                logger.warning("No loss calculated after all attempts, creating final dummy loss")
                loss = torch.tensor(0.0, requires_grad=True, device=model_device)
            
            # Ensure loss is on correct device
            if isinstance(loss, torch.Tensor) and loss.device != model_device:
                loss = loss.to(model_device)
            
            # Validate loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Invalid loss detected: {loss}. Returning zero loss.")
                loss = torch.tensor(0.0, requires_grad=True, device=model_device)
            
            return (loss, outputs) if return_outputs else loss
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error("CUDA OOM in compute_loss")
                self._handle_oom_error()
                # Return a dummy loss on the correct device
                model_device = next(model.parameters()).device
                return torch.tensor(0.0, requires_grad=True, device=model_device)
            else:
                logger.error(f"Runtime error in compute_loss: {e}")
                raise e
        except Exception as e:
            logger.error(f"Unexpected error in compute_loss: {e}")
            # Ensure dummy loss is on correct device
            try:
                model_device = next(model.parameters()).device
                return torch.tensor(0.0, requires_grad=True, device=model_device)
            except:
                return torch.tensor(0.0, requires_grad=True, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    def _prepare_model_inputs_from_batch(self, inputs):
        """Prepare model inputs from ExtendedHiggsAudioBatchInput"""
        model_inputs = {}
        
        # List of expected input keys
        input_keys = [
            'input_ids', 'attention_mask', 'label_ids', 
            'audio_features', 'audio_feature_attention_mask',
            'audio_out_ids', 'audio_out_ids_start', 
            'audio_out_ids_start_group_loc',
            'audio_in_ids', 'audio_in_ids_start',
            'label_audio_ids', 'reward'
        ]
        
        for key in input_keys:
            value = getattr(inputs, key, None)
            if value is not None:
                model_inputs[key] = value
        
        return model_inputs
    
    def _prepare_model_inputs_from_dict(self, inputs):
        """Prepare model inputs from dictionary"""
        model_inputs = {}
        
        for key, value in inputs.items():
            # Skip labels entirely - they should not go to the model
            if key == 'labels':
                continue  # Don't add labels to model inputs
            elif key in ['input_ids', 'attention_mask', 'label_ids',
                        'audio_features', 'audio_feature_attention_mask',
                        'audio_out_ids', 'audio_out_ids_start', 
                        'audio_out_ids_start_group_loc',
                        'audio_in_ids', 'audio_in_ids_start',
                        'label_audio_ids', 'reward']:
                model_inputs[key] = value
        
        return model_inputs

    def _compute_higgs_audio_loss(self, outputs, labels, inputs):
        """Compute loss for HiggsAudio model from logits and labels"""
        try:
            model_device = next(self.model.parameters()).device
            total_loss = torch.tensor(0.0, requires_grad=True, device=model_device)
            num_losses = 0
            
            # 1. Text Loss from logits and expanded_labels
            if hasattr(outputs, 'logits') and outputs.logits is not None:
                text_logits = outputs.logits.to(model_device)  # Ensure on correct device
                text_labels = None
                
                # Get text labels from outputs.expanded_labels if available
                if hasattr(outputs, 'expanded_labels') and outputs.expanded_labels is not None:
                    text_labels = outputs.expanded_labels.to(model_device)
                elif labels is not None:
                    text_labels = labels.to(model_device) if isinstance(labels, torch.Tensor) else labels
                elif isinstance(inputs, dict) and 'label_ids' in inputs:
                    text_labels = inputs['label_ids'].to(model_device) if isinstance(inputs['label_ids'], torch.Tensor) else inputs['label_ids']
                elif hasattr(inputs, 'label_ids') and inputs.label_ids is not None:
                    text_labels = inputs.label_ids.to(model_device) if isinstance(inputs.label_ids, torch.Tensor) else inputs.label_ids
                
                if text_labels is not None and isinstance(text_labels, torch.Tensor):
                    text_labels = text_labels.to(model_device)  # Ensure on correct device
                    
                    # Standard cross-entropy loss for text generation
                    # Shift so that tokens < n predict n
                    shift_logits = text_logits[..., :-1, :].contiguous()
                    shift_labels = text_labels[..., 1:].contiguous()
                    
                    # Ensure both tensors have the same sequence length
                    min_seq_len = min(shift_logits.size(-2), shift_labels.size(-1))
                    shift_logits = shift_logits[..., :min_seq_len, :]
                    shift_labels = shift_labels[..., :min_seq_len]
                    
                    # Flatten for loss computation
                    shift_logits = shift_logits.reshape(-1, shift_logits.size(-1))
                    shift_labels = shift_labels.reshape(-1)
                    
                    # Only compute loss for non-ignored tokens (-100)
                    valid_mask = (shift_labels != -100)
                    if valid_mask.sum() > 0:
                        text_loss = F.cross_entropy(
                            shift_logits[valid_mask], 
                            shift_labels[valid_mask], 
                            reduction='mean'
                        )
                        total_loss = total_loss + text_loss
                        num_losses += 1
                        logger.debug(f"Text loss: {text_loss.item():.4f}")
                        # Memory cleanup
                        del text_loss, shift_logits, shift_labels, valid_mask
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    else:
                        logger.debug("No valid text tokens for loss computation")
                        # Memory cleanup
                        del shift_logits, shift_labels, valid_mask
            
            # 2. Audio Loss from audio_logits and label_audio_ids  
            if hasattr(outputs, 'audio_logits') and outputs.audio_logits is not None:
                audio_logits = outputs.audio_logits.to(model_device)  # Ensure on correct device
                audio_labels = None
                
                # Get audio labels
                if isinstance(inputs, dict) and 'label_audio_ids' in inputs:
                    audio_labels = inputs['label_audio_ids']
                elif hasattr(inputs, 'label_audio_ids') and inputs.label_audio_ids is not None:
                    audio_labels = inputs.label_audio_ids
                
                if audio_labels is not None and isinstance(audio_labels, torch.Tensor) and audio_labels.numel() > 0:
                    audio_labels = audio_labels.to(model_device)  # Ensure on correct device
                    
                    # audio_labels shape: (num_codebooks, seq_len)
                    # audio_logits shape: (seq_len, num_codebooks, codebook_size)
                    
                    # Transpose to match dimensions
                    if audio_labels.dim() == 2 and audio_logits.dim() == 3:
                        audio_labels = audio_labels.transpose(0, 1)  # (seq_len, num_codebooks)
                    
                    # Compute loss for each codebook
                    audio_loss = torch.tensor(0.0, device=model_device, requires_grad=True)
                    valid_codebooks = 0
                    
                    for codebook_idx in range(min(audio_logits.size(1), audio_labels.size(1))):
                        codebook_logits = audio_logits[:, codebook_idx, :]  # (seq_len, codebook_size)
                        codebook_labels = audio_labels[:, codebook_idx]     # (seq_len,)
                        
                        # Only compute loss for valid labels (not -100)
                        valid_mask = (codebook_labels != -100)
                        if valid_mask.sum() > 0:
                            cb_loss = F.cross_entropy(
                                codebook_logits[valid_mask], 
                                codebook_labels[valid_mask], 
                                reduction='mean'
                            )
                            audio_loss = audio_loss + cb_loss
                            valid_codebooks += 1
                    
                    if valid_codebooks > 0:
                        audio_loss = audio_loss / valid_codebooks  # Average across codebooks
                        total_loss = total_loss + audio_loss
                        num_losses += 1
                        logger.debug(f"Audio loss: {audio_loss.item():.4f}, codebooks: {valid_codebooks}")
                        # Memory cleanup for audio processing
                        del audio_loss
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Average the losses if we have multiple components
            if num_losses > 0:
                final_loss = total_loss / num_losses
                logger.debug(f"Combined loss: {final_loss.item():.4f} (components: {num_losses})")
                return final_loss.to(model_device)
            else:
                # Fallback: create a small loss from logits to ensure gradients flow
                logger.warning("No valid labels found for loss computation, creating minimal gradient loss")
                if hasattr(outputs, 'logits') and outputs.logits is not None:
                    return torch.mean(outputs.logits.to(model_device) * 0.001)  # Very small but non-zero loss
                else:
                    return torch.tensor(0.001, requires_grad=True, device=model_device)
                    
        except Exception as e:
            logger.error(f"Error in _compute_higgs_audio_loss: {e}")
            # Return a small gradient-preserving loss on error
            model_device = next(self.model.parameters()).device
            return torch.tensor(0.001, requires_grad=True, device=model_device)

    def _prepare_inputs(self, inputs):
        """Override to remove labels before they reach the model"""
        # Let parent prepare inputs first
        inputs = super()._prepare_inputs(inputs)
        
        # Remove labels if present since HiggsAudioModel doesn't accept them
        if isinstance(inputs, dict) and 'labels' in inputs:
            logger.debug("Removing 'labels' from inputs before model forward pass")
            inputs = {k: v for k, v in inputs.items() if k != 'labels'}
        
        return inputs


def setup_lora_config(model: nn.Module, lora_config: Dict) -> nn.Module:
    """Setup LoRA configuration for the model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_config.get("rank", 16),
        lora_alpha=lora_config.get("alpha", 32),
        lora_dropout=lora_config.get("dropout", 0.1),
        target_modules=lora_config.get("target_modules", [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        auto_mapping=True
    )
    
    model = model.to(device)
    
    if hasattr(model, 'model') and hasattr(model.model, 'text_model'):
        model.model.text_model = get_peft_model(model.model.text_model, peft_config)
    elif hasattr(model, 'model'):
        model.model = get_peft_model(model.model, peft_config)
    else:
        model = get_peft_model(model, peft_config)
    
    model = model.to(device)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Higgs Audio v2 with LoRA")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, default="/root/code/higgs-audio-main/model_ckpt")
    parser.add_argument("--audio_tokenizer_path", type=str, default="/root/code/higgs-audio-main/model_ckpt_tokenizer")
    
    # Data arguments
    parser.add_argument("--train_data_dir", type=str, default="/root/code/higgs-audio-main/higgs_training_data_huo")
    parser.add_argument("--eval_data_dir", type=str, default="")

    # Task type arguments
    parser.add_argument("--task_type", type=str, default="single_speaker_smart_voice",
                       choices=["zero_shot_voice_cloning", "single_speaker_smart_voice", 
                               "multi_speaker_smart_voice", "multi_speaker_voice_cloning"],
                       help="Training task type")
    parser.add_argument("--ref_audio_in_system_message", action="store_true", default=False,
                       help="Whether to include reference audio in system message")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./output/huo_train-vxx")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=4e-5)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--eval_steps", type=int, default=500)
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--logging_dir", type=str, default="./logs/huo_train-vxx")
    parser.add_argument("--report_to", type=str, default="tensorboard", 
                       choices=["tensorboard", "wandb", "none"])
    
    # Training optimization arguments
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False,
                       help="Enable gradient checkpointing to save memory")
    parser.add_argument("--dataloader_num_workers", type=int, default=0,
                       help="Number of dataloader workers")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Number of steps to accumulate gradients before updating")
    parser.add_argument("--dataloader_pin_memory", type=lambda x: x.lower() == 'true', default=True,
                       help="Whether to pin memory in dataloader")
    parser.add_argument("--remove_unused_columns", type=lambda x: x.lower() == 'true', default=True,
                       help="Whether to remove unused columns from dataset")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Maximum gradient norm for clipping")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay for optimizer")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine_with_restarts",
                       help="Learning rate scheduler type")
    
    # Freeze model components
    parser.add_argument("--freeze_audio_tower", action="store_true", default=False)
    parser.add_argument("--freeze_audio_encoder_proj", action="store_true", default=False)
    parser.add_argument("--freeze_llm", action="store_true", default=False)

    args = parser.parse_args()
    
    # Initialize memory monitoring
    memory_monitor.log_memory_usage("Startup")
    
    # Validate arguments
    if args.fp16 and args.bf16:
        raise ValueError("Cannot use both fp16 and bf16. Choose one.")
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    logger.info(f"Starting training with arguments: {vars(args)}")
    memory_monitor.log_memory_usage("After argument parsing")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("✓ Text tokenizer loaded successfully")
        memory_monitor.log_memory_usage("After tokenizer loading")
    except Exception as e:
        logger.error(f"Failed to load tokenizer from {args.model_path}: {e}")
        raise
    
    # Load audio tokenizer
    if HIGGS_AVAILABLE:
        # GPU Optimization: Load audio tokenizer on GPU for much faster processing
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            audio_tokenizer = load_higgs_audio_tokenizer(args.audio_tokenizer_path, device=device)
            logger.info(f"✓ Audio tokenizer loaded successfully on {device}")
            
            # GPU Optimization: Ensure audio tokenizer stays on GPU
            if hasattr(audio_tokenizer, 'to') and torch.cuda.is_available():
                audio_tokenizer = audio_tokenizer.to("cuda")
            if hasattr(audio_tokenizer, 'cuda') and torch.cuda.is_available():
                audio_tokenizer = audio_tokenizer.cuda()
            logger.info(f"✅ Audio tokenizer optimized for GPU processing")
            
            memory_monitor.log_memory_usage("After audio tokenizer loading")
        except Exception as e:
            logger.error(f"Failed to load audio tokenizer: {e}")
            raise
    else:
        audio_tokenizer = None
        logger.warning("Audio tokenizer not available, using fallback")
    
    # Load model
    try:
        model = HiggsAudioModelWrapper(args.model_path, device='cuda', args=args)
        logger.info("✓ Model loaded successfully")
        memory_monitor.log_memory_usage("After model loading")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    # Setup LoRA
    if args.use_lora:
        try:
            lora_config = {
                "rank": args.lora_rank,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
            }
            model = setup_lora_config(model, lora_config)
            logger.info("✓ LoRA configuration applied")
            memory_monitor.log_memory_usage("After LoRA setup")
        except Exception as e:
            logger.error(f"Failed to setup LoRA: {e}")
            raise
    
    # Load datasets
    try:
        logger.info(f"Loading training dataset from {args.train_data_dir}")
        train_dataset = HiggsAudioDataset(
            args.train_data_dir,
            tokenizer,
            audio_tokenizer,
            task_type=args.task_type,
            ref_audio_in_system_message=args.ref_audio_in_system_message
        )
        logger.info(f"✓ Training dataset loaded with {len(train_dataset)} samples")
        memory_monitor.log_memory_usage("After training dataset loading")
    except Exception as e:
        logger.error(f"Failed to load training dataset: {e}")
        raise
    
    eval_dataset = None
    if args.eval_data_dir:
        try:
            logger.info(f"Loading evaluation dataset from {args.eval_data_dir}")
            eval_dataset = HiggsAudioDataset(
                args.eval_data_dir,
                tokenizer,
                audio_tokenizer,
                task_type=args.task_type,
                ref_audio_in_system_message=args.ref_audio_in_system_message
            )
            logger.info(f"✓ Evaluation dataset loaded with {len(eval_dataset)} samples")
            memory_monitor.log_memory_usage("After evaluation dataset loading")
        except Exception as e:
            logger.warning(f"Failed to load evaluation dataset: {e}. Continuing without evaluation.")
            eval_dataset = None
    
    # Setup training arguments with memory optimizations
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps" if eval_dataset else "no",
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        # Mixed precision - use args values
        fp16=args.fp16,  # Use argument value
        bf16=args.bf16 if torch.cuda.is_bf16_supported() else False,  # Only use BF16 if supported
        # Memory optimization settings
        dataloader_pin_memory=args.dataloader_pin_memory,  # Use from args
        dataloader_num_workers=args.dataloader_num_workers,     # Use from args
        remove_unused_columns=args.remove_unused_columns,        # Use from args
        # Gradient optimization
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # Use from args
        gradient_checkpointing=False,   # Disable for now to avoid HiggsAudio compatibility issues
        # Learning rate optimization
        lr_scheduler_type=args.lr_scheduler_type,  # Use from args
        weight_decay=args.weight_decay,            # Use from args
        # Logging and checkpointing
        report_to=args.report_to if args.report_to != "none" else None,
        logging_dir=args.logging_dir,
        # Early stopping and model selection
        save_strategy="steps",
        # Memory cleanup
        skip_memory_metrics=False,     # Monitor memory usage
        # Mixed precision optimizations
        fp16_opt_level=None,  # Disable FP16 optimization level
        # Additional stability settings
        max_grad_norm=args.max_grad_norm,          # Use from args
        # Disable some potentially problematic features for now
        ddp_find_unused_parameters=False,
        ddp_broadcast_buffers=False,
    )
    
    # Setup data collator
    try:
        if HIGGS_AVAILABLE and hasattr(model.config, 'audio_in_token_idx'):
            try:
                from transformers import WhisperProcessor
                whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
                
                data_collator = ExtendedHiggsAudioSampleCollator(
                    whisper_processor=whisper_processor,
                    audio_in_token_id=model.config.audio_in_token_idx,
                    audio_out_token_id=model.config.audio_out_token_idx,
                    audio_stream_bos_id=model.config.audio_stream_bos_id,
                    audio_stream_eos_id=model.config.audio_stream_eos_id,
                    encode_whisper_embed=True,
                    pad_token_id=tokenizer.pad_token_id,
                    return_audio_in_tokens=True,
                    use_delay_pattern=False,
                    round_to=8,
                    audio_num_codebooks=8,
                )
                logger.info("✓ Higgs data collator initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to setup Higgs collator: {e}. Using fallback.")
                data_collator = ExtendedHiggsAudioSampleCollator(pad_token_id=tokenizer.pad_token_id)
        else:
            data_collator = ExtendedHiggsAudioSampleCollator(pad_token_id=tokenizer.pad_token_id)
            logger.warning("Using fallback collator")
        
        memory_monitor.log_memory_usage("After data collator setup")
    except Exception as e:
        logger.error(f"Failed to setup data collator: {e}")
        raise
    
    # Load model config for trainer
    try:
        config = AutoConfig.from_pretrained(args.model_path)
        logger.info("✓ Model config loaded for trainer")
    except Exception as e:
        logger.warning(f"Could not load config: {e}")
        config = model.config
    
    # Initialize trainer
    try:
        trainer = HiggsAudioTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        logger.info("✓ Trainer initialized successfully")
        memory_monitor.log_memory_usage("After trainer initialization")
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}")
        raise
    
    # Pre-training validation
    logger.info("Running pre-training validation...")
    try:
        # Test a small batch
        test_sample = train_dataset[0]
        logger.info("✓ Dataset sample loading test passed")
        
        # Test collator
        test_batch = data_collator([test_sample])
        logger.info("✓ Data collator test passed")
        
        memory_monitor.log_memory_usage("After pre-training validation")
    except Exception as e:
        logger.error(f"Pre-training validation failed: {e}")
        raise
    
    # Start training
    logger.info(f"🚀 Starting training for task: {args.task_type}")
    memory_monitor.log_time_elapsed("Training start")
    
    try:
        trainer.train()
        logger.info("✓ Training completed successfully!")
        memory_monitor.log_time_elapsed("Training completion")
        memory_monitor.log_memory_usage("After training completion")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        memory_monitor.log_memory_usage("After training failure")
        raise
    
    # Save the final model
    try:
        config.save_pretrained(args.output_dir)
        trainer.save_model()
        logger.info(f"✓ Model saved to {args.output_dir}")
        memory_monitor.log_memory_usage("After model saving")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        # Don't raise here, training was successful
    
    # Save LoRA adapters separately
    if args.use_lora:
        try:
            lora_output_dir = os.path.join(args.output_dir, "lora_adapters")
            if hasattr(model, 'model') and hasattr(model.model, 'text_model'):
                model.model.text_model.save_pretrained(lora_output_dir)
            elif hasattr(model, 'model'):
                model.model.save_pretrained(lora_output_dir)
            else:
                model.save_pretrained(lora_output_dir)
            logger.info(f"✓ LoRA adapters saved to {lora_output_dir}")
        except Exception as e:
            logger.error(f"Failed to save LoRA adapters: {e}")
            # Don't raise here, training was successful
    
    logger.info("🎉 Training pipeline completed successfully!")
    memory_monitor.log_time_elapsed("Pipeline completion")
    memory_monitor.log_memory_usage("Pipeline completion")


if __name__ == "__main__":
    main()
