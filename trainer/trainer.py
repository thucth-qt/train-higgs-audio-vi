#!/usr/bin/env python3
"""
Higgs Audio v2 Training Script with LoRA Support and Task-specific Training
Based on the Higgs Audio v2 architecture from Boson AI
"""

import os
import json
import logging
import argparse
import torch
import torch.nn as nn
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加常量定义
AUDIO_PLACEHOLDER_TOKEN = "<|__AUDIO_PLACEHOLDER__|>"

MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE = """You are an AI assistant designed to convert text into speech.
If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.
If no speaker tag is present, select a suitable voice on your own."""


class ExtendedHiggsAudioBatchInput(HiggsAudioBatchInput):
    """
    Extended HiggsAudioBatchInput with __len__ method for Trainer compatibility
    """
    
    def __len__(self):
        """Return the batch size based on input_ids"""
        if self.input_ids is not None:
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


class ExtendedHiggsAudioSampleCollator(HiggsAudioSampleCollator):
    """
    Extended collator that returns our custom batch input class
    """
    
    def __call__(self, batch: List[ChatMLDatasetSample]):
        # 调用父类的 __call__ 方法
        batch_input = super().__call__(batch)
        
        # 收集 label_audio_ids
        label_audio_ids_list = []
        for sample in batch:
            if hasattr(sample, 'label_audio_ids') and sample.label_audio_ids is not None:
                label_audio_ids_list.append(sample.label_audio_ids)
            else:
                # 如果某个样本没有label_audio_ids，用空tensor填充
                label_audio_ids_list.append(torch.tensor([], dtype=torch.long))
        
        # 拼接所有的label_audio_ids
        if label_audio_ids_list and any(ids.numel() > 0 for ids in label_audio_ids_list):
            # 有实际的音频标签数据
            label_audio_ids = torch.cat(label_audio_ids_list, dim=-1)
        else:
            # 没有音频标签数据
            label_audio_ids = None
        
        # 转换为我们的扩展类
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
            label_audio_ids=label_audio_ids,
            reward=batch_input.reward,
        )
        
        return extended_batch


def normalize_chinese_punctuation(text):
    """
    Convert Chinese (full-width) punctuation marks to English (half-width) equivalents.
    """
    # Mapping of Chinese punctuation to English punctuation
    chinese_to_english_punct = {
        "，": ", ",  # comma
        "。": ".",  # period
        "：": ":",  # colon
        "；": ";",  # semicolon
        "？": "?",  # question mark
        "！": "!",  # exclamation mark
        "（": "(",  # left parenthesis
        "）": ")",  # right parenthesis
        "【": "[",  # left square bracket
        "】": "]",  # right square bracket
        "《": "<",  # left angle quote
        "》": ">",  # right angle quote
        """: '"',  # left double quotation
        """: '"',  # right double quotation
        "'": "'",  # left single quotation
        "'": "'",  # right single quotation
        "、": ",",  # enumeration comma
        "--": "-",  # em dash
        "…": "...",  # ellipsis
        "·": ".",  # middle dot
        "「": '"',  # left corner bracket
        "」": '"',  # right corner bracket
        "『": '"',  # left double corner bracket
        "』": '"',  # right double corner bracket
    }

    # Replace each Chinese punctuation with its English counterpart
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
        tokenizer,
        audio_tokenizer,
        task_type: str = "zero_shot_voice_cloning",
        max_audio_length: int = 30,
        max_text_length: int = 512,
        sample_rate: int = 24000,
        use_metadata: bool = True,
        config=None, 
        device=None,
        ref_audio_in_system_message: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.task_type = task_type
        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length
        self.sample_rate = sample_rate
        self.use_metadata = use_metadata
        self.config = config
        self.device = device
        self.ref_audio_in_system_message = ref_audio_in_system_message
        
        # 验证任务类型
        valid_tasks = [
            "zero_shot_voice_cloning", 
            "single_speaker_smart_voice", 
            "multi_speaker_smart_voice", 
            "multi_speaker_voice_cloning"
        ]
        if self.task_type not in valid_tasks:
            raise ValueError(f"Invalid task_type: {self.task_type}. Must be one of {valid_tasks}")
        
        # 检测实际的codebook数量
        self.actual_num_codebooks = self._detect_codebook_size()
        # logger.info(f"Detected {self.actual_num_codebooks} codebooks from audio tokenizer")
        
        # Load samples
        if use_metadata and (self.data_dir / "metadata.json").exists():
            self.samples = self._load_samples_from_metadata()
        else:
            self.samples = self._load_samples_from_directory()
            
        logger.info(f"Loaded {len(self.samples)} samples from {data_dir} for task: {self.task_type}")

    def _detect_codebook_size(self):
        """检测音频tokenizer实际返回的codebook数量"""

        
        # 尝试从一个样本音频中检测
        try:
            # 先加载一个样本来检测codebook数量
            audio_files = list(self.data_dir.glob("*.wav")) + list(self.data_dir.glob("*.mp3"))
            if audio_files:
                test_audio = str(audio_files[0])
                test_tokens = self._encode_audio_tokens(test_audio)
                if test_tokens is not None and len(test_tokens.shape) >= 2:
                    return test_tokens.shape[0]
        except Exception as e:
            logger.warning(f"Failed to detect codebook size: {e}")
        
        return 8  # 默认返回8

    def _to_device(self, tensor):
        """辅助方法：将tensor转换到指定设备"""
        if self.device is not None:
            return tensor.to(self.device)
        return tensor

    def _load_samples_from_metadata(self) -> List[Dict]:
        """Load samples using metadata.json"""
        metadata_path = self.data_dir / "metadata.json"
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        samples = []
        for sample_info in metadata["samples"]:
            audio_path = self.data_dir / sample_info["audio_file"]
            transcript_path = self.data_dir / sample_info["transcript_file"]
            
            if audio_path.exists() and transcript_path.exists():
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript = f.read().strip()
                    
                sample_data = {
                    "audio_path": str(audio_path),
                    "transcript": transcript,
                    "audio_id": sample_info["id"],
                    **{k: v for k, v in sample_info.items() 
                       if k not in ["audio_file", "transcript_file", "id"]}
                }
                
                # 添加参考音频路径（如果有的话）
                if "ref_audio_file" in sample_info:
                    ref_audio_path = self.data_dir / sample_info["ref_audio_file"]
                    if ref_audio_path.exists():
                        sample_data["ref_audio_path"] = str(ref_audio_path)
                        
                samples.append(sample_data)
                
        return samples
    
    def _load_samples_from_directory(self) -> List[Dict]:
        """Load samples by scanning directory"""
        samples = []
        audio_files = list(self.data_dir.glob("*.wav")) + list(self.data_dir.glob("*.mp3"))
        
        for audio_path in audio_files:
            transcript_path = audio_path.with_suffix('.txt')
            if transcript_path.exists():
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript = f.read().strip()
                    
                sample_data = {
                    "audio_path": str(audio_path),
                    "transcript": transcript,
                    "audio_id": audio_path.stem,
                    "scene": "quiet_room",
                    "emotion": "neutral",
                    "language": "en"
                }
                
                # 寻找对应的参考音频文件
                ref_audio_path = audio_path.parent / f"{audio_path.stem}_ref.wav"
                if ref_audio_path.exists():
                    sample_data["ref_audio_path"] = str(ref_audio_path)
                    
                samples.append(sample_data)
                
        return samples
    
    def _load_audio_waveform(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Load audio waveform"""
        try:
            waveform, sr = torchaudio.load(audio_path)
            # 转换为单声道
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            return waveform.squeeze(0), sr
        except Exception as e:
            logger.warning(f"Failed to load audio {audio_path}: {e}")
            # 返回空音频
            return torch.zeros(self.sample_rate), self.sample_rate
    

    def _encode_audio_tokens(self, audio_path: str) -> Optional[torch.Tensor]:
        """Encode audio to tokens using audio tokenizer"""
        if self.audio_tokenizer is None:
            return None
            
        try:
            if hasattr(self.audio_tokenizer, 'encode'):
                audio_tokens = self.audio_tokenizer.encode(audio_path)
                return audio_tokens
            else:
                logger.warning("Audio tokenizer does not have encode method")
                return None
        except Exception as e:
            logger.warning(f"Failed to encode audio {audio_path}: {e}")
            return None

    def _get_scene_description(self, sample: Dict) -> str:
        """Generate scene description based on metadata"""
        scene = sample.get("scene", "quiet_room")
        emotion = sample.get("emotion", "neutral")
        language = sample.get("language", "en")
        
        scene_desc = f"Audio is recorded from a {scene.replace('_', ' ')}."
        if emotion != "neutral":
            scene_desc += f" The speaker sounds {emotion}."
        if language != "en":
            scene_desc += f" The language is {language}."
            
        return scene_desc
    
    def _normalize_transcript(self, transcript: str) -> str:
        """Normalize transcript text"""
        # 应用标点符号标准化
        transcript = normalize_chinese_punctuation(transcript)
        
        # 其他标准化处理
        transcript = transcript.replace("(", " ")
        transcript = transcript.replace(")", " ")
        transcript = transcript.replace("°F", " degrees Fahrenheit")
        transcript = transcript.replace("°C", " degrees Celsius")
        
        # 特殊效果标签标准化
        for tag, replacement in [
            ("[laugh]", "<SE>[Laughter]</SE>"),
            ("[humming start]", "<SE>[Humming]</SE>"),
            ("[humming end]", "<SE_e>[Humming]</SE_e>"),
            ("[music start]", "<SE_s>[Music]</SE_s>"),
            ("[music end]", "<SE_e>[Music]</SE_e>"),
            ("[music]", "<SE>[Music]</SE>"),
            ("[sing start]", "<SE_s>[Singing]</SE_s>"),
            ("[sing end]", "<SE_e>[Singing]</SE_e>"),
            ("[applause]", "<SE>[Applause]</SE>"),
            ("[cheering]", "<SE>[Cheering]</SE>"),
            ("[cough]", "<SE>[Cough]</SE>"),
        ]:
            transcript = transcript.replace(tag, replacement)
            
        # 清理多余空格
        lines = transcript.split("\n")
        transcript = "\n".join([" ".join(line.split()) for line in lines if line.strip()])
        transcript = transcript.strip()
        
        # 确保句子以标点符号结尾
        if not any([transcript.endswith(c) for c in [".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>"]]):
            transcript += "."
            
        return transcript
    
    def _detect_speaker_tags(self, transcript: str) -> List[str]:
        """检测文本中的说话人标签"""
        pattern = re.compile(r"\[(SPEAKER\d+)\]")
        return sorted(set(pattern.findall(transcript)))
    
    def _create_messages_for_task(self, sample: Dict) -> List[Message]:
        """根据任务类型创建消息"""
        transcript = self._normalize_transcript(sample["transcript"])
        speaker_tags = self._detect_speaker_tags(transcript)
        
        messages = []
        
        if self.task_type == "zero_shot_voice_cloning":
            # Zero-shot voice cloning任务
            if "ref_audio_path" in sample and self.ref_audio_in_system_message:
                # 参考音频在系统消息中
                scene_desc = self._get_scene_description(sample) if self.use_metadata else "Audio is recorded from a quiet room."
                system_message = (
                    "Generate audio following instruction."
                    "\n\n"
                    f"<|scene_desc_start|>\n{scene_desc}\n\n"
                    f"SPEAKER0: {AUDIO_PLACEHOLDER_TOKEN}\n<|scene_desc_end|>"
                )
                messages.append(_build_system_message_with_audio_prompt(system_message))
            elif "ref_audio_path" in sample:
                # 参考音频作为对话示例
                scene_desc = self._get_scene_description(sample) if self.use_metadata else "Audio is recorded from a quiet room."
                messages.append(Message(
                    role="system",
                    content=f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_desc}\n<|scene_desc_end|>"
                ))
                
                # 添加参考音频的示例对话
                if hasattr(sample, 'ref_transcript'):
                    ref_transcript = sample['ref_transcript']
                else:
                    ref_transcript = "This is a voice sample."
                    
                messages.append(Message(role="user", content=ref_transcript))
                messages.append(Message(role="assistant", content=AudioContent(audio_url=sample["ref_audio_path"])))
            else:
                # 没有参考音频，使用基本系统消息
                messages.append(Message(
                    role="system",
                    content="Generate audio following instruction."
                ))
                
        elif self.task_type == "single_speaker_smart_voice":
            # 单说话人智能语音任务
            scene_desc_parts = ["Generate audio following instruction."]
            if self.use_metadata:
                scene_desc_parts.append(f"<|scene_desc_start|>\n{self._get_scene_description(sample)}\n<|scene_desc_end|>")
            
            messages.append(Message(
                role="system", 
                content="\n\n".join(scene_desc_parts)
            ))
            
        elif self.task_type == "multi_speaker_smart_voice":
            # 多说话人智能语音任务
            if len(speaker_tags) > 1:
                speaker_desc_l = []
                for idx, tag in enumerate(speaker_tags):
                    if idx % 2 == 0:
                        speaker_desc = "feminine"
                    else:
                        speaker_desc = "masculine"
                    speaker_desc_l.append(f"{tag}: {speaker_desc}")
                
                speaker_desc = "\n".join(speaker_desc_l)
                scene_desc_l = []
                if self.use_metadata:
                    scene_desc_l.append(self._get_scene_description(sample))
                scene_desc_l.append(speaker_desc)
                scene_desc = "\n\n".join(scene_desc_l)
                
                messages.append(Message(
                    role="system",
                    content=f"{MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE}\n\n<|scene_desc_start|>\n{scene_desc}\n<|scene_desc_end|>"
                ))
            else:
                # 回退到单说话人模式
                messages.append(Message(
                    role="system",
                    content="Generate audio following instruction."
                ))
                
        elif self.task_type == "multi_speaker_voice_cloning":
            # 多说话人语音克隆任务
            if "ref_audio_path" in sample and len(speaker_tags) > 1:
                scene_desc = self._get_scene_description(sample) if self.use_metadata else "Audio is recorded from a quiet room."
                speaker_desc_l = [f"SPEAKER{i}: {AUDIO_PLACEHOLDER_TOKEN}" for i in range(len(speaker_tags))]
                
                system_message = (
                    "Generate audio following instruction."
                    "\n\n"
                    f"<|scene_desc_start|>\n{scene_desc}\n\n" + "\n".join(speaker_desc_l) + "\n<|scene_desc_end|>"
                )
                messages.append(_build_system_message_with_audio_prompt(system_message))
            else:
                # 回退到多说话人智能语音
                messages.append(Message(
                    role="system",
                    content=f"{MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE}"
                ))
        
        return messages
    
    def _create_chatml_sample(self, sample: Dict) -> ChatMLSample:
        """Create ChatML format sample based on task type"""
        transcript = self._normalize_transcript(sample["transcript"])
        messages = self._create_messages_for_task(sample)
        
        # 添加用户消息
        messages.append(Message(role="user", content=transcript))
        
        # 添加助手消息（包含音频）
        messages.append(Message(role="assistant", content=AudioContent(audio_url="")))
        
        return ChatMLSample(messages=messages)

    def __len__(self) -> int:
        return len(self.samples)


    def __getitem__(self, idx: int) -> ChatMLDatasetSample:
        """返回 ChatMLDatasetSample 格式的数据"""
        sample = self.samples[idx]
        
        # 创建 ChatML 格式的样本
        chatml_sample = self._create_chatml_sample(sample)
        
        # 使用 prepare_chatml_sample 处理
        input_tokens, label_tokens, audio_contents, audio_label_contents, speaker_id = prepare_chatml_sample(chatml_sample, self.tokenizer)
        
        # 检查是否处理成功
        if input_tokens is None:
            logger.warning(f"Failed to process sample at index {idx}")
            return self._create_empty_sample()
        
        # 转换为tensor
        input_ids = torch.tensor(input_tokens, dtype=torch.long)
        label_ids = torch.tensor(label_tokens, dtype=torch.long) if label_tokens else input_ids.clone()
        
        # 处理音频数据
        audio_ids_list = []
        audio_waveforms_list = []
        audio_sample_rates = []
        audio_speaker_indices = []
        audio_ids_start = []
        audio_waveforms_start = []
        label_audio_ids_list = []
        
        current_audio_offset = 0
        current_waveform_offset = 0
        
        # 使用实际检测到的codebook数量
        expected_num_codebooks = self.actual_num_codebooks
        
        # 处理音频内容
        for i, (audio_content, audio_label_content) in enumerate(zip(audio_contents or [], audio_label_contents or [])):
            # 编码音频为tokens
            if hasattr(audio_content, 'audio_url') and audio_content.audio_url:
                audio_tokens = self._encode_audio_tokens(audio_content.audio_url)
            else:
                audio_tokens = self._encode_audio_tokens(sample["audio_path"])
            
            if audio_tokens is not None and audio_tokens.numel() > 0:
                # 记录起始位置
                audio_ids_start.append(current_audio_offset)
                
                # 添加音频tokens（保持原始维度）
                audio_ids_list.append(audio_tokens)
                current_audio_offset += audio_tokens.shape[-1]
            
            # 处理音频标签
            if audio_label_content is not None:
                if hasattr(audio_label_content, 'audio_url') and audio_label_content.audio_url:
                    label_tokens_audio = self._encode_audio_tokens(audio_label_content.audio_url)
                else:
                    label_tokens_audio = audio_tokens
                
                if label_tokens_audio is not None:
                    label_audio_ids_list.append(label_tokens_audio)
            
            # 处理音频波形
            if hasattr(audio_content, 'audio_url') and audio_content.audio_url:
                waveform, sr = self._load_audio_waveform(audio_content.audio_url)
            else:
                waveform, sr = self._load_audio_waveform(sample["audio_path"])
            
            audio_waveforms_start.append(current_waveform_offset)
            audio_waveforms_list.append(waveform)
            current_waveform_offset += len(waveform)
            
            audio_sample_rates.append(sr)
            
            speaker_idx = 0
            if speaker_id is not None:
                speaker_idx = hash(str(speaker_id)) % 1000
            audio_speaker_indices.append(speaker_idx)
        
        # 如果没有音频内容，添加sample的音频
        if not audio_ids_list:
            audio_waveform, sr = self._load_audio_waveform(sample["audio_path"])
            audio_tokens = self._encode_audio_tokens(sample["audio_path"])
            
            if audio_tokens is not None and audio_tokens.numel() > 0:
                audio_ids_list.append(audio_tokens)
                audio_ids_start.append(0)
                label_audio_ids_list.append(audio_tokens)
            
            audio_waveforms_list.append(audio_waveform)
            audio_waveforms_start.append(0)
            audio_sample_rates.append(sr)
            audio_speaker_indices.append(0)
        
        # 拼接所有音频数据
        if audio_ids_list:
            audio_ids_concat = torch.cat(audio_ids_list, dim=-1)
        else:
            # 使用实际的codebook数量创建空tensor
            audio_ids_concat = torch.zeros((expected_num_codebooks, 0), dtype=torch.long)
        
        if audio_waveforms_list:
            audio_waveforms_concat = torch.cat(audio_waveforms_list, dim=0)
        else:
            audio_waveforms_concat = torch.tensor([], dtype=torch.float32)
        
        if label_audio_ids_list:
            label_audio_ids = torch.cat(label_audio_ids_list, dim=-1)
        else:
            label_audio_ids = audio_ids_concat if audio_ids_concat.numel() > 0 else None
        
        # 确保至少有一个音频
        if len(audio_ids_start) == 0:
            audio_ids_start.append(0)
        if len(audio_waveforms_start) == 0:
            audio_waveforms_start.append(0)
        if len(audio_sample_rates) == 0:
            audio_sample_rates.append(self.sample_rate)
        if len(audio_speaker_indices) == 0:
            audio_speaker_indices.append(0)
        
        # 创建ChatMLDatasetSample
        dataset_sample = ChatMLDatasetSample(
            input_ids=self._to_device(input_ids),
            label_ids=self._to_device(label_ids),
            audio_ids_concat=self._to_device(audio_ids_concat),
            audio_ids_start=self._to_device(torch.tensor(audio_ids_start, dtype=torch.long)),
            audio_waveforms_concat=self._to_device(audio_waveforms_concat),
            audio_waveforms_start=self._to_device(torch.tensor(audio_waveforms_start, dtype=torch.long)),
            audio_sample_rate=self._to_device(torch.tensor(audio_sample_rates, dtype=torch.float32)),
            audio_speaker_indices=self._to_device(torch.tensor(audio_speaker_indices, dtype=torch.long)),
            label_audio_ids=self._to_device(label_audio_ids),
        )
        
        return dataset_sample

    def _create_empty_sample(self):
        """创建一个空的ChatMLDatasetSample作为fallback"""
        expected_num_codebooks = self.actual_num_codebooks
        
        return ChatMLDatasetSample(
            input_ids=torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long),
            label_ids=torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long),
            audio_ids_concat=torch.zeros((expected_num_codebooks, 0), dtype=torch.long),
            audio_ids_start=torch.tensor([0], dtype=torch.long),
            audio_waveforms_concat=torch.tensor([], dtype=torch.float32),
            audio_waveforms_start=torch.tensor([0], dtype=torch.long),
            audio_sample_rate=torch.tensor([self.sample_rate], dtype=torch.float32),
            audio_speaker_indices=torch.tensor([0], dtype=torch.long),
            label_audio_ids=None,
        )


class HiggsAudioModelWrapper(nn.Module):
    """Wrapper for Higgs Audio v2 model to enable training"""
    
    def __init__(self, model_path: str, device: str = 'cuda', args=None):
        super().__init__()
        if HIGGS_AVAILABLE:
            # 使用官方 Higgs Audio 模型
            self.model = HiggsAudioModel.from_pretrained(
                config=HiggsAudioConfig.from_pretrained(model_path),
                pretrained_model_name_or_path=model_path,
                torch_dtype=torch.bfloat16,
                device_map=device,
            )
            self.config = self.model.config
        else:
            # 回退实现
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            self.config = self.model.config
        self.model = self.model.to(device)
        
        #freeze freeze_audio_tower freeze_audio_encoder_proj freeze_llm freeze_text_head
        #
        if args.freeze_audio_tower:
            self.model.freeze_audio_tower = args.freeze_audio_tower
        if args.freeze_audio_encoder_proj:
            self.model.freeze_audio_encoder_proj = args.freeze_audio_encoder_proj
        if args.freeze_llm:
            self.model.freeze_llm = args.freeze_llm
        if args.freeze_text_head:
            self.model.freeze_text_head = args.freeze_text_head

    
    @property
    def device(self):
        return self.model.device
          
    def forward(self, **kwargs):
        #判断model和参数是否在同一个设备上
        if self.model.device != kwargs['input_ids'].device:
            self.model = self.model.to(kwargs['input_ids'].device)

        if HIGGS_AVAILABLE:
            # 使用 Higgs Audio 的前向传播
            return self.model(**kwargs)
        else:
            # 简化的前向传播
            input_ids = kwargs.get('input_ids')
            attention_mask = kwargs.get('attention_mask')
            labels = kwargs.get('label_ids')  # 注意这里可能是 label_ids
            
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            loss = None
            if labels is not None:
                # 简单的语言建模损失
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
            return {"loss": loss, "logits": outputs.logits}


class HiggsAudioTrainer(Trainer):
    """Custom trainer for Higgs Audio v2"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Custom loss computation"""
        # 将 ExtendedHiggsAudioBatchInput 转换为字典，传递给模型
        if isinstance(inputs, ExtendedHiggsAudioBatchInput):
            # 只传递模型需要的参数
            model_inputs = {}
            for attr_name in ['input_ids', 'attention_mask', 'label_ids', 
                            'audio_features', 'audio_feature_attention_mask',
                            'audio_out_ids', 'audio_out_ids_start', 
                            'audio_in_ids', 'audio_in_ids_start',
                            'label_audio_ids']:
                attr_value = getattr(inputs, attr_name, None)
                if attr_value is not None:
                    model_inputs[attr_name] = attr_value
        else:
            # 如果是字典，需要处理 labels -> label_ids 的映射
            model_inputs = {}
            for key, value in inputs.items():
                if key == 'labels':
                    # 将 labels 重命名为 label_ids
                    model_inputs['label_ids'] = value
                elif key in ['input_ids', 'attention_mask', 'label_ids',
                            'audio_features', 'audio_feature_attention_mask',
                            'audio_out_ids', 'audio_out_ids_start', 
                            'audio_in_ids', 'audio_in_ids_start',
                            'label_audio_ids']:
                    model_inputs[key] = value
        
        # 确保所有输入都在相同设备上
        for key, value in model_inputs.items():
            if isinstance(value, torch.Tensor):
                model_inputs[key] = value.to(model.device)
        
        outputs = model(**model_inputs)
        
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        return (loss, outputs) if return_outputs else loss


def check_model_devices(model):
    """检查模型各部分的设备分布"""
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            print(f"{name}: {module.weight.device}")
        if hasattr(module, 'bias') and module.bias is not None:
            print(f"{name}.bias: {module.bias.device}")


def setup_lora_config(model: nn.Module, lora_config: Dict) -> nn.Module:
    """Setup LoRA configuration for the model"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Define LoRA config
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
    
    # 先确保基础模型在GPU上
    model = model.to(device)
    
    # 应用 LoRA
    if hasattr(model, 'model') and hasattr(model.model, 'text_model'):
        model.model.text_model = get_peft_model(model.model.text_model, peft_config)
    elif hasattr(model, 'model'):
        model.model = get_peft_model(model.model, peft_config)
    else:
        model = get_peft_model(model, peft_config)
    
    # 强制所有参数和缓冲区到GPU
    model = model.to(device)
    
    # 递归确保所有子模块都在GPU上
    def ensure_device(module):
        for child in module.children():
            child.to(device)
            ensure_device(child)
    
    ensure_device(model)
    # 使用
    check_model_devices(model)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Higgs Audio v2 with LoRA")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, default="/root/code/higgs-audio-main/model_ckpt")
    parser.add_argument("--audio_tokenizer_path", type=str, default="/root/code/higgs-audio-main/model_ckpt_tokenizer")
    
    # Data arguments
    parser.add_argument("--train_data_dir", type=str, default="/root/code/higgs-audio-main/higgs_training_data")
    parser.add_argument("--eval_data_dir", type=str, default="")
    parser.add_argument("--max_audio_length", type=int, default=30)
    parser.add_argument("--max_text_length", type=int, default=512)
    
    # 新增任务类型参数
    parser.add_argument("--task_type", type=str, default="single_speaker_smart_voice",
                       choices=["zero_shot_voice_cloning", "single_speaker_smart_voice", 
                               "multi_speaker_smart_voice", "multi_speaker_voice_cloning"],
                       help="Training task type")
    parser.add_argument("--ref_audio_in_system_message", action="store_true", default=False,
                       help="Whether to include reference audio in system message")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./higgs_audio_lora")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--logging_dir", type=str, default="./logs")
    parser.add_argument("--report_to", type=str, default="tensorboard", 
                       choices=["tensorboard", "wandb", "none"])
    
    # freeze model
    parser.add_argument("--freeze_audio_tower", action="store_true", default=False)
    parser.add_argument("--freeze_audio_encoder_proj", action="store_true", default=False)
    parser.add_argument("--freeze_llm", action="store_true", default=True)
    parser.add_argument("--freeze_text_head", action="store_true", default=True)

    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load audio tokenizer
    if HIGGS_AVAILABLE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            audio_tokenizer = load_higgs_audio_tokenizer(args.audio_tokenizer_path, device=device)
        except Exception as e:
            logger.warning(f"Failed to load audio tokenizer: {e}")
            audio_tokenizer = None
    else:
        audio_tokenizer = None
        logger.warning("Audio tokenizer not available, using fallback")
    
    # Load model
    model = HiggsAudioModelWrapper(args.model_path, device='cuda', args=args)
    
    # Setup LoRA
    if args.use_lora:
        lora_config = {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
        }
        model = setup_lora_config(model, lora_config)
        logger.info("LoRA configuration applied")
    
    # Load datasets with task-specific configuration
    train_dataset = HiggsAudioDataset(
        args.train_data_dir,
        tokenizer,
        audio_tokenizer,
        task_type=args.task_type,  # 传递任务类型
        max_audio_length=args.max_audio_length,
        max_text_length=args.max_text_length,
        config=model.config,
        ref_audio_in_system_message=args.ref_audio_in_system_message
    )
    
    eval_dataset = None
    if args.eval_data_dir:
        eval_dataset = HiggsAudioDataset(
            args.eval_data_dir,
            tokenizer,
            audio_tokenizer,
            task_type=args.task_type,  # 传递任务类型
            max_audio_length=args.max_audio_length,
            max_text_length=args.max_text_length,
            config=model.config,
            ref_audio_in_system_message=args.ref_audio_in_system_message
        )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps" if eval_dataset else "no",
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        fp16=args.fp16,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=args.report_to if args.report_to != "none" else None,
        logging_dir=args.logging_dir,
    )
    
    # 使用扩展的 collator
    if HIGGS_AVAILABLE and hasattr(model.config, 'audio_in_token_idx'):
        from transformers import WhisperProcessor
        
        # 你需要加载 Whisper processor
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
    else:
        data_collator = None
        logger.warning("Using default collator due to missing config or unavailable Higgs modules")

    
    # Initialize trainer
    trainer = HiggsAudioTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training
    logger.info(f"Starting training for task: {args.task_type}")
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    logger.info(f"Model saved to {args.output_dir}")
    
    # Save LoRA adapters separately
    if args.use_lora:
        lora_output_dir = os.path.join(args.output_dir, "lora_adapters")
        if hasattr(model, 'model') and hasattr(model.model, 'text_model'):
            model.model.text_model.save_pretrained(lora_output_dir)
        elif hasattr(model, 'model'):
            model.model.save_pretrained(lora_output_dir)
        else:
            model.save_pretrained(lora_output_dir)
        logger.info(f"LoRA adapters saved to {lora_output_dir}")


if __name__ == "__main__":
    main()