"""
Audio Support Plugin for FrameTrain v2
======================================
This plugin adds support for audio tasks:
- Speech Recognition (Whisper, Wav2Vec 2.0)
- Text-to-Speech (Tacotron2, FastSpeech)
- Audio Classification
- Audio Feature Extraction

MANIFEST:
{
    "name": "Audio Plugin",
    "description": "Audio processing (Speech Recognition, TTS, Classification)",
    "modality": "audio",
    "required": [
        "torch",
        "torchaudio"
    ],
    "optional": [
        "librosa",
        "transformers",
        "datasets"
    ],
    "python": "3.8"
}

Installation:
    pip install torchaudio librosa transformers datasets

Usage:
    python train_engine.py --config config.json --plugins ./plugins/
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import torchaudio
    import torchaudio.transforms as T
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False

try:
    import librosa
    import numpy as np
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from transformers import (
        AutoProcessor, AutoModelForSpeechSeq2Seq,
        Wav2Vec2ForCTC, Wav2Vec2Processor,
        WhisperProcessor, WhisperForConditionalGeneration
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# Import from main engine
try:
    from train_engine import (
        BaseDataLoader, BaseModelHandler, Modality, 
        REGISTRY, MessageProtocol, TrainingConfig
    )
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from train_engine import (
        BaseDataLoader, BaseModelHandler, Modality, 
        REGISTRY, MessageProtocol, TrainingConfig
    )


# ============================================================================
# AUDIO DATASET
# ============================================================================

class AudioDataset(Dataset):
    """Dataset for audio files with optional transcriptions"""
    
    def __init__(self, audio_files: List[Path], transcriptions: Optional[List[str]] = None,
                 sample_rate: int = 16000, max_duration: float = 30.0):
        self.audio_files = audio_files
        self.transcriptions = transcriptions
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_samples = int(sample_rate * max_duration)
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        
        # Load audio
        if TORCHAUDIO_AVAILABLE:
            waveform, sr = torchaudio.load(audio_path)
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Trim or pad to max_duration
            if waveform.shape[1] > self.max_samples:
                waveform = waveform[:, :self.max_samples]
            elif waveform.shape[1] < self.max_samples:
                padding = self.max_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        elif LIBROSA_AVAILABLE:
            waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Trim or pad
            if len(waveform) > self.max_samples:
                waveform = waveform[:self.max_samples]
            elif len(waveform) < self.max_samples:
                waveform = np.pad(waveform, (0, self.max_samples - len(waveform)))
            
            waveform = torch.from_numpy(waveform).unsqueeze(0)
        
        else:
            raise ImportError("Neither torchaudio nor librosa is available")
        
        # Get transcription if available
        transcription = self.transcriptions[idx] if self.transcriptions else ""
        
        return {
            "audio": waveform.squeeze(0),  # Shape: [samples]
            "transcription": transcription,
            "audio_path": str(audio_path)
        }


# ============================================================================
# AUDIO DATA LOADER
# ============================================================================

class AudioDataLoader(BaseDataLoader):
    """Data loader for audio tasks"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.processor = None
    
    def load(self) -> Tuple[Any, Optional[Any], Optional[Any]]:
        """Load audio datasets"""
        try:
            MessageProtocol.status("loading", "Loading audio datasets...")
            
            dataset_path = Path(self.config.dataset_path)
            train_path = dataset_path / "train"
            
            if not train_path.exists():
                raise ValueError(f"Training path not found: {train_path}")
            
            # Find audio files
            audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
            audio_files = []
            for ext in audio_extensions:
                audio_files.extend(list(train_path.glob(f"*{ext}")))
            
            if not audio_files:
                raise ValueError(f"No audio files found in {train_path}")
            
            MessageProtocol.status("loading", f"Found {len(audio_files)} audio files")
            
            # Check for transcriptions
            transcriptions = None
            transcription_file = train_path / "transcriptions.json"
            
            if transcription_file.exists():
                with open(transcription_file, 'r', encoding='utf-8') as f:
                    transcription_data = json.load(f)
                
                # Map filenames to transcriptions
                transcription_map = {}
                if isinstance(transcription_data, list):
                    for item in transcription_data:
                        transcription_map[item['audio']] = item['text']
                elif isinstance(transcription_data, dict):
                    transcription_map = transcription_data
                
                transcriptions = [
                    transcription_map.get(f.name, "") 
                    for f in audio_files
                ]
                
                MessageProtocol.status("loading", 
                    f"Loaded {len([t for t in transcriptions if t])} transcriptions")
            
            # Create dataset
            train_dataset = AudioDataset(
                audio_files,
                transcriptions,
                sample_rate=self.config.audio_sample_rate,
                max_duration=30.0
            )
            
            # Custom collate function for variable-length audio
            def collate_fn(batch):
                audios = [item['audio'] for item in batch]
                transcriptions = [item['transcription'] for item in batch]
                
                # Pad audios to same length
                max_len = max(a.shape[0] for a in audios)
                padded_audios = []
                for audio in audios:
                    if audio.shape[0] < max_len:
                        padding = max_len - audio.shape[0]
                        audio = torch.nn.functional.pad(audio, (0, padding))
                    padded_audios.append(audio)
                
                return {
                    "audio": torch.stack(padded_audios),
                    "transcription": transcriptions
                }
            
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                collate_fn=collate_fn
            )
            
            # Load validation data if exists
            val_path = dataset_path / "val"
            if val_path.exists():
                val_audio_files = []
                for ext in audio_extensions:
                    val_audio_files.extend(list(val_path.glob(f"*{ext}")))
                
                if val_audio_files:
                    val_transcriptions = None
                    val_transcription_file = val_path / "transcriptions.json"
                    
                    if val_transcription_file.exists():
                        with open(val_transcription_file, 'r', encoding='utf-8') as f:
                            val_transcription_data = json.load(f)
                        
                        val_transcription_map = {}
                        if isinstance(val_transcription_data, list):
                            for item in val_transcription_data:
                                val_transcription_map[item['audio']] = item['text']
                        else:
                            val_transcription_map = val_transcription_data
                        
                        val_transcriptions = [
                            val_transcription_map.get(f.name, "") 
                            for f in val_audio_files
                        ]
                    
                    val_dataset = AudioDataset(
                        val_audio_files,
                        val_transcriptions,
                        sample_rate=self.config.audio_sample_rate
                    )
                    
                    self.val_loader = DataLoader(
                        val_dataset,
                        batch_size=self.config.batch_size,
                        shuffle=False,
                        num_workers=self.config.num_workers,
                        pin_memory=self.config.pin_memory,
                        collate_fn=collate_fn
                    )
                    
                    MessageProtocol.status("loading", 
                        f"Loaded validation set: {len(val_audio_files)} files")
            
            return self.train_loader, self.val_loader, None
            
        except Exception as e:
            MessageProtocol.error("Failed to load audio data", str(e))
            raise
    
    def get_sample_info(self) -> Dict[str, Any]:
        """Get sample information"""
        return {
            "sample_rate": self.config.audio_sample_rate,
            "num_train_samples": len(self.train_loader.dataset) if self.train_loader else 0,
            "num_val_samples": len(self.val_loader.dataset) if self.val_loader else 0
        }


# ============================================================================
# AUDIO MODEL HANDLER (Whisper, Wav2Vec2)
# ============================================================================

class AudioModelHandler(BaseModelHandler):
    """Handler for audio models (Whisper, Wav2Vec2, etc.)"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.processor = None
        self.model_type = None
    
    def load_model(self):
        """Load audio model"""
        try:
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch not available")
            
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Transformers not available. Install with: pip install transformers")
            
            MessageProtocol.status("loading", "Loading audio model...")
            
            # Determine device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                MessageProtocol.status("device", f"Using GPU: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                MessageProtocol.status("device", "Using Apple Silicon GPU (MPS)")
            else:
                self.device = torch.device("cpu")
                MessageProtocol.status("device", "Using CPU")
            
            model_path = self.config.model_path
            
            # Detect model type
            if "whisper" in model_path.lower():
                self.model_type = "whisper"
                MessageProtocol.status("loading", "Loading Whisper model...")
                
                self.processor = WhisperProcessor.from_pretrained(model_path)
                self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
                
            elif "wav2vec" in model_path.lower():
                self.model_type = "wav2vec2"
                MessageProtocol.status("loading", "Loading Wav2Vec2 model...")
                
                self.processor = Wav2Vec2Processor.from_pretrained(model_path)
                self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
            
            else:
                # Try generic audio model
                self.model_type = "generic"
                MessageProtocol.status("loading", f"Loading generic audio model: {model_path}")
                
                self.processor = AutoProcessor.from_pretrained(model_path)
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path)
            
            # Move to device
            self.model = self.model.to(self.device)
            
            # Enable gradient checkpointing for memory efficiency
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                try:
                    self.model.gradient_checkpointing_enable()
                    MessageProtocol.status("loaded", "Gradient checkpointing enabled")
                except:
                    pass
            
            MessageProtocol.status("loaded", f"Audio model loaded: {self.model_type}")
            
        except Exception as e:
            MessageProtocol.error("Audio model loading failed", str(e))
            raise
    
    def forward(self, batch: Any) -> Any:
        """Forward pass"""
        audio = batch["audio"].to(self.device)
        transcriptions = batch["transcription"]
        
        # Process audio
        if self.model_type == "whisper":
            # Whisper expects input_features
            inputs = self.processor(
                audio.cpu().numpy(),
                sampling_rate=self.config.audio_sample_rate,
                return_tensors="pt"
            )
            
            input_features = inputs.input_features.to(self.device)
            
            # Encode transcriptions
            labels = self.processor.tokenizer(
                transcriptions,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).input_ids.to(self.device)
            
            # Replace padding token id's of the labels by -100
            labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)
            
            outputs = self.model(input_features=input_features, labels=labels)
        
        elif self.model_type == "wav2vec2":
            # Wav2Vec2 expects input_values
            inputs = self.processor(
                audio.cpu().numpy(),
                sampling_rate=self.config.audio_sample_rate,
                return_tensors="pt",
                padding=True
            )
            
            input_values = inputs.input_values.to(self.device)
            
            # Encode transcriptions
            with self.processor.as_target_processor():
                labels = self.processor(
                    transcriptions,
                    return_tensors="pt",
                    padding=True
                ).input_ids.to(self.device)
            
            outputs = self.model(input_values=input_values, labels=labels)
        
        else:
            # Generic approach
            inputs = self.processor(
                audio.cpu().numpy(),
                sampling_rate=self.config.audio_sample_rate,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
        
        return outputs
    
    def compute_loss(self, outputs: Any, batch: Any) -> Any:
        """Compute loss"""
        return outputs.loss
    
    def save_model(self, path: str):
        """Save model"""
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model and processor
        self.model.save_pretrained(output_path)
        self.processor.save_pretrained(output_path)
        
        # Save config
        config_dict = {
            "model_type": self.model_type,
            "sample_rate": self.config.audio_sample_rate
        }
        
        with open(output_path / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        MessageProtocol.status("saved", f"Audio model saved to {output_path}")


# ============================================================================
# REGISTER PLUGIN
# ============================================================================

# Register audio components
REGISTRY.register_data_loader(Modality.AUDIO, AudioDataLoader)
REGISTRY.register_model_handler("audio", AudioModelHandler)

MessageProtocol.debug("Audio plugin loaded", {
    "torchaudio_available": TORCHAUDIO_AVAILABLE,
    "librosa_available": LIBROSA_AVAILABLE,
    "transformers_available": TRANSFORMERS_AVAILABLE
})
