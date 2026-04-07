"""
Audio Test Plugin for FrameTrain v2
===================================
This plugin adds testing support for audio tasks:
- Speech Recognition (Whisper, Wav2Vec)
- Computes WER (Word Error Rate), CER (Character Error Rate)

MANIFEST:
{
    "name": "Audio Test Plugin",
    "description": "Testing for Audio/Speech Recognition models",
    "modality": "audio",
    "required": [
        "torch",
        "torchaudio",
        "transformers"
    ],
    "optional": [
        "jiwer"
    ],
    "python": "3.8"
}

Usage:
    python test_engine.py --config test_config.json
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torchaudio
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    TORCH_AVAILABLE = True
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    TRANSFORMERS_AVAILABLE = False

try:
    from jiwer import wer, cer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False

from test_engine import (
    BaseTestLoader, Modality, TEST_REGISTRY,
    MessageProtocol, TestConfig
)


def calculate_wer_manual(reference: str, hypothesis: str) -> float:
    """Calculate WER manually if jiwer not available"""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    
    # Simple Levenshtein distance for words
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + 1)
    
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


class AudioTestLoader(BaseTestLoader):
    """Test loader for audio models"""
    
    def __init__(self, config: TestConfig):
        super().__init__(config)
        self.processor = None
        self.sample_rate = 16000
    
    def load_model(self):
        """Load audio model"""
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            raise ImportError("PyTorch and transformers are required")
        
        MessageProtocol.status("loading", f"Loading audio model from {self.config.model_path}...")
        
        self.device = self.get_device()
        MessageProtocol.status("device", f"Using device: {self.device}")
        
        # Load Whisper model
        self.processor = WhisperProcessor.from_pretrained(self.config.model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.config.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        MessageProtocol.status("loaded", "Audio model loaded")
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        """Load audio test data"""
        MessageProtocol.status("loading", "Loading test audio files...")
        
        test_path = Path(self.config.dataset_path) / "test"
        if not test_path.exists():
            test_path = Path(self.config.dataset_path) / "val"
        if not test_path.exists():
            raise ValueError(f"Test data not found")
        
        test_data = []
        
        # Look for audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        
        for audio_file in test_path.glob("*"):
            if audio_file.suffix.lower() not in audio_extensions:
                continue
            
            # Look for transcript file
            transcript_file = audio_file.with_suffix('.txt')
            transcript = None
            
            if transcript_file.exists():
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    transcript = f.read().strip()
            
            test_data.append({
                "audio_path": str(audio_file),
                "transcript": transcript
            })
        
        if not test_data:
            raise ValueError("No test audio files found")
        
        # Limit samples
        if self.config.max_samples and len(test_data) > self.config.max_samples:
            test_data = test_data[:self.config.max_samples]
        
        MessageProtocol.status("loaded", f"Loaded {len(test_data)} test audio files")
        return test_data
    
    def test_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single audio file"""
        audio_path = sample["audio_path"]
        reference = sample.get("transcript")
        
        start_time = time.time()
        
        try:
            # Load audio
            audio_input, sample_rate = torchaudio.load(audio_path)
            
            # Resample if necessary
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                audio_input = resampler(audio_input)
            
            # Convert to mono if stereo
            if audio_input.shape[0] > 1:
                audio_input = torch.mean(audio_input, dim=0, keepdim=True)
            
            # Prepare input
            audio_input = audio_input.squeeze().numpy()
            input_features = self.processor(
                audio_input,
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            ).input_features.to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.model.generate(input_features)
            
            hypothesis = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            inference_time = time.time() - start_time
            
            # Calculate WER and CER if reference available
            wer_score = None
            cer_score = None
            
            if reference:
                if JIWER_AVAILABLE:
                    wer_score = wer(reference, hypothesis)
                    cer_score = cer(reference, hypothesis)
                else:
                    wer_score = calculate_wer_manual(reference, hypothesis)
            
            return {
                "audio_path": audio_path,
                "predicted_transcript": hypothesis,
                "reference_transcript": reference,
                "wer": wer_score,
                "cer": cer_score,
                "inference_time": inference_time
            }
            
        except Exception as e:
            MessageProtocol.warning(f"Failed to process {audio_path}: {e}")
            return {
                "audio_path": audio_path,
                "error": str(e),
                "inference_time": time.time() - start_time
            }
    
    def compute_metrics(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Compute audio metrics"""
        valid_results = [r for r in all_results if "error" not in r]
        
        if not valid_results:
            return {"error": "No valid predictions"}
        
        # Calculate average WER and CER
        wer_scores = [r['wer'] for r in valid_results if r.get('wer') is not None]
        cer_scores = [r['cer'] for r in valid_results if r.get('cer') is not None]
        
        avg_wer = (sum(wer_scores) / len(wer_scores) * 100) if wer_scores else None
        avg_cer = (sum(cer_scores) / len(cer_scores) * 100) if cer_scores else None
        
        # Average inference time
        inference_times = [r['inference_time'] for r in valid_results]
        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
        
        # Accuracy (1 - WER)
        accuracy = (1 - avg_wer / 100) * 100 if avg_wer is not None else None
        
        return {
            "WER": avg_wer,
            "CER": avg_cer,
            "accuracy": accuracy,
            "total_samples": len(valid_results),
            "samples_with_reference": len(wer_scores),
            "average_inference_time": avg_inference_time
        }


# Register audio test loader
TEST_REGISTRY.register_test_loader(Modality.AUDIO, AudioTestLoader)

MessageProtocol.debug("Audio test plugin loaded", {
    "torch_available": TORCH_AVAILABLE,
    "transformers_available": TRANSFORMERS_AVAILABLE,
    "jiwer_available": JIWER_AVAILABLE
})
