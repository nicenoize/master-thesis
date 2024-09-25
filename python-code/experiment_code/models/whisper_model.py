import torch
import logging
import numpy as np
from pyannote.audio import Pipeline
from typing import Optional
from config import load_config

logger = logging.getLogger(__name__)

class WhisperModel:
    MODEL_MAP = {
        "tiny": "openai/whisper-tiny",
        "base": "openai/whisper-base",
        "small": "openai/whisper-small",
        "medium": "openai/whisper-medium",
        "large": "openai/whisper-large-v2",
    }

    def __init__(self, model_name):
        if model_name not in self.MODEL_MAP:
            raise ValueError(f"Invalid model name. Choose from {', '.join(self.MODEL_MAP.keys())}")
        
        hf_model_name = self.MODEL_MAP[model_name]
        logger.info(f"Loading Whisper model: {hf_model_name}")
        
        # Load the configuration
        self.config = load_config()
        
        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            self.processor = WhisperProcessor.from_pretrained(hf_model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(hf_model_name)
            
            if self.config.HF_TOKEN:
                self.diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                                                     use_auth_token=self.config.HF_TOKEN)
            else:
                logger.warning("No Hugging Face auth token provided in config. Diarization will not be available.")
                self.diarization_pipeline = None
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            raise
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise

    def prepare_audio(self, audio, sampling_rate):
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.shape[0] > audio.shape[1]:
            audio = audio.t()
        if audio.dim() == 2 and audio.shape[0] != 1:
            audio = audio.mean(dim=0, keepdim=True)
        return audio

    async def transcribe(self, audio, sampling_rate=16000, language='en'):
        try:
            # Ensure audio is a 1D numpy array
            if isinstance(audio, torch.Tensor):
                audio = audio.numpy()
            if audio.ndim == 2:
                audio = audio.squeeze()
            if audio.ndim != 1:
                raise ValueError(f"Audio must be 1D array, got shape {audio.shape}")

            logger.info(f"Preprocessed audio shape: {audio.shape}, dtype: {audio.dtype}")
            logger.info(f"Audio min: {audio.min()}, max: {audio.max()}")

            # Convert to 16-bit int if float
            if audio.dtype == np.float32 or audio.dtype == np.float64:
                audio = (audio * 32767).astype(np.int16)
                logger.info(f"Converted audio to int16. New min: {audio.min()}, max: {audio.max()}")

            # Get input features
            input_features = self.processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features
            logger.info(f"Input features shape: {input_features.shape}, dtype: {input_features.dtype}")

            # Get forced decoder ids for language
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task="transcribe")
            logger.info(f"Forced decoder ids type: {type(forced_decoder_ids)}")
            
            # Convert forced_decoder_ids to tensor if it's a list
            if isinstance(forced_decoder_ids, list):
                forced_decoder_ids = torch.tensor(forced_decoder_ids, dtype=torch.long)
            
            logger.info(f"Forced decoder ids shape: {forced_decoder_ids.shape}, dtype: {forced_decoder_ids.dtype}")

            # Ensure shapes are compatible
            batch_size = input_features.shape[0]
            if forced_decoder_ids.shape[0] != batch_size:
                forced_decoder_ids = forced_decoder_ids.repeat(batch_size, 1)
            logger.info(f"Adjusted forced decoder ids shape: {forced_decoder_ids.shape}")

            # Generate transcription
            logger.info("Starting generation...")
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_features, 
                    forced_decoder_ids=forced_decoder_ids,
                    max_length=448,
                    num_beams=5,
                )
            logger.info(f"Generated IDs shape: {generated_ids.shape}, dtype: {generated_ids.dtype}")

            # Decode transcription
            transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            logger.info(f"Transcription length: {len(transcription)}")
            logger.info(f"First 100 characters of transcription: {transcription[:100]}")

            return transcription

        except Exception as e:
            logger.error(f"Error in transcribe method: {e}")
            logger.error(f"Audio shape: {audio.shape if 'audio' in locals() else 'Not available'}")
            logger.error(f"Input features shape: {input_features.shape if 'input_features' in locals() else 'Not available'}")
            logger.error(f"Forced decoder ids: {forced_decoder_ids if 'forced_decoder_ids' in locals() else 'Not available'}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def transcribe_and_diarize(self, audio, sampling_rate=16000, language='en'):
        if self.diarization_pipeline is None:
            raise ValueError("Diarization pipeline is not initialized. Please provide a valid Hugging Face auth token in the config.")

        try:
            # Transcribe
            transcription = await self.transcribe(audio, sampling_rate, language)

            # Prepare audio for diarization
            if isinstance(audio, np.ndarray):
                audio_tensor = torch.from_numpy(audio).float()
            elif isinstance(audio, torch.Tensor):
                audio_tensor = audio.float()
            else:
                raise ValueError(f"Unsupported audio type: {type(audio)}")

            # Ensure audio is 2D (channel, time)
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            elif audio_tensor.dim() > 2:
                raise ValueError(f"Audio tensor has too many dimensions: {audio_tensor.dim()}")

            logger.info(f"Audio tensor shape for diarization: {audio_tensor.shape}")

            # Diarize
            diarization = self.diarization_pipeline({"waveform": audio_tensor, "sample_rate": sampling_rate})

            # Align transcription with diarization
            words = transcription.split()
            word_durations = [len(word) / 15 for word in words]  # Rough estimate of word duration
            total_duration = sum(word_durations)
            
            diarized_transcription = []
            word_index = 0
            current_time = 0

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                turn_words = []
                while word_index < len(words) and current_time < turn.end:
                    turn_words.append(words[word_index])
                    current_time += word_durations[word_index]
                    word_index += 1
                
                if turn_words:
                    diarized_transcription.append(f"Speaker {speaker}: {' '.join(turn_words)}")

            return "\n".join(diarized_transcription)
        except Exception as e:
            logger.error(f"Error in transcribe_and_diarize method: {e}")
            logger.error(f"Audio shape: {audio.shape if hasattr(audio, 'shape') else 'Not available'}")
            logger.error(f"Audio type: {type(audio)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
