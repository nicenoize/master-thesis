from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import logging

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
        
        try:
            self.processor = WhisperProcessor.from_pretrained(hf_model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(hf_model_name)
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            raise

    async def transcribe(self, audio, language='en'):
        input_features = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features

        # Generate the attention mask
        attention_mask = torch.ones_like(input_features)

        # Ensure the language is set to English
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task="transcribe")

        generated_ids = self.model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            attention_mask=attention_mask
        )

        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription