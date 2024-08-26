import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

class WhisperModel:
    def __init__(self, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{model_name}").to(self.device)
        self.processor = WhisperProcessor.from_pretrained(f"openai/whisper-{model_name}")

    async def transcribe(self, audio):
        input_features = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(input_features)
        
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription