import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
import os
from huggingface_hub import snapshot_download
from typing import List, Dict

class GPTModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = self._get_device()
        self.model, self.tokenizer = self._load_model()

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _load_model(self):
        print(f"Loading model {self.model_name} on {self.device}...")
        
        # Automatically download and cache the model
        model_path = snapshot_download(self.model_name)

        if "llama" in self.model_name.lower():
            model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16 if self.device.type != "cpu" else torch.float32)
            tokenizer = LlamaTokenizer.from_pretrained(model_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16 if self.device.type != "cpu" else torch.float32)
            tokenizer = AutoTokenizer.from_pretrained(model_path)

        model = model.to(self.device)
        print(f"Model loaded successfully on {self.device}")
        return model, tokenizer

    async def generate(self, prompt: str, max_length: int = 100) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(input_ids, max_length=max_length, num_return_sequences=1)
        
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    async def translate(self, text: str, target_languages: List[str]) -> Dict[str, str]:
        translations = {}
        for lang in target_languages:
            prompt = f"Translate the following English text to {lang}: {text}"
            translations[lang] = await self.generate(prompt)
        return translations

    async def summarize(self, text: str) -> str:
        prompt = f"Summarize the following text:\n\n{text}\n\nSummary:"
        return await self.generate(prompt)