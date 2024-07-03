import whisper
import numpy as np
import sounddevice as sd
from googletrans import Translator
import threading
import queue
import torch

class LivestreamTranslator:
    def __init__(self, source_language='en', target_languages=['de', 'it'], model_size='base'):
        self.model = whisper.load_model(model_size)
        self.translator = Translator()
        self.source_language = source_language
        self.target_languages = target_languages
        self.is_running = False
        self.audio_queue = queue.Queue()

    def capture_audio(self):
        def callback(indata, frames, time, status):
            if status:
                print(status, file=sys.stderr)
            self.audio_queue.put(indata.copy())

        with sd.InputStream(samplerate=16000, channels=1, callback=callback):
            while self.is_running:
                sd.sleep(100)

    def process_audio(self):
        audio_data = []
        while self.is_running:
            try:
                audio_chunk = self.audio_queue.get(timeout=1)
                audio_data.append(audio_chunk)
                
                if len(audio_data) * 100 > 16000 * 30:  # Process every 30 seconds
                    audio = np.concatenate(audio_data)
                    audio_data = []
                    
                    result = self.model.transcribe(audio, language=self.source_language)
                    self.translate_and_print(result['text'])
            except queue.Empty:
                continue

    def translate_and_print(self, text):
        print(f"Original ({self.source_language}): {text}")
        for lang in self.target_languages:
            try:
                translated = self.translator.translate(text, src=self.source_language, dest=lang)
                print(f"Translated ({lang}): {translated.text}")
            except Exception as e:
                print(f"Translation error for {lang}: {e}")
        print("--------------------")

    def start(self):
        self.is_running = True
        threading.Thread(target=self.capture_audio).start()
        threading.Thread(target=self.process_audio).start()

    def stop(self):
        self.is_running = False

if __name__ == "__main__":
    translator = LivestreamTranslator()
    translator.start()
    
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Stopping the translator...")
        translator.stop()