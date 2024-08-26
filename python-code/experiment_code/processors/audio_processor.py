import librosa
import soundfile as sf
from pydub import AudioSegment
import numpy as np
from scipy.signal import butter, lfilter
from config import config
from api.openai_api import OpenAIAPI

class AudioProcessor:
    def __init__(self, config):
        self.openai_api = OpenAIAPI(config.OPENAI_API_KEY)
        self.speechmatics_api = SpeechmaticsAPI(config.SPEECHMATICS_API_KEY)

    async def api_transcribe(self, audio):
        openai_result = await self.openai_api.transcribe(audio)
        speechmatics_result = await self.speechmatics_api.transcribe(audio)
        return {
            "openai": openai_result,
            "speechmatics": speechmatics_result
        }

    def extract_audio(self, video_path):
        video = AudioSegment.from_file(video_path)
        audio = video.set_channels(1).set_frame_rate(16000)
        audio_path = video_path.rsplit('.', 1)[0] + '_audio.wav'
        audio.export(audio_path, format="wav")
        return audio_path

    def resample_audio(self, audio_path, target_sr=16000):
        audio, sr = librosa.load(audio_path, sr=target_sr)
        sf.write(audio_path, audio, target_sr)
        return audio_path

    async def extract_speech_features(self, audio_chunk):
        # Convert pydub AudioSegment to numpy array
        audio_array = np.array(audio_chunk.get_array_of_samples()).astype(np.float32)
        sample_rate = audio_chunk.frame_rate

        # Extract pitch
        pitches, magnitudes = librosa.piptrack(y=audio_array, sr=sample_rate)
        pitch = np.mean(pitches[magnitudes > np.max(magnitudes) * 0.7])

        # Extract intensity (volume)
        intensity = librosa.feature.rms(y=audio_array)[0]

        # Extract speech rate (syllables per second)
        speech_rate = self.estimate_speech_rate(audio_array, sample_rate)

        # Extract formants
        formants = self.extract_formants(audio_array, sample_rate)

        return {
            "pitch": pitch,
            "intensity": np.mean(intensity),
            "speech_rate": speech_rate,
            "formants": formants
        }

    def estimate_speech_rate(self, audio, sr):
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        return tempo / 60  # Convert BPM to syllables per second

    def extract_formants(self, audio, sr, num_formants=3):
        # Implement formant extraction using linear prediction
        # This is a simplified version and might not be as accurate as specialized software
        lpc_coeffs = librosa.lpc(audio, order=2 + num_formants)
        formants = np.roots(lpc_coeffs)
        formants = formants[np.imag(formants) >= 0]
        formants = np.sort(formants[np.argsort(np.abs(np.imag(formants)))])
        return np.abs(formants[:num_formants])

    def apply_noise_reduction(self, audio, sr):
        # Simple noise reduction using a high-pass filter
        nyquist = 0.5 * sr
        cutoff = 100  # Adjust this value based on your needs
        normal_cutoff = cutoff / nyquist
        b, a = butter(6, normal_cutoff, btype='high', analog=False)
        filtered_audio = lfilter(b, a, audio)
        return filtered_audio

    def detect_silence(self, audio, sr, silence_threshold=-50, min_silence_duration=0.5):
        # Detect silent regions in the audio
        intervals = librosa.effects.split(
            audio, 
            top_db=-silence_threshold, 
            frame_length=int(sr * min_silence_duration),
            hop_length=int(sr * 0.01)
        )
        return intervals