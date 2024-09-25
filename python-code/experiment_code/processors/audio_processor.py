import librosa
import soundfile as sf
from pydub import AudioSegment
import numpy as np
from scipy.signal import butter, lfilter
from api.openai_api import OpenAIAPI
from api.speechmatics_api import SpeechmaticsAPI
import gc
import io
import logging
import os
from pydub.utils import mediainfo

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, config, api_choice=None):
        self.config = config
        self.api_choice = api_choice
        self.openai_api = OpenAIAPI(config.OPENAI_API_KEY)
        self.speechmatics_api = SpeechmaticsAPI(config.SPEECHMATICS_API_KEY)
        
    async def api_transcribe(self, audio, sampling_rate=16000, language='en'):
        openai_result = None
        speechmatics_result = None

        try:
            # Convert audio to the right format if necessary
            audio_bytes = self.prepare_audio(audio)

            if self.api_choice in ["1", "3"]:
                logger.debug("Calling OpenAI Whisper API")
                openai_result = await self.openai_api.transcribe(audio_bytes, sampling_rate, language)
            
            if self.api_choice in ["2", "3"]:
                logger.debug("Calling Speechmatics API")
                speechmatics_result = await self.speechmatics_api.transcribe(audio_bytes, sampling_rate, language)

        except Exception as e:
            logger.error(f"Error in api_transcribe: {str(e)}")
            raise

        finally:
            # Clear the audio data and collect garbage
            audio = None
            audio_bytes = None
            gc.collect()

        return {
            "openai": openai_result,
            "speechmatics": speechmatics_result
        }

    def prepare_audio(self, audio):
        try:
            logger.debug(f"Preparing audio: {audio}")
            
            if isinstance(audio, np.ndarray):
                # Convert numpy array to WAV format (PCM_16)
                buffer = io.BytesIO()
                sf.write(buffer, audio, 16000, format='WAV', subtype='PCM_16')
                buffer.seek(0)
                logger.debug("Converted numpy array to WAV format (PCM_16).")
                return buffer

            elif isinstance(audio, str):
                # If it's a file path, check format and convert if necessary
                audio_format = self.get_audio_format(audio)
                logger.debug(f"Detected audio format: {audio_format}")

                if audio_format not in ['mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm']:
                    logger.debug(f"Converting {audio_format} to WAV format.")
                    audio_segment = AudioSegment.from_file(audio)
                    buffer = io.BytesIO()
                    audio_segment.export(buffer, format="wav")
                    buffer.seek(0)
                    logger.debug(f"Audio successfully converted to WAV.")
                    return buffer
                else:
                    logger.debug(f"Audio format {audio_format} is already supported. Using original file.")
                    return open(audio, 'rb')

            elif isinstance(audio, bytes):
                # If it's already bytes, wrap in BytesIO
                logger.debug("Audio is already in bytes, wrapping in BytesIO.")
                return io.BytesIO(audio)

            else:
                raise ValueError("Unsupported audio format. Expected numpy array, file path, or bytes.")

        except Exception as e:
            logger.error(f"Error in prepare_audio: {str(e)}")
            raise
        

    def get_audio_format(self, file_path):
        try:
            info = mediainfo(file_path)
            format_name = info.get('format_name', '').lower()
            logger.debug(f"Audio format obtained from mediainfo: {format_name}")
            return format_name
        except Exception as e:
            logger.error(f"Error determining audio format: {str(e)}")
            raise


    def extract_audio(self, video_path):
        try:
            video = AudioSegment.from_file(video_path)
            audio = video.set_channels(1).set_frame_rate(16000)
            audio_path = video_path.rsplit('.', 1)[0] + '_audio.wav'
            audio.export(audio_path, format="wav")
            logger.debug(f"Extracted audio to: {audio_path}")
            return audio_path
        except Exception as e:
            logger.error(f"Error extracting audio: {str(e)}")
            raise

    def resample_audio(self, audio_path, target_sr=16000):
        try:
            audio, sr = librosa.load(audio_path, sr=target_sr)
            sf.write(audio_path, audio, target_sr)
            logger.debug(f"Resampled audio to {target_sr} Hz: {audio_path}")
            return audio_path
        except Exception as e:
            logger.error(f"Error resampling audio: {str(e)}")
            raise

    async def extract_speech_features(self, audio_input):
        try:
            if isinstance(audio_input, str):
                audio, sr = librosa.load(audio_input, sr=None)
                logger.debug(f"Loaded audio from file: {audio_input}")
            elif isinstance(audio_input, AudioSegment):
                audio = np.array(audio_input.get_array_of_samples()).astype(np.float32)
                sr = audio_input.frame_rate
                logger.debug("Converted AudioSegment to numpy array")
            else:
                raise ValueError("Invalid audio input type. Expected string (file path) or AudioSegment.")

            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            pitch = np.mean(pitches[magnitudes > np.max(magnitudes) * 0.7])

            intensity = librosa.feature.rms(y=audio)[0]
            speech_rate = self.estimate_speech_rate(audio, sr)
            formants = self.extract_formants(audio, sr)

            logger.debug("Extracted speech features successfully")
            return {
                "pitch": pitch,
                "intensity": np.mean(intensity),
                "speech_rate": speech_rate,
                "formants": formants
            }
        except Exception as e:
            logger.error(f"Error extracting speech features: {str(e)}")
            raise

    def estimate_speech_rate(self, audio, sr):
        try:
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            return tempo / 60  # Convert BPM to syllables per second
        except Exception as e:
            logger.error(f"Error estimating speech rate: {str(e)}")
            raise

    def extract_formants(self, audio, sr, num_formants=3):
        try:
            lpc_coeffs = librosa.lpc(audio, order=2 + num_formants)
            formants = np.roots(lpc_coeffs)
            formants = formants[np.imag(formants) >= 0]
            formants = np.sort(formants[np.argsort(np.abs(np.imag(formants)))])
            return np.abs(formants[:num_formants])
        except Exception as e:
            logger.error(f"Error extracting formants: {str(e)}")
            raise

    def apply_noise_reduction(self, audio, sr):
        try:
            nyquist = 0.5 * sr
            cutoff = 100  # Adjust this value based on your needs
            normal_cutoff = cutoff / nyquist
            b, a = butter(6, normal_cutoff, btype='high', analog=False)
            filtered_audio = lfilter(b, a, audio)
            logger.debug("Applied noise reduction")
            return filtered_audio
        except Exception as e:
            logger.error(f"Error applying noise reduction: {str(e)}")
            raise

    def detect_silence(self, audio, sr, silence_threshold=-50, min_silence_duration=0.5):
        try:
            intervals = librosa.effects.split(
                audio, 
                top_db=-silence_threshold, 
                frame_length=int(sr * min_silence_duration),
                hop_length=int(sr * 0.01)
            )
            logger.debug(f"Detected {len(intervals)} non-silent intervals")
            return intervals
        except Exception as e:
            logger.error(f"Error detecting silence: {str(e)}")
            raise