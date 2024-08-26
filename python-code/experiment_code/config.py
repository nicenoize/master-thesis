import os
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import List, Dict
import yaml

@dataclass
class ExperimentConfig:
    # API Keys
    OPENAI_API_KEY: str = ""
    HF_TOKEN: str = ""
    SPEECHMATICS_API_KEY: str = ""
    ELEVENLABS_API_KEY: str = ""

    # General settings
    CHUNK_SIZE: int = 16000 * 10 * 2
    TARGET_LANGUAGES: List[str] = field(default_factory=lambda: ['ger'])
    OUTPUT_DIR: str = "experiment_results"
    MAX_CHUNK_SIZE: int = 25 * 1024 * 1024
    ENVIRONMENTS: List[str] = field(default_factory=lambda: ["M1 Max", "NVIDIA 4070", "Vultr Cloud", "Hetzner Cloud"])

    # Model configurations
    WHISPER_MODELS: List[str] = field(default_factory=lambda: ["tiny", "base", "small", "medium", "large"])
    GPT_MODELS: List[str] = field(default_factory=lambda: ["gpt-3.5-turbo", "gpt-4"])
    LLAMA_MODELS: List[str] = field(default_factory=lambda: ["7B", "13B", "70B"])
    TTS_MODELS: List[str] = field(default_factory=lambda: ["tacotron", "fastspeech"])

    # Default GPT model for API calls
    DEFAULT_GPT_MODEL: str = "gpt-3.5-turbo"

    # Benchmark settings
    TRANSCRIPTION_BENCHMARK_DATASET: str = "path/to/benchmark/dataset"
    TRANSLATION_BENCHMARK_DATASET: str = "path/to/benchmark/dataset"
    SENTIMENT_BENCHMARK_DATASET: str = "path/to/benchmark/dataset"

    # Load testing settings
    MAX_CONCURRENT_REQUESTS: int = 100
    LOAD_TEST_DURATION: int = 300  # seconds

    # Experiment settings
    USE_LOCAL_MODELS: bool = True
    PERFORM_ADDITIONAL_ANALYSIS: bool = True

    # Hardware settings
    USE_GPU: bool = True
    NUM_GPUS: int = 1
    NUM_CPU_CORES: int = os.cpu_count()

    # Cost settings (for cost-effectiveness calculations)
    TRANSCRIPTION_API_COST: float = 0.006  # Cost per minute for OpenAI Whisper API
    SPEECHMATICS_API_COST: float = 0.005  # Cost per minute for Speechmatics API
    TRANSLATION_API_COST: float = 0.00002  # Cost per character for translation API
    SENTIMENT_API_COST: float = 0.0001  # Cost per request for sentiment analysis API
    GPU_COST_PER_HOUR: float = 0.5  # Estimated cost per hour for GPU usage

    # API-specific settings
    OPENAI_WHISPER_MODEL: str = "whisper-1"
    SPEECHMATICS_LANGUAGE: str = "en"

def load_config(config_file: str = "config.yaml") -> ExperimentConfig:
    load_dotenv()
    
    config = ExperimentConfig(
        OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", ""),
        HF_TOKEN=os.getenv("HUGGING_FACE_TOKEN", ""),
        SPEECHMATICS_API_KEY=os.getenv("SPEECHMATICS_API_KEY", ""),
        ELEVENLABS_API_KEY=os.getenv("ELEVENLABS_API_KEY", "")
    )
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                for key, value in yaml_config.items():
                    setattr(config, key, value)
    
    return config

def save_config(config: ExperimentConfig, config_file: str = "config.yaml"):
    with open(config_file, 'w') as f:
        yaml.dump(config.__dict__, f)

def get_output_structure(config: ExperimentConfig, environment: str, model_type: str, whisper_model: str, gpt_model: str) -> Dict[str, str]:
    base_dir = os.path.join(config.OUTPUT_DIR, environment, model_type)
    model_dir = f"{whisper_model}_{gpt_model}"
    return {
        "transcription": os.path.join(base_dir, model_dir, "transcription"),
        "translation": os.path.join(base_dir, model_dir, "translation"),
        "sentiment": os.path.join(base_dir, model_dir, "sentiment"),
        "performance": os.path.join(base_dir, model_dir, "performance"),
        "plots": os.path.join(base_dir, "plots"),
    }

# Load the configuration
config = load_config()

# Example usage:
# print(config.OPENAI_API_KEY)
# print(get_output_structure(config, "M1 Max", "local", "medium", "gpt-3.5-turbo"))

# To save updated configuration:
# save_config(config)