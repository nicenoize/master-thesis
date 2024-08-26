import asyncio
import json
from config import load_config
from experiments.experiment import Experiment
from experiments.livestream import process_livestream
from utils.logger import setup_logger
from utils.performance_logger import PerformanceLogger
from utils.model_optimizer import optimize_model
import gc

logger = setup_logger()
performance_logger = PerformanceLogger()

async def run_experiment(config, input_source, use_local_models, model_choice, perform_additional_analysis, environment, api_choice=None):
    experiment = Experiment(config, input_source, use_local_models, model_choice, perform_additional_analysis, environment, performance_logger, api_choice)
    results = await experiment.run()

    # Clear experiment object and collect garbage after running
    experiment = None
    gc.collect()

    # Process and log results
    logger.info(f"Experiment Results for {environment}:")
    logger.info(json.dumps(results, indent=2))

async def main():
    config = load_config()
    
    # Environment selection
    print("Choose an environment:")
    for i, env in enumerate(config.ENVIRONMENTS, 1):
        print(f"{i}. {env}")
    env_choice = int(input(f"Enter your choice (1-{len(config.ENVIRONMENTS)}): ")) - 1
    current_environment = config.ENVIRONMENTS[env_choice]

    if config.USE_GPU:
        optimize_model(config)

    # Input source selection
    print("\nChoose an input source:")
    print("1. Process a video file")
    print("2. Process a livestream")
    source_choice = input("Enter your choice (1 or 2): ")

    # Experiment type selection
    print("\nChoose experiment type:")
    print("1. Use local models only")
    print("2. Use API models only")
    print("3. Run both experiments")
    exp_choice = input("Enter your choice (1-3): ")

    use_local_models = exp_choice in ["1", "3"]
    use_api_models = exp_choice in ["2", "3"]

    # Model selection for local experiments
    model_choice = None
    if use_local_models:
        print("\nChoose a model for local experiments:")
        print("1. GPT")
        print("2. LLaMA")
        print("3. Both GPT and LLaMA")
        model_choice = input("Enter your choice (1-3): ")

    # API selection for API experiments
    api_choice = None
    if use_api_models:
        print("\nChoose API for transcription:")
        print("1. OpenAI Whisper")
        print("2. Speechmatics")
        print("3. Both Whisper and Speechmatics")
        api_choice = input("Enter your choice (1-3): ")

    # Additional analysis option
    perform_additional_analysis = input("\nPerform additional analysis (sentiment, video)? (y/n): ").lower() == 'y'

    try:
        if source_choice == "1":
            input_source = input("Enter the path to the video file: ")
            if use_local_models:
                await run_experiment(config, input_source, True, model_choice, perform_additional_analysis, current_environment)
                gc.collect()

            if use_api_models:
                await run_experiment(config, input_source, False, None, perform_additional_analysis, current_environment, api_choice)
                gc.collect()

        else:
            stream_url = input("Enter the livestream URL: ")
            if use_local_models:
                await process_livestream(config, stream_url, True, model_choice, perform_additional_analysis, current_environment, performance_logger)
            if use_api_models:
                await process_livestream(config, stream_url, False, None, perform_additional_analysis, current_environment, performance_logger, api_choice)

    except Exception as e:
        logger.error(f"An error occurred during the experiment: {e}", exc_info=True)
    finally:
        logger.info("Experiment completed")
        performance_logger.generate_report(current_environment, use_local_models)
        performance_logger.plot_performance(current_environment, use_local_models)

if __name__ == "__main__":
    asyncio.run(main())