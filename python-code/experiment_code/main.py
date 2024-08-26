import asyncio
import json
from config import load_config, get_output_structure
from experiments.experiment import Experiment
from experiments.livestream import process_livestream
from utils.logger import setup_logger
from utils.performance_logger import PerformanceLogger
from utils.model_optimizer import optimize_model
from evaluation_metrics.accuracy_metrics import evaluate_accuracy
from evaluation_metrics.latency_metrics import measure_latency
from evaluation_metrics.scalability_metrics import test_scalability
from evaluation_metrics.cost_metrics import calculate_cost_effectiveness
from utils.plot_generator import generate_plots

logger = setup_logger()
performance_logger = PerformanceLogger()

async def run_experiment(config, input_source, use_local_models, model_choice, perform_additional_analysis, environment):
    experiment = Experiment(input_source, use_local_models, model_choice, perform_additional_analysis, environment, performance_logger)
    results = await experiment.run()
    
    # Evaluate results
    accuracy_results = evaluate_accuracy(results['api'], results['local'])
    latency_results = measure_latency(results['api'], results['local'])
    scalability_results = await test_scalability(config, experiment, experiment)  # Using same experiment for both API and local for simplicity
    cost_results = calculate_cost_effectiveness(results['api'], results['local'], config)
    
    # Log results
    logger.info(f"Results for environment: {environment}")
    logger.info(f"Accuracy: {json.dumps(accuracy_results, indent=2)}")
    logger.info(f"Latency: {json.dumps(latency_results, indent=2)}")
    logger.info(f"Scalability: {json.dumps(scalability_results, indent=2)}")
    logger.info(f"Cost-effectiveness: {json.dumps(cost_results, indent=2)}")
    
    # Generate plots
    generate_plots(accuracy_results, latency_results, scalability_results, cost_results, environment)
    
    # Save cross-model analysis
    cross_model_analysis = {
        'accuracy': accuracy_results,
        'latency': latency_results,
        'scalability': scalability_results,
        'cost_effectiveness': cost_results
    }
    with open(f"cross_model_analysis_{environment}.json", "w") as f:
        json.dump(cross_model_analysis, f, indent=2)
    
    print("Cross-model analysis results:")
    print(json.dumps(cross_model_analysis, indent=2))

async def main():
    config = load_config()
    
    # Environment selection
    print("Choose an environment:")
    for i, env in enumerate(config.ENVIRONMENTS, 1):
        print(f"{i}. {env}")
    env_choice = int(input("Enter your choice (1-{len(config.ENVIRONMENTS)}): ")) - 1
    current_environment = config.ENVIRONMENTS[env_choice]

    # Optimize models if using GPU
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

    # Additional analysis option
    perform_additional_analysis = input("\nPerform additional analysis (sentiment, video)? (y/n): ").lower() == 'y'

    try:
        if source_choice == "1":
            input_source = input("Enter the path to the video file: ")
            if use_local_models:
                await run_experiment(config, input_source, True, model_choice, perform_additional_analysis, current_environment)
            if use_api_models:
                await run_experiment(config, input_source, False, None, perform_additional_analysis, current_environment)
        else:
            stream_url = input("Enter the livestream URL: ")
            if use_local_models:
                await process_livestream(config, stream_url, True, model_choice, perform_additional_analysis, current_environment, performance_logger)
            if use_api_models:
                await process_livestream(config, stream_url, False, None, perform_additional_analysis, current_environment, performance_logger)

    except Exception as e:
        logger.error(f"An error occurred during the experiment: {e}", exc_info=True)
    finally:
        logger.info("Experiment completed")
        performance_logger.generate_report(current_environment, use_local_models)
        performance_logger.plot_performance(current_environment, use_local_models)

if __name__ == "__main__":
    asyncio.run(main())