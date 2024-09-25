# main.py

import asyncio
import signal
import config  # Import the config module
from utils import (
    validate_file_path,
    signal_handler,
    generate_performance_plots,
)
from processing import run_experiment

logger = config.logger  # Use logger from config if needed

def main():
    # Remove the global statement if present
    # global CURRENT_ENVIRONMENT

    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)

    print("Choose an environment:")
    for idx, env in enumerate(config.ENVIRONMENTS, start=1):
        print(f"{idx}. {env}")
    env_choice = input("Enter your choice (1-4): ")

    config.CURRENT_ENVIRONMENT = config.ENVIRONMENTS[int(env_choice) - 1]

    print("\nChoose an input source:")
    print("1. Process a livestream")
    print("2. Process a video file")
    choice = input("Enter your choice (1 or 2): ")

    use_local_models = input("Use local models? (y/n): ").lower() == "y"

    if choice == "1":
        stream_url = input("Enter the stream URL: ")
        asyncio.run(run_experiment(stream_url, use_local_models))
    elif choice == "2":
        while True:
            file_path = input("Enter the path to the video file: ")
            is_valid, error_message = validate_file_path(file_path)
            if is_valid:
                break
            print(f"Error: {error_message}")
            retry = input("Do you want to try again? (y/n): ")
            if retry.lower() != "y":
                print("Exiting the program.")
                return
        asyncio.run(run_experiment(file_path, use_local_models))
    else:
        print("Invalid choice. Please run the script again and choose 1 or 2.")
        return

    if config.EXPERIMENT_COMPLETED:
        generate_performance_plots()

if __name__ == "__main__":
    main()
