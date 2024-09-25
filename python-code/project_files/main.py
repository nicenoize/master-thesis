# main.py

import asyncio
import signal
import config
from utils import (
    validate_file_path,
    signal_handler,
    generate_performance_plots,
)
from processing import run_experiment
from models import api_request_queue, process_api_request_queue

logger = config.logger

async def main_async(loop):
    # Start the API request queue processor
    queue_processor = loop.create_task(process_api_request_queue(loop))

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
        await run_experiment(stream_url, use_local_models, loop)
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
        await run_experiment(file_path, use_local_models, loop)
    else:
        print("Invalid choice. Please run the script again and choose 1 or 2.")
        return

    # When done, wait for the queue to be empty
    await api_request_queue.join()
    queue_processor.cancel()

    if config.EXPERIMENT_COMPLETED:
        generate_performance_plots()

def main():
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Create a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Run the async main function
    loop.run_until_complete(main_async(loop))
    loop.close()

if __name__ == "__main__":
    main()