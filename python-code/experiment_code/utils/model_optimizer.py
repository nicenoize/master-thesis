import torch
from transformers import AutoModel, AutoTokenizer

def optimize_model(config):
    if not config.USE_GPU:
        print("GPU optimization skipped as USE_GPU is False")
        return

    if not torch.cuda.is_available():
        print("CUDA is not available. GPU optimization skipped.")
        return

    print(f"Optimizing models for {config.NUM_GPUS} GPU(s)")
    
    # Example optimization for a transformer model
    model_name = "bert-base-uncased"  # Replace with your model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Enable GPU optimization techniques
    if config.NUM_GPUS > 1:
        model = torch.nn.DataParallel(model)
    
    torch.backends.cudnn.benchmark = True

    return model, tokenizer, device

# Usage in main.py:
# model, tokenizer, device = optimize_model(config)