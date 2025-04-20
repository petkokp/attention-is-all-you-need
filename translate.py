# translate.py

import torch
import config  # Your configuration file
import spacy
import argparse
import sys
import os
from modules.transformer import Transformer
from data.dataset import Dataset # Import your Dataset class

# --- Configuration ---
# Default weights path (can be overridden by command-line argument)
DEFAULT_WEIGHTS_PATH = 'experiments/en-de/04_12_2025/12_40_09/weights/99'
# Max length for generated translation to prevent infinite loops
MAX_LEN = 100

def translate_sentence(sentence: str,
                       model: Transformer,
                       dataset: Dataset, # Pass the dataset object
                       device: torch.device,
                       max_len: int = MAX_LEN):
    """
    Translates a single source sentence using the provided model and vocabularies.

    Args:
        sentence (str): The source sentence string.
        model (Transformer): The trained Transformer model.
        dataset (Dataset): The loaded Dataset object containing vocabs, tokenizers, etc.
        device (torch.device): The device to run inference on (CPU or CUDA).
        max_len (int): The maximum length allowed for the generated translation.

    Returns:
        str: The translated sentence string.
    """
    model.eval()  # Set the model to evaluation mode

    # 1. Preprocessing
    # Use the tokenizer from the Dataset object
    src_tokenizer = Dataset._spacy_pipelines[dataset.src_lang].tokenizer
    src_tokens = [dataset.sos_token] + [tok.text.lower() for tok in src_tokenizer(sentence)] + [dataset.eos_token]

    # Numericalize using source vocab
    src_indices = dataset.src_vocab.encode(src_tokens)

    # Convert to tensor and add batch dimension
    src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(device)

    # Create source mask (needed for the encoder)
    # Assuming your Transformer implementation handles mask creation internally,
    # or you might need to create it explicitly based on pad_index.
    # If model.forward handles it, you might not need this here.
    # src_mask = model.make_src_mask(src_tensor) # Example if needed

    # 2. Autoregressive Decoding
    # Start the target sequence with the <sos> token index
    trg_indices = [dataset.trg_vocab.sos_index]

    for i in range(max_len):
        # Convert current target sequence to tensor
        trg_tensor = torch.tensor(trg_indices, dtype=torch.long).unsqueeze(0).to(device)

        # Create target mask (needed for the decoder's self-attention)
        # Again, assuming Transformer handles this or you have a method.
        # trg_mask = model.make_trg_mask(trg_tensor) # Example if needed

        # Run model prediction (no gradients needed)
        with torch.no_grad():
            # The model needs src, target, and potentially masks
            # Adjust the call based on your model's forward signature
            output_logits = model(src_tensor, trg_tensor) # Assuming masks handled internally

        # Get the probabilities for the *last* predicted token in the sequence
        pred_token_logits = output_logits[:, -1, :] # Shape: (1, vocab_size)
        # Find the token index with the highest probability
        pred_token_idx = torch.argmax(pred_token_logits, dim=-1).item()

        # Append the predicted token index to our target sequence
        trg_indices.append(pred_token_idx)

        # If the predicted token is <eos>, stop generation
        if pred_token_idx == dataset.trg_vocab.eos_index:
            break

    # 3. Postprocessing
    # Decode the generated indices (excluding the initial <sos> token)
    trg_tokens = dataset.trg_vocab.decode(trg_indices[1:]) # Exclude SOS index

    # Filter out the <eos> token if it was generated
    trg_tokens = [token for token in trg_tokens if token != dataset.eos_token]

    return " ".join(trg_tokens)


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate sentences using a trained Transformer model.")
    parser.add_argument(
        "--weights",
        type=str,
        default=DEFAULT_WEIGHTS_PATH,
        help=f"Path to the trained model weights file (default: {DEFAULT_WEIGHTS_PATH})"
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=MAX_LEN,
        help=f"Maximum length for generated translations (default: {MAX_LEN})"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force use of CPU even if CUDA is available."
    )
    args = parser.parse_args()

    # --- Setup ---
    # Determine device
    if args.cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    print(f"Using device: {device}")

    # Load dataset (needed for vocabs, tokenizers, indices)
    print("Loading dataset (for vocabularies and tokenizers)...")
    # Use a small batch size as it's not relevant for interactive translation
    try:
        dataset = Dataset(config.LANGUAGE_PAIR, batch_size=1)
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        print("Please ensure the data is downloaded and accessible (e.g., run training once).", file=sys.stderr)
        sys.exit(1)
    print("Dataset loaded.")

    # Initialize model structure
    print("Initializing model structure...")
    model = Transformer(
        config.D_MODEL,
        len(dataset.src_vocab),
        len(dataset.trg_vocab),
        dataset.src_vocab.pad_index,
        dataset.trg_vocab.pad_index
    ).to(device) # Move model structure to device first

    # Load trained weights
    weights_path = args.weights
    print(f"Loading trained weights from: {weights_path}")
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found at {weights_path}", file=sys.stderr)
        sys.exit(1)
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device)) # Load weights onto the correct device
    except Exception as e:
        print(f"Error loading model weights: {e}", file=sys.stderr)
        sys.exit(1)
    print("Model weights loaded successfully.")

    # --- Interactive Translation Loop ---
    print("\nInteractive Translation Ready.")
    print(f"Enter a sentence in '{dataset.src_lang}' to translate into '{dataset.trg_lang}'.")
    print("Type 'quit' or 'exit' to stop.")

    while True:
        try:
            sentence = input(f"\n[{dataset.src_lang}]> ")
            if sentence.lower() in ['quit', 'exit']:
                break
            if not sentence:
                continue

            translation = translate_sentence(sentence, model, dataset, device, args.max_len)
            print(f"[{dataset.trg_lang}]> {translation}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred during translation: {e}")
            # Optional: print traceback for debugging
            # import traceback
            # traceback.print_exc()