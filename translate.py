import argparse
from typing import List
from dataset.tokenizer import Tokenizer
from modules.transformer import Transformer
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate text with a saved Transformer checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model directory (containing config.json & weights)",
    )
    parser.add_argument("--text", type=str, required=True, help="Sentence to translate")
    parser.add_argument(
        "--max_len",
        type=int,
        default=100,
        help="Maximum generated length in tokens (including <sos>/<eos>)",
    )
    return parser.parse_args()


@torch.no_grad()
def greedy_translate(
    model: Transformer,
    src_tok: Tokenizer,
    trg_tok: Tokenizer,
    sentence: str,
    max_len: int = 100,
) -> str:
    """Greedy autoregressive decoding (no beam-search)."""

    model.eval()
    device = model.device

    # Encode source sentence
    src_indices: List[int] = src_tok.encode(sentence)
    src_tensor = torch.tensor(src_indices, dtype=torch.long, device=device).unsqueeze(0)

    # Start with <sos>
    trg_indices: List[int] = [trg_tok.sos_index]

    for _ in range(max_len):
        trg_tensor = torch.tensor(
            trg_indices, dtype=torch.long, device=device
        ).unsqueeze(0)

        # Shape: (1, len_trg, vocab_len)
        logits = model(src_tensor, trg_tensor)
        next_token = int(torch.argmax(logits[0, -1, :]))
        trg_indices.append(next_token)

        if next_token == trg_tok.eos_index:
            break

    # Strip <sos>/<eos>
    trimmed = (
        trg_indices[1:-1] if trg_indices[-1] == trg_tok.eos_index else trg_indices[1:]
    )
    tokens = trg_tok.decode(trimmed, skip_special=True)
    return " ".join(tokens)


def main() -> None:
    args = parse_args()

    # Load tokenizers
    src_tok, trg_tok = Tokenizer.load_from_model(args.model)

    # Load model
    model = Transformer.load(args.model)

    translation = greedy_translate(
        model, src_tok, trg_tok, args.text, max_len=args.max_len
    )
    print(translation)


if __name__ == "__main__":
    main()
