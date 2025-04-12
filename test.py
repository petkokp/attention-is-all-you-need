import config
import torch
from data.dataset import Dataset
from modules.transformer import Transformer
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

WEIGHTS_PATH = 'experiments/en-de/04_12_2025/12_40_09/weights/99' # TODO - path add as an argument to the script, not hardcoding like that

# Load data
dataset = Dataset(config.LANGUAGE_PAIR, batch_size=config.BATCH_SIZE)

# Load saved model
model = Transformer(
    config.D_MODEL,
    len(dataset.src_vocab),
    len(dataset.trg_vocab),
    dataset.src_vocab.pad_index,
    dataset.trg_vocab.pad_index
)
model.load_state_dict(torch.load(WEIGHTS_PATH))

# Evaluate model on test set
with torch.no_grad():
    model.eval()
    valid_loss = 0
    num_batches = 0
    bleu_score = 0
    for data in tqdm(dataset.test_loader, desc='Test'):
        src = data['source'].to(model.device)
        trg = data['target'].to(model.device)

        predictions = model(src, trg[:, :-1])

        # Calculate BLEU score
        batch_size = predictions.size(0)
        batch_bleu = 0
        p_indices = torch.argmax(predictions, dim=-1)
        for i in range(batch_size):
            p_indices_list = p_indices[i].tolist()
            trg_indices_list = trg[i, 1:].tolist()
            
            p_tokens = dataset.trg_vocab.decode(p_indices_list)
            t_tokens = dataset.trg_vocab.decode(trg_indices_list)
                
            # Filter out special tokens
            p_tokens = list(filter(lambda x: '<' not in x, p_tokens))
            t_tokens = list(filter(lambda x: '<' not in x, t_tokens))

            if len(p_tokens) > 0 and len(t_tokens) > 0:
                batch_bleu += sentence_bleu([t_tokens], p_tokens)

        bleu_score += batch_bleu / batch_size
        num_batches += 1
        del src, trg

    print('\nBLEU score:', bleu_score / num_batches * 100)
