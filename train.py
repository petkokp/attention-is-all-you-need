import torch
import config
from dataset.dataset import Dataset
from modules.transformer import Transformer
from nltk.translate.bleu_score import sentence_bleu
from experiment import Experiment
from torch.optim.lr_scheduler import ReduceLROnPlateau

print('[~] Training')
print(f' ~  Using device: {Transformer.device}')

# Download and preprocess data
dataset = Dataset(config.LANGUAGE_PAIR, batch_size=config.BATCH_SIZE)

# Initialize model
model = Transformer(
    config.D_MODEL,
    len(dataset.src_tokenizer),
    len(dataset.trg_tokenizer),
    dataset.src_tokenizer.pad_index,
    dataset.trg_tokenizer.pad_index
)

print(f' ~  Parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

experiment = Experiment(model, category='-'.join(config.LANGUAGE_PAIR))

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.LEARNING_RATE,
    betas=(config.BETA1, config.BETA2),
    eps=config.EPS
)

scheduler = ReduceLROnPlateau(optimizer, factor=config.LR_REDUCTION_FACTOR)

loss_function = torch.nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    train_loss = 0
    num_batches = 0
    for data in dataset.train_loader:
        src = data['source'].to(model.device)
        trg = data['target'].to(model.device)

        # Given the sequence length N, transformer tries to predict the N+1th token.
        # Thus, transformer must take in trg[:-1] as input and predict trg[1:] as output.
        optimizer.zero_grad()
        predictions = model(src, trg[:, :-1])

        # For CrossEntropyLoss, need to reshape input from (batch, seq_len, vocab_len)
        # to (batch * seq_len, vocab_len). Also need to reshape ground truth from
        # (batch, seq_len) to just (batch * seq_len)
        loss = loss_function(
            predictions.reshape(-1, predictions.size(-1)),
            trg[:, 1:].reshape(-1)
        )
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        num_batches += 1
        del src, trg

    experiment.add_scalar('loss/train', epoch, train_loss / num_batches)
    validate(epoch)


# Evaluate against validation set and calculate BLEU
def validate(epoch):
    with torch.no_grad():
        model.eval()
        valid_loss = 0
        num_batches = 0
        bleu_score = 0
        for data in dataset.valid_loader:
            src = data['source'].to(model.device)
            trg = data['target'].to(model.device)

            predictions = model(src, trg[:, :-1])

            loss = loss_function(
                predictions.reshape(-1, predictions.size(-1)),
                trg[:, 1:].reshape(-1)
            )

            # Calculate BLEU score
            batch_size = predictions.size(0)
            batch_bleu = 0
            p_indices = torch.argmax(predictions, dim=-1)
            for i in range(batch_size):
                p_indices_list = p_indices[i].tolist()
                trg_indices_list = trg[i, 1:].tolist()
                
                p_tokens = dataset.trg_tokenizer.decode(p_indices_list)
                t_tokens = dataset.trg_tokenizer.decode(trg_indices_list)

                # Filter out special tokens
                p_tokens = list(filter(lambda x: '<' not in x, p_tokens))
                t_tokens = list(filter(lambda x: '<' not in x, t_tokens))

                if len(p_tokens) > 0 and len(t_tokens) > 0:
                    batch_bleu += sentence_bleu([t_tokens], p_tokens)
            bleu_score += batch_bleu / batch_size

            valid_loss += loss.item()
            scheduler.step(loss.item())
            num_batches += 1
            del src, trg

        experiment.add_scalar('loss/validation', epoch, valid_loss / num_batches)
        experiment.add_scalar('bleu', epoch, bleu_score / num_batches)
        experiment.add_scalar('lr', epoch, next(iter(optimizer.param_groups))['lr'])


experiment.loop(config.NUM_EPOCHS, train)

experiment.save_model("final")             # last checkpoint, optional name
experiment.save_tokenizers(
    "final",
    dataset.src_tokenizer,
    dataset.trg_tokenizer,
)