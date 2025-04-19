import os
import atexit
import math
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from modules.transformer import Transformer
from data.tokenizer import Tokenizer


class Experiment:
    def __init__(self,
                 model: Transformer,
                 category: str | list[str] = None,
                 root: str = 'experiments'):
        self.model = model
        self.writer = SummaryWriter()

        # Set up folder structure
        now = datetime.now()
        dt_path = os.path.join(now.strftime('%m_%d_%Y'), now.strftime('%H_%M_%S'))
        self.path = os.path.join(
            root,
            os.path.join(category) if category else model.__class__.__name__,
            dt_path
        )
        if not os.path.isdir(self.path):
            os.makedirs(self.path)

    def loop(self, n, func, **kwargs):
        # Save model whenever training loop terminates, just in case of crash
        max_digits = int(math.log10(n - 1) + 1)
        atexit.register(lambda: self.save_model(str(step).zfill(max_digits)))

        # Run for N iterations
        kwargs['desc'] = kwargs.get('desc', func.__name__)
        for step in tqdm(range(n), **kwargs):
            func(step)

        self.writer.flush()

    def add_scalar(self, name, step, value):
        name, _ = os.path.splitext(name)

        # Write to Tensorboard
        self.writer.add_scalar(name, value, step)

        # Re-group path b/c name might be a nested path
        folder, file_name = os.path.split(os.path.join(self.path, 'scalars', f'{name}.csv'))
        folder = folder.lower()
        if not os.path.isdir(folder):
            os.makedirs(folder)

        # Append to .csv file
        with open(os.path.join(folder, file_name), 'a') as file:
            file.write(f'{step}, {value}\n')

    def save_model(self, file_name):
        self.model.save(file_name)

    def save_tokenizers(self, file_name, src_tok: Tokenizer, trg_tok: Tokenizer):
        tok_dir = os.path.join(self.path, file_name)
        os.makedirs(tok_dir, exist_ok=True)

        src_tok.save(os.path.join(tok_dir, f"src_{src_tok.language}.json"))
        trg_tok.save(os.path.join(tok_dir, f"trg_{trg_tok.language}.json"))