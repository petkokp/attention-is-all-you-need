from typing import Optional, cast, Tuple, List
import gzip

import torch
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import requests
import tarfile
import logging
from tqdm import tqdm
from .tokenizer import Tokenizer, SpecialTokens

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SOURCE_URL = "https://object.pouta.csc.fi/Tatoeba-Challenge-v2023-09-26/"

class TranslationDataset(Dataset):
    """
    Handles downloading, extracting, loading, and tokenizing Tatoeba Challenge data for any language pair.
    """
    # ----- Dataset interface -----
    def __init__(
        self,
        src: str,
        trg: str,
        data_root: str = 'data',
        min_vocab_freq: int = 2,
        specials: Optional[SpecialTokens] = None
    ):
        self.src_lang = src
        self.trg_lang = trg
        self.data_root = data_root
        self.min_vocab_freq = min_vocab_freq
        self.specials = specials or Tokenizer._default_specials()

        # Directory for raw files
        self.raw_folder = os.path.join(self.data_root, f"{src}-{trg}-data")
        os.makedirs(self.raw_folder, exist_ok=True)

        # Download and extract the .tar archive
        self.extract_dir = self._download_data()

        # Load source and target sentences
        self.src_sentences, self.trg_sentences = self._load_sentences(self.extract_dir)

        # Build tokenizers on the loaded data
        self._build_tokenizers()

    def __len__(self) -> int:
        return len(self.src_sentences)
    
    def __getitem__(
        self,
        idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src_text = self.src_sentences[idx]
        trg_text = self.trg_sentences[idx]
        src_ids = self.src_tokenizer.encode(src_text)
        trg_ids = self.trg_tokenizer.encode(trg_text)
        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(trg_ids, dtype=torch.long)
        )
    
    # ----- DataLoader -----
    def get_dataloader(self, batch_size=32, shuffle=True):
        return DataLoader(self,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        collate_fn=self.collate_fn)

    
    # ----- Other utilities -----
    def _download_data(self) -> str:
        """
        Downloads the .tar for the language pair, extracts its contents, and
        returns the path to the folder containing train.src.gz & train.trg.gz.
        """
        tar_filename = f"{self.src_lang}-{self.trg_lang}.tar"
        tar_path = os.path.join(self.raw_folder, tar_filename)

        # Download if needed
        if not os.path.exists(tar_path):
            url = SOURCE_URL + tar_filename
            logging.info(f"Downloading {url} to {tar_path}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total = int(response.headers.get('content-length', 0))
            with open(tar_path, 'wb') as f, tqdm(
                desc=tar_filename,
                total=total,
                unit='iB', unit_scale=True
            ) as bar:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
                    bar.update(len(chunk))
        else:
            logging.info(f"Archive already exists at {tar_path}. Skipping download.")

        # Extract
        logging.info(f"Extracting {tar_path} to {self.raw_folder}...")
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(path=self.raw_folder)

        extract_dir = os.path.join(self.raw_folder, f"data/release/v2023-09-26/{self.src_lang}-{self.trg_lang}")

        if not os.path.exists(extract_dir+"/train.src.gz"):
            raise FileNotFoundError(f"Extracted files not found in {extract_dir}. Please check the tar file.")

        return extract_dir
    
    def _load_sentences(self, extract_dir: str) -> Tuple[List[str], List[str]]:
        """
        Reads train.src.gz, train.trg.gz, test.src, and test.trg from the extract_dir,
        concatenates them, and returns two lists: source sentences and target sentences.
        """
        src_sents: List[str] = []
        trg_sents: List[str] = []

        # Load training data (gzipped)
        train_src = os.path.join(extract_dir, 'train.src.gz')
        train_trg = os.path.join(extract_dir, 'train.trg.gz')
        logging.info("Loading training data...")
        with gzip.open(train_src, 'rt', encoding='utf-8') as f_src, \
             gzip.open(train_trg, 'rt', encoding='utf-8') as f_trg:
            for src_line, trg_line in zip(f_src, f_trg):
                src_sents.append(src_line.strip())
                trg_sents.append(trg_line.strip())
        logging.info(f"Loaded {len(src_sents)} training pairs.")

        # Load test data (plain text)
        test_src = os.path.join(extract_dir, 'test.src')
        test_trg = os.path.join(extract_dir, 'test.trg')
        if os.path.exists(test_src) and os.path.exists(test_trg):
            logging.info("Loading test data...")
            with open(test_src, 'r', encoding='utf-8') as f_src, \
                 open(test_trg, 'r', encoding='utf-8') as f_trg:
                for src_line, trg_line in zip(f_src, f_trg):
                    src_sents.append(src_line.strip())
                    trg_sents.append(trg_line.strip())
            logging.info(f"Loaded {len(src_sents)} total pairs (including test). ")
        else:
            logging.warning("Test files not found; no test data loaded.")

        return src_sents, trg_sents
    
    def _build_tokenizers(self) -> None:
        """
        Builds Tokenizer instances for source and target languages on all loaded sentences.
        """
        logging.info(f"Building tokenizer for {self.src_lang}...")
        self.src_tokenizer = Tokenizer.build(
            self.src_sentences,
            self.src_lang,
            self.specials,
            self.min_vocab_freq
        )
        logging.info(f"Building tokenizer for {self.trg_lang}...")
        self.trg_tokenizer = Tokenizer.build(
            self.trg_sentences,
            self.trg_lang,
            self.specials,
            self.min_vocab_freq
        )
        logging.info(f"Source vocab size: {len(self.src_tokenizer)}")
        logging.info(f"Target vocab size: {len(self.trg_tokenizer)}")

    def collate_fn(self, batch):
        """
        Pads a batch of (src_tensor, trg_tensor) pairs into a dict of tensors.
        """
        src_batch, trg_batch = cast(List[torch.Tensor], zip(*batch))
        src_padded = pad_sequence(
            src_batch,
            batch_first=True,
            padding_value=self.src_tokenizer.pad_index
        )
        trg_padded = pad_sequence(
            trg_batch,
            batch_first=True,
            padding_value=self.trg_tokenizer.pad_index
        )
        return {'source': src_padded, 'target': trg_padded}
    
