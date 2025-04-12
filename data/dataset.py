import spacy
import torch
from torch.utils.data import Dataset as TorchDataset # Rename to avoid conflict
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter, OrderedDict
import os
import requests
import tarfile
import logging
from tqdm import tqdm
from .vocab import Vocab

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_URLS = {
    'train': 'https://raw.githubusercontent.com/tanjeffreyz/pytorch-multi30k/main/training.tar.gz',
    'valid': 'https://raw.githubusercontent.com/tanjeffreyz/pytorch-multi30k/main/validation.tar.gz',
    'test': 'https://raw.githubusercontent.com/tanjeffreyz/pytorch-multi30k/main/mmt16_task1_test.tar.gz'
}
DATA_ROOT = 'data'
EXTRACTED_PATHS = {
    'train': os.path.join(DATA_ROOT, 'multi30k_train'),
    'valid': os.path.join(DATA_ROOT, 'multi30k_valid'),
    'test': os.path.join(DATA_ROOT, 'multi30k_test')
}


class Dataset:
    # Load spacy models only once
    try:
        _spacy_pipelines = {
            'en': spacy.load('en_core_web_sm'),
            'de': spacy.load('de_core_news_sm')
        }
    except IOError:
        print("Spacy models not found. Please run:")
        print("python -m spacy download en_core_web_sm")
        print("python -m spacy download de_core_news_sm")
        exit()

    class _TorchDataset(TorchDataset):
        def __init__(self, src_sentences, trg_sentences, parent_dataset):
            self.src_sentences = src_sentences
            self.trg_sentences = trg_sentences
            self.parent = parent_dataset # Reference to the outer Dataset instance

        def __len__(self):
            return len(self.src_sentences)

        def __getitem__(self, idx):
            src_text = self.src_sentences[idx]
            trg_text = self.trg_sentences[idx]

            # Use the outer class's tokenize and encode methods/vocabs
            src_tokens = self.parent.tokenize_sentence(src_text, self.parent.src_lang)
            trg_tokens = self.parent.tokenize_sentence(trg_text, self.parent.trg_lang)

            src_indices = self.parent.src_vocab.encode(src_tokens)
            trg_indices = self.parent.trg_vocab.encode(trg_tokens)

            # Return tensors for the collate function
            return torch.tensor(src_indices, dtype=torch.long), torch.tensor(trg_indices, dtype=torch.long)

    def __init__(self,
                 language_pair=('de', 'en'), # Source, Target
                 sos_token='<sos>',
                 eos_token='<eos>',
                 unk_token='<unk>',
                 pad_token='<pad>',
                 batch_size=64,
                 data_root=DATA_ROOT,
                 min_vocab_freq=2): # Min frequency for vocab

        self.src_lang, self.trg_lang = language_pair
        assert self.src_lang in Dataset._spacy_pipelines, f'Unsupported source language: {self.src_lang}'
        assert self.trg_lang in Dataset._spacy_pipelines, f'Unsupported target language: {self.trg_lang}'

        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.special_tokens = [self.unk_token, self.pad_token, self.sos_token, self.eos_token]

        self.batch_size = batch_size
        self.data_root = data_root
        self.min_vocab_freq = min_vocab_freq

        self.src_tokenizer = Dataset._spacy_pipelines[self.src_lang].tokenizer
        self.trg_tokenizer = Dataset._spacy_pipelines[self.trg_lang].tokenizer

        self._download_and_extract_all()

        try:
            train_src_file = self._find_data_file(EXTRACTED_PATHS['train'], 'train', self.src_lang)
            train_trg_file = self._find_data_file(EXTRACTED_PATHS['train'], 'train', self.trg_lang)
            valid_src_file = self._find_data_file(EXTRACTED_PATHS['valid'], 'val', self.src_lang) # Often 'val'
            valid_trg_file = self._find_data_file(EXTRACTED_PATHS['valid'], 'val', self.trg_lang)
            test_src_file = self._find_data_file(EXTRACTED_PATHS['test'], 'test_2016_flickr', self.src_lang) # Often specific name
            test_trg_file = self._find_data_file(EXTRACTED_PATHS['test'], 'test_2016_flickr', self.trg_lang)

            train_src_raw = self._load_sentences(train_src_file)
            train_trg_raw = self._load_sentences(train_trg_file)
            valid_src_raw = self._load_sentences(valid_src_file)
            valid_trg_raw = self._load_sentences(valid_trg_file)
            test_src_raw = self._load_sentences(test_src_file)
            test_trg_raw = self._load_sentences(test_trg_file)
        except FileNotFoundError as e:
             logging.error(f"Error finding data files: {e}")
             logging.error("Please check the filenames inside the extracted folders ('multi30k_train', etc.)")
             logging.error(f"Searched in: {EXTRACTED_PATHS}")
             raise

        logging.info("Building source vocabulary...")
        self.src_vocab = self._build_vocab(train_src_raw, self.src_tokenizer)
        logging.info("Building target vocabulary...")
        self.trg_vocab = self._build_vocab(train_trg_raw, self.trg_tokenizer)
        logging.info(f"Source vocab size: {len(self.src_vocab)}")
        logging.info(f"Target vocab size: {len(self.trg_vocab)}")

        self.train_torch_dataset = self._TorchDataset(train_src_raw, train_trg_raw, self)
        self.valid_torch_dataset = self._TorchDataset(valid_src_raw, valid_trg_raw, self)
        self.test_torch_dataset = self._TorchDataset(test_src_raw, test_trg_raw, self)

        self.train_loader = DataLoader(self.train_torch_dataset, batch_size=self.batch_size,
                                       shuffle=True, collate_fn=self._collate_fn)
        self.valid_loader = DataLoader(self.valid_torch_dataset, batch_size=self.batch_size,
                                       shuffle=False, collate_fn=self._collate_fn)
        self.test_loader = DataLoader(self.test_torch_dataset, batch_size=self.batch_size,
                                      shuffle=False, collate_fn=self._collate_fn)

    def _download_and_extract(self, url, dest_path, extract_path):
        if os.path.exists(extract_path) and os.listdir(extract_path):
             logging.debug(f"Data already extracted at {extract_path}. Skipping download.")
             all_files_exist = True
             try:
                 if 'train' in extract_path:
                      self._find_data_file(extract_path, 'train', self.src_lang)
                      self._find_data_file(extract_path, 'train', self.trg_lang)
                 elif 'val' in extract_path:
                      self._find_data_file(extract_path, 'val', self.src_lang)
                      self._find_data_file(extract_path, 'val', self.trg_lang)
             except FileNotFoundError:
                  all_files_exist = False
                  logging.warning(f"Extraction directory {extract_path} exists but seems incomplete. Re-downloading.")


             if all_files_exist:
                  logging.info(f"Data already extracted at {extract_path}. Skipping download.")
                  return # Skip download if likely complete

        os.makedirs(dest_path, exist_ok=True)
        filename = url.split('/')[-1]
        filepath = os.path.join(dest_path, filename)

        logging.info(f"Downloading {url} to {filepath}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status() # Raise an exception for bad status codes
            total_size = int(response.headers.get('content-length', 0))

            with open(filepath, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)

            logging.info(f"Extracting {filepath} to {extract_path}...")
            os.makedirs(extract_path, exist_ok=True)
            with tarfile.open(filepath, 'r:gz') as tar:
                 # Extract safely, avoid paths starting with '/' or '..'
                 for member in tar.getmembers():
                     member_path = os.path.join(extract_path, member.name)
                     # Check for directory traversal attempts
                     if not os.path.abspath(member_path).startswith(os.path.abspath(extract_path)):
                          raise IOError(f"Attempted Path Traversal in Tar File: {member.name}")
                     tar.extract(member, path=extract_path, set_attrs=False) # set_attrs=False can help on Windows
            logging.info(f"Extraction complete.")
            # Optionally remove the downloaded tar.gz file
            # os.remove(filepath)

        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading {url}: {e}")
            raise
        except tarfile.TarError as e:
            logging.error(f"Error extracting {filepath}: {e}")
            if os.path.exists(extract_path):
                 import shutil
                 shutil.rmtree(extract_path)
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred during download/extract: {e}")
            if os.path.exists(filepath) and 'tar' not in str(e).lower(): # Don't delete if it was a tar error
                try:
                    os.remove(filepath)
                except OSError as rm_err:
                     logging.warning(f"Could not remove temporary file {filepath}: {rm_err}")
            if os.path.exists(extract_path):
                 try:
                     import shutil
                     shutil.rmtree(extract_path)
                 except OSError as rmtree_err:
                      logging.warning(f"Could not remove extraction dir {extract_path}: {rmtree_err}")
            raise


    def _download_and_extract_all(self):
        for split, url in DATA_URLS.items():
            extract_path = EXTRACTED_PATHS[split]
            # Pass the main data_root for storing the tarball temporarily
            self._download_and_extract(url, self.data_root, extract_path)

    def _find_data_file(self, extract_dir, base_name, lang_ext):
        variations = [
            f"{base_name}.{lang_ext}",
            f"{base_name.replace('val', 'validation')}.{lang_ext}",
            f"{base_name.replace('train', 'training')}.{lang_ext}",
            f"{base_name.replace('test_2016_flickr', 'test')}.{lang_ext}",
        ]
        for name in variations:
            filepath = os.path.join(extract_dir, name)
            if os.path.exists(filepath):
                logging.debug(f"Found data file: {filepath}")
                return filepath

        root_filepath = os.path.join(extract_dir, f"{base_name}.{lang_ext}")
        if os.path.exists(root_filepath):
             logging.debug(f"Found data file in root: {root_filepath}")
             return root_filepath

        raise FileNotFoundError(f"Could not find {lang_ext} data file with base '{base_name}' in {extract_dir} or its variations.")


    def _load_sentences(self, filepath):
        logging.info(f"Loading sentences from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
        logging.info(f"Loaded {len(sentences)} sentences.")
        return sentences

    def _build_vocab(self, data_iter, tokenizer):
        counter = Counter()
        for sentence in tqdm(data_iter, desc="Building vocab"):
            counter.update(token.text.lower() for token in tokenizer(sentence))
        return Vocab(counter, specials=self.special_tokens, min_freq=self.min_vocab_freq)

    def tokenize_sentence(self, sentence, lang):
        if lang == self.src_lang:
            tokenizer = self.src_tokenizer
        elif lang == self.trg_lang:
            tokenizer = self.trg_tokenizer
        else:
            raise ValueError(f"Unknown language for tokenization: {lang}")

        tokens = [self.sos_token] + [tok.text.lower() for tok in tokenizer(sentence)] + [self.eos_token]
        return tokens

    def _collate_fn(self, batch):
        src_batch, trg_batch = zip(*batch)

        # Pad sequences in the batch
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=self.src_vocab.pad_index)
        trg_padded = pad_sequence(trg_batch, batch_first=True, padding_value=self.trg_vocab.pad_index)

        return {'source': src_padded, 'target': trg_padded}