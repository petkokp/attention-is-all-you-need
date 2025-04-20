from typing import Optional, cast
import torch
from torch.utils.data import Dataset as TorchDataset # Rename to avoid conflict
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import requests
import tarfile
import logging
from tqdm import tqdm
from .tokenizer import Tokenizer, SpecialTokens

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

    class _TorchDataset(TorchDataset):
        def __init__(self, src_sentences, trg_sentences, parent_dataset: 'Dataset'):
            self.src_sentences = src_sentences
            self.trg_sentences = trg_sentences
            self.parent = parent_dataset # Reference to the outer Dataset instance

        def __len__(self):
            return len(self.src_sentences)

        def __getitem__(self, idx):
            src_text = self.src_sentences[idx]
            trg_text = self.trg_sentences[idx]

            # Use the outer class's tokenize and encode methods/vocabs
            src_tokens = self.parent.src_tokenizer.encode(src_text)
            trg_tokens = self.parent.trg_tokenizer.encode(trg_text)

            # Return tensors for the collate function
            return torch.tensor(src_tokens, dtype=torch.long), torch.tensor(trg_tokens, dtype=torch.long)

    def __init__(self,
                 language_pair=('de', 'en'), # Source, Target
                 batch_size=64,
                 data_root=DATA_ROOT,
                 min_vocab_freq=2,
                 specials: Optional[SpecialTokens]=None,
                 ): # Min frequency for vocab
        
        self.specials = specials or Tokenizer._default_specials()

        self.src_lang, self.trg_lang = language_pair

        self.batch_size = batch_size
        self.data_root = data_root
        self.min_vocab_freq = min_vocab_freq

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
 
        logging.info("Building source tokenizer...")
        self.src_tokenizer = Tokenizer.build(train_src_raw, self.src_lang, self.specials, self.min_vocab_freq)
        logging.info("Building target vocabulary...")
        self.trg_tokenizer = Tokenizer.build(train_trg_raw, self.trg_lang, self.specials, self.min_vocab_freq)
        logging.info(f"Source vocab size: {len(self.src_tokenizer)}")
        logging.info(f"Target vocab size: {len(self.trg_tokenizer)}")

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

    def _collate_fn(self, batch):
        src_batch, trg_batch = cast(list[torch.Tensor], zip(*batch))

        # Pad sequences in the batch
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=self.src_tokenizer.pad_index)
        trg_padded = pad_sequence(trg_batch, batch_first=True, padding_value=self.trg_tokenizer.pad_index)

        return {'source': src_padded, 'target': trg_padded}