from collections import Counter, OrderedDict

class Vocab:
    def __init__(self, counter, specials=['<unk>', '<pad>', '<sos>', '<eos>'], min_freq=1):
        self.counter = counter
        self.specials = specials
        self.min_freq = min_freq
        self._build_vocab()

    def _build_vocab(self):
        self.itos = list(self.specials)
        # Filter tokens by frequency
        for token, freq in self.counter.items():
            if freq >= self.min_freq:
                self.itos.append(token)

        # Create stoi mapping
        self.stoi = OrderedDict([(token, i) for i, token in enumerate(self.itos)])
        self.unk_index = self.stoi.get('<unk>', 0) # Default to 0 if <unk> not in specials
        self.pad_index = self.stoi.get('<pad>', 1) # Default to 1 if <pad> not in specials
        self.sos_index = self.stoi.get('<sos>', 2)
        self.eos_index = self.stoi.get('<eos>', 3)


    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi.get(token, self.unk_index) for token in tokens]

    def decode(self, indices):
        return [self.itos[index] for index in indices if index != self.pad_index] # Exclude pad