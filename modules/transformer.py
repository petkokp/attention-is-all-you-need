import os
import json
import torch
from torch import nn
from .module import Module
from .embedding import Embedding
from .encoder import EncoderLayer
from .decoder import DecoderLayer

class Transformer(Module):
    def __init__(self,
                 d_model,
                 src_vocab_len,
                 trg_vocab_len,
                 src_pad_index,
                 trg_pad_index,
                 num_heads=8,
                 num_layers=6,
                 dropout_rate=0.1,
                 seed=20230815):
        super().__init__()

        self.src_pad_index = src_pad_index
        self.trg_pad_index = trg_pad_index

        # Manually seed to keep embeddings consistent across loads
        torch.manual_seed(seed)

        # Embeddings, pass in pad indices to prevent <pad> from contributing to gradient
        self.src_embedding = Embedding(d_model,
                                       src_vocab_len,
                                       src_pad_index,
                                       dropout_rate=dropout_rate)
        self.trg_embedding = Embedding(d_model,
                                       trg_vocab_len,
                                       trg_pad_index,
                                       dropout_rate=dropout_rate)

        # Encoder
        self.encoder_stack = nn.ModuleList(
            [EncoderLayer(d_model,
                          num_heads=num_heads,
                          dropout_rate=dropout_rate)
             for _ in range(num_layers)]
        )

        # Decoder
        self.decoder_stack = nn.ModuleList(
            [DecoderLayer(d_model,
                          num_heads=num_heads,
                          dropout_rate=dropout_rate)
             for _ in range(num_layers)]
        )

        # Final layer to project embedding to target vocab word probability distribution
        self.linear = nn.Linear(d_model, trg_vocab_len)

        # Re-seed afterward to allow shuffled data
        torch.seed()
        
    @property
    def num_layers(self):
        return len(self.encoder_stack)
    
    @property
    def num_heads(self):
        return self.encoder_stack[0].num_heads
    
    @property
    def dropout_rate(self):
        return self.encoder_stack[0].dropout_rate
    
    def forward(self, source, target):
        # Encoder stack
        enc_out = self.src_embedding(source)
        for layer in self.encoder_stack:
            enc_out = layer(enc_out)

        # Decoder stack
        dec_out = self.trg_embedding(target)
        for layer in self.decoder_stack:
            dec_out = layer(dec_out, enc_out)

        # Final linear layer to get word probabilities
        # DO NOT apply softmax here, as CrossEntropyLoss already does softmax!!!
        return self.linear(dec_out)
    
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        
        weights_path = os.path.join(path, "weights")
        torch.save(self.state_dict(), weights_path)
        
        config_path = os.path.join(path, "config.json")
        config = {
            "d_model": self.src_embedding.d_model,
            "src_vocab_len": self.src_embedding.vocab_len,
            "trg_vocab_len": self.trg_embedding.vocab_len,
            "src_pad_index": self.src_pad_index,
            "trg_pad_index": self.trg_pad_index,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "dropout_rate": self.dropout_rate,
        }
        with open(config_path, "w") as f:
            json.dump(config, f)
    
    @classmethod
    def load(cls, path: str):
        # Load weights
        weights_path = os.path.join(path, "weights")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        
        # Load config
        config_path = os.path.join(path, "config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        # Load config
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Create model
        model = cls(**config)
        
        # Load weights
        model.load_state_dict(torch.load(weights_path, map_location=model.device))
        model.eval()
        return model