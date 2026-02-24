import n_gram
import torch
import torch.nn as nn
import torch.nn.functional as F


class lang_next_pred(nn.Module):
    def __init__(self, n_grams, embedding_dim, context_size) -> None:
        super(lang_next_pred, self).__init__()
        self.embeddings = nn.Embedding(n_grams, embedding_dim)

        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, n_grams)
