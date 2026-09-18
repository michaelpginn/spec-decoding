import torch
import torch.nn as nn


class medusa_heads(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, hidden_state):
        return self.linear(hidden_state)

class madusa(nn.Module):
    def __init__(self, base_model, num_heads=4) -> None:
        super().__init__()
        self.base_model = base_model

        try:
            for param in self.base_model.parameters():
                param.requires_grad = False
        except Exception as e:
            print(f"error raised: {e}")

        hidden_size = self.base_model.config.hidden_size
        vocab_size = self.base_model.config.vocab_size

        self.heads = nn.ModuleList(
            [
                self.medusa_heads(hidden_size, vocab_size)
                for _ in range(num_heads)
            ]
        )

    @property
    def config(self):
        return self.base_model.config

    @property
    def vocab_size(self):
        return self.vocab_size

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self, input_ids, attention_mask=None, past_key_values=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )

        last_hidden = outputs.hidden_states[-1]
        medusa_logits = [
            head(last_hidden) for head in self.heads
        ]

        return outputs.logits, medusa_logits, outputs.past_key_values
