import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4, scale=1.0, lora_only=False, bias=True):
        super().__init__()
        self.lora_only = lora_only
        self.scale = scale
        self.rank = rank

        if not lora_only:
            self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
            nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        else:
            self.weight = None

        self.lora_A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(out_dim, rank) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None

    def forward(self, x):
        delta = self.lora_B @ self.lora_A  # (out_dim x in_dim)
        w_eff = delta * self.scale
        if not self.lora_only:
            w_eff = w_eff + self.weight
        out = x @ w_eff.T
        if self.bias is not None:
            out += self.bias
        return out
