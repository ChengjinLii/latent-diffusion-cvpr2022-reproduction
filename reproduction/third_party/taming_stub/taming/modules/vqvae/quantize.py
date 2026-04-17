import torch
from torch import nn


class VectorQuantizer2(nn.Module):
    """Small compatibility implementation for importing CompVis LDM code.

    The reproduction experiment in this project uses KL autoencoders only, so
    this class is not exercised in the reported runs. It is present because
    ldm.models.autoencoder imports the VQ quantizer at module import time.
    """

    def __init__(self, n_e, e_dim, beta=0.25, remap=None, sane_index_shape=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.remap = remap
        self.sane_index_shape = sane_index_shape
        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

    def forward(self, z):
        z_perm = z.permute(0, 2, 3, 1).contiguous()
        flat = z_perm.view(-1, self.e_dim)
        distances = (
            torch.sum(flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat, self.embedding.weight.t())
        )
        indices = torch.argmin(distances, dim=1)
        quant = self.embedding(indices).view_as(z_perm)
        loss = self.beta * torch.mean((quant.detach() - z_perm) ** 2) + torch.mean(
            (quant - z_perm.detach()) ** 2
        )
        quant = z_perm + (quant - z_perm).detach()
        quant = quant.permute(0, 3, 1, 2).contiguous()
        if self.sane_index_shape:
            indices = indices.view(z.shape[0], z.shape[2], z.shape[3])
        return quant, loss, (None, None, indices)

    def embed_code(self, code_b):
        quant = self.embedding(code_b)
        if quant.ndim == 4:
            quant = quant.permute(0, 3, 1, 2).contiguous()
        return quant

