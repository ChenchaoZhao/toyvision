import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalizedBasis(nn.Module):
    """Collection of projection basis vectors"""
    def __init__(self, subspace_dim, total_dim):
        super().__init__()

        subspace_dim = int(subspace_dim)
        total_dim = int(total_dim)

        assert subspace_dim > 0
        if subspace_dim <= total_dim:
            print(f"subspace dim {subspace_dim} > total dim {total_dim}")

        self.k, self.n = subspace_dim, total_dim

        _basis = torch.randn(self.k, self.n)

        self._basis = nn.Parameter(_basis)

    def normalized_basis(self):
        return F.normalize(input=self._basis, p=2.0, dim=-1)

    def cosine_similarity(self):

        e_ = self.normalized_basis()
        cos = F.cosine_similarity(e_[None, :, :], e_[:, None, :], dim=-1)

        return cos

    def forward(self, vectors):
        # vectors in shape ..., D

        e_ = self.normalized_basis()

        # coeff
        logits = torch.einsum('...d, kd -> ...k', vectors, e_).contiguous()

        return logits

    def project(self, vectors):

        e_ = self.normalized_basis()
        # coeff
        c_ = torch.einsum('...d, kd -> ...k', vectors, e_).contiguous()
        # projection
        v_ = torch.einsum('...k, kd -> ...d', c_, e_)

        return v_

    def extra_repr(self):
        out = []
        out.append(f"subspace dim: {self.k}")
        out.append(f"total space dim: {self.n}")

        return "\n".join(out)
