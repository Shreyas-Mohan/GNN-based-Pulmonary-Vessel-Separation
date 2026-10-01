import torch
import torch.nn as nn
import numpy as np


class LightweightTransformer(nn.Module):
    def __init__(self, patch_size=16, embed_dim=64, num_heads=4, num_layers=2):
        """
        Lightweight Transformer processing 9 orthogonal 2D patches per center point.
        Produces a 9 * embed_dim dimensional visual embedding (e.g. 9 * 64 = 576).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = nn.Linear(patch_size * patch_size, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        Args:
            x (Tensor): Shape [B, 9, 1, patch_size, patch_size]
        Returns:
            Tensor: Shape [B, 9 * embed_dim]
        """
        B = x.size(0)
        x = x.view(B, 9, -1)
        x = self.patch_embed(x)
        out = self.transformer(x)
        return out.view(B, -1)


class PatchExtractor(nn.Module):
    def __init__(self, patch_size=16, embed_dim=64):
        """
        Extracts multi-planar orthogonal 2D patches (Axial, Coronal, Sagittal)
        centered at sparse skeleton nodes and passes them through a transformer encoder.
        """
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.transformer = LightweightTransformer(patch_size=patch_size, embed_dim=embed_dim)

    def extract_patch(self, volume, center):
        """
        Extract 9 multi-planar orthogonal patches from 3D volume at center (x, y, z).
        Offsets: -1, 0, +1 slice across Axial (z), Coronal (y), and Sagittal (x).
        """
        x, y, z = center
        X, Y, Z = volume.shape
        patches = []

        for axis in [0, 1, 2]:
            for offset in [-1, 0, 1]:
                if axis == 0:  # Axial plane (z-axis)
                    slice_idx = int(min(max(z + offset, 0), Z - 1))
                    plane = volume[:, :, slice_idx]
                    h_start = int(max(y - self.patch_size // 2, 0))
                    w_start = int(max(x - self.patch_size // 2, 0))
                    h_end = int(min(h_start + self.patch_size, Y))
                    w_end = int(min(w_start + self.patch_size, X))
                    patch = plane[w_start:w_end, h_start:h_end]
                elif axis == 1:  # Coronal plane (y-axis)
                    slice_idx = int(min(max(y + offset, 0), Y - 1))
                    plane = volume[:, slice_idx, :]
                    h_start = int(max(z - self.patch_size // 2, 0))
                    w_start = int(max(x - self.patch_size // 2, 0))
                    h_end = int(min(h_start + self.patch_size, Z))
                    w_end = int(min(w_start + self.patch_size, X))
                    patch = plane[w_start:w_end, h_start:h_end]
                else:  # Sagittal plane (x-axis)
                    slice_idx = int(min(max(x + offset, 0), X - 1))
                    plane = volume[slice_idx, :, :]
                    h_start = int(max(z - self.patch_size // 2, 0))
                    w_start = int(max(y - self.patch_size // 2, 0))
                    h_end = int(min(h_start + self.patch_size, Z))
                    w_end = int(min(w_start + self.patch_size, Y))
                    patch = plane[w_start:w_end, h_start:h_end]

                if patch.shape != (self.patch_size, self.patch_size):
                    patch = np.pad(
                        patch,
                        ((0, self.patch_size - patch.shape[0]), (0, self.patch_size - patch.shape[1])),
                        mode="constant",
                    )

                patches.append(patch)

        patches = torch.stack([torch.from_numpy(p).unsqueeze(0) for p in patches])
        return patches.float()
