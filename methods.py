from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as T


# Image Augmentation
image_augmentator = T.Compose([T.RandomAffine(degrees=170, translate=(0, 0.2)),
                               T.RandomPerspective(distortion_scale=0.2, p=0.5)])


def stack_image_with_spatialgrid(images: torch.Tensor) -> torch.Tensor:
    us = torch.arange(1, images.shape[-2] + 1, 1, dtype=torch.float32, device=images.device)
    vs = torch.arange(1, images.shape[-1] + 1, 1, dtype=torch.float32, device=images.device)
    grid = torch.meshgrid(us, vs, indexing='ij')
    spatial_grid = torch.stack(grid)

    tiled_spatial_grid = spatial_grid.unsqueeze(dim=0).tile(images.shape[0], 1, 1, 1)

    return torch.cat([images, tiled_spatial_grid], dim=1)


def destack_image_with_spatialgrid(augmented_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    image = augmented_image[:, :3, :, :]
    spatial_grid = torch.round(augmented_image[:, 3:, :])

    return image, spatial_grid


def augment_images_and_map_correspondence(images: torch.Tensor,
                                          n_correspondence: int) -> torch.Tensor:

    augmented_image_a = image_augmentator(stack_image_with_spatialgrid(images))
    augmented_image_b = image_augmentator(stack_image_with_spatialgrid(images))

    augmented_images_a, grids_a = destack_image_with_spatialgrid(augmented_image_a)
    augmented_images_b, grids_b = destack_image_with_spatialgrid(augmented_image_b)

    matches_a: List[torch.Tensor] = []
    matches_b: List[torch.Tensor] = []

    # Compute correspondence from the spatial grid
    for grid_a, grid_b in zip(grids_a, grids_b):

        valid_pixels_a = torch.where(grid_a.sum(dim=0) >= 1.0)
        us, vs = valid_pixels_a[0], valid_pixels_a[1]
        trimming_indices = torch.linspace(0, us.shape[0] - 1, steps=10 * n_correspondence)
        us, vs = us[trimming_indices.long()], vs[trimming_indices.long()]
        valid_pixels_a = torch.vstack([us, vs]).permute(1, 0).type(torch.float32)

        valid_grids_a = grid_a[:, us.long(), vs.long()].permute(1, 0)
        tiled_valid_grids_a = valid_grids_a.view(valid_grids_a.shape[0], valid_grids_a.shape[1], 1, 1)

        spatial_grid_distances = torch.linalg.norm(grid_b - tiled_valid_grids_a, dim=1)

        match_indices_a, ubs, vbs = torch.where(spatial_grid_distances == 0.0)

        mutual_match_a = valid_pixels_a[match_indices_a.long()]
        mutual_match_b = torch.vstack([ubs, vbs]).permute(1, 0)

        trimming_indices = torch.linspace(0, mutual_match_a.shape[0] - 1, steps=n_correspondence)
        trimming_indices = trimming_indices.type(torch.int64)

        matches_a.append(mutual_match_a[trimming_indices])
        matches_b.append(mutual_match_b[trimming_indices])

    return augmented_images_a, torch.stack(matches_a), augmented_images_b, torch.stack(matches_b)
