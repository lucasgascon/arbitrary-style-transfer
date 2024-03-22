import numpy as np
import matplotlib.pyplot as plt
from dataloader import stats
import torch

"""To transform images and visualize
"""


def unnormalize(images, means, stds):
    if len(images.shape) == 4:
        means = torch.tensor(means).reshape(1, 3, 1, 1)
        stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    else:
        means = torch.tensor(means).reshape(3, 1, 1)
        stds = torch.tensor(stds).reshape(3, 1, 1)

    return images * stds + means


def vizualize_preds(content_img, style_img, styled_img, normalize=False):

    if normalize:
        content_img = unnormalize(content_img.detach().cpu(), *stats)
        style_img = unnormalize(style_img.detach().cpu(), *stats)
        styled_img = unnormalize(styled_img.detach().cpu(), *stats)
        styled_img = styled_img.detach().cpu()
    else:
        content_img = content_img.detach().cpu()
        style_img = style_img.detach().cpu()
        styled_img = styled_img.detach().cpu()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(content_img.permute(1, 2, 0).clamp(0, 1))
    ax[0].set_title('Content Image')
    ax[0].axis('off')
    ax[1].imshow(style_img.permute(1, 2, 0).clamp(0, 1))
    ax[1].set_title('Style Image')
    ax[1].axis('off')
    ax[2].imshow(styled_img.permute(1, 2, 0).clamp(0, 1))
    ax[2].set_title('Model output')
    ax[2].axis('off')

    return fig, ax
