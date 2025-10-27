import torch

class TwoHotMixUp:
    """
    Implementation of MixUp that returns both targets as class indices instead of
    class probabilities, matching big_vision implementation.
    """
    def __init__(self, alpha=0.2):
        self._dist = None
        if alpha > 0:
            self._dist = torch.distributions.Beta(alpha, alpha)

    def __call__(self, images, labels):
        if self._dist is None:
            return images, labels

        batch_size = images.size(0)
        indices = torch.randperm(batch_size, device=images.device)
        lam = self._dist.sample().to(images.device)

        mixed_images = lam * images + (1 - lam) * images[indices]
        targets1 = labels
        targets2 = labels[indices]

        return mixed_images, lam, targets1, targets2