import torch
import torchvision.transforms.functional as F
from torchvision.transforms import v2
import math
import random
from typing import List, Optional, Any
from collections import defaultdict

class BigVisionRandAugment(v2.RandAugment):
    """
    A clean implementation of RandAugment matching big_vision's behavior.

    This implementation:
    1. Uses the same transforms as big_vision (adds Invert, SolarizeAdd, Cutout)
    2. Matches the magnitude scaling to big_vision
    3. Uses the same fill value (128,128,128) as big_vision
    4. Fixes the Contrast transform to match the corrected behavior

    Args:
        num_ops (int): Number of augmentation transformations to apply sequentially.
        magnitude (int): Magnitude for all the transformations.
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum.
        fill (List[int] or int): Pixel fill value for the area outside the transformed
            image. Default is [128, 128, 128] to match big_vision.
    """

    def __init__(self,
                 num_ops: int = 2,
                 magnitude: int = 10,
                 num_magnitude_bins: int = 31,
                 interpolation: v2.InterpolationMode = v2.InterpolationMode.BILINEAR,
                 fill: Optional[List[int]] = None):
        if fill is None:
            fill = [128, 128, 128]  # Match big_vision's default

        # Ensure fill is never a defaultdict or other unexpected type
        if isinstance(fill, defaultdict) or not isinstance(fill, (list, int)) and fill is not None:
            fill = [128, 128, 128]  # Force safe value

        super().__init__(num_ops=num_ops,
                         magnitude=magnitude,
                         num_magnitude_bins=num_magnitude_bins,
                         interpolation=interpolation,
                         fill=fill)

        # Override the transforms list to match big_vision
        self._transforms = [
            'Identity', 'AutoContrast', 'Equalize', 'Invert', 'Rotate',
            'Posterize', 'Solarize', 'Color', 'Contrast', 'Brightness',
            'Sharpness', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY',
            'SolarizeAdd', 'Cutout'
        ]

    def _get_transforms(self) -> List[str]:
        """Override to use our custom transform list"""
        return self._transforms

    def _apply_op(self, img: torch.Tensor, op_name: str, magnitude: float) -> torch.Tensor:
        """Apply the selected operation with the given magnitude to the image."""
        # if op_name == 'SolarizeAdd':
        #     return self._solarize_add(img, addition=int(magnitude), threshold=128)
        # elif op_name == 'Cutout':
        #     return self._cutout(img, pad_size=int(magnitude), replace=[128, 128, 128])
        # else:
        #     # Scale the magnitude based on the number of bins
        magnitude_scale = 1.0  # Default scale factor
        if hasattr(self, 'num_magnitude_bins') and self.num_magnitude_bins > 0:
            magnitude_scale = magnitude / self.num_magnitude_bins

        return self._apply_image_transform(
            img,
            op_name,
            magnitude_scale * self.magnitude,
            self.interpolation,
            self.fill
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (Tensor): Image to be transformed.

        Returns:
            Tensor: Transformed image.
        """
        fill = self.fill
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_dimensions(img)[0]
            elif fill is not None:
                fill = [float(f) for f in fill]

        # Select operations at random
        ops = random.choices(self._get_transforms(), k=self.num_ops)

        # Apply operations sequentially
        for op_name in ops:
            magnitude = float(torch.empty(1).uniform_(0, self.magnitude).item())
            img = self._apply_op(img, op_name, magnitude)

        return img

    def _apply_image_transform(self,
                              img: torch.Tensor,
                              transform_id: str,
                              magnitude: float,
                              interpolation: v2.InterpolationMode,
                              fill: Any) -> torch.Tensor:
        """Apply the specified transform to the image with big_vision-compatible behavior."""

        # Ensure fill is a proper type (list or int)
        if isinstance(fill, defaultdict) or not (isinstance(fill, (list, int)) or fill is None):
            fill = [128, 128, 128]  # Use default safe value

        if transform_id == "Identity":
            return img
        elif transform_id == "ShearX":
            # Use arctan like big_vision
            return F.affine(
                img,
                angle=0.0,
                translate=[0, 0],
                scale=1.0,
                shear=[math.degrees(math.atan(magnitude)), 0.0],
                interpolation=interpolation,
                fill=fill,
                center=[0, 0],
            )
        elif transform_id == "ShearY":
            return F.affine(
                img,
                angle=0.0,
                translate=[0, 0],
                scale=1.0,
                shear=[0.0, math.degrees(math.atan(magnitude))],
                interpolation=interpolation,
                fill=fill,
                center=[0, 0],
            )
        elif transform_id == "TranslateX":
            return F.affine(
                img,
                angle=0.0,
                translate=[int(magnitude), 0],
                scale=1.0,
                interpolation=interpolation,
                shear=[0.0, 0.0],
                fill=fill,
            )
        elif transform_id == "TranslateY":
            return F.affine(
                img,
                angle=0.0,
                translate=[0, int(magnitude)],
                scale=1.0,
                interpolation=interpolation,
                shear=[0.0, 0.0],
                fill=fill,
            )
        elif transform_id == "Rotate":
            return F.rotate(img, angle=magnitude, interpolation=interpolation, fill=fill)
        elif transform_id == "Brightness":
            # Use 1.0 + magnitude like big_vision
            return F.adjust_brightness(img, brightness_factor=1.0 + magnitude)
        elif transform_id == "Color":
            return F.adjust_saturation(img, saturation_factor=1.0 + magnitude)
        elif transform_id == "Contrast":
            # Correct implementation matching big_vision's fixed behavior
            return F.adjust_contrast(img, contrast_factor=1.0 + magnitude)
        elif transform_id == "Sharpness":
            # Always sharpen, never blur (like big_vision)
            return F.adjust_sharpness(img, sharpness_factor=1.0 + magnitude)
        elif transform_id == "Posterize":
            return F.posterize(img, bits=int(magnitude))
        elif transform_id == "Solarize":
            # Match big_vision's behavior for solarize
            from torchvision.transforms import _functional_tensor as _FT
            bound = _FT._max_value(img.dtype) if isinstance(img, torch.Tensor) else 255.0
            return F.solarize(img, threshold=bound * magnitude)
        elif transform_id == "AutoContrast":
            return F.autocontrast(img)
        elif transform_id == "Equalize":
            return F.equalize(img)
        elif transform_id == "Invert":
            return F.invert(img)
        elif transform_id == "SolarizeAdd":
            return self._solarize_add(img, addition=int(magnitude))
        elif transform_id == "Cutout":
            return self._cutout(img, pad_size=int(magnitude), replace=fill)
        else:
            raise ValueError(f"No transform available for {transform_id}")

    def _solarize_add(self, img: Any, addition: int = 0, threshold: int = 128) -> Any:
        """Implementation of SolarizeAdd matching big_vision.

        Args:
            img: Image to be transformed (can be PIL Image or tensor)
            addition: Value to add to pixels below threshold
            threshold: Solarization threshold

        Returns:
            Transformed image
        """
        from torchvision.transforms import _functional_tensor as _FT

        # For PIL Image, convert to tensor, process, then convert back
        if not isinstance(img, torch.Tensor):
            from torchvision.transforms import functional as F_pil
            tensor_img = F_pil.to_tensor(img)
            result = self._solarize_add(tensor_img, addition, threshold)
            return F_pil.to_pil_image(result)

        # Get the appropriate max value based on the image dtype
        bound = _FT._max_value(img.dtype)

        # Add the value to the image using int64 for consistent behavior
        added_img = img.to(torch.int64) + addition

        # Clip values and convert back to the original dtype (or uint8 for consistency with big_vision)
        added_img = added_img.clip(0, bound).to(torch.uint8)

        # Apply the threshold mask: use original image for pixels >= threshold
        return torch.where(img < threshold, added_img, img)

    def _cutout(self, img: Any, pad_size: int, replace: Any = None) -> Any:
        """Implementation of Cutout matching big_vision.

        Args:
            img: Image to be transformed (can be PIL Image or tensor)
            pad_size: Size of the cutout (half the side length)
            replace: Fill value for cutout region, defaults to [128, 128, 128]

        Returns:
            Transformed image
        """
        if replace is None:
            replace = [128, 128, 128]  # Match big_vision's default

        # Handle PIL Image by converting to tensor, processing, and converting back
        is_pil = not isinstance(img, torch.Tensor)
        if is_pil:
            from torchvision.transforms import functional as F_pil
            tensor_img = F_pil.to_tensor(img)
            result = self._cutout(tensor_img, pad_size, replace)
            return F_pil.to_pil_image(result)

        # For tensor images, continue with the original implementation
        channels, height, width = F.get_dimensions(img)

        # Sample cutout center
        cutout_center_height = torch.randint(0, height, (1,)).item()
        cutout_center_width = torch.randint(0, width, (1,)).item()

        # Calculate boundaries
        lower_pad = max(0, cutout_center_height - pad_size)
        upper_pad = max(0, height - cutout_center_height - pad_size)
        left_pad = max(0, cutout_center_width - pad_size)
        right_pad = max(0, width - cutout_center_width - pad_size)

        cutout_shape = [height - (lower_pad + upper_pad), width - (left_pad + right_pad)]

        # Create replacement tensor with correct device and dtype
        replace_tensor = torch.tensor(replace, device=img.device, dtype=img.dtype)
        if len(replace_tensor.shape) == 1:
            replace_tensor = replace_tensor.unsqueeze(1).unsqueeze(1)

        # F.erase only works with tensor images
        return F.erase(img, lower_pad, left_pad, cutout_shape[0], cutout_shape[1], replace_tensor)