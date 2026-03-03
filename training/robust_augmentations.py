from __future__ import annotations

import albumentations as A


def build_train_augmentations(
    image_size: int,
    strength: float,
    cfg: dict,
) -> A.Compose:
    s = max(0.0, min(1.0, float(strength)))

    def p(key: str, base: float = 0.25) -> float:
        return max(0.0, min(1.0, float(cfg.get(key, base)) * (0.25 + 0.75 * s)))

    return A.Compose(
        [
            A.RandomResizedCrop(
                height=image_size,
                width=image_size,
                scale=(0.65, 1.0),
                ratio=(0.75, 1.33),
                p=p('random_crop_scale_prob', 0.35),
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.05),
            A.GaussianBlur(blur_limit=(3, 9), p=p('gaussian_blur_prob', 0.35)),
            A.MotionBlur(blur_limit=9, p=p('motion_blur_prob', 0.25)),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=p('brightness_contrast_prob', 0.45),
            ),
            A.RandomRain(
                slant_range=(-15, 15),
                drop_length=18,
                drop_width=1,
                blur_value=3,
                brightness_coefficient=0.88,
                rain_type='heavy',
                p=p('rain_prob', 0.25),
            ),
            A.RandomFog(
                fog_coef_range=(0.15, 0.45),
                alpha_coef=0.12,
                p=p('fog_prob', 0.25),
            ),
            A.ImageCompression(
                quality_lower=22,
                quality_upper=75,
                p=p('jpeg_prob', 0.25),
            ),
            A.RandomShadow(
                shadow_roi=(0, 0.4, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=3,
                shadow_dimension=5,
                p=p('shadow_prob', 0.2),
            ),
            A.Resize(image_size, image_size),
        ]
    )


def build_val_augmentations(image_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
    ])
