import albumentations as albu
from albumentations.pytorch import ToTensorV2

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.Sharpen(p=0.5),
        albu.ColorJitter(p=0.5),
        albu.RGBShift(r_shift_limit=0.05, g_shift_limit=0.05, b_shift_limit=0.05, p=0.7),
        albu.OneOf(
            [
                albu.RandomCrop(height=320, width=320, always_apply=False, p=1),
                albu.CenterCrop(height=400, width=400, always_apply=False, p=0.7)
            ],
            p=0.7
        ),
        albu.OneOf(
            [
                albu.GaussianBlur(p=1),
                albu.RandomFog(p=1),
                albu.GaussNoise(p=1),
                albu.Blur(p=1)
            ],
            p=0.7,
        ),
        albu.Resize(512,512),
        ToTensorV2()
    ]
    return albu.Compose(train_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        ToTensorV2()
    ]
    return albu.Compose(_transform)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        #albu.PadIfNeeded(384, 480),
        ToTensorV2()
    ]
    return albu.Compose(test_transform)