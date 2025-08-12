from .transforms import *


base_trans = ComposeImgAnno([
    ToTensorImgAnno(),
    NormalizeImgAnno()
])


val_trans = ComposeImgAnno([
    ResizeImgAnno(size=560, max_size=1332),
    base_trans
])


train_trans = ComposeImgAnno([
    RandomHorizontalFlipImgAnno(),
    RandomSelectImgAnno(
        RandomResizeImgAnno(sizes=[480, 512, 544, 576, 608, 640, 672], max_size=1333),
        ComposeImgAnno([
            RandomResizeImgAnno(sizes=[400, 500, 600], max_size=1333),
            RandomSizeCropImgAnno(384, 600), # 384
            RandomResizeImgAnno(sizes=[480, 512, 544, 576, 608, 640, 672], max_size=1333),
        ])
    ),
    base_trans
])


weak_aug = ComposeImgAnno([
    SquarePadImgAnno(),
    RandomHorizontalFlipImgAnno(p=0.5),
    ResizeImgAnno(size=560, max_size=1333)
])


weak_trans = ComposeImgAnno([
    weak_aug,
    base_trans
])


strong_aug = ComposeImgAnno([
    RandomApplyImgAnno(
        [ColorJitterImgAnno(0.4, 0.4, 0.4, 0.1)], p=0.8
    ),
    RandomGrayScaleImgAnno(p=0.2),
    RandomApplyImgAnno(
        [GaussianBlurImgAnno([0.1, 2.0])], p=0.5
    )
])


strong_trans = ComposeImgAnno([
    weak_aug,
    strong_aug,
    base_trans
])

def weak_aug(img_size=700):
    return ComposeImgAnno([
        SquarePadImgAnno(),
        RandomHorizontalFlipImgAnno(p=0.5),
        ResizeImgAnno(size=img_size, max_size=1333)
        ])

def strong_aug():
    return ComposeImgAnno([
        RandomApplyImgAnno(
            [ColorJitterImgAnno(0.4, 0.4, 0.4, 0.1)], p=0.8
        ),
        RandomGrayScaleImgAnno(p=0.2),
        RandomApplyImgAnno(
            [GaussianBlurImgAnno([0.1, 2.0])], p=0.5
        )
        ])

def base_trans():
    return ComposeImgAnno([
        ToTensorImgAnno(),
        NormalizeImgAnno()
        ])

def train_trans():
    return ComposeImgAnno([
        SquarePadImgAnno(),
        RandomHorizontalFlipImgAnno(),
        RandomSelectImgAnno(
            RandomResizeImgAnno(sizes=[480, 512, 544, 576, 608, 640, 672], max_size=1333),
            ComposeImgAnno([
                RandomResizeImgAnno(sizes=[400, 500, 600], max_size=1333),
                RandomSizeCropImgAnno(384, 600), # 384
                RandomResizeImgAnno(sizes=[480, 512, 544, 576, 608, 640, 672], max_size=1333),
            ])
        ),
        ToTensorImgAnno(),
        NormalizeImgAnno()
        ])

def val_trans(img_size=700):
    return ComposeImgAnno([
        ResizeImgAnno(size=img_size, max_size=1332),
        ToTensorImgAnno(),
        NormalizeImgAnno()
        ])

def weak_trans(img_size=700):
    return ComposeImgAnno([
        SquarePadImgAnno(),
        RandomHorizontalFlipImgAnno(p=0.5),
        ResizeImgAnno(size=img_size, max_size=1333),
        ToTensorImgAnno(),
        NormalizeImgAnno()
        ])

def strong_trans(img_size=700):
    return ComposeImgAnno([
        SquarePadImgAnno(),
        RandomHorizontalFlipImgAnno(p=0.5),
        ResizeImgAnno(size=img_size, max_size=1333),
        RandomApplyImgAnno(
            [ColorJitterImgAnno(0.4, 0.4, 0.4, 0.1)], p=0.8
        ),
        RandomGrayScaleImgAnno(p=0.2),
        RandomApplyImgAnno(
            [GaussianBlurImgAnno([0.1, 2.0])], p=0.5
        ),
        ToTensorImgAnno(),
        NormalizeImgAnno()
        ])
