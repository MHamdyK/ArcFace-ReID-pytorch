import torchvision.transforms as T

# evaluation / gallery
vgg_transform = T.Compose([
    T.Resize((160, 160)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# training
train_transforms = T.Compose([
    T.RandomResizedCrop(160, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(15),
    T.ColorJitter(0.2, 0.2, 0.2, 0.1),
    vgg_transform,
])
