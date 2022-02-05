from torchvision.transforms import transforms


def cifar10_transforms() -> transforms:
    """
    Creates transformations for CIFAR10 dataset

    Returns:
        Train and Test transformations

    Examples:

        >>> train_transforms, test_transforms = cifar10_transforms()


    """
    train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return train, test
