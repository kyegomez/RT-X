from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image


class EfficientNetFilm(nn.Module):
    """
    EfficientNet with FiLM layer

    Args:
        model (str): EfficientNet model name
        num_classes (int): Number of classes
        num_features (int): Number of features to output from the model
        resize (int): Size to resize the image to

    Attributes:
        model (EfficientNet): EfficientNet model
        num_classes (int): Number of classes
        num_features (int): Number of features to output from the model
        resize (int): Size to resize the image to
        transform (torchvision.transforms.Compose): Image transformations

    Example:
        >>> model = EfficientNetFilm('efficientnet-b0', 10)
        >>> img = Image.open('img.jpeg')
        >>> features = model(img)
        >>> features.shape
        torch.Size([1, 1280])

    """

    def __init__(
        self,
        model,
        num_classes,
        num_features=1280,
        resize=224,
    ):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.num_features = num_features
        self.resize = resize

        self.model = EfficientNet.from_pretrained(model)

        self.transform = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, img: str):
        """
        Extract the feature embeddings from the image

        Args:
            img (str): Path to image
        """
        img = Image.open(img)
        img = self.transform(img).unsqueeze(0)
        print(img.shape)

        features = self.model.extract_features(img)
        print(features.shape)
