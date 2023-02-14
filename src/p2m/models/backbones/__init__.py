# First Party Library
from p2m.models.backbones.resnet import resnet50
from p2m.models.backbones.vgg16 import VGG16P2M
from p2m.options import ModelBackbone

# from p2m.models.backbones.vgg16 import VGG16Recons
# from p2m.models.backbones.vgg16 import VGG16TensorflowAlign


def get_backbone(name: ModelBackbone):
    match name:
        # case ModelBackbone.VGG16:
        #     raise NotImplementedError
        #     nn_encoder = VGG16TensorflowAlign()
        #     nn_decoder = VGG16Recons()
        case ModelBackbone.VGG16P2M:
            nn_encoder = VGG16P2M(pretrained=True)
            nn_decoder = None
        case ModelBackbone.RESNET50:
            nn_encoder = resnet50()
            nn_decoder = None
        case _:
            raise NotImplementedError(f"No implemented backbone called '{name}' found")
    return nn_encoder, nn_decoder
