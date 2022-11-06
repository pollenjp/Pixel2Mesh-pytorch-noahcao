# First Party Library
from p2m.models.backbones.resnet import resnet50
from p2m.models.backbones.vgg16 import VGG16P2M
from p2m.models.backbones.vgg16 import VGG16Recons
from p2m.models.backbones.vgg16 import VGG16TensorflowAlign


def get_backbone(options):
    if options.backbone.startswith("vgg16"):
        if options.align_with_tensorflow:
            nn_encoder = VGG16TensorflowAlign()
        else:
            nn_encoder = VGG16P2M(pretrained="pretrained" in options.backbone)
        nn_decoder = VGG16Recons()
    elif options.backbone == "resnet50":
        nn_encoder = resnet50()
        nn_decoder = None
    else:
        raise NotImplementedError("No implemented backbone called '%s' found" % options.backbone)
    return nn_encoder, nn_decoder
