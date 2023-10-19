import torchvision.models as models
import torch
import torch.nn as nn
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from .unetr import UNETR
from .unet import UNet
from .basic_unet import BasicUNet
from .attention_unet import MAnet

class DeepLabWrapper(nn.Module):
    """This wrapper class allows us to overwrite
    the default behaviour of the deeplab model that returns a dict 
    of the output. Our other models would just output a tensor and adjusting,
    the pl_module would needlessly clutter it, so this makes it easy to keep everything clean.
    """
    def __init__(self, pretrained=True, num_classes=6):
        super(DeepLabWrapper, self).__init__()  # Important! Initialize the parent class
        if pretrained:
            self.model = models.segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        else:
            self.model = models.segmentation.deeplabv3_resnet50(weights=None)
        # Adjust the number of output classes in the classifier
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)

        # Adjust for 1 input channel
        self.model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        output = self.model(x)
        return output['out']

    def __getattr__(self, attr):
        # If an attribute isn't found in this class, it will be searched in self.model
        try:
            return super().__getattr__(attr)
        except AttributeError:
            return getattr(self.model, attr)

class PaddingWrapper(nn.Module):
    def __init__(self, model, model_size=(800, 1104)):
        super(PaddingWrapper, self).__init__()  # Important! Initialize the parent class
        self.model_size = model_size
        self.model = model
        
    def forward(self, x):
        assert x.ndim == 4, f"wrapper input shape had unexpected ndim: {x.ndim} with shape {x.shape}"
        b, c, h, w = x.shape
        padded_input = torch.zeros((b, c) + self.model_size, device=x.device)
        padded_input[:, :, :h, :w] = x
        output = self.model(padded_input)

        return output[:, :, :h, :w]
    
    def __getattr__(self, attr):
        # If an attribute isn't found in this class, it will be searched in self.model
        try:
            return super().__getattr__(attr)
        except AttributeError:
            return getattr(self.model, attr)


def get_model(cfg):
    if cfg.MODEL.NAME == 'deeplab':
        # Get a pretrained FCN with a ResNet-50 backbone
        model = DeepLabWrapper(pretrained=cfg.MODEL.PRETRAINED, num_classes=cfg.MODEL.NUM_CLASSES)

        # if cfg.MODEL.FREEZE_ENCODER:
        #     for param in model.model.encoder.parameters():
        #         param.requires_grad = False

    elif cfg.MODEL.NAME == 'unetr':
        if cfg.MODEL.PRETRAINED:
            print("WARNING: no default pretrained weights available for model Unetr, " +
                  "MODEL.PRETRAINED option is ignored. MODEL.RESUME_PATH can still be used.")
        # different models have different input requirements. The best way to deal with this is probably
        # to find the closest bigger size that is compatable and then pad around the original image
        # with the background class. This allows us to still segment at full resolution.
        model_image_size = (800, 1104)
        model = UNETR(1, 
                      cfg.MODEL.NUM_CLASSES, 
                      img_size=model_image_size, 
                      norm_name='batch', 
                      hidden_size=512, # slightly smaller than the original model (768)
                      num_heads=8, # slightly smaller than the original model (12)
                      mlp_dim=2048,# slightly smaller than the original model (3072)
                      spatial_dims=2)
        model = PaddingWrapper(model, model_size=model_image_size)
    elif cfg.MODEL.NAME == "unet":
        model = BasicUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=cfg.MODEL.NUM_CLASSES
        )
        # freeze gradients of encoder for finetuning of SSL pretrained model
        if cfg.MODEL.FREEZE_ENCODER:
            model.conv_0.requires_grad_(False)
            model.down_1.requires_grad_(False)
            model.down_2.requires_grad_(False)
            model.down_3.requires_grad_(False)
            model.down_4.requires_grad_(False)
        
    elif cfg.MODEL.NAME == "residual_unet":
        model_image_size = (800, 1104)
        model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=cfg.MODEL.NUM_CLASSES,
            channels=(32, 32, 64, 128, 256),
            strides=(1, 2, 2, 2),
        )
        model = PaddingWrapper(model, model_size=model_image_size)

    elif cfg.MODEL.NAME == "attunet":
        if cfg.MODEL.PRETRAINED:
            print("loading pretrained imagenet weights")
            encoder_weights_name = "imagenet"
        else:
            print("not load pretrained imagenet weights")
            encoder_weights_name = None

        model_image_size = (800, 1120)
        model = MAnet(
            encoder_name = 'resnet34',
            in_channels = 1,
            classes = cfg.MODEL.NUM_CLASSES,
            encoder_weights=encoder_weights_name
        )
        model = PaddingWrapper(model, model_size=model_image_size)
        for param in model.model.encoder.parameters():
            param.requires_grad = False
        
        for param in model.model.decoder.center.parameters():
            param.requires_grad = False
    else:
        raise ValueError(f"Model {cfg.MODEL.NAME} not recognized!")
    
    return model
