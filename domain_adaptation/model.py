import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    ResNet18_Weights, resnet18, ResNet34_Weights, resnet34,
    MobileNet_V3_Small_Weights, mobilenet_v3_small,
    SqueezeNet1_1_Weights, squeezenet1_1,
)
from adaptation import DomainDiscriminator, GradientReverseLayer, MultiLinearMap

class SmallCNN(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super(SmallCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        
        self.conv2 = nn.Conv2d(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(base_channels*2)
        
        self.conv3 = nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(base_channels*4)
        
        self.conv4 = nn.Conv2d(base_channels*4, base_channels*4, kernel_size=3, 
                              groups=base_channels*4, padding=1, bias=False)
        self.conv4_pointwise = nn.Conv2d(base_channels*4, base_channels*4, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(base_channels*4)
        
        self.conv5 = nn.Conv2d(base_channels*4, base_channels*4, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(base_channels*4)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.feature_dim = base_channels * 4
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))        
        x = F.relu(self.bn2(self.conv2(x)))        
        x = F.relu(self.bn3(self.conv3(x)))        
        
        x = self.conv4(x)                          
        x = self.conv4_pointwise(x)                
        x = F.relu(self.bn4(x))                   
        
        x = F.relu(self.bn5(self.conv5(x)))        
        
        features = self.adaptive_pool(x)          
        features = features.view(features.size(0), -1)  
        
        return features

class Model(nn.Module):
    def __init__(self, model_name="resnet18", pretrained=True, in_channels=1, args=None):
        super(Model, self).__init__()

        if model_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = resnet18(weights=weights)
            feature_dim = 512
            if in_channels != 3:
                backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        elif model_name == "resnet34":
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            backbone = resnet34(weights=weights)
            feature_dim = 512
            if in_channels != 3:
                backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            
        elif model_name == "mobilenet_v3_small":
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            backbone = mobilenet_v3_small(weights=weights)
            feature_dim = 576
            if in_channels != 3:
                backbone.features[0][0] = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
            self.backbone = nn.Sequential(backbone.features, backbone.avgpool)
            
        elif model_name == "squeezenet1_1":
            weights = SqueezeNet1_1_Weights.DEFAULT if pretrained else None
            backbone = squeezenet1_1(weights=weights)
            feature_dim = 512
            if in_channels != 3:
                backbone.features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1)
            self.backbone = nn.Sequential(backbone.features, nn.AdaptiveAvgPool2d((1, 1)))
            
        elif model_name == "small_cnn":
            base_channels = args.cnn_base_channels if args else 32
            self.backbone = SmallCNN(in_channels=in_channels, base_channels=base_channels)
            feature_dim = base_channels * 4
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Store feature_dim as instance attribute
        self.feature_dim = feature_dim
        
        hidden_dim = feature_dim // 2
        
        self.classification_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        
        self.domain_discriminator = None
        self.grl = None
        self.multilinear_map = None
        
        if args and args.adaptation in ["dann", "cdan"]:
            self._init_domain_adaptation_components(args)
            
    def _init_domain_adaptation_components(self, args):
        if args.adaptation == "dann":
            self.domain_discriminator = DomainDiscriminator(
                in_feature=self.feature_dim, 
                hidden_size=getattr(args, 'domain_discriminator_hidden', self.feature_dim)
            )
            self.grl = GradientReverseLayer(alpha=1.0)
            
        elif args.adaptation == "cdan":
            self.multilinear_map = MultiLinearMap()
            discriminator_input_dim = self.feature_dim * 2
                
            self.domain_discriminator = DomainDiscriminator(
                in_feature=discriminator_input_dim,
                hidden_size=getattr(args, 'domain_discriminator_hidden', self.feature_dim)
            )
            self.grl = GradientReverseLayer(alpha=1.0)
    
    def update_grl_alpha(self, alpha):
        if self.grl is not None:
            self.grl.set_alpha(alpha)
    
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)

        outputs = {
            "features": features,
            "classification": self.classification_head(features),
        }
        return outputs

def create_model(args):
    model = Model(
        model_name=args.model,
        pretrained=args.pretrained,
        in_channels=args.in_channels,
        args=args
    )

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded model from {args.checkpoint}")

    return model