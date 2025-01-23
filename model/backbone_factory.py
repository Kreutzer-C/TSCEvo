import torch
import torch.nn as nn
import torchvision.models as models

def get_backbone(name, pretrained=True):
    if name == "resnet50":
        network = models.resnet50(pretrained=pretrained)
        encoder = nn.Sequential(*list(network.children())[:-1])
        projector = nn.Linear(2048, 512)
        network = nn.Sequential(
            encoder,
            nn.Flatten(),
            projector
        )
        return network

    elif name == "resnet101":
        network = models.resnet101(pretrained=pretrained)
        encoder = nn.Sequential(*list(network.children())[:-1])
        projector = nn.Linear(2048, 512)
        network = nn.Sequential(
            encoder,
            nn.Flatten(),
            projector
        )
        return network

    elif name == "resnet18":
        network = models.resnet18(pretrained=pretrained)
        encoder = nn.Sequential(*list(network.children())[:-1])
        projector = nn.Linear(2048, 512)
        network = nn.Sequential(
            encoder,
            nn.Flatten(),
            projector
        )
        return network

    else:
        raise ValueError(name)

if __name__ == "__main__":
    model = get_backbone('resnet50')
    print(model)