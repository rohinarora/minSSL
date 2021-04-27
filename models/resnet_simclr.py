import torch.nn as nn
import torchvision.models as models
from exceptions.exceptions import InvalidBackboneError

class PreTrain_NN(nn.Module):
    def __init__(self, base_model, out_dim):
        super(PreTrain_NN, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}
        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features #512 for ResNet18
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc) # add mlp projection head

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)
