import torch.nn as nn
import torch
import torchvision.models as models
from exceptions.exceptions import InvalidBackboneError
from torch.nn import Linear
import glob
# from utils import model_dict

class Downstream_NN(nn.Module):
    def __init__(self,args,config,num_classes):
        super(Downstream_NN, self).__init__()
        self.num_classes=num_classes
        assert (args.training_method in ["Scratch","ImageNet","SSL"],"invalid args.training_method")
        pretrained=False if (args.training_method =="Scratch" or args.training_method =="SSL") else True
        self.model_dict = {"resnet18": models.resnet18(pretrained=pretrained), #resnet*(pretrained=True, num_classes=10) doesn't work. hence, decouple pretrained and num_classes
                            "resnet50": models.resnet50(pretrained=pretrained)} #replace by model_dict
        self.model = self._get_basemodel(config.arch)
        if (args.training_method =="Scratch" or args.training_method =="ImageNet"):
            self.model.fc = Linear(in_features=self.model.fc.in_features, out_features=self.num_classes, bias=(self.model.fc.bias is not None)) #https://github.com/pytorch/vision/issues/1040
        else : #SSL
            checkpoint_file=[name for name in glob.glob(args.run_dir+'/checkpoint*')][-1] #picks the last checkpoint if there are > 1
            checkpoint = torch.load(checkpoint_file) #CPU # checkpoint = torch.load(checkpoint_file, map_location=args.device)
            state_dict = checkpoint['state_dict'] #orderedDict
            for k in list(state_dict.keys()):
                if k.startswith('backbone') and not k.startswith('backbone.fc'):
                    state_dict[k[len("backbone."):]] = state_dict[k] # remove prefix from all but fc layer
                del state_dict[k] #delete all layers with names from original pretrained network. 
            log = self.model.load_state_dict(state_dict, strict=False)
            assert log.missing_keys == ['fc.weight', 'fc.bias'] #loaded all but the FC layers. Whether you are loading from a partial state_dict, which is missing some keys, or loading a state_dict with more keys than the model that you are loading into, you can set the strict argument to False in the load_state_dict() function to ignore non-matching keys.
        if args.downstream=="fine_tune":
            parameters = list(filter(lambda p: not p.requires_grad, self.model.parameters()))
            assert len(parameters) == 0  # all params must require_grad
        elif args.downstream=="linear_eval":
            for name, param in self.model.named_parameters(): # freeze all layers but the last fc
                if name not in ['fc.weight', 'fc.bias']:
                    param.requires_grad = False
            parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
            assert len(parameters) == 2  # fc.weight, fc.bias

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
        except KeyError:
            raise InvalidBackboneError("Invalid downstream architecture. Currently supported arch : resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.model(x)




# def get_downstream_model(args,config):
#     elif args.training_method =="SSL": # use SSL pre_trained network
#         if config.arch == 'resnet18': #make the below lines work with any "torchvision" model, any number of classes, any dataset
#             model = torchvision.models.resnet18(pretrained=False, num_classes=10).to(args.device)
#         elif config.arch == 'resnet50':
#             model = torchvision.models.resnet50(pretrained=False, num_classes=10).to(args.device)
#         checkpoint_file=[name for name in glob.glob(args.run_dir+'/checkpoint*')][-1] #picks the last checkpoint if there are > 1
#         checkpoint = torch.load(checkpoint_file, map_location=args.device)
#         state_dict = checkpoint['state_dict'] #orderedDict
#         for k in list(state_dict.keys()):
#             if k.startswith('backbone') and not k.startswith('backbone.fc'):
#                 state_dict[k[len("backbone."):]] = state_dict[k] # remove prefix from all but fc layer
#             del state_dict[k] #delete all layers with names from original pretrained network. 
#         log = model.load_state_dict(state_dict, strict=False)
#         assert log.missing_keys == ['fc.weight', 'fc.bias'] #load all but the FC layers
#     if args.downstream=="fine_tune":
#         parameters = list(filter(lambda p: not p.requires_grad, model.parameters()))
#         assert len(parameters) == 0  # all params must require_grad
#     elif args.downstream=="linear_eval":
#         for name, param in model.named_parameters(): # freeze all layers but the last fc
#             if name not in ['fc.weight', 'fc.bias']:
#                 param.requires_grad = False
#         parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
#         assert len(parameters) == 2  # fc.weight, fc.bias
#     return model