from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection


class CustomDataset: 
    def __init__(self, root_folder):
        self.root_folder = root_folder

    def get_dataset(self, name, train_ds=True):
        valid_datasets = {'cifar10': lambda: (datasets.CIFAR10(self.root_folder, train=True,transform=transforms.ToTensor(),download=True),
                                            datasets.CIFAR10(self.root_folder, train=False,transform=transforms.ToTensor(),download=True)),
                          'stl10': lambda: (datasets.STL10(self.root_folder, split='train',transform=transforms.ToTensor(),download=True),
                                            datasets.STL10(self.root_folder, split='test',transform=transforms.ToTensor(),download=True))}
        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection() 
        else:
            return dataset_fn()