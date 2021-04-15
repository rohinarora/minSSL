from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection


class ContrastiveLearningDataset: #define custom dataset class
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod # here static acts like utility function https://www.programiz.com/python-programming/methods/built-in/staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """transforms can be improved to match well with SimCLR paper. this doesnt seem to be exactly what paper says. can try more transforms as well"""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s) #
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views): # the use of lambda is pretty nice over here. without lambda, each dataset object is created when "valid_datasets" is called. with lambda, just the lambda functions are called. the actual object is created only when dict is called via the key. saves created ALL dataset objects. nice
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator( #ContrastiveLearningViewGenerator is a object with implements __call__ function. Hence can be used to replace typical transforms() function
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          download=True)}

                # some new dataset will be added here
                # feels easy to add dataset. just need to pass in the dataset image size to this function
                # later : https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

        try:
            dataset_fn = valid_datasets[name] #lambda fn
        except KeyError:
            raise InvalidDatasetSelection() #return lambda_fn returns the object`
        else:
            return dataset_fn() #

    def get_dataset_no_lambda(self, name, n_views):
        valid_datasets = {'cifar10': datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),

                          'stl10': datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          download=True)}

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()