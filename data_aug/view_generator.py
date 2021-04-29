import numpy as np
np.random.seed(42)

class ContrastiveLearningViewGenerator(object): #about object. its dont care in python 3 https://stackoverflow.com/a/45062077/5536853
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views
    def __call__(self, x): #define ContrastiveLearningViewGenerator's __call__ function . __call__ -> allows ContrastiveLearningViewGenerator object to behave like function. https://www.geeksforgeeks.org/__call__-in-python/
        return [self.base_transform(x) for i in range(self.n_views)]