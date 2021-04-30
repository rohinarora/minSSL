import numpy as np
from torch.utils.data import Dataset, DataLoader

class RandomDataset(Dataset):
    def __getitem__(self, index):
        return np.random.randint(0, 1000, 3) #returns vector of shape (3,) with ints B/W 0 and 1000

    def __len__(self):
        return 16 #complete dataset shape (16,3)

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    
dataset = RandomDataset()
dataloader = DataLoader(dataset, batch_size=2, num_workers=4)  
# technically mini_batch size is 2. when batch_size=1; its SGD. when batch_size=dataset_size ; full GD. else mini batch GD. In DataLoader();  batch_size is a generic arg for all of them.  https://discuss.pytorch.org/t/performing-mini-batch-gradient-descent-or-stochastic-gradient-descent-on-a-mini-batch/21235/4
for batch in dataloader:
    print(batch) #batch.shape = (2,3)

#all distinct batches with num_workers=1
#with num_workers=4; every worker fetches a minibatch. At every fetch, all num_workers have same random initial seed, and hence all get the same data. (every num_workers batches will have same data)
# partial fix by using worker_init_fn
print ("with fix1\n\n")

dataloader = DataLoader(dataset, batch_size=2, num_workers=4, worker_init_fn=worker_init_fn)

for batch in dataloader:
    print(batch)

# still not correct
print ("\nstill issue\n\n")
for epoch in range(3): #all epochs give same result. want random auugmentations across epochs
    for batch in dataloader:
        print(batch)
    print ("\n")

print ("with fix2\n\n")
initial_seed=42
for epoch in range(3): #all epochs now give different random numbers
    np.random.seed(initial_seed+epoch)
    for batch in dataloader:
        print(batch)
    print ("\n")

