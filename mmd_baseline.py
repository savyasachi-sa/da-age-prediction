from dataloader import *

train_dataset = UTK(file="./Data/part1.tar.gz")
batch_size = 1
_num_workers = 1
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          num_workers=_num_workers,
                          shuffle=False)


