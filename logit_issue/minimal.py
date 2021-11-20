#! /usr/bin/env python3
import torch
import numpy as np
import pickle

import torch.nn as nn
import torch.optim as optim


###################################
# Parameters
###################################
import sys
seed = int(sys.argv[1])
batch_size = 128


import random 
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True) 

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # self.feature = nn.Sequential(
        #     nn.Conv1d(in_channels=2, out_channels=50, kernel_size=7, stride=1),
        #     nn.ReLU(False),
        #     nn.Conv1d(in_channels=50, out_channels=50, kernel_size=7, stride=2),
        #     nn.ReLU(False),
        #     nn.Dropout(),
        #     nn.Flatten(),
        #     nn.Linear(50 * 58, 256),
        #     nn.ReLU(False),
        #     nn.Dropout(),
        #     nn.Linear(256, 80),
        #     nn.ReLU(False),
        #     # nn.Dropout(),
        #     nn.Linear(80, 11),
        #     nn.LogSoftmax(dim=1)
        # )

        self.feature = nn.Sequential(
                nn.Conv1d(in_channels=2, out_channels=50, kernel_size=7, stride=1, padding=0),
                nn.ReLU(True),
                nn.Conv1d(in_channels=50, out_channels=50, kernel_size=7, stride=2, padding=0),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Flatten(),
                nn.Linear(50 * 58, 256),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(256, 80),
                nn.ReLU(True),
                nn.Linear(80, 12),
                nn.LogSoftmax(dim=1)
            )

    def forward(self, input_data):
        return self.feature(input_data)

device = torch.device("cuda")
net = CNNModel().to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

###################################
# Build the dataset
###################################

####
# Real Dataset
####
dataset_path = "/mnt/wd500GB/CSC500/csc500-super-repo/datasets/RML2016.10a_dict.pkl"
Xd = pickle.load(open(dataset_path,'rb'), encoding="latin1")
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])       
modulation_mapping = {
    'AM-DSB': 0,
    'QPSK'  : 1,
    'BPSK'  : 2,
    'QAM64' : 3,
    'CPFSK' : 4,
    '8PSK'  : 5,
    'WBFM'  : 6,
    'GFSK'  : 7,
    'AM-SSB': 8,
    'QAM16' : 9,
    'PAM4'  : 10,
}
data = []
for mod in mods:
    for snr in snrs:
        if snr in [-18, -12, -6, 0, 6, 12, 18]:
            for x in Xd[(mod,snr)]:
                data.append(
                    (
                        x.astype(np.single),
                        modulation_mapping[mod],
                    )
                )



####
# Dummy Dataset
####
# data = []
# for label in range(11):
#     x = np.ones([2,128], dtype=np.single) * label

#     data.append(
#         (x,label)
#     )

# # Replicate the dummy data
# data = data * 1000

dl = torch.utils.data.DataLoader(
    data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=1,
    persistent_workers=True,
    prefetch_factor=50,
    pin_memory=True
)

###################################
# Build the tet jig, train
###################################
best_epoch = (-1, 133700000)
for epoch in range(10):
    total_epoch_loss = 0
    total_batches_in_epoch = 0
    for x,y in dl:
        x = x.to(device)
        y = y.to(device)

        logits = net.forward(x)

        loss = criterion(logits, y)

        total_epoch_loss += loss
        total_batches_in_epoch += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    average_batch_error = total_epoch_loss/total_batches_in_epoch
    if average_batch_error < best_epoch[1]:
        best_epoch = (epoch, average_batch_error)

print("seed,best_epoch,loss:",seed, best_epoch[0], best_epoch[1])