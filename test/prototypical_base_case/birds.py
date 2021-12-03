#! /usr/bin/env python3


from easyfsl.data_tools import EasySet, TaskSampler
from torch.utils.data import DataLoader

train_set = EasySet(specs_file="./data/CUB/train.json", training=True)
train_sampler = TaskSampler(
    train_set, n_way=5, n_shot=5, n_query=10, n_tasks=40000
)
train_loader = DataLoader(
    train_set,
    batch_sampler=train_sampler,
    num_workers=12,
    pin_memory=True,
    collate_fn=train_sampler.episodic_collate_fn,
)



from easyfsl.methods import PrototypicalNetworks
from torch import nn
from torch.optim import Adam
from torchvision.models import resnet18

convolutional_network = resnet18(pretrained=False)
convolutional_network.fc = nn.Flatten()
model = PrototypicalNetworks(convolutional_network).cuda()

optimizer = Adam(params=model.parameters())

model.fit(train_loader, optimizer)



test_set = EasySet(specs_file="./data/CUB/test.json", training=False)
test_sampler = TaskSampler(
    test_set, n_way=5, n_shot=5, n_query=10, n_tasks=100
)
test_loader = DataLoader(
    test_set,
    batch_sampler=test_sampler,
    num_workers=12,
    pin_memory=True,
    collate_fn=test_sampler.episodic_collate_fn,
)

accuracy = model.evaluate(test_loader)
print(f"Average accuracy : {(100 * accuracy):.2f}")
