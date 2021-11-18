from re import U
import torch
import math

def split_dataset_by_percentage(train:float, val:float, test:float, dataset, seed:int):
    assert train < 1.0
    assert val < 1.0
    assert test < 1.0
    assert train + val + test <= 1.0

    num_train = math.floor(len(dataset) * train)
    num_val   = math.floor(len(dataset) * val)
    num_test  = math.floor(len(dataset) * test)

    return torch.utils.data.random_split(dataset, (num_train, num_val, num_test), generator=torch.Generator().manual_seed(seed))



"""
Assumes batch is in the form (X,Y,U) or (X,Y)
"""
def predict_batch(model, device, batch, forward_uses_domain):
    model.eval()
    tup = [t.to(device) for t in batch]

    if forward_uses_domain:
        y_hat, u_hat = model.forward(tup[0], tup[1])
    else:
        y_hat = model.forward(tup[0])
    pred = y_hat.data.max(1, keepdim=True)[1]
    pred = torch.flatten(pred).cpu()

    return pred


"""
Returns dict in the form of
{
    <domain>:
    {
        <y>: {
            <y_hat>: int
        }
    },
}

IE [domain][y][y_hat]:count

IE a confusion matrix for each domain
"""
def confusion_by_domain_over_dataloader(model, device, dl, forward_uses_domain):
    confusion_by_domain = {}

    for batch in dl:
        pred = predict_batch(model, device, batch, forward_uses_domain)

        batch.append(pred)



        for x,y,u,y_hat in zip(*batch):
            x = x.cpu().detach().numpy()
            y = int(y.cpu().detach().numpy())
            u = int(u.cpu().detach().numpy())
            y_hat = int(y_hat.cpu().detach().numpy())

            # Yeah yeah I know...
            if u not in confusion_by_domain:
                confusion_by_domain[u] = {}
            if y not in confusion_by_domain[u]:
                confusion_by_domain[u][y] = {}
            if y_hat not in confusion_by_domain[u][y]:
                confusion_by_domain[u][y][y_hat] = 0

            confusion_by_domain[u][y][y_hat] += 1

    return confusion_by_domain