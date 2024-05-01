import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16
from torch.utils.data import DataLoader

class VGGRegressor(nn.Module):

    def __init__(self):
        super(VGGRegressor, self).__init__()
        self.model_features = vgg16(pretrained=True)
        num_features=self.model_features.classifier[0].in_features
        self.model_features.classifier=nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.Linear(1024, 128), #120
            nn.Linear(128, 64), #64
            nn.Linear(64, 1)
        )

def init():
    global device, model, loss_fn, optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGRegressor().to(device)
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train_model(dataloader: DataLoader):
    size = len(dataloader.dataset)
    model.train()
    averageLoss = 0.0
    n = 0
    for batch, (X, y) in enumerate(dataloader):
        n += 1
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        # print(pred)
        # print(y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss, current = loss.item(), batch * len(X)
        averageLoss += loss

        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    averageLoss = averageLoss / n
    print(f"Average Testing Error: {averageLoss:>7f}")
    return averageLoss

def validate_model(dataloader: DataLoader):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Average Validation Error: {test_loss:>8f} \n")
    return test_loss

def test_model(dataloader: DataLoader):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Average Test Error: {test_loss:>8f} \n")
    return test_loss

# Save model
def save_model(model_path: str = 'src/weights/vgg_regressor/rename_weight.pt'):
    torch.save(model.state_dict(), model_path)


# Load model
def load_model(model_path: str = 'src/weights/vgg_regressor/rename_weight.pt'):
    model.load_state_dict(torch.load(model_path))
