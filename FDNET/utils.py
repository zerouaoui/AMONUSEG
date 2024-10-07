import torch
import numpy as np


def save_checkpoint(state, filename):
    """
    Save model checkpoint for future use
    * filename : path where the model will be saved, the path should end with '.pth.tar'
    """

    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def check_accuracy(loader, model, device="cuda", threshold = 0.5):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    IoU = 0
    precision = 0
    recall = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.float().to(device)
            y = y.unsqueeze(1).to(device)
            prediction = torch.sigmoid(model(x))
            prediction = (prediction > threshold).float()
            num_correct += (prediction == y).sum()
            num_pixels += torch.numel(prediction)
            dice_score += (2 * (prediction * y).sum()) / ((prediction + y).sum() + 1e-8)
            IoU += ((prediction * y).sum()) / ((prediction + y).sum() + 1e-8)
            precision += ((prediction * y).sum()) / (prediction.sum() + 1e-8)
            recall += ((prediction * y).sum()) / (y.sum() + 1e-8)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}")
    print(f"Dice score: {dice_score / len(loader)}")
    model.train()
    return dice_score / len(loader), num_correct / num_pixels * 100, IoU/len(loader), precision/len(loader), recall/len(loader)


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)