import torch
import config
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from data_loader import MyDataLoader

# CHANGE THIS SCRIPT
def save_example(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.float().to(config.DEVICE), y.float().to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake# * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(y, folder + f"/label_{epoch}.png")
        if epoch == 0:
            save_image(x, folder + f"/input_{epoch}.png")
            
    gen.train()

def compare_example(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.float().to(config.DEVICE), y.float().to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        # Concatenate the true and fake images along the width dimension
        combined = torch.cat((y, y_fake), dim=3)
        # Save the combined images in a single image file
        save_image(combined, f"{folder}/combined_{epoch}.png", nrow=1)
        if epoch == 0:
            save_image(x, f"{folder}/input_{epoch}.png", nrow=1)
    gen.train()

def compare_diff(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.float().to(config.DEVICE), y.float().to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        # Ensure that y and y_fake are within [0, 1]
        y = torch.clamp(y, 0, 1)
        y_fake = torch.clamp(y_fake, 0, 1)

        # Compute the difference image
        # Red channel: Highlights incorrect predictions
        red_channel = torch.where(y == 1, 1 - y_fake, y_fake)
        # Green channel: Highlights correct white predictions
        green_channel = torch.where(y == 1, y_fake, torch.zeros_like(y_fake))
        # Blue channel: Remains zero
        blue_channel = torch.zeros_like(y_fake)
        # Stack channels to create an RGB image
        diff_image = torch.cat([red_channel, green_channel, blue_channel], dim=1)  # Shape: [batch, 3, H, W]

        # Convert y and y_fake to RGB by repeating the single channel
        y_rgb = y.repeat(1, 3, 1, 1)
        y_fake_rgb = y_fake.repeat(1, 3, 1, 1)

        # Concatenate true label, generated image, and difference image along the width
        combined = torch.cat([y_rgb, y_fake_rgb, diff_image], dim=3)

        # Save the combined images in a single image file
        save_image(combined, f"{folder}/combined_{epoch}.png", nrow=1)

        if epoch == 0:
            # Optionally, save the input images converted to RGB
            x_rgb = x.repeat(1, 3, 1, 1)
            save_image(x_rgb, f"{folder}/input_{epoch}.png", nrow=1)
    gen.train()


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])


def get_loaders(
        data_dir,
        x_dir,
        y_dir,
        batch_size,
        num_workers=4,
        pin_memory=True
        ):
    
    dataset = MyDataLoader(data_dir,x_dir,y_dir)
    train_set, test_set = torch.utils.data.random_split(dataset,[0.8,0.2])
    train_loader = DataLoader(train_set,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    test_loader = DataLoader(test_set,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=False)

    return train_loader,test_loader

def check_accuracy(loader, model, device='cuda'):
    num_correct = 0
    num_pixels = 0
    dice_score = 0

    model.eval()
    with torch.no_grad():
        for x,y in loader:
            #data = data.float().to(device=cf.DEVICE)
            #target = target.float().to(device=cf.DEVICE)
            x = x.float().squeeze(0).to(device)
            y = y.float().squeeze(0).to(device)

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
    print(
        f"Got {num_correct}/{num_pixels} withh acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice Score: {dice_score/len(loader)}")
    model.train()

