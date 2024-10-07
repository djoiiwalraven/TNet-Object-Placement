# IMPORT DATA
import torch
from torch import nn
from torch import optim
import config as cf
import Unet.config as ucf
from data_loader import MyDataLoader
from Unet.Unet import Unet
from Tnet.Tnet import Tnet
from tqdm import tqdm
from Tnet.dice_loss import composite_loss


from utils import (
    load_checkpoint,
    save_checkpoint,
    save_example,
    compare_example,
    compare_diff,
    get_loaders,
    check_accuracy
)


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data,target) in enumerate(loop):
        data = data.float().to(device=cf.DEVICE)
        target = target.float().to(device=cf.DEVICE)


        optimizer.zero_grad()

        # forward
        with torch.amp.autocast(cf.DEVICE):
            predictions = model(data)
            #predictions = (predictions >= 0.5).int().float()
            #print(predictions)
            #quit()
            loss = loss_fn(predictions,target,bce_weight=.3, dice_weight=.7)

        loss.backward()

        #optional grad clipping
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        #scaler.scale(loss).backward()
        #scaler.step(optimizer)
        #scaler.update()

        #optimizer.zero_grad()

        loop.set_postfix(loss=loss.item())


def main():
    model = Tnet(in_channels=1,out_channels=1,feature_length=2).to(cf.DEVICE)
    
    loss_fn = composite_loss
    #loss_fn = nn.MSELoss() # Binary Cross Entropy

    #optimizer = optim.Adam(model.parameters(),lr=cf.LEARNING_RATE)
    #optimizer = optim.SGD(model.parameters(),lr=cf.LEARNING_RATE, momentum=.7)
    optimizer = torch.optim.Adam(model.parameters(), lr=cf.LEARNING_RATE, weight_decay=1e-5)
    scaler = torch.amp.GradScaler(cf.DEVICE)

    train_loader, _ = get_loaders(cf.DATA_DIR,cf.X_DIR,cf.Y_DIR,ucf.BATCH_SIZE)
    _, test_loader = get_loaders(cf.DATA_DIR,cf.X_DIR,cf.Y_DIR,ucf.BATCH_SIZE//4)

    for epoch in range(ucf.NUM_EPOCHS):
        train_fn(train_loader,model,optimizer,loss_fn,scaler)
        

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }

        #save_checkpoint(checkpoint)
        check_accuracy(test_loader, model, device=cf.DEVICE)
        compare_diff(model,test_loader,epoch,'../results')


        # check accuracy
        # print some examples

if __name__ == "__main__":
    main()