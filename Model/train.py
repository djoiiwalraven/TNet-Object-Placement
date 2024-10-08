# IMPORT DATA
import torch
from torch import nn
from torch import optim

import config as cf
import datasets.data12.config as d12cf
import datasets.mapping.config as mcf

from datasets.data12.data_loader import D12DataLoader
from datasets.mapping.data_loader import MappingDataLoader

from torch.utils.data import DataLoader

from Unet.Unet import Unet
from Tnet.Tnet import Tnet
from Tnet.loss_functions import GDiceLoss, DiceLoss, FocalLossMultiClass, GIoULoss, CompositeLoss
#from Tnet.dice_loss import composite_loss

from tqdm import tqdm

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
            #loss = loss_fn(predictions,target,bce_weight=.3, dice_weight=.7) #special for d12
            loss = loss_fn(predictions,target)

        loss.backward()

        #optional grad clipping
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        #scaler.scale(loss).backward()
        #scaler.step(optimizer)
        #scaler.update()

        #optimizer.zero_grad()

        loop.set_postfix(loss=loss.item())


def double_train_fn(loader,model,optimizer1,optimizer2,loss_fn,epoch):
    loop = tqdm(loader)

    for batch_idx, (data,target) in enumerate(loop):
        data = data.float().to(device=cf.DEVICE)
        target = target.float().to(device=cf.DEVICE)

        if epoch % 2 == 0:
            optimizer1.zero_grad()
        else:
            optimizer2.zero_grad()

        with torch.amp.autocast(cf.DEVICE):
            predictions = model(data)
            loss = loss_fn(predictions,target)
        
        loss.backward()

        if epoch % 2 == 0:
            optimizer1.step()
        else:
            optimizer2.step()
        loop.set_postfix(loss=loss.item())

def main():
    # data12 model
    #model = Tnet(in_channels=1,out_channels=1,feature_length=2).to(cf.DEVICE)
    
    # mapping model
    model = Tnet(in_channels=3,out_channels=3,init_features=64,feature_length=4).to(cf.DEVICE)

    #loss_fn = composite_loss # loss for binary 
    #loss_fn = nn.MSELoss() # Binary Cross Entropy

    l1 = nn.SmoothL1Loss(beta=1.0)
    l2 = nn.MSELoss()
    loss_fn = CompositeLoss(l1,l2,.7)

    optimizer = optim.Adam(model.parameters(),lr=mcf.LEARNING_RATE)
    #optimizer = optim.SGD(model.parameters(),lr=mcf.LEARNING_RATE, momentum=.7)
    #optimizer = torch.optim.Adam(model.parameters(), lr=mcf.LEARNING_RATE, weight_decay=1e-5)
    scaler = torch.amp.GradScaler(cf.DEVICE)

    # Need fix but is for data12 model
    #train_loader, _ = get_loaders(d12cf.DATA_DIR,d12cf.X_DIR,d12cf.Y_DIR,d12cf.BATCH_SIZE)
    #_, test_loader = get_loaders(d12cf.DATA_DIR,d12cf.X_DIR,d12cf.Y_DIR,d12cf.BATCH_SIZE//4)

    # Mapping Model Datasources
    train_set = MappingDataLoader(mcf.DATA_DIR,mcf.TRAIN_DIR,mcf.IMAGE_SIZE)
    train_loader = DataLoader(train_set,batch_size=mcf.BATCH_SIZE,num_workers=mcf.NUM_WORKERS,pin_memory=cf.PIN_MEMORY,shuffle=True)

    test_set = MappingDataLoader(mcf.DATA_DIR,mcf.TEST_DIR,mcf.IMAGE_SIZE)
    test_loader = DataLoader(test_set,batch_size=mcf.BATCH_SIZE,num_workers=mcf.NUM_WORKERS,pin_memory=cf.PIN_MEMORY,shuffle=True)


    for epoch in range(d12cf.NUM_EPOCHS):
        train_fn(train_loader,model,optimizer,loss_fn,scaler)
        

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }

        #save_checkpoint(checkpoint)
        check_accuracy(test_loader, model, device=cf.DEVICE)
        #compare_diff(model,test_loader,epoch,d12cf.OUTPUT_DIR)
        save_example(model,test_loader,epoch,mcf.OUTPUT_DIR)

        # check accuracy
        # print some examples

if __name__ == "__main__":
    main()