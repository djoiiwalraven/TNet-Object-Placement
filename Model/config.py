import torch
import os

first_state_dir = os.getcwd()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = '/mnt/d'
X_DIR = "data12/x"
Y_DIR = "data12/y"
OUTPUT_DIR = "../results"

IMAGE_SIZE = 12
CHANNELS_IMG = 1 # MAKE SURE MODEL IS COMPATIBLE WIHT GREYSCALE


LEARNING_RATE = 3e-4
BATCH_SIZE = 64 # 16?
NUM_WORKERS = 1 # 4?

L1_LAMBDA = 80 # 100
LAMBDA_GP = 10
NUM_EPOCHS = 50

LOAD_MODEL = False
SAVE_MODEL = False

CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"
