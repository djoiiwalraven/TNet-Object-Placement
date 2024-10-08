import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = False

CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"
