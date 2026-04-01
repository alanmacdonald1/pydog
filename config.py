import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Paths
# ----------------------------
DATA_DIR = "data/dogs"
IMAGE_PATH = "test_data/img.png"

# ----------------------------
# Training
# ----------------------------
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.01

# ----------------------------
# System
# ----------------------------
NUM_WORKERS = 0  # keep 0 to avoid multiprocessing bug
PIN_MEMORY = True

# ----------------------------
# Model
# ----------------------------
NUM_CLASSES = 2
