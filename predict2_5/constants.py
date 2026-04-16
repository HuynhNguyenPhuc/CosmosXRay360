"""Constants"""

# ============================================================
# Dataset Paths
# ============================================================

DEFAULT_CT_DIRS = [
    "datasets/NSCLC/processed/train/images/",
    "datasets/MELA2022/raw/train/images/",
    "datasets/MELA2022/raw/val/images/",
    "datasets/TCIA/images/",
]

DEFAULT_XR_DIR = "data/VinDr/v1/processed/"

DEFAULT_EMBEDDING_DIR = "cosmos_reason_embeddings"

# ============================================================
# Cosmos-Reason1 Text Encoder
# ============================================================

CR1_HIDDEN_SIZE = 3584
CR1_NUM_LAYERS = 28
CR1_MAX_LENGTH = 512
CR1_FULL_CONCAT_DIM = CR1_HIDDEN_SIZE * CR1_NUM_LAYERS
CR1_EMBEDDING_DIM = CR1_FULL_CONCAT_DIM

CROSSATTN_EMB_CHANNELS = 1024
CROSSATTN_PROJ_IN_CHANNELS = CR1_FULL_CONCAT_DIM
CROSSATTN_PROJ_OUT_CHANNELS = 1024

# ============================================================
# Volume / Image Configuration
# ============================================================

VOL_SIZE = 256
IMG_HEIGHT = VOL_SIZE
IMG_WIDTH = VOL_SIZE

# ============================================================
# Frame Configuration
# ============================================================

NUM_FRAMES = 93
NUM_LATENT_FRAMES = 24  # 1 + (93-1)//4 = 24

FRAME_HEIGHT = IMG_HEIGHT
FRAME_WIDTH = IMG_WIDTH

# ============================================================
# Checkpoint UUIDs
# ============================================================

COSMOS_TOKENIZER_UUID = "685afcaa-4de2-42fe-b7b9-69f7a2dee4d8"
COSMOS_2B_PRETRAINED_UUID = "d20b7120-df3e-4911-919d-db6e08bad31c"
COSMOS_REASON_UUID = "cb3e3ffa-7b08-4c34-822d-61c7aa31a14f"
T5_11B_UUID = "4dbf13c6-1d30-4b02-99d6-75780dd8b744"

# ============================================================
# Prompts
# ============================================================

PROMPTS = [(
    "A 360-degree rotational view of a chest CT scan showing anatomical "
    "structures from all angles, rotating from 0 to 360 degrees azimuth."
)]
