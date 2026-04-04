# config.py
import os

# -----------------------------
# Model / YOLO settings
# -----------------------------
MODEL_PATH = 'yolo4_weight.h5'
ANCHORS_PATH = 'model_data/yolo4_anchors.txt'
CLASSES_PATH = 'model_data/coco_classes.txt'

SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
MODEL_IMAGE_SIZE = (608, 608)

# -----------------------------
# Run mode
# -----------------------------
MODE = "detect"              # "attack" or "detect"
ATTACK_NAME = "PSO"          # "GA", "PSO", "DE", "SMGM"
ATTACK_TYPE = "untargeted"   # "untargeted" or "targeted"

# -----------------------------
# Input / output paths
# -----------------------------
INPUT_PATH = r"test\original\*.jpg"

BASE_RESULTS_DIR = "results"
ATTACK_OUTPUT_DIR = os.path.join(BASE_RESULTS_DIR, ATTACK_NAME.lower(), ATTACK_TYPE, "attacked")
DETECT_OUTPUT_DIR = os.path.join(BASE_RESULTS_DIR, ATTACK_NAME.lower(), ATTACK_TYPE, "detected")
LOG_DIR = os.path.join(BASE_RESULTS_DIR, ATTACK_NAME.lower(), ATTACK_TYPE, "logs")
CSV_PATH = os.path.join(LOG_DIR, "per_image_results.csv")

# -----------------------------
# Shared metaheuristic settings
# -----------------------------
TOP_K = 5
GRID_SIZE = 4
EPS = 0.08

# -----------------------------
# GA settings
# -----------------------------
GA_POP_SIZE = 12
GA_GENERATIONS = 15
GA_ELITE_SIZE = 2
GA_MUTATION_RATE = 0.2

# -----------------------------
# PSO settings
# -----------------------------
PSO_SWARM_SIZE = 20
PSO_ITERATIONS = 30
PSO_W = 0.9
PSO_C1 = 2.5
PSO_C2 = 0.5

# -----------------------------
# DE settings
# -----------------------------
DE_POP_SIZE = 12
DE_GENERATIONS = 15
DE_F = 0.5
DE_CR = 0.9