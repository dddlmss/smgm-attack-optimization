import os
import glob
from PIL import Image

from config import (
    MODEL_PATH, ANCHORS_PATH, CLASSES_PATH,
    SCORE_THRESHOLD, IOU_THRESHOLD, MODEL_IMAGE_SIZE,
    MODE, ATTACK_NAME, ATTACK_TYPE, INPUT_PATH,
    ATTACK_OUTPUT_DIR, DETECT_OUTPUT_DIR, CSV_PATH,
    TOP_K, GRID_SIZE, EPS,
    GA_POP_SIZE, GA_GENERATIONS, GA_ELITE_SIZE, GA_MUTATION_RATE,
    PSO_SWARM_SIZE, PSO_ITERATIONS, PSO_W, PSO_C1, PSO_C2,
    DE_POP_SIZE, DE_GENERATIONS, DE_F, DE_CR
)

from yolo_attack import Yolo4
from results_logger import ensure_dir

GRADIENT_ATTACKS = ["I-FGSM", "SMGM"]
META_ATTACKS = ["GA", "PSO", "DE"]

def main():
    ensure_dir(ATTACK_OUTPUT_DIR)
    ensure_dir(DETECT_OUTPUT_DIR)
    ensure_dir(os.path.dirname(CSV_PATH))

    yolo4_model = Yolo4(
        score=SCORE_THRESHOLD,
        iou=IOU_THRESHOLD,
        anchors_path=ANCHORS_PATH,
        classes_path=CLASSES_PATH,
        model_path=MODEL_PATH
    )

    ga_params = {
        "pop_size": GA_POP_SIZE,
        "generations": GA_GENERATIONS,
        "elite_size": GA_ELITE_SIZE,
        "mutation_rate": GA_MUTATION_RATE,
        "grid_size": GRID_SIZE,
        "eps": EPS
    }

    pso_params = {
        "swarm_size": PSO_SWARM_SIZE,
        "iterations": PSO_ITERATIONS,
        "grid_size": GRID_SIZE,
        "eps": EPS,
        "w_inertia": PSO_W,
        "c1": PSO_C1,
        "c2": PSO_C2
    }

    de_params = {
        "pop_size": DE_POP_SIZE,
        "generations": DE_GENERATIONS,
        "grid_size": GRID_SIZE,
        "eps": EPS,
        "F": DE_F,
        "CR": DE_CR
    }

    if MODE == "attack":
        print(f"[INFO] Attack method: {ATTACK_NAME}")
        print(f"[INFO] Attack type: {ATTACK_TYPE}")
        print(f"[INFO] Saving attacked images to: {ATTACK_OUTPUT_DIR}")
        print(f"[INFO] Saving logs to: {CSV_PATH}")

        for idx, jpgfile in enumerate(glob.glob(INPUT_PATH), start=1):
            print(f"[ATTACK] {jpgfile}")
            img = Image.open(jpgfile)

            if ATTACK_NAME in GRADIENT_ATTACKS:
                yolo4_model.attack_with_gradient(
                    image=img,
                    attack_name=ATTACK_NAME,
                    attack_type=ATTACK_TYPE,
                    jpgfile=jpgfile,
                    attack_output_dir=ATTACK_OUTPUT_DIR,
                    csv_path=CSV_PATH,
                    model_image_size=MODEL_IMAGE_SIZE
                )

            elif ATTACK_NAME in META_ATTACKS:
                yolo4_model.attack_with_metaheuristic(
                    image=img,
                    attack_name=ATTACK_NAME,
                    attack_type=ATTACK_TYPE,
                    jpgfile=jpgfile,
                    attack_output_dir=ATTACK_OUTPUT_DIR,
                    csv_path=CSV_PATH,
                    top_k=TOP_K,
                    grid_size=GRID_SIZE,
                    eps=EPS,
                    ga_params=ga_params,
                    pso_params=pso_params,
                    de_params=de_params,
                    model_image_size=MODEL_IMAGE_SIZE
                )

            else:
                raise ValueError("Unsupported ATTACK_NAME")

            print(f"[DONE] {idx} images processed.")

    elif MODE == "detect":
        detect_input_path = os.path.join(ATTACK_OUTPUT_DIR, "*.jpg")

        print(f"[INFO] Reading attacked images from: {detect_input_path}")
        print(f"[INFO] Saving detected images to: {DETECT_OUTPUT_DIR}")

        for jpgfile in glob.glob(detect_input_path):
            print(f"[DETECT] {jpgfile}")
            img = Image.open(jpgfile)
            result_img = yolo4_model.detect_image(img)
            result_img.save(os.path.join(DETECT_OUTPUT_DIR, os.path.basename(jpgfile)))

    else:
        print("MODE must be 'attack' or 'detect'")

    yolo4_model.close_session()

if __name__ == "__main__":
    main()