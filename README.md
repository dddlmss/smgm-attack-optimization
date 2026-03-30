## Setup

1. Clone the repository

git clone https://github.com/dddlmss/smgm-attack-optimization.git  
cd smgm-attack-optimization

2. Create the conda environment

conda env create -f environment.yml  
conda activate smgm

3. Download YOLO weights

The model weights are not included in this repository due to size limitations.

Download here:  
https://drive.google.com/file/d/1Zm3Lo9eKm_-qy5GPjABgGvE2kVokwy2M/view?usp=share_link

After downloading, place the file in the project root:

smgm-attack-optimization/yolo4_weight.h5

4. Run the code

python test.py


Notes:
- Model weights (.h5, .pt, .pth, .onnx) are excluded
- Generated outputs (output/, test/result/) are excluded
- Large datasets (data/, datasets/) are excluded


Project Structure:
- test.py → main entry point (attack / detect)
- environment.yml → conda environment
- yolo4_weight.h5 → must be downloaded separately
- model_data/ → class labels and anchors

