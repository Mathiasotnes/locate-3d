###############################################
# run_one_scene.py
# ----------------
# Script to preprocess and cache a single scene from ScanNet and run the Locate3D model on it.
# This is meant to test the cluster setup. Note that the scene from ARKitScenes must be downloaded.
# Usage:
#   git clone https://github.com/apple/ARKitScenes.git
#   cd ARKitScenes
#   python download_data.py 3dod --split Training --video_id 42445211 --download_dir /home/motnes/work/data/arkitscenes
#   cd ../locate-3d
#   python3 scripts/run_one_scene.py
# ---------------------
# Author: Mathias Otnes
###############################################

import subprocess
import sys

from locate3d_data.locate3d_dataset import Locate3DDataset
from models.locate_3d import Locate3D, downsample


###############################################
# Config

ARKIT_DIR       = "../data/arkitscenes"
ANNOTATIONS     = "locate3d_data/dataset/train_arkitscenes.json"
CACHE_PATH      = "cache"
SCENE_INDEX     = 0 # Corresponds to video_id "42445211" in ARKitScenes


###############################################
# Main script

if __name__ == "__main__":
    
    # Preprocessing
    cmd = [
        sys.executable, "-m", "preprocessing.run_preprocessing",
        "--l3dd_annotations_fpath", ANNOTATIONS,
        "--arkitscenes_data_dir", ARKIT_DIR,
        "--cache_path", CACHE_PATH,
        "--start", str(SCENE_INDEX),
        "--end", str(SCENE_INDEX + 1),
    ]

    print(" ".join(cmd))
    completed_process = subprocess.run(cmd, check=True)
    print(f"Process finished with return code {completed_process.returncode}")
    
    # Locate3D inference
    dataset = Locate3DDataset(
        annotations_fpath="locate3d_data/dataset/train_arkitscenes.json",
        return_featurized_pointcloud=True,
        arkitscenes_data_dir=ARKIT_DIR,
    )

    model = Locate3D.from_pretrained("facebook/locate-3d")
    data = dataset[0]

    # Downsample pointcloud (optional) (test with first)
    data["featurized_sensor_pointcloud"] = downsample(
        data["featurized_sensor_pointcloud"], 30000
    )

    output = model.inference(
        data["featurized_sensor_pointcloud"], data["lang_data"]["text_caption"]
    )
    
    print("Model output:", output)
