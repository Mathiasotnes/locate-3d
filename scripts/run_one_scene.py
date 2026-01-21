###############################################
# run_one_scene.py
# ----------------
# Script to preprocess and cache a single scene from ScanNet and run the Locate3D model on it.
# This is meant to test the cluster setup. Note that the scene from ARKitScenes must be downloaded.
# Usage:
#   git clone https://github.com/apple/ARKitScenes.git
#   cd ARKitScenes
#   python3 download_data.py 3dod --split Training --video_id 42445211 --download_dir /data/arkitscenes
#   cd ../locate-3d
#   python3 scripts/run_one_scene.py
# ---------------------
# Author: Mathias Otnes
###############################################

import subprocess
import sys


###############################################
# Config

ARKIT_DIR       = "/data/arkitscenes"
ANNOTATIONS     = "locate3d_data/dataset/train_arkitscenes.json"
CACHE_PATH      = "cache"
SCENE_INDEX     = 0 # Corresponds to video_id "42445211" in ARKitScenes


###############################################
# Main script

if __name__ == "__main__":
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
