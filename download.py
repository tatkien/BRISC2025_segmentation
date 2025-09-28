import kagglehub
import shutil
# Download latest version
path = kagglehub.dataset_download("briscdataset/brisc2025")
# Move to current directory
shutil.move(path, "./dataset/")
DATA_PATH = "./dataset/5/brisc2025/segmentation_task/"
print(f"Data downloaded to {DATA_PATH}")