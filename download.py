from kaggle.api.kaggle_api_extended import KaggleApi
import os

api = KaggleApi()
api.authenticate()

datasets = [
    "alinadilawaiz/dangerous-objects-dataset",
    "trainingdatapro/people-with-guns-segmentation-and-detection"
]

for dataset in datasets:
    download_path = f"./{dataset.split('/')[-1]}"
    os.makedirs(download_path, exist_ok=True)
    print(f"Downloading {dataset} to {download_path} ...")
    api.dataset_download_files(dataset, path=download_path, unzip=True)
    print("Done! Files:", os.listdir(download_path))
