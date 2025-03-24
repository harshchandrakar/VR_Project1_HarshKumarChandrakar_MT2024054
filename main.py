from utils.variables import *
from dataset.data_download import download_from_gdrive
from classification.cnn_classifier import main as task2
from classification.ml_classifiers import main as task1
from segmentation.evaluate_model import main as task4
from segmentation.traditional_segmentation import main as task3
import os


def main(download=False):
    # Create necessary directories if they don't exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("extracted_data", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    if download:
        download_from_gdrive(DATASET_URL_2)

    print("-----------------------Evaluating Task1------------------------")
    # Using correct path relative to project root and setting extract=True to ensure data is available
    task1(extract=False, train=False, output_dir="./extracted_data")

    print("-----------------------Evaluating Task2------------------------")
    # Make sure task2 runs with appropriate training flag
    task2(train=False,data_dir="./data")  # Set to False if models are already trained and you just want to evaluate

    print("-----------------------Evaluating Task3------------------------")
    task3(100, download=False, verbose=False)

    print("-----------------------Evaluating Task4------------------------")
    task4()


if __name__ == '__main__':
    main(download=True)  # Set to True if you need to download data