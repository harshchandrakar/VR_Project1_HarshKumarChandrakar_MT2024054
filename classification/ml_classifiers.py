import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.feature_extraction import ImageFeatureExtractor
from utils.variables import *


create_data = ImageFeatureExtractor(DATASET_URL_1,CLASSES_1,IMAGE_HEIGHT,IMAGE_WIDTH,"combined","../data")

create_data.extract_and_save_data(output_dir="../extracted_data")

