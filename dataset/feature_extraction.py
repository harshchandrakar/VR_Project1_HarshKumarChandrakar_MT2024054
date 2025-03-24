import cv2
import numpy as np
import os
import requests
import zipfile
from sklearn.model_selection import train_test_split
from pathlib import Path
import shutil
import urllib.parse
import time

class ImageFeatureExtractor:
    def __init__(self, dataset_url=None, classes=None, img_h=224, img_w=224, feature_type='hog', data_dir="./dataset"):
        self.DATASET_URL = dataset_url
        self.CLASSES_LIST = classes
        self.IMAGE_HEIGHT, self.IMAGE_WIDTH = img_h, img_w
        self.FEATURE_TYPE = feature_type
        self.data_dir = data_dir
        
    def download_dataset(self):
        if not self.DATASET_URL:
            print("No dataset URL provided. Using local data.")
            return
            
        if not os.path.exists(self.data_dir):
            print(f"Downloading dataset from {self.DATASET_URL}")
            os.makedirs(self.data_dir, exist_ok=True)
            
            try:
                parsed_url = urllib.parse.urlparse(self.DATASET_URL)
                
                if "github.com" in parsed_url.netloc:
                    self._download_from_github()
                elif "kaggle" in parsed_url.netloc:
                    print("Kaggle datasets require authentication. Please download manually.")    
            except Exception as e:
                print(f"Error downloading dataset: {e}")
        else:
            print(f"Using existing dataset at {self.data_dir}")
            
    def _download_from_github(self):
        repo_parts = self.DATASET_URL.split("github.com/")[1].split("/")
        username = repo_parts[0]
        repo_name = repo_parts[1]
        
        branch = "master"
        if len(repo_parts) > 2 and repo_parts[2] == "tree":
            branch = repo_parts[3]
            
        zip_url = f"https://codeload.github.com/{username}/{repo_name}/zip/{branch}"
        
        response = requests.get(zip_url, stream=True)
        if response.status_code == 200:
            zip_path = os.path.join(self.data_dir, f"{repo_name}.zip")
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            os.remove(zip_path)
            
            extracted_dir = os.path.join(self.data_dir, f"{repo_name}-{branch}")
            if os.path.exists(extracted_dir):
                dataset_dirs = ["dataset", "data", "images", "img"]
                for dir_name in dataset_dirs:
                    potential_dir = os.path.join(extracted_dir, dir_name)
                    if os.path.exists(potential_dir):
                        for item in os.listdir(potential_dir):
                            src = os.path.join(potential_dir, item)
                            dst = os.path.join(self.data_dir, item)
                            if os.path.isdir(src):
                                if os.path.exists(dst):
                                    shutil.rmtree(dst)
                                shutil.copytree(src, dst)
                            else:
                                shutil.copy2(src, dst)
                        break
                else:
                    for item in os.listdir(extracted_dir):
                        src = os.path.join(extracted_dir, item)
                        dst = os.path.join(self.data_dir, item)
                        if not os.path.basename(src).startswith('.') and src != zip_path:
                            if os.path.isdir(src):
                                if os.path.exists(dst):
                                    shutil.rmtree(dst)
                                shutil.copytree(src, dst)
                            else:
                                shutil.copy2(src, dst)
                shutil.rmtree(extracted_dir)
            print("Dataset downloaded and extracted successfully")
        else:
            print(f"Failed to download dataset: {response.status_code}")
    
    def extract_hog_features(self, image):
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        resized = cv2.resize(gray, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
        
        win_size = (self.IMAGE_WIDTH, self.IMAGE_HEIGHT)
        cell_size = (8, 8)
        block_size = (16, 16)
        block_stride = (8, 8)
        num_bins = 9
        
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
        hog_features = hog.compute(resized)
        
        return hog_features.flatten()
    
    def extract_lbp_features(self, image):
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        resized = cv2.resize(gray, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
        
        lbp = np.zeros_like(resized)
        for i in range(1, resized.shape[0] - 1):
            for j in range(1, resized.shape[1] - 1):
                center = resized[i, j]
                code = 0
                
                code |= (resized[i-1, j-1] >= center) << 7
                code |= (resized[i-1, j] >= center) << 6
                code |= (resized[i-1, j+1] >= center) << 5
                code |= (resized[i, j+1] >= center) << 4
                code |= (resized[i+1, j+1] >= center) << 3
                code |= (resized[i+1, j] >= center) << 2
                code |= (resized[i+1, j-1] >= center) << 1
                code |= (resized[i, j-1] >= center) << 0
                
                lbp[i, j] = code
        
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        
        return hist
    
    def extract_color_histogram(self, image):
        resized = cv2.resize(image, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
        
        if len(resized.shape) > 2:
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            
            h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256])
            
            h_hist = cv2.normalize(h_hist, h_hist).flatten()
            s_hist = cv2.normalize(s_hist, s_hist).flatten()
            v_hist = cv2.normalize(v_hist, v_hist).flatten()
            
            hist_features = np.concatenate([h_hist, s_hist, v_hist])
        else:
            hist = cv2.calcHist([resized], [0], None, [32], [0, 256])
            hist_features = cv2.normalize(hist, hist).flatten()
        
        return hist_features
    
    def extract_features(self, image):
        if self.FEATURE_TYPE == 'hog':
            return self.extract_hog_features(image)
        elif self.FEATURE_TYPE == 'lbp':
            return self.extract_lbp_features(image)
        elif self.FEATURE_TYPE == 'color_hist':
            return self.extract_color_histogram(image)
        elif self.FEATURE_TYPE == 'combined':
            hog = self.extract_hog_features(image)
            lbp = self.extract_lbp_features(image)
            color_hist = self.extract_color_histogram(image)
            return np.concatenate([hog, lbp, color_hist])
        else:
            return self.extract_hog_features(image)
    
    def discover_classes(self):
        if not os.path.exists(self.data_dir):
            print(f"Data directory {self.data_dir} does not exist")
            return []
        
        classes = []
        for item in os.listdir(self.data_dir):
            item_path = os.path.join(self.data_dir, item)
            if os.path.isdir(item_path):
                has_images = False
                for file in os.listdir(item_path):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        has_images = True
                        break
                
                if has_images:
                    classes.append(item)
        
        return classes
    
    def create_dataset(self):
        features = []
        labels = []
        image_paths = []
        
        if self.DATASET_URL:
            self.download_dataset()
        
        if not self.CLASSES_LIST:
            self.CLASSES_LIST = self.discover_classes()
            print(f"Discovered classes: {self.CLASSES_LIST}")
        
        if not self.CLASSES_LIST:
            print("No classes specified or discovered.")
            return np.array([]), np.array([]), []
        
        for class_index, class_name in enumerate(self.CLASSES_LIST):
            print(f'Extracting features for class: {class_name}')
            
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Directory not found: {class_dir}")
                continue
            
            class_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            total_files = len(class_files)
            processed = 0
            start_time = time.time()
            
            for img_file in class_files:
                img_path = os.path.join(class_dir, img_file)
                
                try:
                    image = cv2.imread(img_path)
                    
                    if image is None:
                        print(f"Failed to read image: {img_path}")
                        continue
                    
                    image_features = self.extract_features(image)
                    
                    features.append(image_features)
                    labels.append(class_index)
                    image_paths.append(img_path)
                    
                    processed += 1
                    if processed % 10 == 0 or processed == total_files:
                        elapsed = time.time() - start_time
                        rate = processed / elapsed if elapsed > 0 else 0
                        print(f"  Processed {processed}/{total_files} images ({rate:.2f} img/s)")
                        
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        if features:
            features = np.array(features)
            labels = np.array(labels)
            
            print(f"Dataset created with {len(features)} samples, {len(self.CLASSES_LIST)} classes")
            print(f"Feature vector dimension: {features.shape[1]}")
        else:
            print("No features extracted. Check your dataset and classes.")
            features = np.array([])
            labels = np.array([])
        
        return features, labels, image_paths
    
    def extract_and_save_data(self, output_dir="../extracted_data"):
        print("Extracting and saving data...")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        features, labels, paths = self.create_dataset()
        
        if len(features) == 0:
            print("No features extracted. Skipping save.")
            return features, labels, paths
        
        np.save(os.path.join(output_dir, f"features_{self.FEATURE_TYPE}.npy"), features)
        np.save(os.path.join(output_dir, "labels.npy"), labels)
        
        with open(os.path.join(output_dir, "image_paths.txt"), 'w') as f:
            for path in paths:
                f.write(f"{path}\n")
        
        with open(os.path.join(output_dir, "metadata.txt"), 'w') as f:
            f.write(f"Feature type: {self.FEATURE_TYPE}\n")
            f.write(f"Image dimensions: {self.IMAGE_HEIGHT}x{self.IMAGE_WIDTH}\n")
            f.write(f"Classes: {', '.join(self.CLASSES_LIST)}\n")
            f.write(f"Samples per class: {[list(labels).count(i) for i in range(len(self.CLASSES_LIST))]}\n")
            f.write(f"Feature dimension: {features.shape[1]}\n")
        
        print(f"Data extraction complete. Features shape: {features.shape}, Labels shape: {labels.shape}")
        
        return features, labels, paths
    
    def load_data(self, output_dir="../extracted_data"):
        features_path = os.path.join(output_dir, f"features_{self.FEATURE_TYPE}.npy")
        labels_path = os.path.join(output_dir, "labels.npy")
        paths_path = os.path.join(output_dir, "image_paths.txt")
        
        if os.path.exists(features_path) and os.path.exists(labels_path):
            features = np.load(features_path)
            labels = np.load(labels_path)
            
            paths = []
            if os.path.exists(paths_path):
                with open(paths_path, 'r') as f:
                    paths = [line.strip() for line in f.readlines()]
            
            print(f"Data loaded. Features shape: {features.shape}, Labels shape: {labels.shape}")
            
            return features, labels, paths
        else:
            print(f"No saved data found for feature type '{self.FEATURE_TYPE}'. Extract data first.")
            return None, None, None
    
    def split_data(self, test_size=0.2, random_state=42,output_dir="../extracted_data"):
        features, labels, _ = self.load_data(output_dir)
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        print(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")
        
        return X_train, X_test, y_train, y_test