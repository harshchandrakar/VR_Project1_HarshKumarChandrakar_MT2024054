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
        """
        Initialize the generalized image feature extraction class.
        
        Args:
            dataset_url: URL to download the dataset from (optional)
            classes: List of classes to extract (e.g., ['class1', 'class2'])
            img_h: Image height for resizing
            img_w: Image width for resizing
            feature_type: Type of features to extract ('hog', 'lbp', 'haralick', 'combined')
            data_dir: Directory to store/access the dataset
        """
        self.DATASET_URL = dataset_url
        self.CLASSES_LIST = classes
        self.IMAGE_HEIGHT, self.IMAGE_WIDTH = img_h, img_w
        self.FEATURE_TYPE = feature_type
        self.data_dir = data_dir
        
    def download_dataset(self):
        """
        Download the dataset if it doesn't exist locally and a URL is provided.
        """
        if not self.DATASET_URL:
            print("No dataset URL provided. Using local data.")
            return
            
        if not os.path.exists(self.data_dir):
            print(f"Downloading dataset from {self.DATASET_URL}")
            
            # Create directory for dataset
            os.makedirs(self.data_dir, exist_ok=True)
            
            try:
                # Parse URL to determine how to handle it
                parsed_url = urllib.parse.urlparse(self.DATASET_URL)
                
                # Handle GitHub repositories
                if "github.com" in parsed_url.netloc:
                    self._download_from_github()
                # Handle Kaggle datasets
                elif "kaggle" in parsed_url.netloc:
                    print("Kaggle datasets require authentication. Please download manually and place in the data directory.")    
                else:
                    self._download_direct()
                    
            except Exception as e:
                print(f"Error downloading dataset: {e}")
        else:
            print(f"Using existing dataset at {self.data_dir}")
            
    def _download_from_github(self):
        """
        Helper method to download datasets from GitHub.
        """
        # For GitHub repositories, we need to download the zip file
        # Convert github.com URL to codeload.github.com format
        repo_parts = self.DATASET_URL.split("github.com/")[1].split("/")
        username = repo_parts[0]
        repo_name = repo_parts[1]
        
        branch = "master"  # Default branch
        if len(repo_parts) > 2 and repo_parts[2] == "tree":
            branch = repo_parts[3]
            
        zip_url = f"https://codeload.github.com/{username}/{repo_name}/zip/{branch}"
        
        # Download zip file
        response = requests.get(zip_url, stream=True)
        if response.status_code == 200:
            zip_path = os.path.join(self.data_dir, f"{repo_name}.zip")
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            # Remove zip file
            os.remove(zip_path)
            
            # Try to find and organize the dataset directory
            extracted_dir = os.path.join(self.data_dir, f"{repo_name}-{branch}")
            if os.path.exists(extracted_dir):
                # Look for common dataset directory names
                dataset_dirs = ["dataset", "data", "images", "img"]
                for dir_name in dataset_dirs:
                    potential_dir = os.path.join(extracted_dir, dir_name)
                    if os.path.exists(potential_dir):
                        # Move files from extracted dataset directory to our data directory
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
                    # If no specific dataset directory found, copy all contents
                    for item in os.listdir(extracted_dir):
                        src = os.path.join(extracted_dir, item)
                        dst = os.path.join(self.data_dir, item)
                        if not os.path.basename(src).startswith('.') and src != zip_path:  # Skip hidden files
                            if os.path.isdir(src):
                                if os.path.exists(dst):
                                    shutil.rmtree(dst)
                                shutil.copytree(src, dst)
                            else:
                                shutil.copy2(src, dst)
                shutil.rmtree(extracted_dir)
                print(f"Removed extracted directory: {extracted_dir}")
            
            print("Dataset downloaded and extracted successfully")
        else:
            print(f"Failed to download dataset: {response.status_code}")
    
    def _download_direct(self):
        """
        Helper method to download datasets from direct links.
        """
        response = requests.get(self.DATASET_URL, stream=True)
        if response.status_code == 200:
            content_type = response.headers.get('content-type', '')
            filename = os.path.basename(urllib.parse.urlparse(self.DATASET_URL).path) or "dataset.download"
            
            # Determine file type and handle accordingly
            if 'zip' in content_type or filename.endswith('.zip'):
                # Download and extract zip file
                zip_path = os.path.join(self.data_dir, filename)
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Extract the zip file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir)
                
                # Remove the zip file
                os.remove(zip_path)
                print(f"Downloaded and extracted {filename}")
            else:
                # Direct download of a file
                file_path = os.path.join(self.data_dir, filename)
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Downloaded {filename}")
        else:
            print(f"Failed to download dataset: {response.status_code}")
    
    def extract_hog_features(self, image):
        """
        Extract HOG (Histogram of Oriented Gradients) features from an image.
        """
        # Convert image to grayscale
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Resize image
        resized = cv2.resize(gray, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
        
        # Extract HOG features
        win_size = (self.IMAGE_WIDTH, self.IMAGE_HEIGHT)
        cell_size = (8, 8)
        block_size = (16, 16)
        block_stride = (8, 8)
        num_bins = 9
        
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
        hog_features = hog.compute(resized)
        
        return hog_features.flatten()
    
    def extract_lbp_features(self, image):
        """
        Extract LBP (Local Binary Pattern) features from an image. for extracting patterns
        """
        # Convert image to grayscale
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Resize image
        resized = cv2.resize(gray, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
        
        # Compute LBP using a simple implementation
        lbp = np.zeros_like(resized)
        for i in range(1, resized.shape[0] - 1):
            for j in range(1, resized.shape[1] - 1):
                center = resized[i, j]
                code = 0
                
                # Compare with 8 neighbors
                code |= (resized[i-1, j-1] >= center) << 7
                code |= (resized[i-1, j] >= center) << 6
                code |= (resized[i-1, j+1] >= center) << 5
                code |= (resized[i, j+1] >= center) << 4
                code |= (resized[i+1, j+1] >= center) << 3
                code |= (resized[i+1, j] >= center) << 2
                code |= (resized[i+1, j-1] >= center) << 1
                code |= (resized[i, j-1] >= center) << 0
                
                lbp[i, j] = code
        
        # Compute histogram of LBP values
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        
        # Normalize histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        
        return hist
    
    def extract_haralick_features(self, image):
        """
        Extract Haralick texture features from an image.
        """
        # Convert image to grayscale
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Resize image
        resized = cv2.resize(gray, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
        
        # Reduce grayscale levels to 8 to speed up GLCM calculation
        resized = (resized // 32).astype(np.uint8)
        levels = 8
        
        # Calculate GLCM (Gray-Level Co-occurrence Matrix) for multiple directions
        glcm = np.zeros((levels, levels, 4), dtype=np.float32)
        directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]  # 0°, 90°, 45°, 135°
        
        h, w = resized.shape
        for di, (dx, dy) in enumerate(directions):
            for i in range(h):
                for j in range(w):
                    if 0 <= i + dx < h and 0 <= j + dy < w:
                        glcm[resized[i, j], resized[i+dx, j+dy], di] += 1
        
        # Normalize GLCM
        for di in range(4):
            glcm[:, :, di] /= glcm[:, :, di].sum() + 1e-7
        
        # Calculate Haralick features for each direction
        features = []
        
        for di in range(4):
            # Contrast
            contrast = np.sum(np.outer(np.arange(levels), np.arange(levels)) * glcm[:, :, di])
            
            # Energy
            energy = np.sum(glcm[:, :, di] ** 2)
            
            # Homogeneity
            i_indices, j_indices = np.meshgrid(np.arange(levels), np.arange(levels), indexing='ij')
            homogeneity = np.sum(glcm[:, :, di] / (1 + (i_indices - j_indices) ** 2))
            
            # Add features
            features.extend([contrast, energy, homogeneity])
        
        return np.array(features)
    
    def extract_color_histogram(self, image):
        """
        Extract color histogram features from an image.
        """
        # Resize image
        resized = cv2.resize(image, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
        
        # Convert to HSV color space (better for color representation)
        if len(resized.shape) > 2:
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            
            # Calculate histogram for each channel
            h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256])
            
            # Normalize histograms
            h_hist = cv2.normalize(h_hist, h_hist).flatten()
            s_hist = cv2.normalize(s_hist, s_hist).flatten()
            v_hist = cv2.normalize(v_hist, v_hist).flatten()
            
            # Concatenate histograms
            hist_features = np.concatenate([h_hist, s_hist, v_hist])
        else:
            # Grayscale image
            hist = cv2.calcHist([resized], [0], None, [32], [0, 256])
            hist_features = cv2.normalize(hist, hist).flatten()
        
        return hist_features
    
    def extract_features(self, image):
        """
        Extract features from an image based on the specified feature type.
        """
        if self.FEATURE_TYPE == 'hog':
            return self.extract_hog_features(image)
        elif self.FEATURE_TYPE == 'lbp':
            return self.extract_lbp_features(image)
        elif self.FEATURE_TYPE == 'haralick':
            return self.extract_haralick_features(image)
        elif self.FEATURE_TYPE == 'color_hist':
            return self.extract_color_histogram(image)
        elif self.FEATURE_TYPE == 'combined':
            # Combine multiple feature types
            hog = self.extract_hog_features(image)
            lbp = self.extract_lbp_features(image)
            color_hist = self.extract_color_histogram(image)
            
            # Return combined features
            return np.concatenate([hog, lbp, color_hist])
        else:
            # Default to HOG if feature type is not recognized
            return self.extract_hog_features(image)
    
    def discover_classes(self):
        """
        Automatically discover classes from directory structure.
        Returns list of class names based on subdirectories.
        """
        if not os.path.exists(self.data_dir):
            print(f"Data directory {self.data_dir} does not exist")
            return []
        
        classes = []
        for item in os.listdir(self.data_dir):
            item_path = os.path.join(self.data_dir, item)
            if os.path.isdir(item_path):
                # Check if directory contains images
                has_images = False
                for file in os.listdir(item_path):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        has_images = True
                        break
                
                if has_images:
                    classes.append(item)
        
        return classes
    
    def create_dataset(self):
        """
        Create dataset from images stored locally.
        """
        features = []
        labels = []
        image_paths = []
        
        # Ensure the dataset is downloaded if URL is provided
        if self.DATASET_URL:
            self.download_dataset()
        
        # If no classes are specified, try to discover them
        if not self.CLASSES_LIST:
            self.CLASSES_LIST = self.discover_classes()
            print(f"Discovered classes: {self.CLASSES_LIST}")
        
        if not self.CLASSES_LIST:
            print("No classes specified or discovered.")
            return np.array([]), np.array([]), []
        
        # Iterating through all class folders
        for class_index, class_name in enumerate(self.CLASSES_LIST):
            print(f'Extracting features for class: {class_name}')
            
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Directory not found: {class_dir}")
                continue
            
            # Get all image files in the class directory
            class_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            # Show progress
            total_files = len(class_files)
            processed = 0
            start_time = time.time()
            
            for img_file in class_files:
                img_path = os.path.join(class_dir, img_file)
                
                try:
                    # Read the image
                    image = cv2.imread(img_path)
                    
                    if image is None:
                        print(f"Failed to read image: {img_path}")
                        continue
                    
                    # Extract features
                    image_features = self.extract_features(image)
                    
                    # Add to dataset
                    features.append(image_features)
                    labels.append(class_index)
                    image_paths.append(img_path)
                    
                    # Update progress
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
        """
        Extract features from images and save them to files.
        """
        print("Extracting and saving data...")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Extract features and labels
        features, labels, paths = self.create_dataset()
        
        if len(features) == 0:
            print("No features extracted. Skipping save.")
            return features, labels, paths
        
        # Save extracted data
        np.save(os.path.join(output_dir, f"features_{self.FEATURE_TYPE}.npy"), features)
        np.save(os.path.join(output_dir, "labels.npy"), labels)
        
        # Save paths as text file
        with open(os.path.join(output_dir, "image_paths.txt"), 'w') as f:
            for path in paths:
                f.write(f"{path}\n")
        
        # Save feature type and classes for reference
        with open(os.path.join(output_dir, "metadata.txt"), 'w') as f:
            f.write(f"Feature type: {self.FEATURE_TYPE}\n")
            f.write(f"Image dimensions: {self.IMAGE_HEIGHT}x{self.IMAGE_WIDTH}\n")
            f.write(f"Classes: {', '.join(self.CLASSES_LIST)}\n")
            f.write(f"Samples per class: {[list(labels).count(i) for i in range(len(self.CLASSES_LIST))]}\n")
            f.write(f"Feature dimension: {features.shape[1]}\n")
        
        print(f"Data extraction complete. Features shape: {features.shape}, Labels shape: {labels.shape}")
        
        return features, labels, paths
    
    def load_data(self, output_dir="../extracted_data"):
        """
        Load previously extracted data.
        """
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
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.
        """
        features, labels, _ =   self.load_data()
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        print(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")
        
        return X_train, X_test, y_train, y_test