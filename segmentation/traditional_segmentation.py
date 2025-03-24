import cv2
import numpy as np
import pandas as pd
import os
from sklearn.metrics import precision_score, recall_score, f1_score
import glob
import matplotlib.pyplot as plt
import sys
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.data_download import download_from_gdrive
from utils.variables import *

def load_dataset(csv_path):
    """
    Load dataset and filter images with masks
    """
    print(f"Reading CSV file from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Filter images with masks
    mask_images = df[df['with_mask'] == True]['filename'].tolist()
    print(f"Images with masks: {len(mask_images)}")
    
    print(f"Selected samples: {len(mask_images)}")
    return mask_images

def calculate_metrics(pred_mask, gt_mask):
    """
    Calculate evaluation metrics
    """
    pred_binary = (pred_mask > 127).astype(np.uint8)
    gt_binary = (gt_mask > 127).astype(np.uint8)
    
    pred_flat = pred_binary.flatten()
    gt_flat = gt_binary.flatten()
    
    intersection = np.logical_and(pred_binary, gt_binary)
    union = np.logical_or(pred_binary, gt_binary)
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
    
    precision = precision_score(gt_flat, pred_flat, zero_division=0)
    recall = recall_score(gt_flat, pred_flat, zero_division=0)
    f1 = f1_score(gt_flat, pred_flat, zero_division=0)
    
    return iou, precision, recall, f1

def detect_face(image):
    """
    Detect face in the image using Haar Cascade
    """
    height, width = image.shape[:2]
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Not necessary to convert to grayscale only to save time
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        return max(faces, key=lambda rect: rect[2] * rect[3])
    else:
        return (0, 0, width, int(height * 0.8))

def segment_mask_region(image):
    """
    Segment mask regions using simple region-based methods
    """
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    face_rect = detect_face(image)
    x, y, w, h = face_rect
    
    # Focus on the lower part of the face (generally where the mask is)
    mask_region_y = y + int(0.5 * h) 
    mask_region_h = int(0.7 * h)      # Cover lower portion of face
    
    mask_region_bottom = min(height, mask_region_y + mask_region_h)
    
    # ROI
    roi = image[mask_region_y:mask_region_bottom, x:x+w]
  
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Basic thresholding to extract potential mask regions
    # Edge-based approach
    blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Color-based approach for common mask colors
    # Blue masks
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv_roi, lower_blue, upper_blue)
    
    # White masks
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv_roi, lower_white, upper_white)
    
    # Black masks
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    black_mask = cv2.inRange(hsv_roi, lower_black, upper_black)
    
    color_mask = cv2.bitwise_or(blue_mask, white_mask)
    color_mask = cv2.bitwise_or(color_mask, black_mask)
    
    combined_mask = cv2.bitwise_or(edges, color_mask)
    
    # Morphological operations 
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    # Remove small noise
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        min_area = 0.005 * (width * height)  # 0.5% of the total image area
        significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        top_contours = sorted(significant_contours, key=cv2.contourArea, reverse=True)[:3]
        
        temp_mask = np.zeros_like(combined_mask)
        cv2.drawContours(temp_mask, top_contours, -1, 255, -1)
        combined_mask = temp_mask
    
    mask[mask_region_y:mask_region_bottom, x:x+w] = combined_mask
    
    return mask

def create_mask_overlay(image, mask, color=(0, 255, 0), alpha=0.3):
    """
    Creates a colored overlay of the mask on the original image.
    """
    mask_rgb = np.zeros_like(image)
    mask_rgb[mask > 0] = color
    return cv2.addWeighted(image, 1 - alpha, mask_rgb, alpha, 0)

def create_directories(base_folder, subdirs):
    """
    Create multiple subdirectories within a base folder.
    """
    os.makedirs(base_folder, exist_ok=True)
    for subdir in subdirs:
        os.makedirs(os.path.join(base_folder, subdir), exist_ok=True)

def safe_image_read(path):
    """
    Reads an image safely, raising an exception if it fails.
    """
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not read image at {path}")
    return image

def find_matching_file(folder, base_name, extension="*.jpg"):
    """
    Finds files matching the base name and extension in a given folder.
    """
    matching_files = glob.glob(os.path.join(folder, f"{base_name}{extension}"))
    return matching_files[0] if matching_files else None

def visualize_results(image, mask, gt_mask=None, save_path=None):
    """
    Visualize segmentation results and save a single combined image with all three views side by side.
    """
    # RGB format for consistency
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    if gt_mask is not None:
        gt_rgb = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)
    else:
        gt_rgb = create_mask_overlay(image, mask)
    
    combined_result = np.hstack([original_rgb, mask_rgb, gt_rgb])
    
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(combined_result, cv2.COLOR_RGB2BGR))
    else:
        plt.imshow(combined_result)
        plt.axis('off')
        plt.show()

def process_dataset(input_folder, output_folder, gt_folder, num_samples,verbose=False):
    """
    Process the dataset and perform segmentation with visualization and evaluation
    """

    create_directories(output_folder, ['masks', 'overlays', 'visualizations'])
    
    image_files = load_dataset('MFSD_dataset/MSFD/1/dataset.csv')
    image_files = image_files[:num_samples]
    
    total_images = len(image_files)
    print(f"\nFound {total_images} images with masks")
    
    has_ground_truth = os.path.exists(gt_folder)
    if not has_ground_truth:
        print(f"\nWarning: Ground truth directory {gt_folder} not found. Metrics will not be calculated.")
    
    metrics = {metric: [] for metric in ['iou', 'precision', 'recall', 'f1']}
    
    for idx, image_file in enumerate(image_files, 1):
        if verbose:
            print(f"\nProcessing image {idx}/{total_images}: {image_file}")
        
        base_name = os.path.splitext(image_file)[0]
        
        input_path = find_matching_file(input_folder, base_name)
        gt_path = find_matching_file(gt_folder, base_name) if has_ground_truth else None
        
        if not input_path:
            print(f"Error: No matching files found for {image_file}")
            continue
            
        if not gt_path and has_ground_truth:
            print(f"Warning: No ground truth file found for {image_file}")
        
        try:
            image = safe_image_read(input_path)
            
            # Segment mask
            mask = segment_mask_region(image)
            
            # Save mask
            mask_path = os.path.join(output_folder, 'masks', image_file)
            cv2.imwrite(mask_path, mask)
            
            # Create and save overlay
            overlay = create_mask_overlay(image, mask)
            overlay_path = os.path.join(output_folder, 'overlays', image_file)
            cv2.imwrite(overlay_path, overlay)
            
            gt_mask = None
            if gt_path:
                gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                
                if gt_mask is not None:
                    # Same dimensions
                    if gt_mask.shape != mask.shape:
                        gt_mask = cv2.resize(gt_mask, (mask.shape[1], mask.shape[0]))
                    
                    iou, precision, recall, f1 = calculate_metrics(mask, gt_mask)
                    
                    metrics['iou'].append(iou)
                    metrics['precision'].append(precision)
                    metrics['recall'].append(recall)
                    metrics['f1'].append(f1)
                    if verbose:
                        print(f"Metrics for {image_file}:")
                        print(f"IoU: {iou:.4f}")
                        print(f"Precision: {precision:.4f}")
                        print(f"Recall: {recall:.4f}")
                        print(f"F1 Score: {f1:.4f}")
            
            # Visualize Results
            vis_path = os.path.join(output_folder, 'visualizations', f"{base_name}_vis.png")
            visualize_results(image, mask, gt_mask, save_path=vis_path)
            
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue

    # Summary of metrics
    if metrics['iou']:
        print("\nSegmentation Evaluation Summary:")
        print(f"Total images evaluated: {len(metrics['iou'])}")
        
        for metric, values in metrics.items():
            mean_val = np.mean(values)
            median_val = np.median(values)
            std_val = np.std(values)
            print(f"Average {metric.upper()}: {mean_val:.4f} (Median: {median_val:.4f}, Std: {std_val:.4f})")
        

def main(images,download = False,verbose=False):
    if download:
        download_from_gdrive(DATASET_URL_2)
    input_folder = "MFSD_dataset/MSFD/1/face_crop"
    output_folder = "results/region_segmentation"
    gt_folder = "MFSD_dataset/MSFD/1/face_crop_segmentation"
    
    process_dataset(input_folder, output_folder, gt_folder, images,verbose=verbose)
    print("\nSegmentation and evaluation completed.")

if __name__ == "__main__":
    main(100,download=True,verbose=False)