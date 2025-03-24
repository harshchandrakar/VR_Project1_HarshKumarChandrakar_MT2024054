import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from segmentation.mask_segmentation import UNet, SegmentationDataset, evaluate_model
import os

def main():
    # Set paths
    image_dir = "MFSD_dataset/MSFD/1/face_crop"
    mask_dir = "MFSD_dataset/MSFD/1/face_crop_segmentation"
    model_path = "models/best_unet_model.pth"
    
    # Define transforms
    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader
    dataset = SegmentationDataset(image_dir, mask_dir, transform=img_transform)
    val_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS device (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device")
    else:
        device = torch.device("cpu")
        print(f"Using CPU device")
    
    # Load model
    model = UNet().to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded model from epoch {checkpoint['epoch']+1}")
    
    # Evaluate model
    print("\nEvaluating model performance...")
    avg_iou, avg_dice = evaluate_model(model, val_loader, device)
    print(f"Average IoU Score: {avg_iou:.4f}")
    print(f"Average Dice Score: {avg_dice:.4f}")

if __name__ == "__main__":
    main() 