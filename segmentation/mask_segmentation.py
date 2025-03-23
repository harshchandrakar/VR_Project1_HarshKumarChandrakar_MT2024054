import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

# Define the UNet architecture
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = DoubleConv(3, 64)
        self.encoder2 = DoubleConv(64, 128)
        self.encoder3 = DoubleConv(128, 256)
        self.encoder4 = DoubleConv(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.decoder3 = DoubleConv(512 + 256, 256)
        self.decoder2 = DoubleConv(256 + 128, 128)
        self.decoder1 = DoubleConv(128 + 64, 64)
        
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        
        # Decoder
        d3 = self.decoder3(torch.cat([self.upsample(e4), e3], dim=1))
        d2 = self.decoder2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.decoder1(torch.cat([self.upsample(d2), e1], dim=1))
        
        return torch.sigmoid(self.final_conv(d1))

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # Get all image files
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.mask_files = set(os.listdir(mask_dir))  # Use a set for O(1) lookup
        
        # Filter to only include images that have corresponding masks
        self.valid_images = []
        self.skipped_images = []
        
        for img_file in self.image_files:
            mask_file = img_file
            if mask_file in self.mask_files:
                try:
                    # Try to open both image and mask to validate them
                    img_path = os.path.join(image_dir, img_file)
                    mask_path = os.path.join(mask_dir, mask_file)
                    
                    img = Image.open(img_path)
                    mask = Image.open(mask_path).convert('L')  # Convert to grayscale
                    
                    self.valid_images.append(img_file)
                        
                except Exception as e:
                    self.skipped_images.append((img_file, str(e)))
            else:
                self.skipped_images.append((img_file, "No corresponding mask file"))

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        img_name = self.valid_images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Apply transformations first
        if self.transform:
            image = self.transform(image)
            # For mask, use separate transform without normalization
            mask = transforms.Resize((256, 256))(mask)
            mask = transforms.ToTensor()(mask)
        
        # Ensure mask is binary (0 or 1)
        mask = (mask > 0.5).float()
        
        return image, mask

# Enhanced training function with validation
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20, save_path='best_model.pth'):
    model.to(device)
    best_val_loss = float('inf')
    
    # Create a directory for model checkpoints if it doesn't exist
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'dice_score': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]') as t:
            for images, masks in t:
                images = images.to(device)
                masks = masks.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                t.set_postfix(loss=loss.item())
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        dice_score = 0.0
        
        with torch.no_grad():
            with tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Valid]') as t:
                for images, masks in t:
                    images = images.to(device)
                    masks = masks.to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
                    
                    # Calculate Dice coefficient (F1 score)
                    pred = (outputs > 0.5).float()
                    intersection = (pred * masks).sum()
                    dice = (2. * intersection) / (pred.sum() + masks.sum() + 1e-6)
                    dice_score += dice.item()
                    
                    t.set_postfix(loss=loss.item(), dice=dice.item())
        
        val_loss /= len(val_loader)
        dice_score /= len(val_loader)
        
        history['val_loss'].append(val_loss)
        history['dice_score'].append(dice_score)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Valid Loss: {val_loss:.4f}, Dice Score: {dice_score:.4f}')
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'dice_score': dice_score,
            }, save_path)
            print(f'Model saved to {save_path}')
            
        print('-' * 60)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['dice_score'], label='Dice Score')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.title('Validation Dice Score')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    return model, history

# Function to visualize predictions
def visualize_predictions(model, val_loader, device, num_samples=4):
    model.eval()
    images, masks, predictions = [], [], []
    
    with torch.no_grad():
        for img, mask in val_loader:
            if len(images) >= num_samples:
                break
                
            img = img.to(device)
            pred = model(img)
            pred = (pred > 0.5).float()
            
            # Move tensors to CPU for visualization
            images.append(img.cpu())
            masks.append(mask.cpu())
            predictions.append(pred.cpu())
    
    # Concatenate batch samples
    images = torch.cat(images)[:num_samples]
    masks = torch.cat(masks)[:num_samples]
    predictions = torch.cat(predictions)[:num_samples]
    
    # Plot results
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    for i in range(num_samples):
        # Original image
        img = images[i].permute(1, 2, 0)  # Change from CxHxW to HxWxC
        # Denormalize image
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        img = torch.clamp(img, 0, 1)
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis('off')
        
        # Ground truth mask
        axes[i, 1].imshow(masks[i].squeeze(), cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')
        
        # Predicted mask
        axes[i, 2].imshow(predictions[i].squeeze(), cmap='gray')
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('segmentation_results.png')
    plt.show()

# Add a class for combined loss function
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCELoss(weight=weight, reduction='mean')
    
    def forward(self, inputs, targets):
        # BCE Loss
        bce_loss = self.bce(inputs, targets)
        
        # Dice Loss
        smooth = 1.0
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1 - ((2. * intersection + smooth) / 
                     (inputs_flat.sum() + targets_flat.sum() + smooth))
        
        # Combined loss (you can adjust the weighting)
        return bce_loss + dice_loss
    
def calculate_iou(pred, target):
    """Calculate Intersection over Union (IoU) between prediction and target."""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()

def evaluate_model(model, val_loader, device):
    """Evaluate model performance using IoU and Dice scores."""
    model.eval()
    total_iou = 0.0
    total_dice = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            # Calculate IoU
            iou = calculate_iou(outputs, masks)
            total_iou += iou
            
            # Calculate Dice score
            pred = (outputs > 0.5).float()
            intersection = (pred * masks).sum()
            dice = (2. * intersection) / (pred.sum() + masks.sum() + 1e-6)
            total_dice += dice.item()
            
            num_samples += 1
    
    avg_iou = total_iou / num_samples
    avg_dice = total_dice / num_samples
    
    return avg_iou, avg_dice

def main():
    # Set paths
    image_dir = "MFSD_dataset/MSFD/1/face_crop"
    mask_dir = "MFSD_dataset/MSFD/1/face_crop_segmentation"
    
    # Define transforms
    # For images - includes normalization
    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset with normalized images
    dataset = SegmentationDataset(image_dir, mask_dir, transform=img_transform)
    
    # Print dataset summary
    print(f"Total images found: {len(dataset.image_files)}")
    print(f"Valid image-mask pairs: {len(dataset.valid_images)}")
    print(f"Skipped images: {len(dataset.skipped_images)}")
    
    # Print details of skipped images
    if dataset.skipped_images:
        print("\nSkipped images details:")
        for img_name, reason in dataset.skipped_images[:10]:  # Print only first 10
            print(f"{img_name}: {reason}")
        if len(dataset.skipped_images) > 10:
            print(f"...and {len(dataset.skipped_images) - 10} more")
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Set up device - add MPS support for Apple Silicon
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS device (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device")
    else:
        device = torch.device("cpu")
        print(f"Using CPU device")
    
    # Optimized parameters for M3 Pro Apple Silicon
    batch_size = 16   # M3 Pro can handle larger batches
    num_workers = 4  
    learning_rate = 3e-4  
    epochs = 10 # takes 2 hours if GPU cores are used and display is always on
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Number of workers: {num_workers}")
    print(f"Learning rate: {learning_rate}")
    print(f"Number of epochs: {epochs}")
    
    # Check the first batch to ensure binary masks
    for images, masks in train_loader:
        print(f"\nBatch shape:")
        print(f"Images: {images.shape}")
        print(f"Masks: {masks.shape}")
        print(f"Mask unique values: {torch.unique(masks)}")
        break
    
    # Initialize model, loss, and optimizer
    model = UNet()
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create directories for outputs
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Train the model
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=epochs,
        save_path='models/best_unet_model.pth'
    )
    
    # Load best model for visualization
    checkpoint = torch.load('models/best_unet_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1} with validation loss: {checkpoint['val_loss']:.4f} and Dice score: {checkpoint['dice_score']:.4f}")
    
    # Visualize predictions
    visualize_predictions(model, val_loader, device, num_samples=6)
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'models/final_unet_model.pth')
    
    print("Training completed!")

if __name__ == "__main__":
    main()