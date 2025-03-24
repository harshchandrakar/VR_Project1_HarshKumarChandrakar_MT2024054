import os
import sys
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import glob
import shutil
import random
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.feature_extraction import ImageFeatureExtractor
from utils.variables import *

def split_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    # Ensures the ratios add up to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
    
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    class_names = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    print(f"Found {len(class_names)} classes: {', '.join(class_names)}")
    
    for split_dir in [train_dir, val_dir, test_dir]:
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
        for class_name in class_names:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
    
    for class_name in class_names:
        class_dir = os.path.join(source_dir, class_name)
        
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(class_dir, ext)))
        
        print(f"Class '{class_name}': Found {len(image_files)} images")
        
        if len(image_files) == 0:
            print(f"Warning: No images found for class {class_name}")
            continue
        
        # Split into train and temporary sets
        train_files, temp_files = train_test_split(
            image_files, 
            train_size=train_ratio,
            random_state=random_state
        )
        
        # Further split the temporary set into validation and test sets
        relative_val_ratio = val_ratio / (val_ratio + test_ratio)
        val_files, test_files = train_test_split(
            temp_files,
            train_size=relative_val_ratio,
            random_state=random_state
        )
        
        for src_file, dest_dir, split_name in [
            (train_files, train_dir, "train"),
            (val_files, val_dir, "validation"),
            (test_files, test_dir, "test")
        ]:
            for file_path in src_file:
                file_name = os.path.basename(file_path)
                dst_path = os.path.join(dest_dir, class_name, file_name)
                shutil.copy2(file_path, dst_path)
            
            print(f"Class '{class_name}': Copied {len(src_file)} images to {split_name} set")
    
    return output_dir

class CNNClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(CNNClassifier, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate)
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            for img_path in glob.glob(os.path.join(class_path, '*.jpg')) + \
                           glob.glob(os.path.join(class_path, '*.png')) + \
                           glob.glob(os.path.join(class_path, '*.jpeg')):
                self.samples.append((img_path, self.class_to_idx[class_name]))
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # Convert RGBA images to RGB for consistent input to model
        image = Image.open(img_path).convert('RGBA').convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def train_model(model, train_loader, val_loader, criterion, optimizer, device, class_to_idx, model_name, num_epochs=10, patience=20, model_save_path='models'):
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
        
    model.to(device)
    
    best_val_acc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(model_save_path, model_name)
    
    # Clear GPU memory if using MPS
    if device.type == 'mps' and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct / total
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
        
        # Save intermediate checkpoints
        if (epoch + 1) % 10 == 0:
            checkpoint_filename = f"{os.path.splitext(model_name)[0]}_epoch_{epoch+1}.pth"
            checkpoint_path = os.path.join(model_save_path, checkpoint_filename)
            save_model(model, checkpoint_path, epoch, optimizer, val_accuracy, class_to_idx)
        
        # Check for improvement and implement early stopping
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
            save_model(model, best_model_path, epoch, optimizer, val_accuracy, class_to_idx)
            print(f"New best model saved with validation accuracy: {val_accuracy:.2f}%")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
            
        if device.type == 'mps' and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    return best_model_path

def save_model(model, filepath, epoch=None, optimizer=None, val_accuracy=None, class_mapping=None):
    # Create a dictionary with all necessary model information
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_architecture': type(model).__name__,
        'input_size': model.classifier[1].in_features,
        'num_classes': model.classifier[-1].out_features,
    }
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if val_accuracy is not None:
        checkpoint['validation_accuracy'] = val_accuracy
    if class_mapping is not None:
        checkpoint['class_mapping'] = class_mapping
        
    torch.save(checkpoint, filepath)

def load_model(filepath, device=None):
    # Auto-detect device if not specified
    if device is None:
        try:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        except Exception as e:
            device = torch.device("cpu")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    num_classes = checkpoint.get('num_classes', 2)
    input_size = checkpoint.get('input_size', 128*28*28)
    
    model = CNNClassifier(num_classes=num_classes)
    
    feature_map_size = int((input_size / 128) ** 0.5)
    model.classifier[1] = nn.Linear(input_size, 256)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint.get('class_mapping', None)

def evaluate_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions) * 100
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    print(cm)
    
    report = classification_report(all_labels, all_predictions)
    print("Classification Report:")
    print(report)
    
    return accuracy, cm, report

def predict_single_image(model, image_path, transform, device, class_mapping):
    # Convert index to class name mapping
    idx_to_class = {v: k for k, v in class_mapping.items()}
    
    image = Image.open(image_path).convert('RGBA').convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        
    predicted_class_idx = predicted.item()
    predicted_class = idx_to_class.get(predicted_class_idx, f"Unknown class {predicted_class_idx}")
    
    # Calculate confidence using softmax
    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    confidence = probabilities[predicted_class_idx].item() * 100
    
    return predicted_class, confidence, probabilities.cpu().numpy()

def setup_device():
    # Select the best available computation device
    try:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS acceleration on Mac")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA acceleration")
        else:
            device = torch.device("cpu")
            print("Using CPU (no GPU acceleration available)")
    except Exception as e:
        device = torch.device("cpu")
        print("Falling back to CPU")
    
    return device

def prepare_dataset(data_dir, download_dataset=False, split_data=False, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    if download_dataset:
        print("Downloading dataset...")
        try:
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
                
            extractor = ImageFeatureExtractor(
                DATASET_URL_1, 
                CLASSES_1, 
                IMAGE_HEIGHT, 
                IMAGE_WIDTH, 
                "combined", 
                data_dir
            )
            extractor.download_dataset()
            print("Dataset downloaded successfully.")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return None
    
    dataset_path = data_dir
    if split_data:
        # Create train/val/test split for the dataset
        split_dir = data_dir + "_split"
        try:
            dataset_path = split_dataset(
                data_dir,
                split_dir,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio
            )
        except Exception as e:
            print(f"Error splitting dataset: {e}")
            return None
    
    return dataset_path

def train_models(data_dir, hyperparams_list, model_names, download_dataset=False, split_data=False, models_dir="models"):
    assert len(hyperparams_list) == len(model_names), "Number of hyperparameter sets must match number of model names"
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    device = setup_device()
    
    prepared_data_dir = prepare_dataset(data_dir, download_dataset, split_data)
    if prepared_data_dir is None:
        print("Failed to prepare dataset. Aborting training.")
        return []
    
    results = []
    
    for i, (hyperparams, model_name) in enumerate(zip(hyperparams_list, model_names)):
        print(f"\n[{i+1}/{len(hyperparams_list)}] Training model with {model_name}")
        
        # Define training transforms with data augmentation
        train_transform = transforms.Compose([
            transforms.Resize((hyperparams['img_size'], hyperparams['img_size'])),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Define validation transforms without augmentation
        val_transform = transforms.Compose([
            transforms.Resize((hyperparams['img_size'], hyperparams['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = ImageFolderDataset(os.path.join(prepared_data_dir, 'train'), transform=train_transform)
        val_dataset = ImageFolderDataset(os.path.join(prepared_data_dir, 'val'), transform=val_transform)
        
        # Adjust number of workers based on device
        num_workers = 2 if device.type == 'mps' else 4
        pin_memory = (device.type != 'cpu')
        
        train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], 
                                  shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'], 
                                shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        
        num_classes = len(train_dataset.classes)
        print(f"Number of classes: {num_classes}")
        
        model = CNNClassifier(num_classes=num_classes, dropout_rate=hyperparams['dropout_rate'])
        
        # Adjust linear layer dimensions based on input image size
        feature_map_size = hyperparams['img_size'] // 8 
        in_features = 128 * feature_map_size * feature_map_size
        
        model.classifier[1] = nn.Linear(in_features, 256)
        
        criterion = nn.CrossEntropyLoss()
        
        # Select optimizer based on hyperparameters
        if hyperparams['optimizer'].lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'], 
                                  weight_decay=hyperparams['weight_decay'])
        elif hyperparams['optimizer'].lower() == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=hyperparams['learning_rate'],
                                 momentum=0.9, weight_decay=hyperparams['weight_decay'])
        else:
            optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'], 
                                  weight_decay=hyperparams['weight_decay'])
        
        best_model_path = train_model(
            model, 
            train_loader, 
            val_loader,
            criterion, 
            optimizer, 
            device,
            train_dataset.class_to_idx,
            model_name,
            num_epochs=hyperparams['num_epochs'],
            patience=hyperparams['patience'],
            model_save_path=models_dir
        )
        
        checkpoint = torch.load(best_model_path, map_location=device)
        val_accuracy = checkpoint.get('validation_accuracy', 0.0)
        
        results.append((best_model_path, val_accuracy))
        
        # Free up GPU memory after training
        if device.type in ["cuda", "mps"]:
            torch.cuda.empty_cache() if device.type == "cuda" else torch.mps.empty_cache()
    
    return results, prepared_data_dir

def test_models(data_dir, model_names, models_dir="models"):
    device = setup_device()
    
    # Default test transforms, will be adjusted for each model
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = ImageFolderDataset(os.path.join(data_dir, 'test'), transform=test_transform)
    
    results = {}
    
    for model_name in model_names:
        print(f"\nTesting model: {model_name}")
        model_path = os.path.join(models_dir, model_name)
        
        try:
            model, class_mapping = load_model(model_path, device)
            
            # Adjust test transform to match model's input size
            in_features = model.classifier[1].in_features
            feature_map_size = int((in_features / 128) ** 0.5)
            img_size = feature_map_size * 8
            
            test_transform.transforms[0] = transforms.Resize((img_size, img_size))
            test_dataset.transform = test_transform
            
            num_workers = 2 if device.type == 'mps' else 4
            pin_memory = (device.type != 'cpu')
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, 
                                     num_workers=num_workers, pin_memory=pin_memory)
            
            test_accuracy, confusion_mat, classification_rep = evaluate_model(model, test_loader, device)
            
            results[model_name] = {
                'accuracy': test_accuracy,
                'confusion_matrix': confusion_mat,
                'classification_report': classification_rep
            }
            
            # Test on a single example
            if len(test_dataset.samples) > 0:
                sample_image_path = test_dataset.samples[0][0]
                predicted_class, confidence, _ = predict_single_image(
                    model, 
                    sample_image_path, 
                    test_transform, 
                    device, 
                    test_dataset.class_to_idx
                )
                print(f"Sample prediction: {predicted_class} with {confidence:.2f}% confidence")
                
        except Exception as e:
            print(f"Error testing model {model_name}: {e}")
            results[model_name] = {'error': str(e)}
        
        # Free up GPU memory
        if device.type in ["cuda", "mps"]:
            torch.cuda.empty_cache() if device.type == "cuda" else torch.mps.empty_cache()
    
    print("\n===== Test Results Summary =====")
    for model_name, result in results.items():
        if 'accuracy' in result:
            print(f"{model_name}: Accuracy = {result['accuracy']:.2f}%")
        else:
            print(f"{model_name}: Failed - {result.get('error', 'Unknown error')}")
    
    return results

def main(train = True,data_dir = "../data"):
    
    
    hyperparams_list = [
        {
            'learning_rate': 0.001,
            'batch_size': 64,  
            'num_epochs': 30,
            'dropout_rate': 0.3,
            'optimizer': 'adam',
            'weight_decay': 1e-5,
            'patience': 15,
            'img_size': IMAGE_HEIGHT
        },
        {
            'learning_rate': 0.0005,
            'batch_size': 32,  
            'num_epochs': 30,
            'dropout_rate': 0.5,
            'optimizer': 'sgd',
            'weight_decay': 1e-4,
            'patience': 20,
            'img_size': IMAGE_HEIGHT
        },
        {
            'learning_rate': 0.0003,
            'batch_size': 64,  
            'num_epochs': 40,
            'dropout_rate': 0.4,
            'optimizer': 'adam',
            'weight_decay': 1e-4,
            'patience': 20,
            'img_size': IMAGE_HEIGHT
        }
    ]
    
    model_names = [
        "model_adam_lr001.pth",
        "model_sgd_lr0005.pth",
        "model_adam_lr0003.pth"
    ]

    if train:
    
        trained_models, prepared_data_dir = train_models(
            data_dir, 
            hyperparams_list, 
            model_names, 
            download_dataset=True, 
            split_data=True
        )
        
        print("\n===== Training Results =====")
        for model_path, val_accuracy in trained_models:
            print(f"{os.path.basename(model_path)}: Validation Accuracy = {val_accuracy:.2f}%")
    else:
        prepared_data_dir = prepare_dataset(data_dir, True, True)
        if prepared_data_dir is None:
            print("Failed to prepare dataset. Aborting training.")
            return []
        
    test_results = test_models(
        prepared_data_dir,
        model_names
    )


if __name__ == "__main__":
    main(train=True)