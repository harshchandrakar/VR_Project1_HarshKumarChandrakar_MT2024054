import sys
import os
import torch
import numpy as np
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.feature_extraction import ImageFeatureExtractor
from utils.variables import *

def svm_model(X_train, X_test, y_train, y_test,kernel):
    svm_model = SVC(kernel=kernel) 
    svm_model.fit(X_train, y_train)


    y_pred = svm_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of SVM on {kernel} kernel: {accuracy:.4f}")

#neural network implementation
class BinaryClassifier(nn.Module):
    def __init__(self, input_size=8404):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 1) 
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x)).squeeze(1)
        return x

def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        epoch_acc = 100 * correct / total
        if ((epoch+1)%50 == 0):
            print(f'Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}, Accuracy: {epoch_acc:.2f}%')

    return model


def evaluate(model, dataloader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            outputs = model(batch_x)
            predicted = (outputs > 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

def main(extract=False):
    create_data = ImageFeatureExtractor(DATASET_URL_1,CLASSES_1,IMAGE_HEIGHT,IMAGE_WIDTH,"combined","../data")

    if extract:
        create_data.extract_and_save_data(output_dir="../extracted_data")

    X_train, X_test, y_train, y_test = create_data.split_data(test_size=0.2,random_state=42)

    # because rbf works better if there is complex data
    svm_model(X_train, X_test, y_train, y_test,'rbf')
    #sigmoid if data is suitable for neural network classification problem 
    svm_model(X_train, X_test, y_train, y_test,'sigmoid')
    
    svm_model(X_train, X_test, y_train, y_test,'linear')


    # Neural network
    model = BinaryClassifier()
    criterion = nn.BCELoss() # Binary class classification loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Train using training data
    model = train_model(model, train_dataloader, criterion, optimizer, num_epochs=300)
    # Evaluate on test dataset
    evaluate(model, test_dataloader)



if __name__ == "__main__":
    main()







