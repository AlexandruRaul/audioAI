from django.db import models
import sqlite3
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class Chanson(models.Model):
    titre = models.CharField(max_length=200)
    artiste = models.ForeignKey('Artiste', on_delete=models.CASCADE)
    # autres champs ici

class Artiste(models.Model):
    nom = models.CharField(max_length=200)
    # autres champs ici

    def __str__(self):
        return f"{self.titre} - {self.artiste}"
    


# Define the CNN model
class AudioRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(AudioRecognitionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 30 * 30, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# Define a custom dataset for loading audio data from the database
class AudioDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        audio = self.data[index]
        label = self.labels[index]

        if self.transform:
            audio = self.transform(audio)

        return audio, label

# Train the model
def train_model(model, train_loader, criterion, optimizer, device, num_epochs):
    model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {running_loss / len(train_loader)}")

# Test the model
def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy}%")

# Load audio data from the database
def load_audio_data():
    conn = sqlite3.connect('path/to/your/music_database.db')
    cursor = conn.cursor()
    
    # Execute a query to retrieve audio data and labels from the database
    cursor.execute("SELECT audio, music_name, composer_name FROM audio_table")
    results = cursor.fetchall()

    # Initialize lists to store the retrieved data
    audio_data = []
    labels = []

    # Loop through the results and extract audio data and labels
    for result in results:
        audio_path = result[0]
        music_name = result[1]
        composer_name = result[2]

        # Load the audio file and preprocess it
        audio = load_and_preprocess_audio(audio_path)
        
        # Append the preprocessed audio data to the audio_data list
        audio_data.append(audio)

        # Create a label based on the music name and composer name
        label = f"{music_name} - {composer_name}"
        
        # Append the label to the labels list
        labels.append(label)

    # Close the database connection
    cursor.close()
    conn.close()

    return audio_data, labels


# Preprocess the audio data
def load_and_preprocess_audio(audio_path):
    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_path)

    # Apply audio preprocessing techniques
    # Example: Convert waveform to spectrogram
    spectrogram_transform = transforms.Spectrogram()
    spectrogram = spectrogram_transform(waveform)

    # Normalize the spectrogram
    normalized_spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()

    return normalized_spectrogram

# Main function
def main():
    # Set the 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load audio data and labels from the database
    audio_data, labels = load_audio_data()

    # Preprocess the audio data
    transformed_data = []
    for audio in audio_data:
        preprocessed_audio = load_and_preprocess_audio(audio)
        transformed_data.append(preprocessed_audio)

    # Split the data into training and test sets
    split_ratio = 0.8
    split_index = int(len(transformed_data) * split_ratio)
    train_data = transformed_data[:split_index]
    train_labels = labels[:split_index]
    test_data = transformed_data[split_index:]
    test_labels = labels[split_index:]

    # Create data loaders for training and test sets
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = AudioDataset(train_data, train_labels, transform=transform)
    test_dataset = AudioDataset(test_data, test_labels, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Create the model
    num_classes = len(set(labels))
    model = AudioRecognitionModel(num_classes)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model
    num_epochs = 10
    train_model(model, train_loader, criterion, optimizer, device, num_epochs)

    # Test the model
    test_model(model, test_loader, device)

    # Save the model
    model_path = "audio_recognition_model.pth"
    torch.save(model.state_dict(), model_path)

