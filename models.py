import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import sqlite3
import os


# Définir le modèle CNN
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

# Définir un jeu de données personnalisé pour charger les données audio depuis la base de données
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

# Entraîner le modèle
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

# Tester le modèle
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

# Charger les données audio depuis la base de données
def load_audio_data():
    conn = sqlite3.connect('path/to/your/music_database.db')
    cursor = conn.cursor()
    
    # Exécuter une requête pour récupérer les données audio et les étiquettes de la base de données
    cursor.execute("SELECT audio, music_name, composer_name FROM audio_table")
    results = cursor.fetchall()

    # Initialiser des listes pour stocker les données récupérées
    audio_data = []
    labels = []

    # Parcourir les résultats et extraire les données audio et les étiquettes
    for result in results:
        audio_path = result[0]
        music_name = result[1]
        composer_name = result[2]

        # Charger le fichier audio et le prétraiter
        audio = load_and_preprocess_audio(audio_path)
        
        # Ajouter les données audio prétraitées à la liste audio_data
        audio_data.append(audio)

        # Créer une étiquette basée sur le nom de la musique et le nom du compositeur
        label = f"{music_name} - {composer_name}"
        
        # Ajouter l'étiquette à la liste labels
        labels.append(label)

    # Fermer la connexion à la base de données
    cursor.close()
    conn.close()

    return audio_data, labels


# Prétraiter les données audio
def load_and_preprocess_audio(audio_path):
    # Charger le fichier audio
    waveform, sample_rate = torchaudio.load(audio_path)

    # Appliquer des techniques de prétraitement audio
    # Exemple : Convertir la forme d'onde en spectrogramme
    spectrogram_transform = transforms.Spectrogram()
    spectrogram = spectrogram_transform(waveform)

    # Normaliser le spectrogramme
    normalized_spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()

    return normalized_spectrogram

# Fonction principale
def main():
    # Définir le périphérique (GPU si disponible, sinon CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Charger les données audio et les étiquettes depuis la base de données
    audio_data, labels = load_audio_data()

    # Prétraiter les données audio
    transformed_data = []
    for audio in audio_data:
        preprocessed_audio = load_and_preprocess_audio(audio)
        transformed_data.append(preprocessed_audio)

    # Diviser les données en ensembles d'entraînement et de test
    split_ratio = 0.8
    split_index = int(len(transformed_data) * split_ratio)
    train_data = transformed_data[:split_index]
    train_labels = labels[:split_index]
    test_data = transformed_data[split_index:]
    test_labels = labels[split_index:]

    # Créer des chargeurs de données pour les ensembles d'entraînement et de test
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = AudioDataset(train_data, train_labels, transform=transform)
    test_dataset = AudioDataset(test_data, test_labels, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Créer le modèle
    num_classes = len(set(labels))
    model = AudioRecognitionModel(num_classes)

    # Définir la fonction de perte et l'optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Entraîner le modèle
    num_epochs = 10
    train_model(model, train_loader, criterion, optimizer, device, num_epochs)

    # Tester le modèle
    test_model(model, test_loader, device)

    # Sauvegarder le modèle
    model_path = "audio_recognition_model.pth"
    torch.save(model.state_dict(), model_path)

