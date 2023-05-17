import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from .utils import load_audio_data_function, load_and_preprocess_audio_function
from music_recognitio.music.models import AudioRecognitionModel


def load_audio_data():
    return load_audio_data_function()

def load_and_preprocess_audio():
    return load_and_preprocess_audio_function()

class AudioRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(AudioRecognitionModel, self).__init__()
        self.model = nn.Sequential(
            # Model layers...
        )

    def forward(self, x):
        return self.model(x)

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        super(AudioDataset, self).__init__()
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        label = self.labels[index]

        if self.transform is not None:
            item = self.transform(item)

        return item, label

def train_model(model, train_loader, num_epochs):
    return train_model_function(model, train_loader, num_epochs)

def test_model(model, test_loader):
    return test_model_function(model, test_loader)

def identifier_chanson(request):
    if request.method == 'POST':
        artist_name = request.POST.get('artist_name')
        song_name = request.POST.get('song_name')

        try:
            from .models import Chanson  # Import moved here
            chanson = Chanson.objects.get(artist_name=artist_name, song_name=song_name)
            chanson_identifiee = f"{chanson.artist_name} - {chanson.song_name}"
            return render(request, 'identifier_chanson.html', {'chanson_identifiee': chanson_identifiee})
        except Chanson.DoesNotExist:
            return render(request, 'erreur.html', {'message': "La chanson n'a pas été trouvée."})

    return None  # Return None if no POST request is received

def train_audio_model(request):
    # Rest of the code for training the audio model...
    from .models import Chanson  # Import moved here

    # Save the model
    model_path = "audio_recognition_model.pth"
    torch.save(model.state_dict(), model_path)

    render(request, 'train_audio_model.html', {'accuracy': accuracy})

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

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
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

    # Save the model
    model_path = "audio_recognition_model.pth"
    torch.save(model.state_dict(), model_path)

    render(request, 'train_audio_model.html', {'accuracy': accuracy})

