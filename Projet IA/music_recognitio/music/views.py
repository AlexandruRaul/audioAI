from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from .models import Chanson
from .models import File
import magic
import os
import torchaudio
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout
from . import models

def index(request):
    if request.method == 'POST':
        file = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(file.name, file)
        uploaded_file_url = fs.path(filename)

        file_type = magic.from_file(uploaded_file_url, mime=True)
        if "audio" not in file_type:
            os.remove(uploaded_file_url)
            return render(request, 'erreur.html', {'message': "Le fichier téléchargé n'est pas un fichier audio."})

        song_name = "Nom de la chanson"
        artist_name = "Nom de l'artiste"

        if song_name and artist_name:
            return render(request, 'resultats.html', {'song_name': song_name, 'artist_name': artist_name})
        else:
            return render(request, 'erreur.html', {'message': "La chanson n'a pas pu être reconnue."})
    else:
        return render(request, 'index.html')

def charger_chanson(request):
    if request.method == 'POST':
        file = request.FILES['file']

        fs = FileSystemStorage()
        filename = fs.save(file.name, file)
        uploaded_file_url = fs.url(filename)

        file_type = magic.from_file(uploaded_file_url, mime=True)
        if "audio" not in file_type:
            os.remove(uploaded_file_url)
            return render(request, 'erreur.html', {'message': "Le fichier téléchargé n'est pas un fichier audio."})

        waveform, sample_rate = torchaudio.load(uploaded_file_url)

        song_name = "Nom de la chanson"
        artist_name = "Nom de l'artiste"

        if song_name and artist_name:
            return render(request, 'resultats.html', {'song_name': song_name, 'artist_name': artist_name})
        else:
            return render(request, 'erreur.html', {'message': "La chanson n'a pas pu être reconnue."})

    return render(request, 'charger_chanson.html')

def identifier_chanson(request):
    if request.method == 'POST':
        artist_name = request.POST.get('artist_name')
        song_name = request.POST.get('song_name')

        try:
            chanson = Chanson.objects.get(artist_name=artist_name, song_name=song_name)
            chanson_identifiee = f"{chanson.artist_name} - {chanson.song_name}"
            return render(request, 'identifier_chanson.html', {'chanson_identifiee': chanson_identifiee})
        except Chanson.DoesNotExist:
            return render(request, 'erreur.html', {'message': "La chanson n'a pas été trouvée."})

    # Execute the code for identifying the song from models.py
    result = models.identifier_chanson(request)
    
    # Check if the song was successfully identified
    if result is None:
        return render(request, 'erreur.html', {'message': "La chanson n'a pas pu être identifiée."})
    
    # Retrieve the identified song name and artist name
    song_name, artist_name = result

    return render(request, 'identifier_chanson.html', {'song_name': song_name, 'artist_name': artist_name})



@login_required
def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return render(request, 'index.html')
        else:
            return render(request, 'erreur.html', {'message': "Nom d'utilisateur ou mot de passe incorrect."})
    else:
        return render(request, 'login.html')

@login_required
def logout_view(request):
    logout(request)
    # Ajoutez ici le code de redirection ou de rendu approprié après la déconnexion
    return redirect('home')  # Remplacez 'home' par le nom de votre vue d'accueil

def erreur_view(request):
    return render(request, 'erreur.html', {'message': "Une erreur s'est produite."})