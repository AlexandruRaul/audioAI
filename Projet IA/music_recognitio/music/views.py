from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .models import Chanson
from .models import File
import magic
import os
from django.conf import settings

def index(request):
    if request.method == 'POST':
        file = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(file.name, file)
        uploaded_file_url = fs.path(filename)

        # vérifiez que le fichier téléchargé est un fichier audio
        file_type = magic.from_file(uploaded_file_url, mime=True)
        if "audio" not in file_type:
            os.remove(uploaded_file_url)  # supprime le fichier s'il n'est pas audio
            return render(request, 'erreur.html', {'message': "Le fichier téléchargé n'est pas un fichier audio."})

        # ici vous pouvez ajouter la logique de votre algorithme de reconnaissance de musique 
        # pour l'instant, nous supposons simplement que la chanson a été reconnue avec succès
        song_name = "Nom de la chanson"
        artist_name = "Nom de l'artiste"
        
        if song_name and artist_name:
            return render(request, 'resultats.html', {'song_name': song_name, 'artist_name': artist_name})
        else:
            return render(request, 'erreur.html', {'message': "La chanson n'a pas pu être reconnue."})
    else:
        return render(request, 'index.html')
