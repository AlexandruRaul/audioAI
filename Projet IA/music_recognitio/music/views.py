from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, HttpResponseBadRequest, Http404
from .models import Chanson
from . import some_audio_library
from django.contrib.auth import authenticate, login
from django.shortcuts import redirect
from .forms import LoginForm
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required


def charger_chanson(request):
    # logique pour charger une chanson ici
    return render(request, 'charger_chanson.html')

def identifier_chanson(request):
    if request.method == 'POST':
        try:
            # Tentez de récupérer la chanson du formulaire
            chanson_id = request.POST['chanson_id']
            chanson = get_object_or_404(Chanson, pk=chanson_id)

            # Tentez d'identifier la chanson
            audio_file = request.FILES['audio_file']
            identified_song = some_audio_library.identify_song(audio_file)

            # Si tout se passe bien, retournez la chanson identifiée
            return HttpResponse(f'Chanson identifiée : {identified_song}')

        except KeyError:
            # Si 'chanson_id' ou 'audio_file' n'est pas dans le formulaire, retournez une erreur 400
            return HttpResponseBadRequest('Requête mal formée : chanson_id ou audio_file manquant')

        except Chanson.DoesNotExist:
            # Si la chanson n'existe pas, retournez une erreur 404
            raise Http404('Chanson non trouvée')

        except some_audio_library.AudioFormatError:
            # Si le fichier audio est mal formaté, retournez une erreur 400
            return HttpResponseBadRequest('Fichier audio mal formaté')

    else:
        return HttpResponseBadRequest('Méthode non autorisée')
    
def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('home')
    else:
        form = LoginForm()
    return render(request, 'login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('login')

@login_required
def protected_view(request):
    # Cette vue nécessite une authentification
    pass
