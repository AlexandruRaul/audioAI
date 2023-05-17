from django.urls import path
from . import views
from music.views import charger_chanson
from music.views import identifier_chanson

app_name = 'music'

urlpatterns = [
    path('charger/', views.charger_chanson, name='charger_chanson'),
    path('identifier/', views.identifier_chanson, name='identifier_chanson'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('erreur/', views.erreur_view, name='erreur'),	
]
