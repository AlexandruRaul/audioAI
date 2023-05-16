from django.urls import path
from . import views

app_name = 'music'

urlpatterns = [
    path('charger/', views.charger_chanson, name='charger_chanson'),
    path('identifier-chanson/', views.identifier_chanson, name='identifier_chanson'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
]
