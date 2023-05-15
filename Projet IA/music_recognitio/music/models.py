from django.db import models

class Chanson(models.Model):
    titre = models.CharField(max_length=200)
    artiste = models.ForeignKey('Artiste', on_delete=models.CASCADE)
    # autres champs ici

class Artiste(models.Model):
    nom = models.CharField(max_length=200)
    # autres champs ici

    def __str__(self):
        return f"{self.titre} - {self.artiste}"
    
    