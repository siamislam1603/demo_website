from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse



class info(models.Model):
    detection_name = models.CharField(max_length=300, blank=True, null=True)
    slug = models.CharField(max_length=130,default="")
    image_1 = models.ImageField(upload_to="images", blank=True, null=True)
    image_2 = models.ImageField(upload_to="images", blank=True, null=True)
    video = models.FileField(upload_to='videos', blank=True, null=True)


    def get_absolute_url(self):
        return reverse('det_index')
