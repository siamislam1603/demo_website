from django import forms
from .models import info

class detectionForm(forms.ModelForm):
    class Meta:
        model = info
        fields = ('detection_name','slug', 'image_1', 'image_2', 'video')
        widgets = {
            'detection_name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Title of the Blog'}),
            'slug': forms.TextInput(
                attrs={'class': 'form-control', 'placeholder': 'Copy the title with no space and a hyphen in between'}),
        }
