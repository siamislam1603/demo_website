from django import forms
from .models import Profile, BlogPost
from django.contrib.auth.models import User

class ProfileForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ('phone_no', 'bio', 'facebook', 'instagram', 'linkedin', 'image' )
        widgets = {
            'phone_no': forms.TextInput(attrs={'class':'form-control', 'placeholder':'Enter your phone number'}),
            'bio': forms.Textarea(attrs={'class':'form-control', 'placeholder':'Enter your bio'}),
            'facebook': forms.TextInput(attrs={'class':'form-control', 'placeholder':'Enter your facebook link'}),
            'bio': forms.Textarea(attrs={'class':'form-control', 'placeholder':'Enter your bio'}),
            'instagram': forms.TextInput(attrs={'class':'form-control', 'placeholder':'Enter your instagram link'}),
            'bio': forms.Textarea(attrs={'class':'form-control', 'placeholder':'Enter your bio'}),
            'linkedin': forms.TextInput(attrs={'class':'form-control', 'placeholder':'Enter your linkedin link'}),
        }
        
class BlogPostForm(forms.ModelForm):
    class Meta:
        model = BlogPost
        fields = ('title', 'slug', 'content', 'image')
        widgets = {
            'title': forms.TextInput(attrs={'class':'form-control', 'placeholder':'Title of the Blog'}),
            'slug': forms.TextInput(attrs={'class':'form-control', 'placeholder':'Copy the title with no space and a hyphen in between'}),
            'content': forms.Textarea(attrs={'class':'form-control', 'placeholder':'Content of the Blog'}),
        }