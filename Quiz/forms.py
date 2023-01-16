from django import forms
from .models import Quiz, Question, Answer
from django.contrib import admin

class QuizForm(forms.ModelForm):
    class Meta:
        model = Quiz
        fields = ('name','slug', 'desc', 'number_of_questions', 'time')

        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Title of the Blog'}),
            'slug': forms.TextInput(
                attrs={'class': 'form-control', 'placeholder': 'Copy the title with no space and a hyphen in between'}),
        }

    
class QuestionForm(forms.ModelForm):
    class Meta:
        model = Question
        fields = ('content','slug','image', 'quiz')

        widgets = {
            'content': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Title of the Blog'}),
            'slug': forms.TextInput(
                attrs={'class': 'form-control', 'placeholder': 'give a unique name with no space and a hyphen in between'}),
        }
        
