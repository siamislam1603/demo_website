from django.db import models
from django.contrib.auth.models import User
import random
from django.urls import reverse


class Quiz(models.Model):
    name = models.CharField(max_length=50)
    slug = models.CharField(max_length=130, default="")
    desc = models.CharField(max_length=500)
    number_of_questions = models.IntegerField(default=1)
    time = models.IntegerField(help_text="Duration of the quiz in seconds", default="1")

    def __str__(self):
        return self.name

    def get_questions(self):
        return self.question_set.all()

    def get_absolute_url(self):
        return reverse('add_quiz')

class Question(models.Model):
    content = models.CharField(max_length=200)
    slug = models.CharField(max_length=130, default="")
    image = models.ImageField(upload_to="images/quiz", blank=True, null=True)
    quiz = models.ForeignKey(Quiz, on_delete=models.CASCADE)

    def __str__(self):
        return self.content

    def get_answers(self):
        return self.answer_set.all()

    def get_absolute_url(self):
        return reverse('add_question')


class Answer(models.Model):
    content = models.CharField(max_length=200)
    correct = models.BooleanField(default=False)
    question = models.ForeignKey(Question, on_delete=models.CASCADE)

    def __str__(self):
        return f"question: {self.question.content}, answer: {self.content}, correct: {self.correct}"


class Marks_Of_User(models.Model):
    quiz = models.ForeignKey(Quiz, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    score = models.FloatField()
    time_taken = models.IntegerField(default=1)

    def __str__(self):
        return str(self.quiz)


