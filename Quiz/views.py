from django.shortcuts import render, redirect
from .models import *
from django.http import JsonResponse
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from .forms import QuizForm, QuestionForm
from django.forms import inlineformset_factory
from django.views.generic import UpdateView
import random


# This is function is for running the server request active all the time
def is_ajax(request):
    return request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest'

# this function fetch all of the quiz and show in a table view
def index(request):
    quiz = Quiz.objects.all()
    print("quiz",quiz)
    para = {'quiz': quiz}
    return render(request, "index_page.html", para)


# this function get a specific quiz according to the detection that the user has chosen
@login_required(login_url='/login')  # this line ensures that the quiz will be available only when a user gets logged in
def quiz(request, slug):
    quiz = Quiz.objects.get(slug=slug) # slug is the primary key based on which we search a specific quiz
    # print(quiz.name)
    return render(request, "quiz.html", {'quiz': quiz})


# this function takes a quiz and fetch all of the questions, options and images and send them to the html page
def quiz_data_view(request, slug):
    quiz = Quiz.objects.get(slug=slug)

    questions = []
    images = []

    for q in quiz.get_questions():
        answers = []
        images.append(q.image.url) # adding the images of each ques
        for a in q.get_answers():
            # print(a)
            answers.append(a.content) # adding all the options
        try:
            random.shuffle(answers) # shuffling options
        except:
            pass
        questions.append({str(q): answers})

    temp = list(zip(questions, images))
    random.shuffle(temp) # shuffling questions and answers
    questions, images = zip(*temp)
    print(images)

    return JsonResponse({
        'data': questions[:7], # sending only seven ques
        'time': quiz.time,
        'images':images[:7],
    })


# this function evaluate the quiz score
def save_quiz_view(request, slug):
    if is_ajax(request):
        questions = []
        data = request.POST
        data_ = dict(data.lists())

        data_.pop('csrfmiddlewaretoken')
        time_taken=int(data_['time_taken'][0])
        data_.pop('time_taken')

        for k in data_.keys():
            print('key: ', k)
            question = Question.objects.get(content=k)
            questions.append(question)

        user = request.user
        quiz = Quiz.objects.get(slug=slug)

        score = 0
        marks = []
        correct_answer = None

        for q in questions:
            a_selected = request.POST.get(q.content)

            if a_selected != "":
                question_answers = Answer.objects.filter(question=q)
                for a in question_answers:
                    if a_selected == a.content:
                        if a.correct:
                            score += 1
                            correct_answer = a.content
                    else:
                        if a.correct:
                            correct_answer = a.content

                marks.append({str(q): {'correct_answer': correct_answer, 'answered': a_selected}})
            else:
                marks.append({str(q): 'not answered'})

        Marks_Of_User.objects.create(quiz=quiz, user=user, score=score,time_taken=time_taken) # creating entry by using the user quiz score

        return JsonResponse({'passed': True, 'score': score, 'marks': marks, 'time_taken':time_taken})


# this added new quiz
def add_quiz(request):
    quizs = Quiz.objects.all()
    quizs = Quiz.objects.filter().order_by('-id')
    if request.method == "POST":
        form = QuizForm(request.POST)
        if form.is_valid():
            form.save()
            return render(request, "add_quiz.html")
    else:
        form = QuizForm()
    return render(request, "add_quiz.html", {'form': form, 'quizs': quizs})

# for deleting a quiz
def delete_quiz(request, myid):
    quiz = Quiz.objects.get(id=myid)
    if request.method == "POST":
        quiz.delete()
        return redirect('/add_quiz')
    return render(request, "delete_quiz.html", {'question': quiz})


def add_question(request):
    questions = Question.objects.all()
    questions = Question.objects.filter().order_by('-id')
    if request.method == "POST":
        form = QuestionForm(request.POST)
        if form.is_valid():
            form.save()
            return render(request, "add_question.html")
    else:
        form = QuestionForm()
    return render(request, "add_question.html", {'form': form, 'questions': questions})


def delete_question(request, myid):
    question = Question.objects.get(id=myid)
    if request.method == "POST":
        question.delete()
        return redirect('/add_question')
    return render(request, "delete_question.html", {'question': question})


class QuizUpdatePostView(UpdateView):
    model = Quiz
    template_name = 'edit_quiz.html'
    fields = ['name', 'slug', 'desc', 'number_of_questions','time']

class UpdatePostView(UpdateView):
    model = Question
    template_name = 'add_options.html'
    fields = ['content', 'slug', 'image', 'quiz']

# multiple choices adding module for quizs
def add_options(request, myid):
    question = Question.objects.get(id=myid)
    QuestionFormSet = inlineformset_factory(Question, Answer, fields=('content', 'correct', 'question'), extra=4)
    if request.method == "POST":
        formset = QuestionFormSet(request.POST, instance=question)
        if formset.is_valid():
            formset.save()
            alert = True
            return render(request, "add_ans.html", {'alert': alert})
    else:
        formset = QuestionFormSet(instance=question)
    return render(request, "add_ans.html", {'formset': formset, 'question': question})

# shows top 10 scores as leaderboard
def LeaderBoard(request):
    marks = Marks_Of_User.objects.all()
    marks = Marks_Of_User.objects.filter().order_by('-score')[:10]
    return render(request, "result_user.html", {'marks': marks})

# result page for admin
def results(request):
    marks = Marks_Of_User.objects.all()
    marks = Marks_Of_User.objects.filter().order_by('-score')
    return render(request, "results.html", {'marks': marks})

# delete user result when logged in as an admin
def delete_result(request, myid):
    marks = Marks_Of_User.objects.get(id=myid)

    if request.method == "POST":
        marks.delete()
        return redirect('/results')
    return render(request, "delete_result.html", {'marks': marks})
