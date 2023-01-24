from django.urls import path
from . import views

# these are all quiz urls
urlpatterns = [
    path("quiz/", views.index, name="index"),
    path("quiz/<str:slug>/", views.quiz, name="quiz"),
    path('quiz/<str:slug>/data/', views.quiz_data_view, name='quiz-data'),
    path('quiz/<str:slug>/save/', views.save_quiz_view, name='quiz-save'),
    
    # path("quiz/signup/", views.Signup, name="signup"),
    # path("quiz/login/", views.Login, name="login"),
    # path("quiz/logout/", views.Logout, name="logout"),
    
    path('add_quiz/', views.add_quiz, name='add_quiz'),
    path('edit_quiz/<str:slug>/', views.QuizUpdatePostView.as_view(), name='edit_quiz'),
    path('delete_quiz/<int:myid>/', views.delete_quiz, name='delete_quiz'),
    path('add_question/', views.add_question, name='add_question'),  
    path('add_options/<str:slug>/', views.UpdatePostView.as_view(), name='add_options'),
    path('add_ans/<int:myid>/', views.add_options, name='add_ans'),
    path('results/', views.results, name='results'),
    path('leaderboard/', views.LeaderBoard, name='leaderboard'),
    path('delete_question/<int:myid>/', views.delete_question, name='delete_question'),  
    path('delete_result/<int:myid>/', views.delete_result, name='delete_result'),    
]