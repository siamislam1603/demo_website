import json
from . import models
from django.shortcuts import render, redirect, HttpResponse
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from .models import *
from django.contrib.auth.decorators import login_required
from .forms import ProfileForm, BlogPostForm
from django.views.generic import UpdateView
from django.contrib import messages
# from django.views.decorators.csrf import csrf_exempt
# from PIL import Image
# import cv2
# import numpy
# import requests
from django.http.response import StreamingHttpResponse
from .camera import VideoCamera,bangla_det,Number_detection
from .camera import test


def firstpage(request):
    return render(request, "home.html")


def gen(camera):
    while True:
        try:

            frame = camera.get_frame()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        except:
            camera.stop()

def english_stream(request):
    return render(request, "streamapp/english.html")

def bangla_stream(request):
    return render(request, "streamapp/home.html")

def number_stream(request):
    return render(request, "streamapp/number.html")

def bangla_feed(request):
    return StreamingHttpResponse(gen(Number_detection()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

def video_stream(request):
    return StreamingHttpResponse(gen(VideoCamera()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

def number_feed(request):
    return StreamingHttpResponse(gen(bangla_det()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


@login_required(login_url='/login')
def blogs(request):
    posts = BlogPost.objects.all()
    posts = BlogPost.objects.filter().order_by('-dateTime')
    return render(request, "blog.html", {'posts': posts})


@login_required(login_url='/login')
def blogs_comments(request, slug):
    post = BlogPost.objects.filter(slug=slug).first()
    comments = Comment.objects.filter(blog=post)
    if request.method == "POST":
        user = request.user
        content = request.POST.get('content', '')
        blog_id = request.POST.get('blog_id', '')
        comment = Comment(user=user, content=content, blog=post)
        comment.save()
    return render(request, "blog_comments.html", {'post': post, 'comments': comments})


@login_required(login_url='/login')
def Delete_Blog_Post(request, slug):
    posts = BlogPost.objects.get(slug=slug)
    if request.method == "POST":
        posts.delete()
        return redirect('/')
    return render(request, 'delete_blog_post.html', {'posts': posts})


@login_required(login_url='/login')
def search(request):
    if request.method == "POST":
        searched = request.POST['searched']
        blogs = BlogPost.objects.filter(title__contains=searched)
        return render(request, "search.html", {'searched': searched, 'blogs': blogs})
    else:
        return render(request, "search.html", {})


@login_required(login_url='/login')
def add_blogs(request):
    if request.method == "POST":
        form = BlogPostForm(data=request.POST, files=request.FILES)
        if form.is_valid():
            blogpost = form.save(commit=False)
            blogpost.author = request.user
            blogpost.save()
            obj = form.instance
            alert = True
            return render(request, "add_blogs.html", {'obj': obj, 'alert': alert})
    else:
        form = BlogPostForm()
    return render(request, "add_blogs.html", {'form': form})


class UpdatePostView(UpdateView):
    model = BlogPost
    template_name = 'edit_blog_post.html'
    fields = ['title', 'slug', 'content', 'image']


def user_profile(request, myid):
    post = BlogPost.objects.filter(id=myid)
    return render(request, "user_profile.html", {'post': post})


def Profile_v(request, *args, **kwargs):
    return render(request, "profile.html")


def edit_profile(request):
    import urllib
    # request = urllib.request.Request(request, None, headers)
    try:
        profile = request.user.profile
        # print(request.user.password)
    except Exception as e:
        profile = Profile(request)
        # print("profile created....")

    print(request.method)
    if request.method == "POST":
        print("Post Method")
        profile = Profile(user=request.user)
        form = ProfileForm(request.POST, request.FILES,instance=profile)
        # form.fields['user'] = request.user

        # print(form)
        if form.is_valid():
            form.save()
            alert = True
            return render(request, "edit_profile.html", {'alert': alert,'form': form})
    else:
        try:

            form = ProfileForm(instance = profile)
        except:
            profile = Profile(request)
            form = ProfileForm(profile)
    return render(request, "edit_profile.html", {'form': form})

def view_user_account(request):
    users = User.objects.all()
    users = User.objects.filter().order_by('-id')
    return render(request, "user_accounts.html", { 'users': users})


def delete_user(request, username):

    u = User.objects.get(username=username)

    if request.method == "POST":
        u = User.objects.get(username=username)
        u.delete()
        return redirect('/view_user/')

    return render(request, "delete_user.html", {'user': u})


import re


# Define a function for
# for validating an Email
def check(s):
    pat = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    if re.match(pat, s):
        return False
    else:
        return True


def Register(request):
    alert = False
    email_st = False
    if request.method == "POST":
        username = request.POST['username']
        email = request.POST['email']
        if check(email):
            email_st = True
        first_name = request.POST['first_name']
        last_name = request.POST['last_name']
        password1 = request.POST['password1']
        password2 = request.POST['password2']

        if password1 != password2:
            alert = True
            messages.error(request, "Passwords do not match.")
            return render(request, "register.html",{'info':' Passwords do not match.','alert':alert})
        if email_st:
            messages.error(request, "Invalid Email.")
            return render(request, "register.html", {'info': ' Invalid Email.', 'email': email_st})

        else:
            alert = False

        user = User.objects.create_user(username, email, password1)
        user.first_name = first_name
        user.last_name = last_name
        user.save()
        return render(request, 'login.html')
    return render(request, "register.html")


def Login(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(username=username, password=password)

        if user is not None:
            login(request, user)
            messages.success(request, "Successfully Logged In")
            return redirect("/")
        else:
            alert = True
            messages.error(request, "Invalid Credentials")
            return render(request, "login.html",{'info':' Please enter valid info','alert':alert})
        # return render(request, 'blog.html')
    return render(request, "login.html")


def Logout(request):
    logout(request)
    messages.success(request, "Successfully logged out")
    return redirect('/login')

def Verify(request):
    return render(request,"verify.html")

def change_password(request):
    if request.method == "POST":
        username = request.POST['username']
        password1 = request.POST['password1']
        password2 = request.POST['password2']
        if password1 != password2:
            alert = True
            messages.error(request, "Passwords do not match.")
            return render(request, "change_password.html",{'info':' Passwords do not match.','alert':alert})
        # password = request.POST['confirmpassword']
        from django.contrib.auth.models import User
        try:

            u = User.objects.get(username__exact=username)
            u.set_password(password1)
            u.save()
            return redirect('/login')
        except:
            alert_u = True
            return render(request, "change_password.html", {'info': ' Passwords do not match.', 'alert_u': alert_u})

    return render(request, "change_password.html")

