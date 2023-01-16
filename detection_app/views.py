from django.shortcuts import render, redirect
from .models import *
from django.http import JsonResponse
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from .forms import detectionForm
from django.forms import inlineformset_factory
from django.views.generic import UpdateView


def view_page_eng(request):
    det = info.objects.get(detection_name='english')
    print("det",det)
    # para = {'det': det}
    return render(request, "view_page.html",{'det':det})

def view_page_ban(request):
    det = info.objects.get(detection_name='bangla')
    print("det",det)
    # para = {'det': det}
    return render(request, "view_page_ban.html",{'det':det})

def view_page_num(request):
    det = info.objects.get(detection_name='number')
    print("det",det)
    # para = {'det': det}
    return render(request, "view_page_num.html",{'det':det})

# def edit_det(request,myid):
#     det = info.objects.get(id =myid)
#     print("det",det)
#     para = {'det': det}
#     return render(request, "detection_page.html", para)

class UpdatePostView(UpdateView):
    model = info
    template_name = 'edit_det_post.html'
    fields = ['detection_name','slug', 'image_1', 'image_2', 'video']

def edit_det(request,myid):
    det = info.objects.get(id=myid)
    print(det.detection_name)
    # det_form = detectionForm(instance=det)
    if request.method == "POST":
        formset = detectionForm(request.POST, instance=det)
        if formset.is_valid():
            formset.save()
            alert = True
            return render(request, "edit_det_post.html", {'alert': alert})
    else:
        formset = detectionForm(instance=det)
    return render(request, "edit_det_post.html", {'formset': formset, 'detection': det})

def index(request):

    detections = info.objects.all()
    detections = info.objects.filter().order_by('-id')
    if request.method == "POST":
        form = detectionForm(request.POST,request.FILES)
        if form.is_valid():
            form.save()
            return render(request, "edit_detection.html")
    else:
        form = detectionForm()
    return render(request, "edit_detection.html", {'form': form, 'detections': detections})

def delete_detection(request, myid):
    detection = info.objects.get(id=myid)
    if request.method == "POST":
        detection.delete()
        return redirect('/det')
    return render(request, "delete_question.html", {'question': detection})

class UpdateDetView(UpdateView):
    model = info
    template_name = 'edit_det_post.html'
    fields = ['detection_name', 'image', 'video']
