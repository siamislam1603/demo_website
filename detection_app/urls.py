from django.urls import path
from . import views

urlpatterns = [
    path("det/", views.index, name="det_index"),
    path("view_det_eng/", views.view_page_eng, name="det_eng"),
    path("view_det_ban/", views.view_page_ban, name="det_ban"),
    path("view_det_num/", views.view_page_num, name="det_num"),
    path("det/<str:slug>/", views.UpdatePostView.as_view(), name="edit_det"),
    path("delete_detection/<int:myid>/", views.delete_detection, name="del_det"),
    # path("edit_blog_post/<str:slug>/", views.UpdateDetView.as_view()),
    # path("det/<int:myid>/", views.edit_det, name="det"),
]