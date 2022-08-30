
from django.urls import path
from django.views.generic import TemplateView
from . import views
  
urlpatterns = [
    path('', views.index, name='index'),
    path('open_jupiter_notbook/', TemplateView.as_view(template_name ="Project1.html"), name='open_jupiter_notbook'),
    path('Face_Mask', views.Face_Mask, name='Face_Mask'),
    path('facecam_feed', views.facecam_feed, name='facecam_feed'),
    path('Face_Mask_train', views.Face_Mask_train, name='Face_Mask_train'),
]