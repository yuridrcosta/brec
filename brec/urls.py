from django.urls import path

from . import views

urlpatterns = [
    path('',views.homeView),
    path('text/<int:id>',views.textView,name="text-view"),
]
