# coding=utf-8
from django.urls import path
from . import views

urlpatterns = [
    path('register/', views.index_view),
    path('login/', views.login_view),
    path('test/', views.test_view),
    path('main/', views.main_view),

    path('homepage/', views.homepage_view),
    path('director/', views.director_view),
    path('news/', views.news_view),
    path('knowledge/', views.knowledge_view),
    path('warning/', views.warning_view)
]
