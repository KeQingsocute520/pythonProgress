# coding=utf-8
from django.urls import path
from . import views

urlpatterns = [
    path('register/', views.index_view),
    path('login/', views.login_view),
    path('test/', views.test_view),
    path('main/', views.main_view),
    path('home/', views.home_view),

    path('homepage/', views.homepage_view),
    path('director/', views.director_view),
    path('news/', views.news_view),
    path('knowledge/', views.knowledge_view),
    path('warning/', views.warning_view),
    path('chinaMap/', views.chinaMap_view),
    path('jiangsu/', views.jiangsu_view),
    path('search/', views.search_view),

    path('nanjing/', views.nanjing_view),
    path('wuxi/', views.wuxi_view),
    path('yancheng/', views.yancheng_view),
    path('nantong/', views.nantong_view),
    path('suzhou/', views.suzhou_view),
    path('yangzhou/', views.yangzhou_view),
    path('lianyungang/', views.lianyungang_view),
    path('zhenjiang/', views.zhenjiang_view),
    path('taizhou/', views.taizhou_view),
    path('changzhou/', views.changzhou_view),
    path('huaian/', views.huaian_view),
    path('suqian/', views.suqian_view),
    path('xuzhou/', views.xuzhou_view)

]
