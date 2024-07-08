from django.http import HttpResponse
from django.shortcuts import render
from .models import *
# Create your views here.

def nanjing_view(request):
    return render(request, 'nanjing.html')
def wuxi_view(request):
    return render(request, 'wuxi.html')

def yancheng_view(request):
    return render(request, 'yancheng.html')

def nantong_view(request):
    return render(request, 'nantong.html')

def suzhou_view(request):
    return render(request, 'suzhou.html')

def yangzhou_view(request):
    return render(request, 'yangzhou.html')

def lianyungang_view(request):
    return render(request, 'lianyungang.html')

def zhenjiang_view(request):
    return render(request, 'zhenjiang.html')

def taizhou_view(request):
    return render(request, 'taizhou.html')

def changzhou_view(request):
    return render(request, 'changzhou.html')

def huaian_view(request):
    return render(request, 'huaian.html')

def suqian_view(request):
    return render(request, 'suqian.html')

def xuzhou_view(request):
    return render(request, 'xuzhou.html')



def home_view(request):
    return render(request, 'home.html')
def login_view(request):
    return render(request, 'login.html')

def main_view(request):
    return render(request, 'main.html')

def homepage_view(request):
    return render(request, 'homepage.html')

def director_view(request):
    return render(request, 'director.html')

def news_view(request):
    return render(request, 'news.html')

def knowledge_view(request):
    return render(request, 'knowledge.html')

def warning_view(request):
    return render(request, 'warning.html')

def chinaMap_view(request):
    return render(request, 'chinaMap.html')

def jiangsu_view(request):
    return render(request, 'jiangsu.html')

def search_view(request):
    return render(request, 'search.html')

def index_view(request):
    m = request.method
    if m == 'GET':
        return render(request, 'register.html')
    else:
# 获取请求参数
        uname = request.POST.get('uname', '')
        pwd = request.POST.get('pwd', '')
        print(uname)
        print(pwd)
# 判断
        if uname and pwd:
# 创建模型对象
            stu = Student(sname=uname, spwd=pwd)
# 插入数据库
            stu.save()
            return HttpResponse('注册成功')
        else:
            return HttpResponse('注册失败')

def test_view(request):
    m = request.method
    if m == 'GET':
        return render(request, 'test.html')
    else:
        uname = request.POST.get('uname','')
        truedata = []
        all_student = Student.objects.filter(sname=uname)
        print(all_student)
        for i in all_student:
            result = {
                'name':i.sname,
                'pwd':i.spwd,
                'sex':i.sex
            }
            truedata.append(result)
        print(truedata)
        truedata = {
            'info':truedata[0]
        }
        return render(request, 'test.html', {'fontdata': truedata})