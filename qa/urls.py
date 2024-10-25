from django.contrib.auth import views as auth_views
from django.urls import path
from . import views

urlpatterns = [
    path('', views.mainpage,name="mainpage"),
    path('signup/',views.signup,name='signup'),
    path('qa_page/', views.qa_page, name='qa_page'),
    path('login/',views.login_views,name='login'),
    path('',views.logout_views,name='logout'),
]
