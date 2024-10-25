from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, authenticate, logout
from django.shortcuts import render,redirect
from .forms import SignUpForm,QuestionForm
from .models import QAModel
from .utils import get_google_ai_response, fake_news_detection, sentimentanalysis, security_alert

# Create your views here.
def mainpage(request):
    logout(request)
    return render(request,"qa/index.html")

def signup(request):
    if request.method=='POST':
        form=SignUpForm(request.POST)

        if form.is_valid():
            user=form.save()
            
            security_alert(user)

            login(request,user)
            return redirect('qa_page')
    else:
        form=SignUpForm()
    return render(request,'qa/signup.html',{'form':form})



def qa_page(request):
    if request.user.is_authenticated:
        answer=sentiment=score=fact= None
        if request.method == 'POST':
            form = QuestionForm(request.POST)
            if form.is_valid():
                question = form.cleaned_data['question']
                answer = fake_news_detection(question)
                # Save to the database
                sentiment,score=sentimentanalysis(question)
                fact=get_google_ai_response(question)
                # QAModel.objects.create(
                #     user=request.user,
                #     question=question,
                #     answer=answer
                # )
        else:
            form = QuestionForm()

        return render(request, 'qa/qa_page.html', {
            'form': form,
            'answer': answer,
            'sentiment_result': sentiment,
            'score':score,
            'fact': fact
        })
    else:
        return redirect('login')
        
    
def login_views(request):
    logout(request)
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('qa_page')
        else:
            pass
    return render(request, 'qa/login.html')

def logout_views(request):
    logout(request)
    return redirect('mainpage')