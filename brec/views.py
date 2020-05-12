from django.shortcuts import render,get_object_or_404,redirect
from . import neuralnet

from .forms import TextEvaluationForm
from .models import TextEvaluation

# Create your views here.

def homeView(request):
    if request.method == 'POST':
        form = TextEvaluationForm(request.POST)

        if form.is_valid():
            task = form.save(commit=False)
            task.save()
            return redirect('text/{}'.format(task.id))
    else:
        form = TextEvaluationForm()
        return render(request,'brec/index.html',{'form':form})

def textView(request,id):
    tbd = get_object_or_404(TextEvaluation,pk=id)
    text_evaluated = neuralnet.run_evaluation(tbd.text_content)
    return render(request,'brec/text.html',{'tbd': tbd,'text_evaluated':text_evaluated})
