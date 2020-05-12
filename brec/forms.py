from django import forms

from .models import TextEvaluation

class TextEvaluationForm(forms.ModelForm):

    class Meta:
        model = TextEvaluation
        fields = ('text_content',)
