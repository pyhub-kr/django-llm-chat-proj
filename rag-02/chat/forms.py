from django import forms
from .models import PaikdabangMenuDocument


class PaikdabangMenuDocumentForm(forms.ModelForm):
    class Meta:
        model = PaikdabangMenuDocument
        fields = ["page_content", "metadata"]
