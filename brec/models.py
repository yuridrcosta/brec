from django.db import models


class TextEvaluation(models.Model):

    text_content = models.TextField()

    def __str__(self):
        return self.text_content[:15]

