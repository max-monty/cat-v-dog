import gradio as gr
from fastai.vision.all import *


def is_cat(x): return x[0].isupper()


learn = load_learner('model.pkl')
labels = learn.dls.vocab


def predict(img):
    img = PILImage.create(img)
    pred, pred_idx, p = learn.predict(img)
    labels = ['dog', 'cat']
    return {labels[i]: float(p[i]) for i in range(len(labels))}


title = "Pet Breed Classifier"
description = "A pet breed classifier trained on the Oxford Pets dataset with fastai. Created as a demo for Gradio and HuggingFace Spaces."
article = "<p style='text-align: center'><a href='https://tmabraham.github.io/blog/gradio_hf_spaces_tutorial' target='_blank'>Blog post</a></p>"
examples = ['siamese.jpg']

gr.Interface(fn=predict, inputs=gr.Image(), outputs=gr.Label(num_top_classes=2), title=title,
             description=description, article=article, examples=examples).launch(share=True)
