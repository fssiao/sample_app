import streamlit as st
import pandas as pd
import os
#import tkinter

from fastai.vision.all import *
from fastai.text import *


def load_model(model_path):
    return load_learner(model_path)

def predict_text(model, text_input):
    pred, _, _ = model.predict(text_input)
    return pred


def main():
    model_path = 'export.pkl'
    loaded_model = load_model(model_path)

    st.header('Customer Inquiry Classification')
    st.text('Enter inquiry')
    user_input = st.text_input('Inquiry')

    inquiry_class = predict_text(loaded_model, user_input)
    st.text('Classification: {}'.format(inquiry_class))

    with st.sidebar:
        st.text('Inquiry history')
        st.text(user_input)
        st.text('Classification: {}'.format(inquiry_class))


if __name__ == '__main__':
    main()