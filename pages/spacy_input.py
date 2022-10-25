import os, sys
import streamlit as st
import string

PATH = os.path.dirname(os.path.realpath(__file__)).replace('/pages', '') # PATH
sys.path.append(PATH)

from modules.spacy import TrainingTest

st.markdown("# SpaCy Process Text 📊")
st.sidebar.markdown("# Processando o texto 📊")

text = st.text_area('Texto')

container_button = st.container()
    
if container_button.button("Processar"):
    print(text)
    a = text.apply(lambda x: [token for token in x if token not in string.punctuation and not token.isnumeric()])
    a = a.apply(lambda x: ''.join(x))
    print(a)
