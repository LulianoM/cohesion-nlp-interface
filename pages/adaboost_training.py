import os, sys

import streamlit as st

PATH = os.path.dirname(os.path.realpath(__file__)).replace('/pages', '') # PATH
sys.path.append(PATH)

from modules.spacy_adaboost import TrainingTest

st.markdown("# Adaboost Training Page 📊")
st.sidebar.markdown("# Processando os pesos 📊")

file = st.file_uploader('Selecione seu arquivo .xlsx para coleta dos dados para treinamento')
    
container_button = st.container()
    
if container_button.button("Processar"):
    tt = TrainingTest()
    
    grouped = tt.process(file)
    
    st.dataframe(tt.df)

    st.dataframe(grouped.mean())
    st.dataframe(grouped.std())
