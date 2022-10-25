from modules.openai import OpenAI
import streamlit as st

st.markdown("# OpenAI Module ❄️")
st.sidebar.markdown("# Page 1 ❄️")

def instert_text():
    key =st.text_input("Insira sua key do openai")
    txt = st.text_area("Insira o texto", height=250)
    colum1, colum2,colum3,colum4,colum5 = st.columns([1,1,1,1,1])
    
    if colum1.button("Corrigir"):
        with st.spinner(text='en progreso'):
            
            new_txt, status = OpenAI.connect_openai(txt, key)
            
            if status == 200:
                st.text_area(label="Nota do texto:", value=new_txt)
                st.success("Corrigido!!")  
            else:
                st.text_area(label="Error:", value=new_txt["Error"])
                st.error(new_txt["Error"]) 
    
    if colum2.button("Limpar texto"):
        st.info("limpando")

instert_text()
