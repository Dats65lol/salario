import pandas as pd 
import streamlit as st 
import plotly.express as px
import pickle as pkl
import numpy as np

wage = pd.read_csv('wage.csv')

st.title("Salario Esperado")

with open("model.pickle", "rb") as m:
    modelo = pkl.load(m)

tab1, tab2, tab3 = st.tabs(['Análisis Univariado', 'Análisis bivariado', 'Modelo'])

with tab1: 

    st.subheader("Estadísticas descriptivas - Variables numéricas")
    st.dataframe(wage[['wage', 'educ', 'exper', 'Permanencia']].describe())
 
    st.subheader("Distribución de variables numéricas")

    fig = px.histogram(wage, x="wage")
    st.plotly_chart(fig)

    fig2 = px.histogram(wage, x="educ")
    st.plotly_chart(fig2)

    fig3 = px.histogram(wage, x="exper")
    st.plotly_chart(fig3)

    fig4 = px.histogram(wage, x="Permanencia")
    st.plotly_chart(fig4)

    st.subheader("Frecuencias - Variables categóricas")

    st.write("Género")
    st.dataframe(wage['sexo'].value_counts())
 
    st.write("Estado_civil")
    st.dataframe(wage['Estado_civil'].value_counts())
 
    st.subheader("Distribución de variables categóricas")

    fig5 = px.bar(wage, x="sexo")
    st.plotly_chart(fig5)

    fig6 = px.bar(wage, x="Estado_civil")
    st.plotly_chart(fig6)


with tab2:
    
    st.subheader("Relación entre variables y salario")
    fig1 = px.scatter(wage, x='educ', y='wage', title='Educación vs Salario')
    st.plotly_chart(fig1)
 
    fig2 = px.scatter(wage, x='exper', y='wage', title='Experiencia vs Salario')
    st.plotly_chart(fig2)
 
    fig3 = px.box(wage, x='sexo', y='wage', title='Salario por Género')
    st.plotly_chart(fig3)
 
    fig4 = px.box(wage, x='Estado_civil', y='wage', title='Salario por Estado Civil')
    st.plotly_chart(fig4)




with tab3:

    
    st.title("Modelo")
 
    educ = st.slider("Años de educación", 0, 18)
 
    exper = st.slider("Años de experiencia", 1, 51)
 
    Permanencia= st.slider("Años en la empresa", 0, 44)
 
    sexo = st.selectbox('sexo', ['mujer', 'hombre'])

    if sexo == 'mujer':
        sexo = 1
    else:
        sexo = 0

    Estado_civil = st.selectbox('Estado_civil', ['casado', 'soltero'])

    if Estado_civil == 'casado':
        Estado_civil = 1 
    else:
        Estado_civil = 0
 
    if st.button ("Predecir"):
        Predecir = modelo.predict(np.array([[educ, exper, Permanencia, sexo, Estado_civil]]))
        st.write(Predecir [0])

print(10)
