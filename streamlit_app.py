import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write('''
# Application de Visualisation de la prédiction des données Iris
Cette démo prédit la catégorie des fleurs d'Iris
''')


st.sidebar.header("Les paramètres d'entrée")

#Paramétrage du modèle
def user_input():
    sepal_length=st.sidebar.slider('La longeur du Sepal',4.3,7.9,4.3)
    sepal_width=st.sidebar.slider('La lareur du Sepal',2.0,4.4,3.3)
    petal_length=st.sidebar.slider('La longueur du Petal',1.0,6.9,2.3)
    petal_width=st.sidebar.slider('La largeur du Petal',0.1,2.5,1.3)
    data={'sepal_length': sepal_length,
    'sepal_width': sepal_width,
    'petal_length': petal_length,
    'petal_width': petal_width
    }
    fleurs_parameters=pd.DataFrame(data,index=[0])
    return fleurs_parameters

df=user_input()

st.subheader('on veut trouver la catégorie de cette fleur')
st.write(df)

#Appel de notre modèle
#Import des données
iris=datasets.load_iris()
clf=RandomForestClassifier()
clf.fit(iris.data,iris.target)

prediction=clf.predict(df)

st.subheader("La catégorie de la fleur est:")
st.write(iris.target_names[prediction])

