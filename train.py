import streamlit as st
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from jcopml.pipeline import cat_pipe, num_pipe
from sklearn.naive_bayes import GaussianNB
from jcopml.plot import plot_confusion_matrix
from plotly.subplots import make_subplots
import pickle


def train(X,Y):
    clf = GaussianNB()
    clf = clf.fit(X,Y)
    pickle.dump(clf, open("naive_bayes_model.p", "wb"))
    
    
def model(data, pickle_filename):
    Z = data.sar_diff, data.ma_diff
    data.drop(columns="trend")
    clf = pickle.load(open(pickle_filename,"rb"))
    data['trend'] = clf.predict(Z)
    st.write(data)



def DataTraining():
    X = df.drop(columns="trend")
    y = df.trend
    X_train, X_test , y_train, y_test = train_test_split(X,y)
    preprocessor = ColumnTransformer([('numeric', num_pipe(),["ma_diff","sar_diff"]),])
    pipeline = Pipeline([('prep', preprocessor),('algo',GaussianNB())])
    pipeline.fit(X_train,y_train)
    display_results(X_train,X_pred)
    plot_confusion_matrix(X_train,y_train,X_test,y_test, pipeline)
    ShowNB()
   

def ShowNB():
    st.write('')
    st.subheader('Plot Confussion Matrix') 
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig = plt.show()
    st.pyplot(fig)

   

def display_results(X_train,X_pred):
    st.subheader('Data Training')
    st.write(X_train)
    

   
if __name__=="__main__":
    df = pd.read_excel('datatest.xlsx')
    X_pred = pd.read_excel('datatest.xlsx')
    st.header("Brekdown Dataset dengan Niave Bayes")
    st.write(df)
    
    model(X_pred,"naive_bayes_model.p")
        
   
  
   
   







                                                                                                   
