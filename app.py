#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import pickle
import warnings
import streamlit as st


print("welcome");
model_loaded = []

st.markdown('''
<body style="background-color:white;">
      <center>  <h1>Book Recommedation App</h1>  </center>
       <h3> Welcome to the book Recommedation System </h3>
         
</body>
   ''',unsafe_allow_html=True)



st.sidebar.title("Text mining - Lib data \n \n")

st.sidebar.subheader("Project Contributors \n\n");
st.sidebar.text(" Muhammad Umar(Id Number)")
st.sidebar.text(" Manish kumar (Id Number)")

st.sidebar.subheader("**Project Details**");
st.sidebar.markdown(" *This Project is designed to recommends books to the user by its similar nature and top rating which means user just need to type its favourite book and our system eventually recommended best books for them based on similar nature and top rating.*");


@st.cache(suppress_st_warning=True)  # 👈 This function will be cached
def load():
    us_canada_user_rating  =pd.read_csv("us_canada_user_rating.csv")
    us_canada_user_rating_pivot = us_canada_user_rating.reset_index().pivot_table(index = 'bookTitle', columns= 'userID', values = 'bookRating').fillna(0)
    us_canada_user_rating_pivot2 = us_canada_user_rating.reset_index().pivot_table(index = 'userID', columns= 'bookTitle', values = 'bookRating').fillna(0)
    #st.write("function loaded")
    return us_canada_user_rating,us_canada_user_rating_pivot,us_canada_user_rating_pivot2 
    
    
us_canada_user_rating,us_canada_user_rating_pivot,us_canada_user_rating_pivot2 =load();   




us_canada_book_title = us_canada_user_rating_pivot2.columns
us_canada_book_list = list(us_canada_book_title)

print("Data Loaded");

# load the model from disk
loaded_model = pickle.load(open('finalized_knn_model.sav', 'rb'))

#DISPLAY TITLE "My Diabetes Prediction App"
#st.title('Book Recommedation App  \n\n ')

#st.subheader('Welcome to the book Recommedation System ')

book = st.text_input('Which is your favourite Book:')
num =  st.number_input("How many Recommedation you want?",key="Recommedations",value=int(5))

def predict(book_name,num):    
    
    
    query_index1= us_canada_book_list.index(book_name)
    

    distances, indices = loaded_model.kneighbors(us_canada_user_rating_pivot.iloc[query_index1,:].values.reshape(1, -1), n_neighbors = num+1)
    
    	
    return query_index1,distances,indices


#CREATING A BUTTON TO EXECUTE MODEL PREDICTION
st.write('\n','\n')

if st.button("Check out the Recommedations"):

    st.write('\n','\n')
    query_index1,distances,indices=predict(book,num)
    
    #DISPLAYING MODEL OUTPUT AND PREDICTION PROBABILITY 
    for i in range(0, len(distances.flatten())):
        if i == 0:
            st.subheader('Recommendations for {0}:\n'.format(us_canada_user_rating_pivot.index[query_index1]))
        else:
            st.write('{0}: is  **{1}** '.format(i, us_canada_user_rating_pivot.index[indices.flatten()[i]]))
