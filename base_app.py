"""

    Simple Streamlit webserver application for serving developed classification
    models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
    application. You are expected to extend the functionality of this script
    as part of your predict project.

    For further help with the Streamlit framework, see:

    https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os


# Data dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# Load data
df = pd.read_csv('https://github.com/TomK24/classification-predict-Team-1/blob/master/classification/train.csv?raw=true')

#Split sentiment accordingly
def convert_sent(val):
    if val == -1:
        return 'Anti'
    elif val == 0:
        return 'Neutral'
    elif val == 1:
        return 'Pro'
    else:
        return 'News'
df["sentiment"] = df["sentiment"].apply(convert_sent)
df.head()

# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """
    # Creates a main title and subheader on your page -
    # these are static across all pages
    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Make Predictions","Explore The Data", "Model Information"]
    selection = st.sidebar.selectbox("Choose Option", options)
    # Creating a page with m
    #Building out the 'About This App' page
    # Building out the predication page
    if selection == "Make Predictions":
        st.image("https://i.imgur.com/vUdzKY3.png")
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Tweet Below⬇️","")
        options = ["Naive Bayes Classifier","Linear Support Vector Classifier", "Multinomial Logistical Regression"]
        selection = st.selectbox("Choose Your Model😬", options)

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
            predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
            prediction = predictor.predict(vect_text)

            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success("Text Categorized as: {}".format(prediction))

    #Building out the 'EDA' page
    if selection == 'Explore The Data':
        st.title("Explore the Data!")
        st.image("https://imgur.com/j8EXXWA.jpg")
        st.sidebar.subheader("Exploring The Data")
        st.image("https://static.thenounproject.com/png/2321738-200.png")
        st.markdown("Click the boxes on the side panel to explore the dataset.")
        if st.sidebar.checkbox('Dataset'):
            st.subheader('Overview of dataset:')
            st.write(df.head())
            st.info("The dataset consists of three columns namely: Sentiment, Tweet and Tweet ID")
        if st.sidebar.checkbox('Popular words for each sentiment in a word cloud'):
            st.subheader('Word cloud of popular tweets for each sentiment')
            st.image("https://i.imgur.com/M70a1Vd.png")
        if st.sidebar.checkbox('Top Hashtags in different tweet sentiments'):
            options = ["Top hashtags in news sentiment","Top hashtags in pro sentiment", "Top hashtags for neutral sentiment", "Top hashtags for anti sentiment"]
            selection = st.selectbox("Choose sentiment to view", options)
            if selection == "Top hashtags in news sentiment":
                st.subheader('Bar graph depicting popular hashtag words for news sentiment')
                st.image("https://i.imgur.com/vHbhWIn.png")
            if selection == "Top hashtags in pro sentiment":
                st.subheader('Bar graph depicting popular hashtag words for pro sentiment')
                st.image("https://i.imgur.com/JYOz7nz.png")	
            if selection == "Top hashtags in neutral sentiment":
                st.subheader('Bar graph depicting popular hashtag words for neutral sentiment')
                st.image("https://i.imgur.com/qvZyh6Z.png")
            if selection == "Top hashtags in anti sentiment":
                st.subheader('Bar graph depicting popular hashtag words for anti sentiment')
                st.image("https://i.imgur.com/03DU2hR.png")	
    # Building out the "Information" page
    if selection == "Model Information": 
        st.info("General Information")
        # You can read a markdown file from supporting resources folder
        st.markdown("Some information here")

        st.subheader("Raw Twitter data and label")
        if st.checkbox('Show raw data'): # data is hidden if box is unchecked
           st.write(raw[['sentiment', 'message']]) # will write the df to the page

    

# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()        

















    