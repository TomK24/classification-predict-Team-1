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
import pickle

import re

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# New Vectorizer we'll actually be using
tweet_vectorizer = open("models+vectors/vector_1.pkl","rb")
vect = joblib.load(tweet_vectorizer)

# Section to load models -- this is now done when the classify button is clicked. Much simpler
# cnb = pickle.load(open("models+vectors/CNB_model.pkl", 'rb'))
# svc = pickle.load(open("models+vectors/SVC_model.pkl", 'rb'))
# mlr = pickle.load(open("models+vectors/MLR_model.pkl", 'rb'))

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
# df.head()

def preprocess1(tweet):
    '''Method for pre-processing a single tweet entered using the text box'''
    pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
    normal_chars = 'abcdefghijklmnopqrstuvwxyz0123456789 '
    tweet2 = re.sub(pattern_url, 'url-web', tweet)
    tweet3 = tweet2.lower()
    tweet_4 = ''.join([l for l in tweet3 if l in normal_chars])
    np_tweet = np.array([tweet_4])
    return tweet_4

def preprocess2(data):
    '''Method for pre-processing an uploaded csv file. Takes a dataframe as input and returns one as output'''
    def remove_weird_chars(post):
        return ''.join([l for l in post if l in normal_chars])
    pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
    normal_chars = 'abcdefghijklmnopqrstuvwxyz0123456789 '
    subs_url = r'url-web'
    data['message'] = data['message'].replace(to_replace = pattern_url, value = subs_url, regex = True)
    data['message'] = data['message'].str.lower()
    data['message'] = data['message'].apply(remove_weird_chars)
    return data
# Create a dictionary for tweet prediction outputs
dictionary_tweets = {'[-1]': "Not supporting man-made climate change",
                     '[0]': "Neither supporting nor refuting the belief of man-made climate change",
                     '[1]': "Being pro climate change",
                     '[2]': "Linking factual news about climate change"}
    


# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """
    # Creates a main title and subheader on your page -
    # these are static across all pages
    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Make Predictions","Model Explinations", "Explore The Data"]
    st.sidebar.image("https://im.ezgif.com/tmp/ezgif-1-d73edcc34959.gif")
    selection = st.sidebar.selectbox("Choose Option Below⬇️", options)
    # Creating a page with m
    #Building out the 'About This App' page
    # Building out the predication page
    if selection == "Make Predictions":
        st.image("https://i.imgur.com/wZdJqho.png")
        pred_type = st.radio("Predict sentiment of a single tweet or submit a csv for multiple tweets", ('Single tweet', 'Multiple'))
        if pred_type == 'Single tweet':

            # Creating a text box for user input
            tweet_text = st.text_area("Enter Tweet Below⬇️","")
        # upload = st.file_uploader('Upload a csv file here', type='csv', accept_multiple_files=False, key=None, help='Only CSV files are accepted', on_change=None, args=None, kwargs=None)
        # df = None
        # x_pred = None
        # if upload is not None:
        #     df = pd.read_csv(upload)
        #     processed_df = preprocess2(df)
        #     X_pred = processed_df['message']


            options = ["Naive Bayes Classifier","Linear Support Vector Classifier", "Multinomial Logistical Regression"]
            selection = st.selectbox("Choose Your Model😬", options)

       
            if st.button("Classify tweet"):
                # Transforming user input with vectorizer
                processed_text = preprocess1(tweet_text)
                # vect_text = tweet_cv.transform([processed_text]).toarray()
                # Load your .pkl file with the model of your choice + make predictions
                # Try loading in multiple models to give the user a choice
                # predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
                # prediction = predictor.predict(vect_text)
                predictor = None
                if selection == "Naive Bayes Classifier":
                    cnb = pickle.load(open("models+vectors/CNB_model.pkl", 'rb'))
                    predictor = cnb
                elif selection == "Linear Support Vector Classifier":
                    svc = pickle.load(open("models+vectors/SVC_model.pkl", 'rb'))
                    predictor = svc
                elif selection == "Multinomial Logistical Regression":
                    mlr = pickle.load(open("models+vectors/MLR_model.pkl", 'rb'))
                    predictor = mlr
                vect_text = vect.transform([tweet_text]).toarray()

                prediction = predictor.predict(vect_text)
                prediction_str = np.array_str(prediction)

                # When model has successfully run, will print prediction
                # You can use a dictionary or similar structure to make this output
                # more human interpretable.
                st.success("Tweet Categorized as: {}".format(dictionary_tweets[prediction_str]))
        
        elif pred_type == 'Multiple':
            upload = st.file_uploader('Upload a csv file here', type='csv', accept_multiple_files=False, key=None, help='Only CSV files are accepted', on_change=None, args=None, kwargs=None)
            df_uploaded = None
            x_pred = None
            if upload is not None:
                df_uploaded = pd.read_csv(upload)
                processed_df = preprocess2(df_uploaded)
                X_pred = processed_df['message']
            
            options = ["Naive Bayes Classifier","Linear Support Vector Classifier", "Multinomial Logistical Regression"]
            selection = st.selectbox("Choose Your Model😬", options)

            if st.button("Classify csv"):
                # Transforming user input with vectorizer
                # processed_text = preprocess1(tweet_text)
                # vect_text = tweet_cv.transform([processed_text]).toarray()
                # Load your .pkl file with the model of your choice + make predictions
                # Try loading in multiple models to give the user a choice
                # predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
                # prediction = predictor.predict(vect_text)
                predictor = None
                if selection == "Naive Bayes Classifier":
                    cnb = pickle.load(open("models+vectors/CNB_model.pkl", 'rb'))
                    predictor = cnb
                elif selection == "Linear Support Vector Classifier":
                    svc = pickle.load(open("models+vectors/SVC_model.pkl", 'rb'))
                    predictor = svc
                elif selection == "Multinomial Logistical Regression":
                    mlr = pickle.load(open("models+vectors/MLR_model.pkl", 'rb'))
                    predictor = mlr
                vect_text = vect.transform(X_pred)
                prediction = predictor.predict(vect_text)
                df_download = df_uploaded.copy()
                df_download['sentiment'] = prediction
                # When model has successfully run, will print prediction
                # You can use a dictionary or similar structure to make this output
                # more human interpretable.
                st.success("Tweets succesfully classified")
                st.dataframe(data=df_download, width=None, height=None)
                st.download_button(label='Download csv with sentiment predictions', data=df_download.to_csv(),file_name='sentiment_predictions.csv',mime='text/csv')
    #Building out 'Model Explination' page
    if selection == "Model Explinations":
        st.image("https://i.imgur.com/MDxSN4d.png")
        options = ["Multinomial Logistical Regression Model","Linear Support Vector Classifier Model", "Naive Bayes Classifier Model"]
        selection = st.selectbox("Which model would you like to learn more about?", options)
        if selection == "Multinomial Logistical Regression Model":
            st.info("This model doesn't do particuraly well at predicting tweets that have a neutral or negative sentiment")
            st.markdown("Logistic regression is a statistical analysis method used to predict a data value based on prior observations of a data set. ")
            st.markdown("A multinomial logistic regression model predicts two or more dependent variables by analyzing the relationship between one or more existing independent variables.")
            st.image("https://miro.medium.com/max/1400/1*dVsfG6i-Y93prmLgTcxRwA.jpeg")
        if selection == "Linear Support Vector Classifier Model":
            st.info("This model is the best overall and does the best at predicting each of the sentiments")
            st.markdown("The goal of the SVM algorithm is to create the best line or decision boundary that can seperate multi dimensional space into classes so that we can easily put the new data point in the correct category in the future. This best decision boundary is called a hyperplane.")
            st.markdown("SVM chooses the extreme points/vectors that help in creating the hyperplane. These extreme cases are called as support vectors, and hence the algorithm is termed as Support Vector Machine.")
            st.image("https://static.javatpoint.com/tutorial/machine-learning/images/support-vector-machine-algorithm.png")
        if selection == "Naive Bayes Classifier Model":
            st.info("This model is better than the Multinomial Logistical Regression Model at predicting tweets that have a neutral or negative sentiment.")
            st.markdown("Naive Bayes classifier assumes that the presence of a particular independent variable in a set is unrelated to the presence of any other independent variable.")
            st.markdown("For example, a fruit may be thought to be an apple if it is red/green, round, and about 7.5cm in diameter. Even if these variables depend on each other or upon the existence of the other variables, all of these properties independently contribute to the probability that this fruit is an apple and that is why it is known as ‘Naive’.")
            st.image("https://miro.medium.com/max/468/1*IGwM9cb8W-gyJW5rkiVQPw.jpeg")




       


    #Building out the 'EDA' page
    if selection == 'Explore The Data':
        st.image("https://i.imgur.com/Ce2r5oh.png")
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
# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()        

















    