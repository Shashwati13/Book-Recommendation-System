import streamlit as st
import os
import base64
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import requests
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from PIL import Image
from io import BytesIO
import base64



# st.set_page_config(page_title='Book Recommendation using Sentiment Analysis', page_icon=':smiley:', layout='wide')

# create SentimentIntensityAnalyzer object
sia = SentimentIntensityAnalyzer()
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
background_image = "background2.jpg"
add_bg_from_local(background_image) 


st.markdown("<h1 style='font-size: 50px; color: black;font-style: italic; font-family: \"Palatino\"'>Enter a review!!</h1>", unsafe_allow_html=True)
input_text = st.text_input("")
# input_text = st.text_input("Enter a review:")

# perform sentiment analysis on the input
if input_text:
    sentiment_score = sia.polarity_scores(input_text)['compound']
    col1, col2,col3 = st.columns(3)

    with col1:
        st.markdown("<div style='font-size: 30px; color: black;font-style: italic; font-family: \"Palatino\"'>Sentiment Score:", unsafe_allow_html=True)

    with col2:
        st.write("<div style='font-size: 30px;display: flex; align-items: center; color: black; font-family: \"Palatino\";'>{} {}".format("", sentiment_score), unsafe_allow_html=True)
    with col3:
        if sentiment_score > 0:
            st.markdown('<div style="font-size:25px;display: inline-block; align-items: center;color:green; background-color:white; padding: 5px; font-family: \'Palatino\'">Positive</div>', unsafe_allow_html=True)
        elif sentiment_score < 0:
            st.markdown('<div style="font-size:25px;display: inline-block; align-items: center;color:red; background-color:white; padding: 5px; font-family: \'Palatino\'">Negative</div>', unsafe_allow_html=True)
           
        else:
            st.markdown('<div style="font-size:25px;display: inline-block; align-items: center;color:blue; background-color:white; padding: 5px; font-family: \'Palatino\'">Neutral</div>', unsafe_allow_html=True)



@st.cache_data()
def popular_books(df, n=100):
    rating_count=df.groupby("Title").count()["review/score"].reset_index()
    rating_count.rename(columns={"review/score":"NumberOfVotes"},inplace=True)
    
    rating_average=df.groupby("Title")["review/score"].mean().reset_index()
    rating_average.rename(columns={"review/score":"AverageRatings"},inplace=True)
    
    popularBooks=rating_count.merge(rating_average,on="Title")
    
    def weighted_rate(x):
        v=x["NumberOfVotes"]
        R=x["AverageRatings"]
        
        return ((v*R) + (m*C)) / (v+m)
    
    C=popularBooks["AverageRatings"].mean()
    m=popularBooks["NumberOfVotes"].quantile(0.90)
    
    popularBooks=popularBooks[popularBooks["NumberOfVotes"] >=250]
    popularBooks["Popularity"]=popularBooks.apply(weighted_rate,axis=1)
    popularBooks=popularBooks.sort_values(by="Popularity",ascending=False)
    return popularBooks[["Title","Popularity"]].reset_index(drop=True).head(n)

@st.cache_data()
def load_data():
    df = pd.read_csv("popularity.csv")
    return df

df = load_data()

recommendations = popular_books(df, n=10)

# def app():
#     if st.button("Show Popular Books"):

#         st.write(recommendations)

# if __name__ == "__main__":
    
#     app()


#recommendations = popular_books(df, n=10)
def getPopularBooks():
    n=10
    top_ten=pd.DataFrame(popular_books(df,10))
    fig, ax = plt.subplots(1,10,figsize=(17, 5))
    fig.suptitle("MOST POPULAR 10 BOOKS",fontsize=40,color="brown")
    for i in range(len(top_ten["Title"].tolist())):
        url=df.loc[df["Title"]==top_ten["Title"].tolist()[i],"image"][:1].values[0]
        if url != 'Image not found': 
            img=Image.open(requests.get(url,stream=True).raw)
            ax[i].imshow(img)
            ax[i].axis("off")
            ax[i].set_title("RATING: {}".format(round(df[df["Title"]==top_ten["Title"].tolist()[i]]["review/score"].mean(),1)),y=-0.20,color="brown",fontsize=10)
        else:
            img = Image.open(requests.get('https://as1.ftcdn.net/v2/jpg/04/34/72/82/1000_F_434728286_OWQQvAFoXZLdGHlObozsolNeuSxhpr84.jpg',stream=True).raw)
            ax[i].imshow(img)
            ax[i].axis("off")
            ax[i].set_title("RATING: {} ".format(round(df[df["Title"]==top_ten["Title"].tolist()[i]]["review/score"].mean(),1)),y=-0.20,color="brown",fontsize=10)
    st.pyplot(fig)
    
    
    
def content_based(bookTitle):
    bookTitle=str(bookTitle)
    
    if bookTitle in df["Title"].values:
        rating_count=pd.DataFrame(df["Title"].value_counts())
        rare_books=rating_count[rating_count["Title"]<=200].index
        common_books=df[~df["Title"].isin(rare_books)]
        
        if bookTitle in rare_books:
            most_common=pd.Series(common_books["Title"].unique()).sample(3).values
            st.write("No Recommendations for this Book ☹️ \n ")
            st.write("YOU MAY TRY: \n ")
            st.write("{}".format(most_common[0]), "\n")
            st.write("{}".format(most_common[1]), "\n")
            st.write("{}".format(most_common[2]), "\n")
        else:
            common_books=common_books.drop_duplicates(subset=["Title"])
            common_books.reset_index(inplace=True)
            common_books["index"]=[i for i in range(common_books.shape[0])]
            targets=["Title","authors","publisher"]
            common_books["all_features"] = [" ".join(common_books[targets].iloc[i,].values) for i in range(common_books[targets].shape[0])]
            vectorizer=CountVectorizer()
            common_booksVector=vectorizer.fit_transform(common_books["all_features"])
            similarity=cosine_similarity(common_booksVector)
            index=common_books[common_books["Title"]==bookTitle]["index"].values[0]
            similar_books=list(enumerate(similarity[index]))
            similar_booksSorted=sorted(similar_books,key=lambda x:x[1],reverse=True)[1:6]
            books=[]
            for i in range(len(similar_booksSorted)):
                
                books.append(common_books[common_books["index"]==similar_booksSorted[i][0]]["Title"].item())
            fig,ax=plt.subplots(1,5,figsize=(17,5))
            fig.suptitle("YOU MAY ALSO LIKE THESE BOOKS",fontsize=40,color="chocolate")
                
            for i in range(len(books)):
                
                url=common_books.loc[common_books["Title"]==books[i],"image"][:1].values[0]
                if url != 'Image not found': 
                    img=Image.open(requests.get(url,stream=True).raw)
                    ax[i].imshow(img)
                    ax[i].axis("off")
                    ax[i].set_title("RATING: {}".format(round(df[df["Title"]==books[i]]["review/score"].mean(),1)),y=-0.20,color="mediumorchid",fontsize=22)
                else:
                    img = Image.open(requests.get('https://as1.ftcdn.net/v2/jpg/04/34/72/82/1000_F_434728286_OWQQvAFoXZLdGHlObozsolNeuSxhpr84.jpg',stream=True).raw)
            ax[i].imshow(img)
            ax[i].axis("off")
            ax[i].set_title("RATING: {} ".format(round(df[df["Title"]==top_ten["Title"].tolist()[i]]["review/score"].mean(),1)),y=-0.20,color="mediumorchid",fontsize=10)
                    
            st.pyplot(fig)

    else:
        st.write("❌ COULD NOT FIND ❌") 

def app():
    # if st.button("Show Popular Books"):
    if st.button("Show Popular Books", '<div style={"font-size": "24px", "padding": "10px"}>'):
        getPopularBooks()
    
   
    st.markdown("<h1 style='font-size: 50px; color: black;font-style: italic; font-family: \"Palatino\"'>Enter book name!!</h1>", unsafe_allow_html=True)
    input_text1 = st.text_input("",key="text_input2")
    if st.button("Recommend Similar Books"):
        if bookTitle != "":
            content_based(bookTitle)
        else:
            st.write('Please enter a book name')

if __name__ == "__main__":
    
    app()

