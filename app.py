import streamlit as st
import os
import base64
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from PIL import Image
import requests
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from PIL import Image
from io import BytesIO
import base64
from stop_words import get_stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise import Dataset, Reader
from surprise import KNNWithMeans
from surprise.model_selection import train_test_split
from surprise import accuracy

################
import string

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
translating = str.maketrans('', '', string.punctuation)
input_text = input_text.translate(translating)
# perform sentiment analysis on the input
if input_text:
    sentiment_score = sia.polarity_scores(input_text)['compound']
    col1, col2,col3 = st.columns(3)

    with col1:
        st.markdown("<div style='font-size: 30px; color: black;font-style: italic; font-family: \"Palatino\"'>Sentiment Score:", unsafe_allow_html=True)

    with col2:
        st.write("<div style='font-size: 30px;display: flex; align-items: center; color: black; font-family: \"Palatino\";'>{} {}".format("", sentiment_score), unsafe_allow_html=True)
    with col3:
        if sentiment_score > 0.3:
            st.markdown('<div style="font-size:25px;display: inline-block; align-items: center;color:green; background-color:white; padding: 5px; font-family: \'Palatino\'">Positive</div>', unsafe_allow_html=True)
        elif sentiment_score < -0.3:
            st.markdown('<div style="font-size:25px;display: inline-block; align-items: center;color:red; background-color:white; padding: 5px; font-family: \'Palatino\'">Negative</div>', unsafe_allow_html=True)
           
        else:
            st.markdown('<div style="font-size:25px;display: inline-block; align-items: center;color:blue; background-color:white; padding: 5px; font-family: \'Palatino\'">Neutral</div>', unsafe_allow_html=True)


@st.cache_data()
def popular_books(df, n=10):
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
    return popularBooks[["Title","NumberOfVotes","AverageRatings","Popularity"]].reset_index(drop=True).head(n)

@st.cache_data()
def load_data():
    df = pd.read_csv("popularity.csv")
    return df

df = load_data()

# recommendations = popular_books(df, n=10)

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
    
    

@st.cache_data()
def load_data_user():
    df_user_data = pd.read_csv("user_data.csv")
    return df_user_data
df_user_data = load_data_user()

def user_based(user_id):
    
    # Get the list of all items (books) in the dataset
    items = df_user_data['Id'].unique()
    # Predict the rating the user would give to each item and store in a dictionary
    item_ratings = {}
    for item in items:
        predicted_rating = loaded_model.predict(user_id, item).est
        item_ratings[item] = predicted_rating
    # Sort the items by predicted rating in descending order and select the top 5
    top_items = sorted(item_ratings.items(), key=lambda x: x[1], reverse=True)[:40]
    # Print the top 5 recommended books
    prev_search_books = []
    count = 0
    for i, item in enumerate(top_items):
        if count == 5:
            break
        if df_user_data[df_user_data['Id']==item[0]].iloc[0].Title not in prev_search_books:
            prev_search_books.append(df_user_data[df_user_data['Id']==item[0]].iloc[0].Title)
            count += 1
        fig,ax=plt.subplots(1,5,figsize=(17,5))
        fig.suptitle("RECOMMENDATIONS FOR YOU",fontsize=40,color="chocolate")
    for i in range(len(prev_search_books)):
        url=df_user_data.loc[df_user_data["Title"]==prev_search_books[i],"image"][:1].values[0]
        if url != 'Image not found': 
            img=Image.open(requests.get(url,stream=True).raw)
            ax[i].imshow(img)
            ax[i].axis("off")
            ax[i].set_title("RATING: {}".format(round(df_user_data[df_user_data["Title"]==prev_search_books[i]]["review/score"].mean(),1)),y=-0.20,color="mediumorchid",fontsize=22)
        else:
            img = Image.open(requests.get('https://as1.ftcdn.net/v2/jpg/04/34/72/82/1000_F_434728286_OWQQvAFoXZLdGHlObozsolNeuSxhpr84.jpg',stream=True).raw)
            ax[i].imshow(img)
            ax[i].axis("off")
            ax[i].set_title("RATING: {}".format(round(df_user_data[df_user_data["Title"]==prev_search_books["Title"].tolist()[i]]["review/score"].mean(),1) ),y=-0.20,color="mediumorchid",fontsize=10)
    st.pyplot(fig)
loaded_model = pickle.load(open('knnWithMeansModel', 'rb'))

@st.cache_data()
def load_data_content_based():
    df_content_based = pd.read_csv("contentBased.csv")
    return df_content_based
df_content_based = load_data_content_based()
def item_based(bookTitle):
    bookTitle=str(bookTitle)
    
    if bookTitle in df_content_based["Title"].values:
        rating_count=pd.DataFrame(df_content_based["Title"].value_counts())
        rare_books=rating_count[rating_count["Title"]<=200].index
        common_books=df_content_based[~df_content_based["Title"].isin(rare_books)]
        
        if bookTitle in rare_books:
            most_common=pd.Series(common_books["Title"].unique()).sample(3).values
            # st.write("No Recommendations for this Book ☹️ \n ")
            # st.write("YOU MAY TRY: \n ")
            # st.write("{}".format(most_common[0]), "\n")
            # st.write("{}".format(most_common[1]), "\n")
            # st.write("{}".format(most_common[2]), "\n")
            
            # st.write("No Recommendations for this Book ☹️ \n ")
            st.write("<div style='font-size:24px; color:white; background-color:red; padding:8px'>No Recommendations for this Book ☹️</div>", unsafe_allow_html=True)
            st.write("<div style='font-size:24px; color:white; padding:8px'>YOU MAY TRY: </div>", unsafe_allow_html=True)
            # st.write("YOU MAY TRY: \n ")
            st.write("<div style='color: black; font-size: 20px;background-color:white'>{}</div>".format(most_common[0]),unsafe_allow_html=True)
            st.write("<div style='color: black; font-size: 20px;background-color:white'>{}</div>".format(most_common[1]), unsafe_allow_html=True)
            st.write("<div style='color: black; font-size: 20px;background-color:white'>{}</div>".format(most_common[2]), unsafe_allow_html=True)

        else:
            common_books_pivot=common_books.pivot_table(index=["Id"],columns=["Title"],values="review/score")
            title=common_books_pivot[bookTitle]
            recommendation_df=pd.DataFrame(common_books_pivot.corrwith(title).sort_values(ascending=False)).reset_index(drop=False)
            
            if bookTitle in [title for title in recommendation_df["Title"]]:
                recommendation_df=recommendation_df.drop(recommendation_df[recommendation_df["Title"]==bookTitle].index[0])
                
            less_rating=[]
            for i in recommendation_df["Title"]:
                if df[df["Title"]==i]["review/score"].mean() < 5:
                    less_rating.append(i)
            if recommendation_df.shape[0] - len(less_rating) > 5:
                recommendation_df=recommendation_df[~recommendation_df["Title"].isin(less_rating)]
                
            recommendation_df=recommendation_df[0:5]
            recommendation_df.columns=["Title","Correlation"]
            
            fig, ax = plt.subplots(1,5,figsize=(17,5))
            fig.suptitle("WOULD YOU LIKE to TRY THESE BOOKS?",fontsize=40,color="deepskyblue")
            for i in range(len(recommendation_df["Title"].tolist())):
                url=df_content_based.loc[df_content_based["Title"]==recommendation_df["Title"].tolist()[i],"image"][:1].values[0]
                if url != 'Image not found':
                    
                    img=Image.open(requests.get(url,stream=True).raw)
                    ax[i].imshow(img)
                    ax[i].axis("off")
                    ax[i].set_title("RATING: {} ".format(round(df_content_based[df_content_based["Title"]==recommendation_df["Title"].tolist()[i]]["review/score"].mean(),1)),y=-0.20,color="mediumorchid",fontsize=22)
                else:
                    img = Image.open(requests.get('https://as1.ftcdn.net/v2/jpg/04/34/72/82/1000_F_434728286_OWQQvAFoXZLdGHlObozsolNeuSxhpr84.jpg',stream=True).raw)
                    ax[i].imshow(img)
                    ax[i].axis("off")
                    ax[i].set_title("RATING: {} ".format(round(df_content_based[df_content_based["Title"]==recommendation_df["Title"].tolist()[i]]["review/score"].mean(),1)),y=-0.20,color="mediumorchid",fontsize=22)
            st.pyplot(fig) 
    else:
        st.write("<div style='font-size:24px; color:red; background-color:white; padding:8px'>❌ COULD NOT FIND ❌</div>", unsafe_allow_html=True)
        # st.write("❌ COULD NOT FIND ❌")

def content_based(bookTitle):
    bookTitle=str(bookTitle)
    
    if bookTitle in df_content_based["Title"].values:
        rating_count=pd.DataFrame(df_content_based["Title"].value_counts())
        rare_books=rating_count[rating_count["Title"]<=200].index
        common_books=df_content_based[~df_content_based["Title"].isin(rare_books)]
        if bookTitle in rare_books:
            most_common=pd.Series(common_books["Title"].unique()).sample(3).values
            st.write("<div style='font-size:24px; color:white; background-color:red; padding:8px'>No Recommendations for this Book ☹️</div>", unsafe_allow_html=True)
            st.write("<div style='font-size:24px; color:white; padding:8px'>YOU MAY TRY: </div>", unsafe_allow_html=True)
            # st.write("YOU MAY TRY: \n ")
            st.write("<div style='color: black; font-size: 20px;background-color:white'>{}</div>".format(most_common[0]),unsafe_allow_html=True)
            st.write("<div style='color: black; font-size: 20px;background-color:white'>{}</div>".format(most_common[1]), unsafe_allow_html=True)
            st.write("<div style='color: black; font-size: 20px;background-color:white'>{}</div>".format(most_common[2]), unsafe_allow_html=True)

            # st.write("No Recommendations for this Book ☹️ \n ")
            # st.write("YOU MAY TRY: \n ")
            # st.write("{}".format(most_common[0]), "\n")
            # st.write("{}".format(most_common[1]), "\n")
            # st.write("{}".format(most_common[2]), "\n")
        else:
            common_books=common_books.drop_duplicates(subset=["Title"])
            common_books.reset_index(inplace=True)
            common_books["index"]=[i for i in range(common_books.shape[0])]
            targets=["Title","authors","publisher"]
            common_books["all_features"] = [" ".join(str(common_books[targets].iloc[i,].values)) for i in range(common_books[targets].shape[0])]
            #vectorizer=CountVectorizer(stop_words='english', min_df=1)
            
            #vectorizer=TfidfVectorizer(stop_words='english', min_df=10)
 		            #vectorizer=TfidfVectorizer(stop_words='english', min_df=10)
            #non_empty_docs = [doc for doc in common_books["all_features"] if doc.strip() != ""]
            #st.write(non_empty_docs)
            stop_words = get_stop_words('english')
            # Create the TF-IDF vectorizer
            vectorizer = TfidfVectorizer(stop_words=stop_words, min_df=10)
            # Remove documents that contain only stop words
            non_empty_docs = []
            for doc in common_books["all_features"]:
                words = doc.strip().split()
                if len(set(words) - set(stop_words)) > 0:
                    non_empty_docs.append(doc)
            #non_empty_docs = [doc for doc in common_books["all_features"] if doc.strip() != ""]
            if len(non_empty_docs) != 0:
                common_booksVector=vectorizer.fit_transform(non_empty_docs)
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
                        ax[i].set_title("RATING: {}".format(round(df_content_based[df_content_based["Title"]==books[i]]["review/score"].mean(),1)),y=-0.20,color="mediumorchid",fontsize=22)
                    else:
                        img = Image.open(requests.get('https://as1.ftcdn.net/v2/jpg/04/34/72/82/1000_F_434728286_OWQQvAFoXZLdGHlObozsolNeuSxhpr84.jpg',stream=True).raw)
                        ax[i].imshow(img)
                        ax[i].axis("off")
                        ax[i].set_title("RATING: {} ".format(round(df_content_based[df_content_based["Title"]==top_ten["Title"].tolist()[i]]["review/score"].mean(),1)),y=-0.20,color="mediumorchid",fontsize=10)
                st.pyplot(fig)
    else:
        st.write("❌ COULD NOT FIND ❌")
def app():
    st.markdown("<h1 style='font-size: 50px; color: black;font-style: italic; font-family: \"Palatino\"'>Popular Books!!</h1>", unsafe_allow_html=True)
    if st.button("Show Popular Books"):
        getPopularBooks()
    # input_text2 = st.text_input("Enter book name", key="book_input")
    st.markdown("<h1 style='font-size: 50px; color: black;font-style: italic; font-family: \"Palatino\"'>Enter Book Name!!</h1>", unsafe_allow_html=True)
    input_text2 = st.text_input("",key="input_text2")
    
    # bookTitle = st.text_input('Enter Book Name')
    bookTitle=input_text2
    if st.button("Recommend Similar Book Items"):
        if bookTitle != "":
            item_based(bookTitle)
        else:
            st.write('Please enter a book name')
    st.markdown("<h1 style='font-size: 50px; color: black;font-style: italic; font-family: \"Palatino\"'>Enter User ID!!</h1>", unsafe_allow_html=True)    
    user_id = st.text_input("",key="user_id")
    if st.button("Recommend Books for User ID"):
        if user_id != "":
            user_based(user_id)
        else:
            st.write('Please enter a user ID')
if __name__ == "__main__":
    
    app()
