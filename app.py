from flask import Flask, render_template, request, redirect
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
movies_data = pd.read_csv("movies_data.csv")
movies = pd.read_csv("movies_tags.csv")
cv = CountVectorizer(max_features=5000,stop_words='english')
vectors = cv.fit_transform(movies.tags).toarray()
similarity = cosine_similarity(vectors)

def recommender(movie):
    movie_index = movies[movies['title']==movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    x = []
    for i in movies_list:
        x.append(movies.iloc[i[0]].title)
    return x

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def Home():
    movies_title = list(movies['title'].values)
    return render_template("index.html",  movies_data=movies_title)
@app.route('/result',methods=['GET','POST'])
def result():
    return render_template("recommended.html")

@app.route("/predict", methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        mov = str(request.form['mov'])
        y = recommender(mov)
        movie_index = movies_data[movies_data['title']==mov].index[0]
        search_movie = movies_data.iloc[movie_index]
        return render_template("recommended.html",y=y,title = search_movie.title,overview = search_movie.overview,genres=search_movie.genres,cast=search_movie.cast,crew=search_movie.crew)

    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)