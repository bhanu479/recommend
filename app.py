from flask import Flask, render_template, request, redirect
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
movies_data = pd.read_csv("movies_data.csv")
movies = pd.read_csv("movies_tags.csv")
poster = pd.read_csv("poster.csv")
def create_similarity(data):
    cv = CountVectorizer(max_features=5000,stop_words='english')
    vectors = cv.fit_transform(data.tags).toarray()
    similarity = cosine_similarity(vectors)
    return similarity
def recommender(movie):
    similarity = create_similarity(movies)
    movie_index = movies[movies['title']==movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:11]
    x = pd.DataFrame()
    for i in movies_list:
        y = {'id':movies.iloc[i[0]].id,'movies':movies.iloc[i[0]].title}
        x = x.append(y,ignore_index=True)
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
        poster_index = poster[poster['id']==search_movie.id].index[0]
        poster_data = poster.iloc[poster_index]
        recommender_data = pd.DataFrame()
        for i in y.id:
            recommender_data= recommender_data.append(poster[poster['id']==i],ignore_index=True)
        return render_template("recommended.html",title = search_movie.title,overview = search_movie.overview,genres=search_movie.genres,cast=search_movie['cast'].split(","),crew=search_movie.crew,recommended_id=y.id,recommended_movies=y.movies,cast_poster=poster_data.cast.split(","),crew_poster=poster_data.crew,movie_poster=poster_data.poster,recommender_poster=recommender_data.poster)
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)