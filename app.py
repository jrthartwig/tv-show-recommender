import pandas as pd
import numpy as np
import requests
from flask import Flask, jsonify, request, make_response
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from flask_cors import CORS

from fuzzywuzzy import process


def find_closest_match(title, titles, score_cutoff=90):
    match, score = process.extractOne(title, titles, score_cutoff=score_cutoff)
    if score >= score_cutoff:
        return match
    else:
        return None


tv_shows = pd.read_csv('tv_shows_with_genres.csv')

# Clean the data
# Clean the data
columns_to_drop = ['ID', 'Age', 'IMDb']
tv_shows = tv_shows.drop(
    [col for col in columns_to_drop if col in tv_shows.columns], axis=1)
tv_shows = tv_shows.fillna('')


def get_genres(title, api_key, counter, total):
    print(f"Fetching genre information for show {counter} of {total}: {title}")

    url = f"https://api.themoviedb.org/3/search/tv?api_key={api_key}&query={title}"
    response = requests.get(url)
    data = response.json()

    if data["total_results"] > 0:
        show_id = data["results"][0]["id"]

        show_url = f"https://api.themoviedb.org/3/tv/{show_id}?api_key={api_key}"
        show_response = requests.get(show_url)
        show_data = show_response.json()

        genres = [genre["name"] for genre in show_data["genres"]]
        return " ".join(genres)
    else:
        return ""


# Fetch genre information and print progress updates
total_shows = len(tv_shows)
# tv_shows["genres"] = [get_genres(title, api_key, counter + 1, total_shows)
#                       for counter, title in enumerate(tv_shows["Title"])]

# tv_shows.to_csv('tv_shows_with_genres.csv', index=False)


features = ['Year', 'Netflix', 'Hulu', 'Prime Video',
            'Disney+', 'Rotten Tomatoes', 'genres']
tv_shows['features'] = tv_shows[features].apply(
    lambda x: ' '.join(x.astype(str)), axis=1)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(tv_shows['features'])
cosine_sim = cosine_similarity(tfidf_matrix)

app = Flask(__name__)
CORS(app)


def no_cache(func):
    def wrapped(*args, **kwargs):
        response = make_response(func(*args, **kwargs))
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
    return wrapped


@app.route('/recommend', methods=['POST'])
@no_cache
def recommend():
    data = request.get_json()
    user_shows = data['shows']

    indices = []
    closest_matches = []
    for show in user_shows:
        closest_match = find_closest_match(show, tv_shows['Title'].tolist())
        if closest_match:
            closest_matches.append(closest_match)
            idx = tv_shows.index[tv_shows['Title'] == closest_match].tolist()
            if len(idx) > 0:
                indices.append(idx[0])

    similarity_scores = cosine_sim[indices].mean(axis=0)
    # Add small random noise
    noise = np.random.normal(0, 0.0001, similarity_scores.shape)
    similarity_scores += noise
    top_indices = similarity_scores.argsort()[::-1][:6]

    recommended_shows = []
    for idx in top_indices:
        show = tv_shows.iloc[idx]['Title']
        if show not in closest_matches:
            available_on = []
            if tv_shows.iloc[idx]['Netflix'] == 1:
                available_on.append('Netflix')
            if tv_shows.iloc[idx]['Hulu'] == 1:
                available_on.append('Hulu')
            if tv_shows.iloc[idx]['Prime Video'] == 1:
                available_on.append('Prime Video')
            if tv_shows.iloc[idx]['Disney+'] == 1:
                available_on.append('Disney+')

            recommended_shows.append({
                'title': show,
                'available_on': available_on
            })

    return jsonify({'recommended_shows': recommended_shows[:5]})


if __name__ == '__main__':
    import os
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=True)
