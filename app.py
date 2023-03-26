import pandas as pd
from flask import Flask, jsonify, request
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from flask_cors import CORS

tv_shows = pd.read_csv('tv_shows.csv')

# Clean the data
tv_shows = tv_shows.drop(['ID', 'Age', 'IMDb'], axis=1)
tv_shows = tv_shows.fillna('')

# Create the feature matrix
features = ['Title', 'Year', 'Netflix', 'Hulu',
            'Prime Video', 'Disney+', 'Rotten Tomatoes']
tv_shows['features'] = tv_shows[features].apply(
    lambda x: ' '.join(x.astype(str)), axis=1)

# Create the count matrix
count = CountVectorizer().fit_transform(tv_shows['features'])
cosine_sim = cosine_similarity(count)

app = Flask(__name__)
CORS(app)


@app.route('/recommend', methods=['POST'])
def recommend():
    # Get the data from the request
    data = request.get_json()
    user_shows = data['shows']

    # Get the indices of the user shows
    indices = []
    for show in user_shows:
        idx = tv_shows.index[tv_shows['Title'] == show].tolist()
        if len(idx) > 0:
            indices.append(idx[0])

    # Get the cosine similarity scores for the user shows
    similarity_scores = cosine_sim[indices].mean(axis=0)

    # Get the indices of the top 5 shows
    top_indices = similarity_scores.argsort()[::-1][:6]

    # Get the top 5 recommended shows
    recommended_shows = []
    for idx in top_indices:
        show = tv_shows.iloc[idx]['Title']
        if show not in user_shows:
            recommended_shows.append(show)

    return jsonify({'recommended_shows': recommended_shows[:5]})


if __name__ == '__main__':
    import os
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=True)
