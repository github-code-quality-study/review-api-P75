import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# Load the reviews data
reviews = pd.read_csv('data/reviews.csv').to_dict('records')

# Define the valid locations
VALID_LOCATIONS = [
    "Albuquerque, New Mexico", "Carlsbad, California", "Chula Vista, California", 
    "Colorado Springs, Colorado", "Denver, Colorado", "El Cajon, California", 
    "El Paso, Texas", "Escondido, California", "Fresno, California", "La Mesa, California", 
    "Las Vegas, Nevada", "Los Angeles, California", "Oceanside, California", 
    "Phoenix, Arizona", "Sacramento, California", "Salt Lake City, Utah", 
    "San Diego, California", "Tucson, Arizona"
]

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        if environ["REQUEST_METHOD"] == "GET":
            query_params = parse_qs(environ['QUERY_STRING'])
            location = query_params.get('location', [None])[0]
            start_date = query_params.get('start_date', [None])[0]
            end_date = query_params.get('end_date', [None])[0]

            filtered_reviews = []

            for review in reviews:
                review_sentiment = self.analyze_sentiment(review['ReviewBody'])
                review['sentiment'] = review_sentiment

                if location and review['Location'] != location:
                    continue

                review_date = datetime.strptime(review['Timestamp'], "%Y-%m-%d %H:%M:%S")

                if start_date and review_date < datetime.strptime(start_date, '%Y-%m-%d'):
                    continue

                if end_date and review_date > datetime.strptime(end_date, '%Y-%m-%d'):
                    continue

                filtered_reviews.append(review)

            # Sort by compound sentiment in descending order
            filtered_reviews.sort(key=lambda x: x['sentiment']['compound'], reverse=True)

            response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")
            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            return [response_body]

        if environ["REQUEST_METHOD"] == "POST":
            try:
                content_length = int(environ.get('CONTENT_LENGTH', 0))
                post_data = environ['wsgi.input'].read(content_length).decode('utf-8')
                parsed_data = parse_qs(post_data)

                review_body = parsed_data.get('ReviewBody', [None])[0]
                location = parsed_data.get('Location', [None])[0]

                if not review_body or not location:
                    start_response("400 Bad Request", [("Content-Type", "application/json")])
                    return [json.dumps({"error": "ReviewBody and Location are required fields."}).encode("utf-8")]

                if location not in VALID_LOCATIONS:
                    start_response("400 Bad Request", [("Content-Type", "application/json")])
                    return [json.dumps({"error": "Invalid location provided."}).encode("utf-8")]

                new_review = {
                    "ReviewId": str(uuid.uuid4()),
                    "ReviewBody": review_body,
                    "Location": location,
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                reviews.append(new_review)

                response_body = json.dumps(new_review, indent=2).encode("utf-8")
                start_response("201 Created", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                return [response_body]
            except Exception as e:
                start_response("500 Internal Server Error", [("Content-Type", "application/json")])
                return [json.dumps({"error": str(e)}).encode("utf-8")]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = int(os.environ.get('PORT', 8000))
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()
