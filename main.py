import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load and preprocess the dataset
data = pd.read_csv("events.csv")
data = data[data["event"] == "view"]
data["session_id"] = data["visitorid"].astype("category").cat.codes

# Convert item ids to strings because gensim's Word2Vec expects words (strings), not integers.
data["itemid"] = data["itemid"].astype(str)

# Group by session id and transform each session into a list of itemids.
# The result is a list of sessions, and each session is a list of itemids.
sessions = data.groupby('session_id')['itemid'].apply(list).tolist()

# Train Item2Vec model.
model = Word2Vec(sessions, vector_size=100, window=5, min_count=1, workers=4)

# Function to recommend the next item
def recommend_next_item(session):
    most_similar_items = model.wv.most_similar(session[-1], topn=1)
    return most_similar_items[0][0] if most_similar_items else None

# Example: Get a user input and recommend the next item
user_session = []
print("Enter 5 item ids, one at a time:")
for i in range(5):
    item_id = input(f"Item {i + 1}: ")
    user_session.append(item_id)

recommended_item = recommend_next_item(user_session)
print(f"Recommended next item: {recommended_item}")
