# Recommender-System-with-Item2Vec

This repository contains a simple implementation of a recommender system based on Item2Vec. Item2Vec is a technique that applies Word2Vec methodology to items (in this case, products viewed by users) to identify similar items based on co-occurrences in the same context (i.e., viewed in the same session).
Dependencies

This code depends on several Python libraries, including:

    numpy
    pandas
    gensim
    sklearn

How to Run

To run the code, you need a CSV dataset named events.csv in the same directory. The dataset should contain the following columns:

    event: the type of event (this script assumes 'view' events)
    visitorid: the ID of the visitor
    itemid: the ID of the item being viewed

Once you have the data set up, you can run the code directly. The script will preprocess the data, train the Item2Vec model, and then prompt you to enter a sequence of 5 item IDs. It will then use the model to recommend the next item.

python

python item2vec_recommender.py

When running the script, you will be prompted to input 5 item ids one by one. After the inputs, the script will output the recommended next item based on the Item2Vec model.
Understanding the Code

The script trains an Item2Vec model on sessions extracted from your data. A session is defined as a sequence of items viewed by the same visitor, and each item is represented by its item ID.

Item IDs are converted to strings because Word2Vec (which we're using as Item2Vec) expects words, not integers.

The script uses the gensim implementation of Word2Vec to train the model. It groups the data by session ID, transforming each session into a list of item IDs, and feeds these lists of item IDs to the Word2Vec model. The resulting model can then find items that are similar to a given item.

The recommend_next_item function uses the Item2Vec model to recommend an item. It takes as input a list of item IDs, and returns the item that is most similar to the last item in the list, according to the Item2Vec model.
Note

Please remember that the quality of recommendations strongly depends on the quality and quantity of the data. The more diverse and representative your data is of the items and sessions you're interested in, the better the Item2Vec model will be.
