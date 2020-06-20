import csv
import json
import logging
import os
import pickle
from pathlib import Path

import feedparser
import newspaper
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


class news_topic_text_classifier:
    '''
    News topic text classification model. Based off: https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f
    '''

    # The dataframe that contains the CSV data
    _data_frame = None
    # The model used for text classification
    _text_classifier = None
    # Used in text classification
    _tfidf = None

    # Data used for predicting and testing the model
    _y_pred = None
    _y_test = None

    # relative file paths
    _script_dir = Path(__file__).parent
    _models_dir = os.path.join(_script_dir, "models")
    _data_dir = os.path.join(_script_dir, "data")

    def __init__(self):
        '''
        The constructor
        '''

        # Set Logging Level
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Checks if the model files exist and if so initialises this class with them. If not the class will need to be re trained.
        classifier_file_path = os.path.join(self._models_dir, "news_text_classifier.class")
        vector_file_path = os.path.join(self._models_dir, "tfidf.class")
        data_frame_file_path = os.path.join(self._models_dir, "data_frame.class")
        y_pred_file_path = os.path.join(self._models_dir, "y_pred.class")
        y_test_file_path = os.path.join(self._models_dir, "y_test.class")

        if os.path.exists(vector_file_path) and os.path.exists(classifier_file_path) and os.path.exists(
                data_frame_file_path):
            classifier_file = open(classifier_file_path, "rb")
            vector_file = open(vector_file_path, "rb")
            data_frame_file = open(data_frame_file_path, "rb")
            y_pred_file = open(y_pred_file_path, "rb")
            y_test_file = open(y_test_file_path, "rb")

            self._text_classifier = pickle.load(file=classifier_file)
            self._tfidf = pickle.load(file=vector_file)
            self._data_frame = pickle.load(file=data_frame_file)

            self._y_test = pickle.load(file=y_test_file)
            self._y_pred = pickle.load(file=y_pred_file)

    def create_data_set(self, dataset=os.path.join(_data_dir, "dataset.csv")):
        """
        A function for creating a dataset following the format used in this model.
        :param dataset:
        """
        deny_words = ["Image copyright", "Getty Images", "image caption", "reuters", "getty"]

        with open(os.path.join(self._data_dir, "rss_feeds.json")) as json_file:
            dict_of_rss_feeds = json.load(json_file)

        logging.info("Starting downloading of RSS Feed data. This may take some time.")
        with open(dataset, 'w', newline='', encoding="utf-8") as csv_file:
            fieldnames = ['url', 'category', 'body']
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()

            # loop through lists of rss feed topics
            length = len(dict_of_rss_feeds)
            iterator = 0

            for category in dict_of_rss_feeds:

                iterator = iterator + 1
                logging.info("Aggregating RSS feeds: {} out of {}".format(iterator, length))

                rss_feeds = dict_of_rss_feeds[category]

                # Loop through list of rss feeds for topic
                for feed in rss_feeds:

                    news_feed = feedparser.parse(feed)

                    # Loop through all URLs in RSS feed
                    for entry in news_feed.entries:
                        # Download article

                        try:
                            url = entry.link
                            article = newspaper.Article(url)
                            article.download()
                            article.parse()
                        except newspaper.article.ArticleException as e:
                            continue

                        article_body = article.text.lower()

                        # Remove deny words
                        for phrase in deny_words:
                            article_body = article_body.replace(phrase.lower(), "")

                        # Removes new lines, leading spaces, and tabs
                        article_body = article_body.replace("\n", " ").replace("\r", " ").replace("\t", "").lstrip()

                        # Write to CSV file
                        csv_writer.writerow({'url': url, 'category': category, 'body': article_body})

    def print_model_feature_data(self):

        if self._text_classifier is not None or os.path.exists(
                os.path.join(self._models_dir, "news_text_classifier.class")):
            N = 2

            category_id_df = self._data_frame[['category', 'category_id']].drop_duplicates().sort_values('category_id')
            category_to_id = dict(category_id_df.values)

            for category, category_id in sorted(category_to_id.items()):
                indices = np.argsort(self._text_classifier.coef_[category_id])
                feature_names = np.array(self._tfidf.get_feature_names())[indices]
                unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
                bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
                print("# '{}':".format(category))
                print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams)))
                print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))

            print("-" * 20)
            print("\n")

            print(metrics.classification_report(self._y_test, self._y_pred,
                                                target_names=self._data_frame['category'].unique()))

        else:
            raise Exception("Unable to retrieve model data as model does not exist. Please re-train the model.")

    def re_train(self, dataset=os.path.join(_data_dir, "dataset.csv")):
        '''
        Trains the text classifier model if it needs re-training or has not already been trained
        :param dataset: csv file location
        '''

        if os.path.exists(os.path.join(self._data_dir, "dataset.csv")):

            self._data_frame = pd.read_csv(dataset)
            self._data_frame.head()
            col = ['category', 'body']
            self._data_frame = self._data_frame[col]
            self._data_frame = self._data_frame[pd.notnull(self._data_frame['body'])]
            self._data_frame.columns = ['category', 'body']
            self._data_frame['category_id'] = self._data_frame['category'].factorize()[0]

            model = LinearSVC()

            self._tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1',
                                          ngram_range=(1, 2),
                                          stop_words='english')

            features = self._tfidf.fit_transform(self._data_frame.body).toarray()
            labels = self._data_frame.category_id
            X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels,
                                                                                             self._data_frame.index,
                                                                                             test_size=0.33,
                                                                                             random_state=0)

            model.fit(X_train, y_train)
            model.fit(features, labels)
            self._text_classifier = model

            self._y_test = y_test
            self._y_pred = model.predict(X_test)

            # saves the classifier
            classifier_file_path = os.path.join(self._models_dir, "news_text_classifier.class")
            classifier_file = open(classifier_file_path, "wb")
            pickle.dump(obj=self._text_classifier, file=classifier_file)

            # saves the vector file
            vector_file_path = os.path.join(self._models_dir, "tfidf.class")
            vector_file = open(vector_file_path, "wb")
            pickle.dump(obj=self._tfidf, file=vector_file)

            # saves the data frame
            data_frame_file_path = os.path.join(self._models_dir, "data_frame.class")
            data_frame_file = open(data_frame_file_path, "wb")
            pickle.dump(obj=self._data_frame, file=data_frame_file)

            # saves y_pred
            data_frame_file_path = os.path.join(self._models_dir, "y_pred.class")
            data_frame_file = open(data_frame_file_path, "wb")
            pickle.dump(obj=self._y_pred, file=data_frame_file)

            # saves y_test
            data_frame_file_path = os.path.join(self._models_dir, "y_test.class")
            data_frame_file = open(data_frame_file_path, "wb")
            pickle.dump(obj=self._y_test, file=data_frame_file)
        else:
            raise Exception("Cannot train model without dataset, please download first.")

    def get_category(self, text):
        '''
        Returns the category of the given piece of text.
        :param text: the text to identify the topic of
        :return:
        '''

        if self._text_classifier is not None and self._tfidf is not None:
            category_id_df = self._data_frame[['category', 'category_id']].drop_duplicates().sort_values('category_id')
            id_to_category = dict(category_id_df[['category_id', 'category']].values)

            text = [text]
            text_features = self._tfidf.transform(text)
            predictions = self._text_classifier.predict(text_features)

            for text, predicted in zip(text, predictions):
                return id_to_category[predicted]
        else:
            raise Exception("Model not found. Please re-train model.")

    def get_all_categories(self):
        '''
        Returns a set of all unique topics possible in the model.
        '''

        if self._data_frame is not None or os.path.exists(os.path.join(self._models_dir, "data_frame.class")):
            return set(self._data_frame["category"].tolist())
        else:
            raise Exception("Attempted to use data frame without it existing. Please re-train model.")
