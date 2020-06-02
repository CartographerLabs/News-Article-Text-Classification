import csv
import os
import pickle
from pathlib import Path

import feedparser
import newspaper
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import chi2
import numpy as np

class news_topic_text_classifier:
    '''
    News topic text classification model. Based off: https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f
    '''

    # The dataframe that contains the CSV data
    _data_frame = None
    # The model used for text classification
    _text_classifier = None
    # Used in text classification
    _count_vect = None

    # relative file paths
    _script_dir = Path(__file__).parent
    _models_dir = os.path.join(_script_dir, "models")
    _data_dir = os.path.join(_script_dir, "data")

    def __init__(self):
        '''
        The constructor
        '''

        # Checks if the model files exist and if so initialises this class with them. If not the class will need to be re trained.
        classifier_file_path = os.path.join(self._models_dir, "news_text_classifier.class")
        vector_file_path = os.path.join(self._models_dir, "count_vect.class")
        data_frame_file_path = os.path.join(self._models_dir, "data_frame.class")

        if os.path.exists(vector_file_path) and os.path.exists(classifier_file_path) and os.path.exists(data_frame_file_path):
            classifier_file = open(classifier_file_path, "rb")
            vector_file = open(vector_file_path, "rb")
            data_frame_file = open(data_frame_file_path, "rb")

            self._text_classifier = pickle.load(file=classifier_file)
            self._count_vect = pickle.load(file=vector_file)
            self._data_frame = pickle.load(file=data_frame_file)

    @staticmethod
    def create_data_set(dataset = os.path.join(_data_dir, "dataset.csv")):
        """
        A function for creating a dataset following the format used in this model.
        :param dataset:
        """
        deny_words = ["Image copyright", "Getty Images", "image caption", "reuters"]

        dict_of_bbc_feeds = {
            "business": ["http://feeds.bbci.co.uk/news/business/rss.xml", "https://www.dailymail.co.uk/money/index.rss",
                         "http://www.independent.co.uk/news/business/rss",
                         "https://www.wired.com/feed/category/business/latest/rss",
                         "http://rss.cnn.com/rss/money_news_international.rss"],
            "politics": ["http://feeds.bbci.co.uk/news/politics/rss.xml"],
            "health": ["http://feeds.bbci.co.uk/news/health/rss.xml", "https://www.dailymail.co.uk/health/index.rss"],
            "family_and_education": ["http://feeds.bbci.co.uk/news/education/rss.xml"],
            "science_and_enviroment": ["http://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
                                       "https://www.dailymail.co.uk/sciencetech/index.rss",
                                       "http://www.independent.co.uk/news/science/rss",
                                       "https://www.wired.com/feed/category/science/latest/rss",
                                       "http://rss.cnn.com/rss/edition_space.rss"],
            "technology": ["http://feeds.bbci.co.uk/news/technology/rss.xml",
                           "http://www.independent.co.uk/life-style/gadgets-and-tech/rss",
                           "https://www.wired.com/feed/category/gear/latest/rss",
                           "http://rss.cnn.com/rss/edition_technology.rss"],
            "entertainment_and_arts": ["http://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml",
                                       "https://www.dailymail.co.uk/news/arts/index.rss",
                                       "https://www.wired.com/feed/category/culture/latest/rss",
                                       "http://rss.cnn.com/rss/edition_entertainment.rss"],
            "sport": ["http://feeds.bbci.co.uk/sport/football/rss.xml?edition=uk",
                      "http://feeds.bbci.co.uk/sport/football/rss.xml?edition=uk",
                      "http://feeds.bbci.co.uk/sport/golf/rss.xml?edition=uk",
                      "http://www.independent.co.uk/sport/general/athletics/rss",
                      "http://www.independent.co.uk/sport/rugby/rugby-union/rss",
                      "http://rss.cnn.com/rss/edition_football.rss"],
            "travel": ["https://www.dailymail.co.uk/travel/index.rss",
                       "https://www.wired.com/feed/category/transportation/latest/rss"],
            "food_and_drink": ["http://www.independent.co.uk/life-style/food-and-drink/rss"]
        }

        with open(dataset, 'w', newline='', encoding="utf8") as csv_file:
            fieldnames = ['url', 'category', 'body']
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()

            # loop through lists of rss feed topics
            for category in dict_of_bbc_feeds:
                rss_feeds = dict_of_bbc_feeds[category]

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

        if self._text_classifier is not None or os.path.exists(os.path.join(self._models_dir,"news_text_classifier.class")):
            tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                                    stop_words='english')
            features = tfidf.fit_transform(self._data_frame.body).toarray()
            print(features.shape)
            print("-"*20)
            category_id_df = self._data_frame[['category', 'category_id']].drop_duplicates().sort_values('category_id')
            category_to_id = dict(category_id_df.values)

            N = 2
            labels = self._data_frame.category_id

            for Product, category_id in sorted(category_to_id.items()):
                features_chi2 = chi2(features, labels == category_id)
                indices = np.argsort(features_chi2[0])
                feature_names = np.array(tfidf.get_feature_names())[indices]
                unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
                bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
                print("# '{}':".format(Product))
                print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
                print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

        else:
            raise Exception("Unable to retrieve model data as model does not exist. Please re-train the model.")

    def re_train(self, dataset = os.path.join(_data_dir,"dataset.csv")):
        '''
        Trains the text classifier model if it needs re-training or has not already been trained
        :param dataset: csv file location
        '''

        self._data_frame = pd.read_csv(dataset)
        self._data_frame.head()
        col = ['category', 'body']
        self._data_frame = self._data_frame[col]
        self._data_frame = self._data_frame[pd.notnull(self._data_frame['body'])]
        self._data_frame.columns = ['category', 'body']
        self._data_frame['category_id'] = self._data_frame['category'].factorize()[0]

        X_train, X_test, y_train, y_test = train_test_split(self._data_frame['body'], self._data_frame['category'], random_state=0)
        self._count_vect = CountVectorizer()
        X_train_counts = self._count_vect.fit_transform(X_train) #todo pickle
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

        self._text_classifier = MultinomialNB().fit(X_train_tfidf, y_train)

        # Saves the created models in the models dir

        # saves the classifier
        classifier_file_path = os.path.join(self._models_dir, "news_text_classifier.class")
        classifier_file = open(classifier_file_path, "wb")
        pickle.dump(obj=self._text_classifier,file=classifier_file)

        # saves the vector file
        vector_file_path = os.path.join(self._models_dir, "count_vect.class")
        vector_file = open(vector_file_path, "wb")
        pickle.dump(obj=self._count_vect, file=vector_file)

        # saves the data frame
        data_frame_file_path = os.path.join(self._models_dir, "data_frame.class")
        data_frame_file = open(data_frame_file_path, "wb")
        pickle.dump(obj=self._data_frame, file=data_frame_file)

    def get_category(self, text):
        '''
        Returns the category of
        :param text: the text to identify the topic of
        :return:
        '''

        if self._text_classifier is not None and self._count_vect is not None:
            count_vect = self._count_vect
            return self._text_classifier.predict(count_vect.transform([text]))[0]
        else:
            raise Exception("Model not found. Please re-train model.")

    def get_all_categories(self):
        '''
        Returns a set of all unique topics possible in the model.
        '''

        if self._data_frame is not None or os.path.exists(os.path.join(self._models_dir,"data_frame.class")):
            return set(self._data_frame["category"].tolist())
        else:
            raise Exception("Attempted to use data frame without it existing. Please re-train model.")