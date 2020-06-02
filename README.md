# News Article Text Classification
A pre-trained Multi-Class Text Classification model for identifying the topic of news articles. The purpose of this model is to provide it with a given piece of text and to be provided with it's topic/ category. These categories are based off common news website categories. 

This model was created with the help of Susan Li, [who wrote this article](https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f).

## Instalation 
```shell
python -m pip install git+https://github.com/user1342/News-Article-Text-Classification.git
```

## Usage 
Using the class:
```python 
from news_classification.news_topic_text_classifier import news_topic_text_classifier
model = news_topic_text_classifier()
```

Print model data:
```python
model.print_model_feature_data()
```

Identify the topic of a given piece of text:
```python
# Get all categories currently used in the model
print(model.get_all_categories())
# Get the category of a given piece of text
print(model.get_category(r"The introduction of the General Data Protection Regulation (GDPR), the EU is enacting a set of mandatory regulations for businesses that go into effect soon, on May 25, 2018. Organisations found in non-compliance could face hefty penalties of up to 20 million euros, or 4 percent of worldwide annual turnover, whichever is higher. Simply put, GDPR was enacted to give citizens and residents more control over their personal data and puts strict data handling rules in place governing “controllers” that collect data from EU residents, and “processors” that process the data on behalf of controllers, such as cloud providers."))
```

While the model is pre-trained you can re-download a new set of training data and re-train the model. The creation of the dataset uses news website RSS feeds to download their most recent articles in specific categories. These news websites include: BBC, The Daily Mail, The Independant, Wired, and CNN.  Re-training the model will use this new data set. If no paramiters are given they default to the deafult location and will over-write the existing dataset. 
```python 
new_data_set_location = "new_data_set.csv"
model.create_data_set(dataset=new_data_set_location)
model.re_train(dataset=new_data_set_location)

print(model._data_frame.head())
```

## Categories 
The categories used as part of the multi-class text classification are:
- Business
- Politics
- Health
- Family And Education
- Science And Enviroment
- Technology
- Entertainment And Arts
- Sport
- Travel
- Food And Drink
