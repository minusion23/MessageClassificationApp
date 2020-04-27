#Importing packages needed to service the file
import sys
import nltk
import pickle
nltk.download(['punkt', 'wordnet'])
from sqlalchemy import create_engine
from sqlalchemy.engine import reflection
import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    ''' Loading the database prepared in the previous step with merged messages andcategories assinged to them. This is crucial for training and evaluating the model'''

    # Establish engine connection
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    con = engine.connect()

#   Read the table from the database that contains the data
    df = pd.read_sql_table("DatabTable", con )
    
#   Divide the information into messages and their classification. This division is crucial for the model to fit and train on the response vector Y and related
#   data X

    X = df['message']
    Y = df.iloc[:, 4:40]
    category_names = Y.columns

#   The function returns the data needed to train the model as well as category_names for a clearer evaluation output

    return X, Y, category_names

def tokenize(text):
#   Tokenize inserted text message, with stripping, forcing the lower case and lemmatizing
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
#   Set up the pipeline and work to tweak the model to achieve the best result. Setting n_jobs to -1 to allow for multicorep rocessing to speed up processing
    pipeline =  Pipeline([
    
              ('vect', CountVectorizer(tokenizer=tokenize)),
              ('tfidf', TfidfTransformer()),            
              ('clf', MultiOutputClassifier(KNeighborsClassifier(),n_jobs = -1 ))
    ])
    
#   Parameters to test the model with GridSearch CV to find the best combination stemming from the provided data
    parameters = {
    'clf__estimator__leaf_size': [15, 30],
    'clf__estimator__n_neighbors': [3, 5],
    }

    # Returning the model that is going to train
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv
def evaluate_model(model, X_test, Y_test, category_names):
    
#   Testing the predictions made by the model against the test values
    y_predicted = model.predict(X_test)
    print((Y_test == y_predicted).mean())
    for i in range(35):
        print(classification_report(Y_test.iloc[:,i],y_predicted[:,i], target_names = category_names[[i]]))
    
def save_model(model, model_filepath):

#  Saving the model with the best options providedby GridSearchCV optimization

    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
