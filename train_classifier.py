import sys
import pandas as pd
import nltk
import pickle
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression
from sqlalchemy import create_engine
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('jzproj2', con = engine)
    X = df['message']
    Y = df[df.columns[4:]]
    category_names = df.columns[4:].values
    return X,Y,category_names

def tokenize(text):
    #convert a piece of text into clean tokens
    tokens =  nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    #build the multi classification model
    #First create the pipeline
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    #then test some parameters with GridSearch to find a better model
    #paremeters to test
    parameters = {
        'vect__ngram_range': ((1, 1),(1,2)),
    #    'vect__max_features': (None,5000,10000),
    #    'tfidf__use_idf': (True,False),
    #    'clf__estimator__min_samples_split': (2,3,4),
        'clf__estimator__n_estimators': [10,30,50]
    }
    
    #Grid searched model
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv
    



def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    #convert Y_test to numerical array
    Ytest = np.array(Y_test)

    for i in range(Y_pred.shape[1]):
        print(category_names[i]+':')
        print(classification_report(Ytest[:,i],Y_pred[:,i]))
    


def save_model(model, model_filepath):
    #save the model as pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


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
