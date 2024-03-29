import sys
import math
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    messages.head()
    #messages.shape
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    categories.head()
    # merge datasets into one dataframe
    df = pd.merge(messages, categories,left_index=True,right_index=True,how='outer')
    return df

def clean_data(df):
    #This function clean the data we loaded and ready for modeling of the data
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";",expand=True)
    categories.head()
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    # use this row to extract a list of new column names for categories.
    row = row.map(lambda x:x[:-2]).tolist()
    #print(row)
    #Asssign row to catagary names
    category_colnames = row
    #print(category_colnames)
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    categories.head()

    #Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].map(lambda x : x[-1])
    
        # convert column from string to numeric
        categories[column] = categories[column].map(lambda x : int(x))

    #categories.head()

    #Replace categories column in df with new category columns.
    # drop the original categories column from `df`
    df= df.drop(labels=['id_y','categories'],axis=1)
    #df.head()
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.merge(df,categories,left_index=True,right_index=True,how='outer')
    #df.head()
    # check number of duplicates and drop the duplicate
    df.duplicated().sum()
    df = df.drop_duplicates()

    return df
    

def save_data(df, database_filename):
    #save the cleaned dataframe to the sql databese
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('jzproj2', engine, index=False)
    

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
