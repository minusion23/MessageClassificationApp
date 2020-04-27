# Importing packages needed to load and process data needed for training and
# outputting a classifier
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    ''' Loads raw messages and the related categories assigned to them returns
    a dataframe containing both the messages and assigned categories'''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on = 'id')
    return df
    
def clean_data(df):
# create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)

# Adding a clear list that will host the names without any additional details

    column_names = []
    columns = categories.iloc[1]
    columns = columns.str.split("-", n = 1)

# Iterating thorough the seires to have clear column names 
    for name, index in columns:
        column_names.append(name)
        
# Updating the dataframe which row will serve as the base for the column names
    col_names = pd.Series(column_names)

# Changing the column_names

    categories.rename(columns = col_names ,inplace = True)

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        
        # convert column from string to numeric
        
        categories[column] = categories[column].astype(int)

# drop the original categories column from `df` to use the new updated version
    
    df.drop(columns = 'categories',inplace = True)

# concatenate the original dataframe with the new `categories` dataframe

    df = pd.concat([df, categories], axis = 1, join = 'inner', sort= False, ignore_index = False)

# drop duplicates
    dup_rows = df.duplicated()[df.duplicated() > 0]
    df.drop(index = dup_rows.index, inplace = True)

    return df

def save_data(df, database_filename):

    ''' Saving the loaded and cleaned data into an sql databse to be used for
    classification '''
    
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DatabTable', engine, index=False)

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
