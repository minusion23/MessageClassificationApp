# Disaster Response Pipeline Project

The following project works to categorize text messages received from invidviduals in need of help.
A database with messages and their categorization has been provided for analysisc (disaster_messages.csv - pure messages, disaster_categories.csv - messages categorized)

1. The process_data.py scriptimports the datbases and clean the data. The script unpacks the information from the categories file and merges the information with the original messages
to prepare for a training process to be performed on the dataset. Duplicates are deleted as well and only the necessary information is left with the categories one-hot encoded.
The result of these operations is saved as an SQLite database.

2. The train_classifier.py script creates an ML pipeline that takes the message, tokenizes the text and then feeds it to the ML classification engine. GridSearchCV is used to optimize
the model. After the training and validation process has been completed the model is saved in a pickle file for futher use on a web page

3. Run.py runs a webpage through flask displaying some basics metrics about the training dataset on the homepage, as well as allowing the user to enter a text message that is then
evaluated by the model and categorized.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

Summary

The accuracy of the model still needs to increase but based on the provided data the results of the model are satisfactory.