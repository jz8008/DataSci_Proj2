#
This is the submission for the Disaster Response Pipeline project.

In this project, a machine learning pipeline was created based on real messages that were sent during disaster events, the pipeline categorize these events so that the messages can be sent to an appropriate disaster relief agency. There are three components in this project:
ETL(Extract, Transform, Load) pipeline to load and clean the data.
ML(Machine Learning) pipeline to train and evaluate the model.
Flask Web App to display the visualization.

The repository holds three python scripts:
process_data.py is ETL and will read in the raw data and merge/clean the data and save the data into a databese
train_classifier.py is ML pipeline and will read the clean data from database and train the model with it and evaluate the model
run.py will run the web app and create visualizations.

Instructions for runing the project:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

cd app
3. Go to http://0.0.0.0:3001/

You should see something like this image.

 



