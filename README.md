# Setting up Python Virtual Environment and Installing Virtualenv

## Prerequisites

This project was run on Windows.
Before proceeding, make sure you have the following installed on your system:

1. Python (version 3.9 or higher)
2. pip (Python package manager)

## Setup virtual Environment

Open your terminal or command prompt and run the following command to install `virtualenv` globally:

```bash

# Install virtual env globally
pip install virtualenv

# To create the virtual environment
python -m venv his_project_env

#For Windows
#Activate the virtual environment
his_project_env/Scripts/activate


#For Windows
#Run the Project
python src/main.py

# Install requirements packages also
pandas
nltk
gensim
pyLDAvis
matplotlib
wordcloud
ipython

# Deactivate virtual Env
deactivate
```
