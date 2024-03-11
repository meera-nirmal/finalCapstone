# finalCapstone

## Overview
A Python programme that performs sentiment analysis on a dataset of reviews of Amazon products using NLP. This programme uses the simple English language model for natural language processing (NLP) and the spacytextblob library to gather the sentiment towards Amazon's products based on a database of reviews provided by Datafiniti.

## Contents
| Section | Description        |
| ------- | ------------------ |
| [Installation](#installation) | Provides information about how to install the project. |
| [Usage](#usage)                | Provides information about how to use the project.        |
| [Author](#author)              | Provides information about the author of the project.     |


## Installation <a name="installation"></a>
Download python from the following link:
https://www.python.org/downloads/

If you wish to use VS Code which I will be referring to here, if you do not have it downloaded yet, you can download it from:
https://code.visualstudio.com/download

Ensure you have the Python package manager pip installed. Open your terminal and type the following into your terminal (cmd if using Windows and zsh if using macOS:
python3 -m pip --version

If it has not been installed, type the following into your terminal:
python3 -m pip install --upgrade pip

Once installed, install the package virtualenv:
python3 -m pip install --user virtualenv

Go to your project folder, the environment that we create is going to be located in any specific path where your project is located:

Type the following command into your terminal:

cd /Users/username/Documents/

Create the virtual environment:

python3 -m venv _env1

(Here _env1 refers to the name of the environment, you can choose any name that you wish to use.)

Activate the virtual environment:
source _env1/bin/activate

In order to deactivate the virtual environment after finishing with the project, type:
deactivate

To install spacy via pip type the following commands on at a time in your terminal:

```
pip install -U pip setuptools wheel
```
```
pip install -U spacy
```
```
python -m spacy download en_core_web_sm
```

The final command downloads the small English package. You could alternatively download the medium sized package by replacing "en_core_web_sm" with "en_core_web_md" to get more accurate results when running the programme. If you choose to do so, anytime the 'en_core_web_sm' is used, just replace 'sm' with 'md'.

Once the download is complete, navigate to VS Code.

Since the virtual environment has been created manually here, navigate to "View" then "Command Palette".
Select "Python: Select Interpreter". This command will display all available global environments and virtual environments. 

Select the virtual environment (_env1). 
If the environment is not listed you may specify the path by clicking "Enter interpreter path" then select "Find" to browse the file system.

### Install the necessary modules

To download the other essential libraries the following one at a time in your terminal in the virtual environment you intend to use:

1. To create a dataframe using the data provided
```
pip3 install pandas
```

2. To perform sentiment analysis
```
pip3 install spacytextblob
```

3. To download the additional data for spacytextblob
```
python3 -m textblob.download_corpora
```

4. To visualise polarity and sentiment
```
pip3 install wordcloud
```

5. To aid the visualisation
```
pip3 install matplotlib
```

6. For categorising positive and negative words in a default dictionary
```
pip3 install collections
```

Clone the repsitory by typing the following into your terminal within your virtual environment and in the relevant folder that you want the project to be in:
git clone https://github.com/meera-nirmal/finalCapstone/.git

## Usage <a name="usage"></a>

In order to use this programme, either:
a) download the file in this repository
b) download a dataset provided by Datafinity from Kaggle: 
```
https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products
```

Ensure that you have specified the path of where you have saved the dataset of reviews that you will be using for the sentiment analysis here:

<img width="530" alt="read_file" src="https://github.com/meera-nirmal/finalCapstone/assets/152800399/a1c6a3b3-3b0d-4bb7-8a47-425759e75b76">

To ensure completeness of the dataset and for easier analysis; selecting all relevant columns for analysis, removing any duplicates from the column used for the sentiment analysis (reveiws.text column) and dropping any rows with missing values makes for direct comparison later:
<img width="863" alt="drop and remove duplicates" src="https://github.com/meera-nirmal/finalCapstone/assets/152800399/0d69a8cb-4649-4995-a090-18fbbc6579eb"> 

<img width="424" alt="missing_removed" src="https://github.com/meera-nirmal/finalCapstone/assets/152800399/b48ae66c-d87f-412e-b9d7-cdfa09f14258">

The following updates the boolean values for the reviews.doRecommend column to ensure that the column is easier to understand:
<img width="920" alt="recommend" src="https://github.com/meera-nirmal/finalCapstone/assets/152800399/00a92e97-6599-4b5e-927c-b2afe03298d0">

<img width="391" alt="summary" src="https://github.com/meera-nirmal/finalCapstone/assets/152800399/94735219-2502-4b25-8a31-2a3e000d8f93">


The following shows the processed reveiws which have stop words removed and words that have been lemmatised (reduced to their base words) ready for sentiment analysis. The polarity scores match with whether the consumers recommended or did not recommend the relevant product the review was written for where negative polarity scores represent a negative sentiment towards the product and a positive score represents a positive sentiment towards it. Furthermore, the closer the polarity score to positive one (+1), the more psoitive the sentiment towards the product and vice versa:
<img width="663" alt="Understanding polarity" src="https://github.com/meera-nirmal/finalCapstone/assets/152800399/c3f01225-8579-43a1-ae8d-d355ce8dc418">

The following shows a tuple of both polarity and subjectivity scores. The higher the score (i.e. the closer the score is to 1), the more subjective the review is:

<img width="708" alt="sentiment_and_subjectivity" src="https://github.com/meera-nirmal/finalCapstone/assets/152800399/ee765452-9680-458d-a8f8-31153c7f21d0">

The following shows the sentiment in words based on the polarity scores of the processed reviews:
<img width="675" alt="overall" src="https://github.com/meera-nirmal/finalCapstone/assets/152800399/c87d29b8-8498-4217-84d1-36367472b5f0">

Comparing the sentiment of the titles of the reviews and the processed reviews shows if they align and how accurate the programme is (assuming that all data has been entered correctly):
<img width="737" alt="sentiment_reviews_titles" src="https://github.com/meera-nirmal/finalCapstone/assets/152800399/09e35034-83f0-42cc-9232-1ec666a3eff8">

Comparing the sentiment between the second and fourth review (which we would expect to have some similarity since they are both positive reviews about the same product):
<img width="519" alt="similarity_2_reviews" src="https://github.com/meera-nirmal/finalCapstone/assets/152800399/3b147517-c2a3-4ce5-9924-d7ca71eaf501">

Similarity between two random reveiws:
<img width="849" alt="random_reveiws1" src="https://github.com/meera-nirmal/finalCapstone/assets/152800399/33c9bafd-faef-4165-a40b-dfee1d25cc40">
<img width="392" alt="randome_reviews2" src="https://github.com/meera-nirmal/finalCapstone/assets/152800399/b8639f4a-8791-4361-8465-0c82e4de3b2a">

The sentiment of both of the reviews above are positive and thus have some similarity, however, they refer to different Amazon products which suggests why the similarity score is not closer to 1.

The following shows which words were identified as having a positive and negative sentiment in the model:
![Figure_1_pos_and_neg_words_identified](https://github.com/meera-nirmal/finalCapstone/assets/152800399/1a83a208-b053-4360-8320-1b010978d724)

## Author <a name="author"></a>

This programme was written my Meera Nirmal.
The link to the profile is: 
```
https://github.com/meera-nirmal
```
