import spacy
import pandas as pd
from spacytextblob.spacytextblob import SpacyTextBlob

# For the report.
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import defaultdict

# Start
# Import spaCy (Python's natural language processing library).
# Import pandas to read in the csv file and to create a data frame of the dataset.
# Import spacytextblob from the TextBlob library for polarity and sentiment analysis.
# Import WordCloud, matplotlib.pyplot and defaultdict for analysing the effectiveness of identifying the positive and
# negative words for the sentiment analysis.
# Load the simple english language model to enable natural language processing tasks.
# Read in the amazon product reviews file and call the data frame 'products'.
# Creating a function to preprocess the reviews from the 'reviews.text' column for lemmatisation.
# Create a function to get a polarity score of the reviews.
# Create a function to get polarity and subjectivity scores of the reviews.
# Create a function to get the sentiment in words based off the polarity scores of the reviews.
# Select only the relevant columns for analysis.
# Check for any duplicate reviews and drop rows with duplicate reviews.
# Remove any rows with null values for a complete dataset as long as it does not affect the integrity and representativeness
# of the dataset largely and assign this to a new DataFrame called 'clean_data'.
# Ensure that all values are easily understood under each column and adjust otherwise.
# Apply the preprocess function to the 'reviews.text' column to preprocess the text and store the preprocessed reviews
# in a new column called 'processed_reviews'.
# Apply the function to get the polarity scores to the preprocessed reviews and store in a new column.
# Apply the sentiment score function to get the tuples of polarity and subjectivity on the preprocessed reviews and store 
# in a new column.
# Apply the sentiment function on the processed reviews to identify whether the sentiment of the reviews are
# positive/negative/neutral and store in a new column.
# In order to compare the effectiveness of these, compare the sentiments in words against the doRecommend column
# and the reviews.title column to easily identify if the sentiments identified are along the same lines as the 
# reviews.title is a summary of what the review represents.
# To further assess the effectiveness of this analysis, conduct the sentiment analysis on the 'reviews.title' column 
# after preprocessing to check whether the sentiment in words matches our own intuitive understanding of the sentiment.
# Select two reviews that have been identified to have the same sentiment and check their similarity scores.
# Select two reviews at random and check their similarity scores and assess whether it is in alignment
# to the polarity scores.
# Initialise positive and negative words dictionaries to store positive and negative words identified from the
# processed reviews.
# Create a function to enable the above.
# Use WordCloud and matplotlib.pyplot to create a visualisation of the positive and negative words identified.
# End

nlp = spacy.load('en_core_web_sm')

# Download additional data for TextBlob using the following:
#python3 -m textblob.download_corpora

# Add the spacytextblob object to the spaCy pipeline.
nlp.add_pipe('spacytextblob')

products = pd.read_csv('amazon_product_reviews.csv', low_memory=False)

# Preprocess to remove any trailing whitespaces and turn words into lowercase letters for all the lemmatised words 
# for the tokens that are not stop words nor punctuation marks and those that are alphabetical.
def preprocess(text):
    try:
        # Runs the text through the simple English language model.
        doc = nlp(text)
        processed_text = ' '.join([token.lemma_.lower().strip() for token in doc if not token.is_stop and not token.is_punct and token.is_alpha])
        return processed_text
    # returns errors occurred, if they do.
    except Exception as e:
        print(f"Error processing text: {e}")
        return ""
    
# Function to analyse the overall polarity of each of the reviews using spacytextblob.
def analyse_polarity(text):
    # Process the review using spaCy and TextBlob through the simple English language model.
    doc_polarity = nlp(text)
    
    # Initialise the variables to add the polarity scores and the number of tokens.
    total_polarity = 0
    num_tokens = 0
    
    # Iterate over each token in the processed review.
    for token in doc_polarity:
        
        # Analyse polarity of the token.
        polarity = token._.blob.polarity
        
        # Accumulate polarity score and count of tokens.
        total_polarity += polarity
        num_tokens += 1
    
    # Calculate the average polarity for the entire review.
    if num_tokens > 0:
        average_polarity = total_polarity / num_tokens
    else:
        # To avoid the division by zero error.
        average_polarity = 0 
    
    return average_polarity

# Function to get the tuple of polarity and subjectivity scores:
def analyse_sentiment_scores(text):
    doc_sentiment = nlp(text)
    
    # Initialise the variables to accumulate the sentiment and subjectivity scores and count the number of
    # tokens at each iteration.
    total_sentiment = 0
    total_subjectivity = 0
    num_tokens = 0
    
    # Iterate over each token in the processed text.
    for token in doc_sentiment:
        # Calculate the sentiment and subjectivity scores of the token.
        sentiment = token._.blob.sentiment.polarity
        subjectivity = token._.blob.sentiment.subjectivity
        
        # Add the sentiment and subjectivity scores and count the number of tokens.
        total_sentiment += sentiment
        total_subjectivity += subjectivity
        num_tokens += 1
    
    # Calculate the average sentiment and subjectivity scores for the entire string.
    if num_tokens > 0:
        average_sentiment = total_sentiment / num_tokens
        average_subjectivity = total_subjectivity / num_tokens
    else:
        # Avoid the division by zero error.
        average_sentiment = 0 
        average_subjectivity = 0
    
    return (average_sentiment, average_subjectivity)

# Function to get the sentiment in words:
def analyse_sentiment(polarity_score):
   
    if polarity_score > 0:
        sentiment = 'positive'
    
    elif polarity_score < 0:
        sentiment = 'negative'
    
    else:
        sentiment = 'neutral'

    return sentiment

# Function for the report to be used for WordCloud when identifying positive and negative words:
def find_sentiment_words(processed_text):
    
    doc = nlp(processed_text)
    for token in doc:

        polarity = token._.blob.polarity

        # adds the tokens in text form to the corresponding dictionaries.
        if polarity > 0:
            positive_words[token.text] += 1
        elif polarity < 0:
            negative_words[token.text] += 1


# Print the top 5 rows to get an overview of the dataset.
print(products.head())

# Summary to observe how many values there are and the data types for each column.
print(products.info())

# To find how many null values there are for each column.
print(products.isnull().sum())

# Select all the relevant columns required and call the new data frame 'products'.
products = products[['id', 'name', 'reviews.doRecommend', 'reviews.rating', 'reviews.title', 'reviews.text']]
print(products.info())

# View all duplicate reviews.
duplicate_reviews = products[products.duplicated(subset = ['reviews.text'])]
print(duplicate_reviews['reviews.text'])

# Remove any duplicates from the 'reviews.text' column and keep the first instances of the duplicate. 
# Return and save this in a new data frame called 'products_no_review_duplicates'.
products_no_review_duplicates = products.drop_duplicates(subset = ['reviews.text'], keep='first', inplace=False)
print(products_no_review_duplicates)

# View a summary of this new data frame to see if it is a complete dataset with any null values or not.
print(products_no_review_duplicates.info())

# You could do the following to remove any duplicates just from the reviews.text column:
#clean_products_data = products_no_review_duplicates.dropna(subset=['reviews.text'])
#print(clean_products_data.info())

# However, since I wish to conduct a comparison analysis of the sentiment of the reviews.title
# (the review title acts almost as a small summary of the corresponding review from the reviews.text column) 
# against the reviews.text column values, I will remove the rows with missing values as there are 11 remaining 
# missing values under the reviews.title column to ensure completeness of the dataset and for a comparison.

# Remove all rows with missing values:
clean_data = products_no_review_duplicates.dropna()
print(clean_data.info())

# Replace boolean values with text for the 'reviews.doRecommend' column values to make it easier to understand and to
# compare later to see if it aligns with the reviews and their sentiment.
# True is replaced by 'recommend' and False is replaced by 'not recommend'.
clean_data['reviews.doRecommend'] = clean_data['reviews.doRecommend'].replace({True: 'recommend', False: 'not recommend'})
print(clean_data['reviews.doRecommend'].head())

# Apply the preprocess function to the 'reviews.text' column.
clean_data['processed_reviews'] = clean_data['reviews.text'].apply(preprocess)

# Display the first few rows of the 'review.text' and 'processed_reviews' columns to observe the changes.
print(clean_data[['reviews.text','processed_reviews']].head())

# Check to see the datatypes and the number of values.
print(clean_data.info())

# Add a new column with the polarity scores with the 'analyse_polarity' function applied to the 'processed_reviews'.
# Compare the different columns to see if the entries are entered correctly and if the programme analyses the
# reviews correctly by selecting the top 10 rows.
clean_data['polarity_score'] = clean_data['processed_reviews'].apply(analyse_polarity)
print(clean_data[['reviews.doRecommend', 'processed_reviews', 'polarity_score']].head(10))

# Get and display the tuples of the polarity scores and subjectivity scores of the processed reviews.
# Save in a new column called 'sentiment_scores'.
clean_data['sentiment_scores'] = clean_data['processed_reviews'].apply(analyse_sentiment_scores)
print(clean_data['sentiment_scores'].head())

# Get and display the sentiment in words using the 'analyse_sentiment' function defined earlier based off the 
# polarity scores obtained. Save in a new column called 'reviews_sentiment'.
clean_data['reviews_sentiment'] = clean_data['polarity_score'].apply(analyse_sentiment)
print(clean_data[['reviews.doRecommend', 'reviews.title', 'reviews_sentiment']].head())

# To see the entire DataFrame.
print(clean_data)

# To see how many times the word 'positive' appeared under the 'sentiment' column.
positive_count = len(clean_data[clean_data['reviews_sentiment'] == 'positive'])
print(f"\nThere are a total of {positive_count} positive reviews in the dataset")

# To see how many times the word 'negative' appeared under the 'sentiment' column.
negative_count = len(clean_data[clean_data['reviews_sentiment'] == 'negative'])
print(f"\nThere are a total of {negative_count} negative reviews in the dataset")

# To see how many times the word 'negative' appeared under the 'sentiment' column.
neutral_count = len(clean_data[clean_data['reviews_sentiment'] == 'neutral'])
print(f"\nThere are a total of {neutral_count} neutral reviews in the dataset")


# To compare the polarity and sentiment of the titles against the reviews:
print("Comparing the sentiment between product reviews and their titles: ")
clean_data['preprocessed_title'] = clean_data['reviews.title'].apply(preprocess)
print(clean_data['preprocessed_title'].head())

clean_data['polarity_score_title'] = clean_data['preprocessed_title'].apply(analyse_polarity)
print(clean_data['polarity_score_title'].head())

clean_data['title_sentiment'] = clean_data['polarity_score_title'].apply(analyse_sentiment)
print(clean_data['title_sentiment'].head())

# To compare the sentiments in words of the reviews and titles to see if the functions work correctly.
print(clean_data[['reviews.doRecommend', 'reviews.title', 'reviews_sentiment', 'title_sentiment']].head())

# Similarity Testing of Reviews.

print("\nTesting Similarity between 2 reviews: ")
# Testing similarity between reviews (2 and 4 at indexes 1 and 3).
my_review_of_choice_1 = nlp(clean_data['reviews.text'][1])
my_review_of_choice_2 = nlp(clean_data['reviews.text'][3])

# Calculate similarity:
similarity = my_review_of_choice_1.similarity(my_review_of_choice_2)
print(f"Similarity between 2 reviews of my choice (Reviews 2 and 4): {similarity:.2f}")
# Get the polarity scores of the 2 reviews chosen to compare them against the similarity score.
polarity_score_choice = clean_data['polarity_score'][[1, 3]]
print("\nThe polarity scores of the two reviews are: ")
print(polarity_score_choice)

# Testing similarity between 2 random reviews:
clean_data_sample_testing = clean_data.sample(2, random_state=42)
print(clean_data_sample_testing)

review_1 = nlp(clean_data_sample_testing.iloc[0]['reviews.text'])
review_2 = nlp(clean_data_sample_testing.iloc[1]['reviews.text'])

# Calculate similarity:
similarity = review_1.similarity(review_2)
print(f"Similarity between 2 reviews selected at random: {similarity:.2f}")


# For the word cloud analysis for the report on what words were identified as positive an negative.
# The graphic generated has been included in the report.

# Initialise dictionaries to hold positive and negative words
positive_words = defaultdict(int)
negative_words = defaultdict(int)

# Apply the function defined earlier to categorise words from the processed reviews as positive or negative.
clean_data['processed_reviews'].apply(find_sentiment_words)

# Generate word clouds of positive and negative words based on the frequency of the words in the reviews.
pos_wordcloud = WordCloud(width=400, height=200, background_color ='white').generate_from_frequencies(positive_words)
neg_wordcloud = WordCloud(width=400, height=200, background_color ='white').generate_from_frequencies(negative_words)

# The below creates a visual representation of the words clouds of positive and negative words next to each other
# in one image using matplotlib.pyplot.
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(pos_wordcloud, interpolation='bilinear')
ax[0].set_title('Positive Words')
ax[0].axis('off')

ax[1].imshow(neg_wordcloud, interpolation='bilinear')
ax[1].set_title('Negative Words')
ax[1].axis('off')

plt.show()