from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from tqdm import tqdm
import nltk
import re
import random 
# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def process_text(text):
    # Initialize variables
    stop_words = set(stopwords.words('english'))
    custom_stop_words = {'would', 'use', 'dont', 'think', 'like', 'time', 'work', 'could', 'one', 'people', 'use', 'year', 'way', 'thing', 'u', 'im', 'may', 'look', 'use', 'u', 'want'}
    stop_words.update(custom_stop_words)
    lemmatizer = WordNetLemmatizer()

    # Remove punctuation, numbers, and non-word characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize the text
    tokens = nltk.word_tokenize(text.lower())

    # Remove stop words and lemmatize the tokens
    processed_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(token))
                        for token in tokens
                        if token not in stop_words]

    # Keep only nouns
    noun_tokens = [token for token in processed_tokens if get_wordnet_pos(token) == wordnet.NOUN]

    # Join the processed tokens back into a string
    processed_text = ' '.join(noun_tokens)

    return processed_text

def process_20newsgroups(categories = ['comp.graphics', 'sci.med', 'rec.sport.baseball', 'talk.politics.misc','rec.autos','talk.politics.mideast']):
    # Load the 20 Newsgroups dataset with specific categories
    newsgroups = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

    processed_data = []

    # Process each document in the dataset
    for doc in tqdm(newsgroups.data, desc="Processing documents"):
        processed_doc = process_text(doc)
        processed_data.append(processed_doc)
    random.shuffle(processed_data)

    return processed_data


if __name__ == "__main__":
    # Specify the desired categories
    categories = ['comp.graphics', 'sci.med', 'rec.sport.baseball', 'talk.politics.misc','rec.autos','talk.politics.mideast']

    # Call the function to process the 20 Newsgroups dataset with specific categories
    data = process_20newsgroups(categories)[:1000]

    # Print the number of processed documents
    print(f"Processed {len(data)} documents.")

    # Write the processed data to a file
    with open("20_news_group_nouns_topics.txt", "w", encoding="utf-8") as file:
        for doc in data:
            file.write(doc + "\n")

    print("Processed data has been written to 20_news_group_nouns_topics.txt.")