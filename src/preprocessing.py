"""
SMS Spam Detection System - Preprocessing Module
Provides reusable NLP preprocessing functions using NLTK.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure required NLTK resources are available
def _ensure_nltk_resources():
    resources = ['punkt', 'punkt_tab', 'stopwords']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
        except (LookupError, IndexError):
            try:
                nltk.download(resource, quiet=True)
            except Exception as e:
                # Log or ignore if already downloaded or network restricted
                pass

_ensure_nltk_resources()

# Initialize stemmer and stopword list once for performance
_stemmer = PorterStemmer()
try:
    _stop_words = set(stopwords.words('english'))
except Exception:
    nltk.download('stopwords', quiet=True)
    _stop_words = set(stopwords.words('english'))


def preprocess_text(text: str) -> str:
    """
    Cleans and preprocesses a raw SMS message for NLP modeling.
    
    Steps applied:
    1. Type validation and empty string handling
    2. Lowercase conversion
    3. URL and non-alphanumeric/special character cleaning
    4. Tokenization using NLTK
    5. Stopword removal
    6. Stemming using PorterStemmer
    7. Rejoining tokens into a single clean string
    
    Parameters:
        text (str): The raw text of the SMS message.
        
    Returns:
        str: Preprocessed, cleaned, and stemmed text.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Lowercase conversion
    text = text.lower()

    # 2. Remove URLs (http, https, www)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # 3. Remove punctuation and special characters (retain only alphanumeric and whitespace)
    # Using regex to replace non-alphanumeric characters with a space
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # 4. Tokenization
    try:
        tokens = nltk.word_tokenize(text)
    except Exception:
        # Fallback to whitespace split if tokenizer has unexpected error
        tokens = text.split()

    # 5. Stopword removal & 6. Stemming
    cleaned_tokens = []
    for token in tokens:
        token = token.strip()
        if token and token not in _stop_words:
            stemmed = _stemmer.stem(token)
            if stemmed:
                cleaned_tokens.append(stemmed)

    # 7. Return joined cleaned text
    return " ".join(cleaned_tokens)


if __name__ == "__main__":
    test_message = "Congratulations! You have won a FREE $1,000 Walmart Gift Card. Call 0800-123-4567 or visit http://win.com now!"
    print("Original:", test_message)
    print("Preprocessed:", preprocess_text(test_message))
