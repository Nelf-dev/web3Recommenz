import nltk

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')