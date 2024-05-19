#Gerekli kütüphaneleri indir
!pip install stanza
!pip install tpdm
!pip install nltk
!python -m nltk.downloader wordnet
!pip install TurkishStemmer
!pip install fasttext

import pandas as pd
import stanza
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from TurkishStemmer import TurkishStemmer
from sklearn.model_selection import train_test_split
import nltk
import string
import re
from sklearn.model_selection import train_test_split
import fasttext
nltk.download('stopwords')
nltk.download('punkt')

# NLTK Türkçe stop words listesini yükleyin
stop_words = set(stopwords.words('turkish'))

# Stanza'yı başlatın
stanza.download('tr')
nlp = stanza.Pipeline('tr')

# data dosyasını yükleme
sample_data = pd.read_csv("data.csv")

# Create TurkishStemmer and TurkishWordNetLemmatizer instances
stemmer = TurkishStemmer()
lemmatizer = WordNetLemmatizer()

# Preprocessing function with stemming and lemmatization for Turkish text
def preprocess_text(text):

    # Noktalama işaretlerini ve sayıları kaldır
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)

    # Lowercasing
    tokens = [word.lower() for word in tokens]

    # Removing punctuation and stopwords
    table = str.maketrans('', '', string.punctuation)
    words = [word.translate(table) for word in tokens if word not in stop_words]

    # Stemming
    stemmed_words = [stemmer.stem(word) for word in words]

    # Lemmatization
    lemmatized_words = [lemmatizer.lemmatize(word) for word in stemmed_words]

    return ' '.join(lemmatized_words)

tqdm.pandas()
sample_data['Haber Gövdesi'] = sample_data['Haber Gövdesi'].progress_apply(preprocess_text)
sample_data.to_csv('sample_data.csv',index=False)

# İlk olarak, veri setini train + validation ve test olarak ayır
train_data, test_data = train_test_split(sample_data, test_size=0.3, random_state=50)

# Sonra, train + validation verisini train ve validation olarak ayır
train_data, validation_data = train_test_split(train_data, test_size=0.2, random_state=42)  # 0.25 * 0.8 = 0.2

def save_to_fasttext_format(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for index, row in data.iterrows():
            line = str(row["Haber Gövdesi"]) + " __label__" +str( row["Sınıf"]) + "\n"
            f.write(line)

# Dosyaları kaydet
save_to_fasttext_format(train_data, "train.txt")
save_to_fasttext_format(validation_data, "validation.txt")
save_to_fasttext_format(test_data, "test.txt")


# FastText modelini eğit
model = fasttext.train_supervised(input="train.txt", epoch=25, lr=1.0, wordNgrams=2)


# Modeli değerlendir
result = model.test("test.txt")
print(f"Test Samples: {results[0]} Precision@{k} : {results[1]*100:2.4f} Recall@{k} : {results[2]*100:2.4f}")

# Test verisini tahmin et
with open("validation.txt", "r", encoding="utf-8") as f:
    for line in f:
        text = line.strip()
        predicted_label = model.predict(text)
        print("Text:", text)
        print("Predicted Label:", predicted_label[0][0])
