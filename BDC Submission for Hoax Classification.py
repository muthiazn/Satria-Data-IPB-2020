import pandas as pd

train_set = pd.read_csv('./Data Latih BDC.csv')
test_set = pd.read_csv('./Data Uji BDC.csv')

train_set.drop(columns=['ID', 'tanggal', 'nama file gambar'], inplace=True)
train_set['judul'] = train_set['judul'].astype(str)
train_set['narasi'] = train_set['narasi'].astype(str)

test_set.drop(columns=['ID', 'tanggal', 'nama file gambar'], inplace=True)
test_set['judul'] = test_set['judul'].astype(str)
test_set['narasi'] = test_set['narasi'].astype(str)

!pip
install
Sastrawi
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

indo = stopwords.words('indonesian')

for i in range(0, len(train_set)):
    review_judul = re.sub('[^a-zA-Z]', ' ', train_set['judul'][i])
    review_judul = review_judul.lower()  # lower case all text
    review_judul = review_judul.split()

    psi = StemmerFactory()
    stemmer = psi.create_stemmer()
    review_judul = [stemmer.stem(word) for word in review_judul if not word in indo]
    review_judul = ' '.join(review_judul)
    train_set.loc[
        i, 'judul'] = review_judul

for i in range(0, len(train_set)):
    review_narasi = re.sub('[^a-zA-Z]', ' ',
                           train_set['narasi'][i])
    review_narasi = review_narasi.lower()
    review_narasi = review_narasi.split()

    psi = StemmerFactory()
    stemmer = psi.create_stemmer()
    review_narasi = [stemmer.stem(word) for word in review_narasi if not word in indo]
    review_narasi = ' '.join(review_narasi)
    train_set.loc[
        i, 'narasi'] = review_narasi

for i in range(0, len(test_set)):
    review_judul = re.sub('[^a-zA-Z]', ' ', test_set['judul'][i])
    review_judul = review_judul.lower()
    review_judul = review_judul.split()

    psi = StemmerFactory()
    stemmer = psi.create_stemmer()
    review_judul = [stemmer.stem(word) for word in review_judul if
                    not word in indo]
    review_judul = ' '.join(review_judul)
    test_set.loc[i, 'judul'] = review_judul

for i in range(0, len(test_set)):
    review_narasi = re.sub('[^a-zA-Z]', ' ', test_set['narasi'][i])
    review_narasi = review_narasi.lower()
    review_narasi = review_narasi.split()

    psi = StemmerFactory()
    stemmer = psi.create_stemmer()
    review_narasi = [stemmer.stem(word) for word in review_narasi if
                     not word in indo]
    review_narasi = ' '.join(review_narasi)
    test_set.loc[i, 'narasi'] = review_narasi

# ---

corpus = []
for i in range(0, len(train_set)):
    corpus.append(train_set.loc[i, 'judul'] + ' ' + train_set.loc[i, 'narasi'])

# Creating Sparse Matrix for Train Set
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
X1 = cv.fit_transform(corpus)
y1 = train_set.loc[:, 'label'].values

df = pd.DataFrame.sparse.from_spmatrix(X1, columns=cv.get_feature_names())
df['label'] = y1

corpus_2 = []
for i in range(0, len(test_set)):
    corpus_2.append(test_set.loc[i, 'judul'] + ' ' + test_set.loc[i, 'narasi'])

# Creating Sparse Matrix for Test Set
X2 = cv.fit_transform(corpus_2)
test_df = pd.DataFrame.sparse.from_spmatrix(X2, columns=cv.get_feature_names())

df = df.filter([kata for kata in test_df.columns])
test_df = test_df.filter([kata2 for kata2 in df.columns])
test_df.drop(columns=['label'], inplace=True)

# Dataset Splitting
X = df.drop(columns=['label'])
y = df.label

train_df = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# Training Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

classifierRF = RandomForestClassifier(n_estimators=500, criterion='entropy', random_state=0)
classifierRF.fit(X_train, y_train)

# Calculating F1-Score
import sklearn.metrics

y_pred_RF = classifierRF.predict(X_val)
f1 = sklearn.metrics.f1_score(y_val, y_pred_RF)
print(f1)

# Predicting Test Set
test_pred = classifierRF.predict(test_df)