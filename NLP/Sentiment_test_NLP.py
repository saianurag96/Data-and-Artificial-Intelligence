#Importing necessary libraries

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import glob2
import nltk
import warnings
warnings.filterwarnings('ignore') 
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM,Embedding, SpatialDropout1D, Flatten, Dropout,SimpleRNN
from keras.models import Sequential
from sklearn.metrics import accuracy_score
import pickle



#Reading all the positive review files present in the test data folder

read_pos_files=glob2.glob(r'data\aclImdb\test\pos\*.txt')

#Writing all the positive reviews present in the test data folder to the file result_pos.txt with delimiter as "\n"

with open(r"data\aclImdb\test\result_pos.txt", "w",encoding="utf8") as outfile:
    for f in read_pos_files:
        with open(f, "r",encoding="utf8") as infile:
            outfile.write(infile.read()+"\n")

#Reading all the negative review files present in the test data folder

read_neg_files=glob2.glob(r'data\aclImdb\test\neg\*.txt')

#Writing all the negative reviews present in the test data folder to the file result_neg.txt with delimiter as "\n"

with open(r"data\aclImdb\test\result_neg.txt", "w",encoding="utf8") as outfile:
    for f in read_neg_files:
        with open(f, "r",encoding="utf8") as infile:
            outfile.write(infile.read()+"\n")

#Reading all the delimted positive reviews present in the file result_pos.txt as a dataframe

x_pos=pd.read_csv(r"data\aclImdb\test\result_pos.txt",header=None,sep='\n')

#Converting the dataframe to a list

x_pos_list = x_pos[0].tolist()
#x_pos_list=x_pos_list[0:200]

#Reading all the delimted negative reviews present in the file result_neg.txt as a dataframe

x_neg=pd.read_csv(r"data\aclImdb\test\result_neg.txt",header=None,sep='\n')

#Converting the dataframe to a list
x_neg_list = x_neg[0].tolist()
#x_neg_list=x_neg_list[0:200]

x_pos_list1=[]
for i in range(len(x_pos_list)):
    x_pos_list[i] = x_pos_list[i].lower() 
    text = re.sub('\n', ' ', x_pos_list[i])
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text=re.sub('[^abcdefghijklmnopqrstuvwxyz\s]', '',text)
    text = re.sub(r'\s+',' ',text)
    x_pos_list1.append(text)
    
lemmatizer = WordNetLemmatizer()
for i in range(len(x_pos_list1)):
    words = nltk.word_tokenize(x_pos_list1[i])
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    x_pos_list1[i] = ' '.join(words)  
    
x_neg_list1=[]
for i in range(len(x_neg_list)):
    x_neg_list[i] = x_neg_list[i].lower() 
    text = re.sub('\n', ' ', x_neg_list[i])
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text=re.sub('[^abcdefghijklmnopqrstuvwxyz\s]', '',text)
    text = re.sub(r'\s+',' ',text)
    x_neg_list1.append(text)
    
lemmatizer = WordNetLemmatizer()
for i in range(len(x_neg_list1)):
    words = nltk.word_tokenize(x_neg_list1[i])
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    x_neg_list1[i] = ' '.join(words) 

x_test_list = x_pos_list1+x_neg_list1

Y_test = [1 if j < 12500 else 0 for j in range(25000)]
Y_test = np.array(Y_test)


#Reading the pickle file for Tokenizer Object
tokenizer_obj = pickle.load(open("data/tokenizer_obj.pkl", 'rb'))

#Preparing the test dataset for the model.

data_test = pd.DataFrame(x_test_list, columns=['review',])
X_test = tokenizer_obj.texts_to_sequences(data_test['review'].values)
X_test = pad_sequences(X_test,maxlen=250)

#loading the saved model

from keras.models import load_model
loaded_model = load_model(r"models\Group55_model_NLP.h5")

acc = loaded_model.evaluate(X_test, Y_test, verbose = True, batch_size = 64)
print("Accuracy on Test Data: {}".format(round(acc[1],4)))