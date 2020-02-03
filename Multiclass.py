#author Eric Ávila
# Version 1.0, not optimized for the dataset yet

#Hyperparameters:

#filters:32,16
#batch_size:32,16
#kernel_size=8
#patience_lr=5,patience_earlyStop=10

import os
import string
import keras
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import re
import numpy as np
import tensorflow as tf
from numpy import array
from sklearn.model_selection import StratifiedKFold
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

from numpy.random import seed
seed(1)

tf.random.set_seed(1)

#root="D:/bananaCorpus"
root="D:/CEFR_corpus"
train_dir=os.path.join(root,"train")

texts=[]
labels=[]

for label in ["A1","A2","B1","B2"]:
     directory=os.path.join(train_dir,label)
     for fname in os.listdir(directory):
         if fname[-4:]==".txt":
             f = open(os.path.join(directory, fname),encoding="utf-8")
             texts.append(f.read())
             f.close()
             if label == "A1":
                 labels.append(0)
             elif label=="A2":
                       labels.append(1)
             elif label=="B1":
                  labels.append(2)
             else:
                labels.append(3)

print(texts)
print(labels)
print("Corpus Length", len( root), "\n")
print("The total number of reviews in the train dataset is", len(texts),"\n")
stops = set(stopwords.words("english"))
print("The number of stopwords used in the beginning: ", len(stops),"\n")
print("The words removed from the corpus will be",stops,"\n")


## This adds new words or terms from words_to_add list to the stop_words
words_to_add=[]
[stops.append(w) for w in words_to_add]

##This removes the words or terms from the words_to_remove list,
##so that they are no longer included in stopwords
words_to_remove=["wasn't","did","didn't","weren't","were","should","should've","doesn","hasn't"]#adjust with real stopwords
[stops.remove(w) for w in words_to_remove ]

texts=[[w.lower() for w  in word_tokenize("".join(str(review))) if  w not in stops and w not in string.punctuation and len(w)>1 and w.isalpha()]for review in texts ]#play with  len(w)

print("costumized stopwords: ", stops,"\n")
print("count of costumized stopwords",len(stops),"\n")
print("**********",texts,"\n")

#vectorization
#tokenizing the raw data
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

maxlen = 50
max_words = 10000


tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

print("Tokens:", sequences,"\n")
word_index = tokenizer.word_index
print("Unique tokens:",word_index,"\n") 
print(' %s unique tokens in total.' % len(word_index,),"\n")
print("Unique tokens: ", word_index,"\n")
print("Dictionary of words and their count:", tokenizer.word_counts,"\n" )
print(" Number of docs/seqs used to fit the Tokenizer:", tokenizer.document_count,"\n")
print(tokenizer.word_index,"\n")
print("Dictionary of words and how many documents each appeared in:",tokenizer.word_docs,"\n")

data = pad_sequences(sequences, maxlen=maxlen, padding="post")
print("padded data","\n")
print(data)

#checking the encoding with a new document
text2="I like to study english in the morning and play games in the afternoon"
text2=[w.lower() for w  in word_tokenize("".join(str(text2))) if  w not in stops and w not in string.punctuation
          and len(w)>2 and w.isalpha()]
sequences = tokenizer.texts_to_sequences([text2])
text2 = pad_sequences(sequences, maxlen=maxlen, padding="post")
print("padded text2","\n")
print(text2)


#K fold cross-validation
labels = np.asarray(labels)

print('Shape of data tensor:', data.shape,"\n")
print('Shape of label tensor:', labels.shape,"\n")
print("labels",labels,"\n")


from keras.models import Sequential#del if nes
from keras.layers import Embedding,Flatten,Dense#del if nes
from keras import layers#del if nes
from keras.layers.convolutional import Conv1D#del if nes
from keras.layers.convolutional import MaxPooling1D#del if nes
from keras.layers.normalization import BatchNormalization#delete if not used
embedding_dim = 100#del if nes


#Pretrained embedding
glove_dir = 'D:\glove'

embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'),encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print("Loading Glove...")
print("Found %s word vectors fom GLOVE."% len(embeddings_index))

#Preparing the Glove word-embeddings matrix to pass to the embedding layer(max_words, embedding_dim)
embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

           
#Setting optimizer externally                       
Adam=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)#del if nes

def create_model():
     
   
    model = Sequential()

    model.add(Embedding(max_words, embedding_dim, input_length=maxlen, weights=[embedding_matrix], trainable=False))#vocabulary size + the size of glove version +max len of input documents.
    model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(BatchNormalization(weights=None, epsilon=1e-06, momentum=0.9))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer=Adam,
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
    
    plot_model(model, to_file='model.png', show_shapes=True)
    return model

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
#callbacks
def get_callbacks(name_weights, patience_lr,patience_earlyStop):
     mpc_save = ModelCheckpoint(name_weights,save_best_only= True,monitor="val_loss",mode="min")
     reduce_lr_loss = ReduceLROnPlateau(monitor="val_loss",patience=patience_lr, verbose=1,mode="min")
     early_stopping = EarlyStopping(monitor="val_sparse_categorical_accuracy", patience=patience_earlyStop)
     return [mpc_save, reduce_lr_loss,early_stopping]

#K-fold
n_folds = 10
kf = StratifiedKFold(n_folds, random_state=None, shuffle=True)
kf.get_n_splits(data)

print("skfold",kf)

for j, (train_index, test_index) in enumerate(kf.split(data,labels)):
    print("\nFold\n",j) 
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    name_weights = "final_model_fold" + str(j) + "_weights.h5"
    callbacks = get_callbacks(name_weights = name_weights,patience_lr=5,patience_earlyStop=10)
    model=create_model()
    history = model.fit(X_train, y_train,
                    epochs=25,
                    batch_size=32,
                    verbose=1, 
                    callbacks=callbacks,
                    validation_data=(X_test, y_test))
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss: %s Test acc: %s' %(loss,acc))
    
    # summarize history for accuracy
    #Plotting the result
    plt.plot(history.history["sparse_categorical_accuracy"])
    plt.plot(history.history["val_sparse_categorical_accuracy"])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
#print(history.history.keys()) uncomment to see the training metrics

print(model.summary())
    
    
     




  



