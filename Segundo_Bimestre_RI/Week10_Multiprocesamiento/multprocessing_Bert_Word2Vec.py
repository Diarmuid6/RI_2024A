from multiprocessing import Pool, TimeoutError
from random import randrange
import time
import os
import kaggle
import pandas as pd
import numpy as np
import gensim.downloader as api
from tqdm import tqdm
from transformers import BertTokenizer, TFBertModel
from gensim.models import KeyedVectors



# Cargar modelo Word2Vec y guardarla en variables globales
def initialize_model_word2vec():
    global word2vec_model
    word2vec_model= KeyedVectors.load('word2vec-google-news-300.model')


# Cargar modelos BertTokenizery TFBertModel, luego guardarlas en variables globales
def initialize_model_bert():
    global tokenizer
    global model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertModel.from_pretrained('bert-base-uncased')

#generar los embeddings Wor2vec
def generate_word2vec_embeddings(texts):
    global word2vec_model
    embeddings = []
    tokens = texts.lower().split()
    word_vectors = [word2vec_model[word] for word in tokens if word in word2vec_model]  
    if word_vectors:
        embeddings.append(np.mean(word_vectors, axis=0))
    else:
        embeddings.append(np.zeros(word2vec_model.vector_size))
    return np.array(embeddings)

#generar los embeddings Bert
def generate_bert_embeddings(text):
    global tokenizer
    global model
    inputs = tokenizer(text, return_tensors='tf', padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]


if __name__ == '__main__':

    wine_df = pd.read_csv('data/winemag-data_first150k.csv')
    corpus_df = wine_df['description']

    print(corpus_df.shape)


    # BERT EMBEDDINGS Multiprocesamiento

    pool = Pool(processes=5, initializer=initialize_model_bert)

    with tqdm(total=len(corpus_df)) as pbar:
        results = []
        for result in pool.imap(generate_bert_embeddings, corpus_df):
            results.append(result)
            pbar.update()

    pool.close()  # Cerrar el pool
    pool.join()   

    bert=np.array(results).transpose(0,2,1)
    np.save("bert_embeddings_total.npy", bert)    # guardar el resultado para luego usarlo en .ipynb

    # WORD2VEC EMBEDDINGS Multiprocesamiento

    poolVEC = Pool(processes=5, initializer=initialize_model_word2vec)

    with tqdm(total=len(corpus_df)) as pbar:
        resultsVEC = []
        for result in poolVEC.imap(generate_word2vec_embeddings, corpus_df):
            resultsVEC.append(result)
            pbar.update()

    poolVEC.close()  # Cerrar el pool
    poolVEC.join()  

    vecResutls=np.array(results).transpose(0,2,1) 
    np.save("vec_embeddings.npy", vecResutls)   # guardar el resultado para luego usarlo en .ipynb
