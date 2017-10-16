import tensorflow as tf
import numpy as np
import glob #this will be useful when reading reviews from file
import os
import tarfile
import re
import math

batch_size = 50
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
maxColumn = 40

numClasses = 2
data = []
def cleanSentences(string):
    stopwords=["a", "an", "and", "are", "as", "at", "be", "but", "by", "for","from","if", "in", "into", "is", "it",
               "no", "not", "of", "on", "or", "such",
               "that", "the", "their", "then", "there", "these",
               "they", "this", "to", "was", "will", "with"]
    words = [w.replace(w, '') if w in stopwords else w for w in string.split()]
    string = ' '.join(words)
    string = string.lower().replace("<br />", " ")
    #REMOVE STOP WORDS
    return re.sub(strip_special_chars, "", string.lower())

def load_data(glove_dict):
    """
    Take reviews from text files, vectorize them, and load them into a
    numpy array. Any preprocessing of the reviews should occur here. The first
    12500 reviews in the array should be the positive reviews, the 2nd 12500
    reviews should be the negative reviews.
    RETURN: numpy array of data with each row being a review in vectorized
    form"""
    global data
    folders=['pos','neg']

    maxReviews = 12500
    count=0
    if not (os.path.isdir(folders[0]) and os.path.isdir(folders[1])):
        tar = tarfile.open("reviews.tar")
        tar.extractall()
        tar.close()
    for folder in folders:
        for element in os.listdir(folder):
            with open(folder + '/'+element,'r',encoding="utf-8") as file:
                review = [glove_dict[word] for word in cleanSentences(file.read()).split() if word in glove_dict][:maxColumn]
                if len(review) < maxColumn:
                    review.extend([0]*(maxColumn - len(review)))
                data.append(review)
    data = np.array(data)
    return data


def load_glove_embeddings():
    """
    Load the glove embeddings into a array and a dictionary with words as
    keys and their associated index as the value. Assumes the glove
    embeddings are located in the same directory and named "glove.6B.50d.txt"
    RETURN: embeddings: the array containing word vectors
            word_index_dict: a dictionary matching a word in string form to
            its index in the embeddings array. e.g. {"apple": 119"}
    """
    #data = open("glove.6B.50d.txt",'r',encoding="utf-8")
    #if you are running on the CSE machines, you can load the glove data from here
    #data = open("/home/cs9444/public_html/17s2/hw2/glove.6B.50d.txt",'r',encoding="utf-8")
    word_index_dict={};
    embeddings=[];
    temp=[];
    i=0
    with open("glove.6B.50d.txt",'r',encoding="utf-8") as file:
        for line in file:
             temp = line.strip().split(" ",1)
             embeddings.append(np.fromstring(temp[1], dtype=np.float32, sep=' '))
             word_index_dict[temp[0]]=i
             i=i+1
    embeddings.append(np.array([0] * 50,dtype=np.float32))
    word_index_dict['UNK']=len(embeddings)-1
    return np.array(embeddings), word_index_dict


def define_graph(glove_embeddings_arr):
    """
    Define the tensorflow graph that forms your model. You must use at least
    one recurrent unit. The input placeholder should be of size [batch_size,
    40] as we are restricting each review to it's first 40 words. The
    following naming convention must be used:
        Input placeholder: name="input_data"
        labels placeholder: name="labels"
        accuracy tensor: name="accuracy"
        loss tensor: name="loss"

    RETURN: input placeholder, labels placeholder, optimizer, accuracy and loss
    tensors"""

    #data = glove_embeddings_arr
    lstmUnits = 64
    tf.reset_default_graph()
    labels = tf.placeholder(tf.float32, [batch_size, numClasses],name="labels")
    input_data = tf.placeholder(tf.int32, [batch_size, maxColumn], name="input_data")
    datax = tf.Variable(tf.zeros([batch_size, maxColumn,50]),dtype=tf.float32)
    datax = tf.nn.embedding_lookup(glove_embeddings_arr,input_data)
    dropout_keep_prob = tf.placeholder_with_default(1.0, shape=())
    with tf.device('/cpu:0'):
        lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)

        lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=dropout_keep_prob)
        value, state = tf.nn.dynamic_rnn(lstmCell, datax, dtype=tf.float32)
        weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
        bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
        value = tf.transpose(value, [1, 0, 2])
        last = tf.gather(value, int(value.get_shape()[0]) - 1)
        prediction = (tf.matmul(last, weight) + bias)
        correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
        accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32),name="accuracy")
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
        #optimizer = tf.train.GradientDescentOptimizer(0.25).minimize(loss)
        optimizer = tf.train.AdamOptimizer().minimize(loss)
    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss
