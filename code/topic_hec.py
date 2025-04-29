import os.path as path
import string   
import nltk
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
from pprint import pprint
import pandas as pd  # For data handling
import spacy  # For preprocessing
from sklearn.feature_extraction.text import CountVectorizer
#!pip install umap-learn
import umap
import multiprocessing
from gensim.models import Word2Vec
from pprint import pprint
import hdbscan
from hdbscan.flat import (HDBSCAN_flat,
                          approximate_predict_flat,
                          membership_vector_flat,
                          all_points_membership_vectors_flat)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import gensim
import utility
import matplotlib
import pickle
from sklearn.manifold import TSNE
from numpy import var 
from matplotlib.text import TextPath
import matplotlib.patches as mpatches
import metrics

def dic_hastag_frequency(texts):
    """
    :return dictionary of the hashtag
    """
    freq = {}
    for val in texts:
        for item in val:
            if item[0]=='#':
                if (item in freq):
                    freq[item] += 1
                else:
                    freq[item] = 1
    return freq


def return_list_frequency(texts):
    """
    return list of the hashtag
    """
    min_freq=10#stessa frequenza utilizzata nel preprocessing
    l=[]
    for key, value in dic_hastag_frequency(texts).items():
        if value>=min_freq:
            l.append(key)
    return set(l)
    
   
def topic_HEC(name_w2v_model,text,min_size,top_words,filename_result):
    """
    name_w2v_model= name of word2Vec model
    text= list of sentence
    min_size= min size of cluster
    top_words= top word to consider in evaluation
    filename_result= name of file for save result
    return list oh topic and save in pdf format the clustering
    """
    
    w2v_model = Word2Vec.load(name_w2v_model)
    frequency=return_list_frequency(text)
    l_e=[]#lista degli embeddings
    l_hashtag=[]
    for hashtag in frequency:
        if hashtag in w2v_model.wv:
            em=w2v_model.wv[hashtag]
            l_e.append(em)
            l_hashtag.append(hashtag)    
    lx=np.array(l_e)
    embeddings=TSNE(init="pca",n_components=2, verbose=0,perplexity=50, angle=0.5,learning_rate=20, n_iter=3000, metric="cosine").fit_transform(lx)    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size,min_samples=1)
    clusterer.fit(embeddings)   
    
    color_palette = sns.color_palette('husl', (clusterer.labels_.max()+1))
    cluster_colors = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in clusterer.labels_]
    alphas = [ 0.2 if x < 0 else 1 for x in clusterer.labels_]
    index=[i for i in range(clusterer.labels_.max()+1) ]
    plt.figure(figsize=(14,14))
    colors=[]
    indexs=[index[clusterer.labels_[x]] if clusterer.labels_[x]!=-1 else -1 for x in range(len(clusterer.labels_))]
    recs=[]
    classes=[]
    for x,y,col,a,i in zip(*embeddings.T,cluster_colors,alphas,indexs): 
        plt.scatter(x,y,marker="o",s=70,linewidth=0.1,color=col,alpha=a)
        if col not in colors:
            recs.append(mpatches.Rectangle((0,0),1,1,fc=col))
            classes.append(i)
            plt.text(x,y, str(i),horizontalalignment='center', verticalalignment='center',fontsize=14)
            colors.append(col) 
    plt.legend(recs,classes,loc=4)           
    plt.savefig(filename_result, format="pdf", bbox_inches="tight", dpi=400) 
    d_frequency=dic_hastag_frequency(text)
    topics=extract_topic(clusterer,l_hashtag,top_words,d_frequency)
    result=[]
    for index in range(len(topics)):
        tuple_=(index,topics[index])
        result.append(tuple_)
    print("numero topic ritornati : "+str(len(result)))
    print(result)
   
     
    
def topic_HEC_flat(name_w2v_model,text,num_topics,min_size):
    """
    name_w2v_model= name of word2Vec model
    text= list of sentence
    num_topics= number of topic
    min_size= min_cluster_size
    return cluster and list of hashtag relative to clustering flat
    """
    w2v_model = Word2Vec.load(name_w2v_model)
    frequency=return_list_frequency(text)
    l_e=[]#lista degli embeddings
    l_hashtag=[]
    for hashtag in frequency:
        if hashtag in w2v_model.wv:
            em=w2v_model.wv[hashtag]
            l_e.append(em)
            l_hashtag.append(hashtag)    
    lx=np.array(l_e)
    embeddings=TSNE(init="pca",n_components=2, verbose=0,perplexity=50,angle=0.5,learning_rate=20,n_iter=3000,metric="cosine").fit_transform(lx)    
    new_clusterer = HDBSCAN_flat (embeddings,cluster_selection_method='leaf',n_clusters =num_topics,min_cluster_size=min_size,min_samples=1) 
    return new_clusterer,l_hashtag
  
    
def extract_topic(clusterer,list_hashtag,top_words,dic_frequency):
    topics=[]
    """
    clusterer= cluster of HDBSCAN
    list_hashtag= list of hashtag 
    top_words= top word to considerer
    dic_frequency= dictionary whit frequency of the hashtags
    return topic most hight frequency
    """
    for i in range(clusterer.labels_.max()+1):
        x=clusterer.labels_==i
        hashtag_i=np.array(list_hashtag)[x]
        dic={}
        for k in dic_frequency:
            if k in hashtag_i:
                dic[k]=dic_frequency[k]
        order_dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)
        ret_list=[ val[0] for val in order_dic[:top_words]]   
        topics.append(ret_list)
    return topics




def value_coherence(name_w2v_model,clean_text,start, stop, step,num_words):
    """
    name_w2v_model: name of the word2Vec model
    clean_text= list of list of token
    start,stop,step: range 
    num_words: top word to consider in evaluation
    return coherence score
    """
    tipo=['c_v','c_npmi']
    dic_frequency=dic_hastag_frequency(clean_text)
    dictionary,corpus=utility.prepare_corpus(clean_text)
    stop=stop+step #utile per considerare estremo valido stop    
    for val in tipo:
        final_score=[]     
        for i in range(0,10):
            coherences=[]
            for num_topics in range(start, stop, step):
                new_clusterer,l_hashtag= topic_HEC_flat(name_w2v_model,clean_text,num_topics,num_words)
                topics=extract_topic( new_clusterer,l_hashtag,num_words,dic_frequency)
                if(len(topics)!=num_topics):
                    print("warning size topics")
                coherence_per_topic = metrics.coherence(topics,clean_text, val, dictionary)
                coherence_model=sum(coherence_per_topic)/len(coherence_per_topic)
                coherences.append(coherence_model)
            final_score.append(sum(coherences)/len(coherences))
        print("topic coherence "+val+": "+str(sum(final_score)/len(final_score)))  


def value_topic_diversity(name_w2v_model,clean_text,start, stop, step,num_words):
    """
    name_w2v_model name of the model to evaluate word embedding
    clean_text: list of list of token
    start,stop,step: range 
    num_words: top word to consider in evaluation
    return diversity score
    """
    types=['PUW','JD','SIL_PW','SIL_CB']
    dictionary,corpus=utility.prepare_corpus(clean_text)
    dic_frequency=dic_hastag_frequency(clean_text)
    stop=stop+step #utile per considerare estremo valido stop
    wv = Word2Vec.load(name_w2v_model)
    for tipo in types:
        final_score=[]
        for i in range(0,10):
            td=[]
            for num_topics in range(start, stop, step):
                new_clusterer,l_hashtag= topic_HEC_flat(name_w2v_model,clean_text,num_topics,num_words)
                topics=extract_topic( new_clusterer,l_hashtag,num_words,dic_frequency)
                if(len(topics)!=num_topics):
                    print("warning size topics")
                if tipo=='PUW':
                    val=metrics.proportion_unique_words(topics, topk=num_words)
                    td.append(val)
                if tipo=='JD':
                    val=metrics.pairwise_jaccard_diversity(topics, topk=num_words)
                    td.append(val)
                if tipo=='SIL_PW':
                    val=metrics.sil_pw(topics,wv,topk=num_words)
                    td.append(val)
                if tipo=='SIL_CB':
                    val=metrics.sil_cb(topics,wv,topk=num_words)
                    td.append(val)
            final_score.append(sum(td)/len(td)) 
        print("topic diversity "+tipo+" : "+str(sum(final_score)/len(final_score)))  


















