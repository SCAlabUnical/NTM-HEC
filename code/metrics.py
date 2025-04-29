from gensim.models.coherencemodel import CoherenceModel
import numpy as np
from scipy.spatial import distance
from itertools import combinations
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from sklearn.metrics.pairwise import cosine_similarity



def coherence(topics,clean_text,value,dictionary):
    """
    compute the coherence score
    Parameters
    ----------
    topics: a list of lists of words
    clean_text: list of list of tokens
    value: type of coherence to be evaluated
    dictionary: corpus dictionary
    """
    cm = CoherenceModel(topics=topics,texts=clean_text, coherence=value, dictionary=dictionary)
    return cm.get_coherence_per_topic()

def proportion_unique_words(topics, topk=10):
    """
    compute the proportion of unique words
    Parameters
    ----------
    topics: a list of lists of words
    topk: top k words on which the topic diversity will be computed
    """
    if topk > len(topics[0]):
        raise Exception('Words in topics are less than '+str(topk))
    else:
        unique_words = set()
        for topic in topics:
            unique_words = unique_words.union(set(topic[:topk]))
        puw = len(unique_words) / (topk * len(topics))
        return puw

def pairwise_jaccard_diversity(topics, topk=10):
    '''
    compute the average pairwise jaccard distance between the topics 
  
    Parameters
    ----------
    topics: a list of lists of words
    topk: top k words on which the topic diversity will be computed
    '''
    dist = 0
    count = 0
    for list1, list2 in combinations(topics, 2):
        js = 1 - len(set(list1).intersection(set(list2)))/len(set(list1).union(set(list2)))
        dist = dist + js
        count = count + 1
    return dist/count


def ste_lem_topics(word,w2v_model):
    '''
    suitable word at vocabolary of w2v_model
  
    Parameters
    ----------
    word: word to preprocessing
    w2v_model: word2Vec model 
    '''
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    if word in w2v_model.wv:
        return word
    lemm_word=lemmatizer.lemmatize(word)# trasforma feet in foot
    stem_word=stemmer.stem(lemm_word) #toglie ing,y,etc dalle parole
    if stem_word in w2v_model.wv:
        return stem_word
    if '#'+str(stem_word) in w2v_model.wv:
        return '#'+str(stem_word)
    return None


        
def sil_pw(topics, word_embedding_model,topk):
    """
    topics: list of topics
    word_embedding_model : model of embedding
    param topk: how many most likely words to consider in the evaluation
    return: silhouette score computed on the word embeddings of word_embedding_model, based on parwise model
    """
    if topk > len(topics[0]):
        raise Exception('Words in topics are less than topk')
    else:
        dist_intra = []
        for topic in topics:
            dist = 0.0
            count_intra=0
            for i in range(len(topic)):
                for j in range(len(topic)):
                    if topic[i] == topic[j] and i!=j:
                        dist=dist+0
                        count_intra=count_intra+1
                    else:
                        w1=ste_lem_topics(topic[i],word_embedding_model)
                        w2=ste_lem_topics(topic[j],word_embedding_model)
                        if w1 is not None and w2 is not None:
                            dist = dist + (1-cosine_similarity([word_embedding_model.wv[w1]],[word_embedding_model.wv[w2]]))
                            count_intra=count_intra+1
            dist_intra.append(dist/count_intra)
        inter_list=[]
        for (index_1,topic_i), (index_2,topic_j) in combinations(enumerate(topics), 2):
            inter_dist = 0.0
            count_inter=0
            for i in range(len(topic_i)):
                for j in range(len(topic_j)):
                    if topic_i[i] == topic_j[j]:
                        inter_dist = inter_dist + 0
                        count_inter=count_inter+1
                    else:
                        w1=ste_lem_topics(topic_i[i],word_embedding_model)
                        w2=ste_lem_topics(topic_j[j],word_embedding_model)
                        if w1 is not None and w2 is not None:
                            inter_dist = inter_dist + (1-cosine_similarity([word_embedding_model.wv[w1]],[word_embedding_model.wv[w2]]))
                            count_inter=count_inter+1
            inter_list.append(inter_dist/count_inter)
        avg_intra=sum(dist_intra)/len(dist_intra)    
        avg_inter=sum(inter_list)/len(inter_list)    
        res=(avg_inter-avg_intra)/max(avg_inter,avg_intra)             
        return res

def sil_cb(topics, word_embedding_model,topk):
    """
    topics: list of topics
    word_embedding_model : model of embedding
    param topk: how many most likely words to consider in the evaluation
    return: silhouette score computed on the word embeddings of word_embedding_model, based on centroid model
    """
    if topk > len(topics[0]):
        raise Exception('Words in topics are less than topk')
    else:
        dist_intra = []
        for topic in topics:
            dist = 0.0
            count_intra=0
            count_centr=0
            centroid = np.zeros(word_embedding_model.vector_size)
            for i in range(len(topic)):
                w1=ste_lem_topics(topic[i],word_embedding_model)
                if w1 is not None:
                    centroid = centroid + word_embedding_model.wv[w1]
                    count_centr=count_centr+1
            point_centroid=centroid/count_centr
            
            for j in range(len(topic)):
                w2=ste_lem_topics(topic[j],word_embedding_model)
                if w2 is not None:
                    dist = dist + (1-cosine_similarity([point_centroid],[word_embedding_model.wv[w2]]))
                    count_intra=count_intra+1
            dist_intra.append(dist/count_intra)
        inter_list=[]
        for list1, list2 in combinations(topics, 2):
            centroid1 = np.zeros(word_embedding_model.vector_size)
            centroid2 = np.zeros(word_embedding_model.vector_size)
            count1=0
            count2=0
            for word1 in list1:
                word1=ste_lem_topics(word1,word_embedding_model)
                if word1 is not None: 
                    count1=count1+1
                    centroid1 = centroid1 + word_embedding_model.wv[word1]
            for word2 in list2:
                word2=ste_lem_topics(word2,word_embedding_model)
                if word2 is not None:
                    count2=count2+1
                    centroid2 = centroid2 + word_embedding_model.wv[word2]
            centroid1 = centroid1 / count1
            centroid2 = centroid2 / count2
            inter_list.append(distance.cosine(centroid1, centroid2))
        avg_intra=sum(dist_intra)/len(dist_intra)    
        avg_inter=sum(inter_list)/len(inter_list)    
        res=(avg_inter-avg_intra)/max(avg_inter,avg_intra)             
        return res



