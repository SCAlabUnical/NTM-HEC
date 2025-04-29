import os.path as path
import nltk
import multiprocessing
from gensim.models import Word2Vec
from top2vec import Top2Vec
from nltk.corpus import stopwords
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
import preprocessing
import utility
import metrics
import topic_hec
import lsa_
import lda_
import top2Vec_
import berTopic



def run_lsa(path_data, name_w2v_model, file_name, min_num_topic, max_num_topic, step, top_words):
    print("*****LSA*****")
    clear_filename=preprocessing.clear_data(path_data,file_name) 
    clean_text=utility.list_list_token(path_data,clear_filename) 
    lsa_.coherence_lsa(clean_text,min_num_topic,max_num_topic, step,top_words)
    lsa_.topic_diversity_lsa(name_w2v_model,clean_text,min_num_topic, max_num_topic, step,top_words)
    

def run_lda(path_data, name_w2v_model, file_name, min_num_topic, max_num_topic, step, top_words):
    print("*****LDA*****")
    clear_filename=preprocessing.clear_data(path_data,file_name) 
    clean_text=utility.list_list_token(path_data,clear_filename)
    lda_.coherence_lda(clean_text,min_num_topic,max_num_topic, step,top_words)
    lda_.topic_diversity_lda(name_w2v_model,clean_text,min_num_topic, max_num_topic, step,top_words)
   
   
def run_top2Vec(path_data,path_model, name_w2v_model, file_name, min_num_topic, max_num_topic, step, top_words):
    print("*****top2Vec*****")
    clear_filename=preprocessing.shallow_clear_data(path_data,file_name) 
    clean_text=utility.list_list_token(path_data,clear_filename) 
    text=utility.list_sentence(path_data,clear_filename) 
    name_top2vec_model=path_model+"top2vec_"+file_name
    if not path.exists(name_top2vec_model):
        model=Top2Vec(documents=text,embedding_model='universal-sentence-encoder')
        model.save(name_top2vec_model) 
    top2Vec_.coherence_top2vec(name_top2vec_model,clean_text,min_num_topic,max_num_topic, step,top_words)
    top2Vec_.topic_diversity_top2vec(name_top2vec_model,name_w2v_model,min_num_topic,max_num_topic, step,top_words)


def run_berTopic(path_data,path_model, name_w2v_model, file_name, min_num_topic, max_num_topic, step, top_words):
    print("*****berTopic*****")
    clear_filename=preprocessing.shallow_clear_data(path_data,file_name) 
    clean_text=utility.list_list_token(path_data,clear_filename) 
    text=utility.list_sentence(path_data,clear_filename) 
    name_berTopic_model=path_model+"berTopic_"+file_name
    if not path.exists(name_berTopic_model):
        l_stopwords = stopwords.words('english')   
        l_stopwords.append('')   
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        vectorizer_model = CountVectorizer(stop_words=l_stopwords)
        repr_model = MaximalMarginalRelevance(diversity=0.2, top_n_words=top_words)
        model = BERTopic(min_topic_size=100,vectorizer_model=vectorizer_model,ctfidf_model=ctfidf_model, representation_model=repr_model, verbose=True, top_n_words=top_words)
        topics, probs = model.fit_transform(text)
        model.save(name_berTopic_model)
    berTopic.coherence_berTopic(name_berTopic_model,text,clean_text,min_num_topic,max_num_topic,step,top_words)
    berTopic.topic_diversity_berTopic(name_berTopic_model,name_w2v_model,text,min_num_topic,max_num_topic, step,top_words)
  
    
def run_topic_HEC(path_data, name_w2v_model, file_name,min_cluster_size,show_words, min_num_topic, max_num_topic, step, top_words):
    print("*****topic-HEC*****")
    clear_filename=preprocessing.clear_data(path_data,file_name)
    clean_text=utility.list_list_token(path_data,clear_filename)
    topic_hec.value_coherence(name_w2v_model,clean_text,min_num_topic,max_num_topic,step,top_words)
    topic_hec.value_topic_diversity(name_w2v_model,clean_text,min_num_topic, max_num_topic, step,top_words)
    filename_result="topic_HEC_"+file_name+".pdf"
    topic_hec.topic_HEC(name_w2v_model,clean_text,min_cluster_size,show_words,filename_result)

if __name__ == "__main__":  
    # define hyper-paramaters
    path_data = "../data/"
    path_model = "../models/"
    file_name="tweets"
    min_num_topic,max_num_topic,step=5,10,5 #range defined for evaluate coherence and diversity
    top_words=10 #top word analized for evaluate coherence and diversity
    min_cluster_size=10 # min size of clustering for topic-HEC
    show_words=10 # top words returned on console for topic-HEC
    w2v_min_count=5
    w2v_size=150

    # preprocessing for word2vec model
    clear_filename=preprocessing.clear_data(path_data,file_name) 
    clean_text=utility.list_list_token(path_data,clear_filename) 
    """
    word2vec defined for evaluate topic diversity in all types
    """
    name_w2v_model=path_model+"word2vec_"+file_name
    if not path.exists(name_w2v_model):
        cores = multiprocessing.cpu_count() 
        w2v_model = Word2Vec(min_count=w2v_min_count, window=5, vector_size=w2v_size, sample=6e-5, alpha=0.01, min_alpha=0.001, negative=20, workers=cores-1)
        w2v_model.build_vocab(clean_text, progress_per=10000)
        w2v_model.train(clean_text, total_examples=w2v_model.corpus_count, epochs=50, report_delay=1)
        w2v_model.save(name_w2v_model)
   
    run_lsa(path_data, name_w2v_model, file_name, min_num_topic, max_num_topic, step, top_words)
    run_lda(path_data, name_w2v_model, file_name, min_num_topic, max_num_topic, step, top_words)
    run_top2Vec(path_data,path_model, name_w2v_model, file_name, min_num_topic, max_num_topic, step, top_words)
    run_berTopic(path_data,path_model, name_w2v_model, file_name, min_num_topic, max_num_topic, step, top_words)
    run_topic_HEC(path_data,name_w2v_model,file_name,min_cluster_size,show_words,min_num_topic, max_num_topic,step, top_words)











