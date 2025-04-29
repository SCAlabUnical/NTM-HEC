from top2vec import Top2Vec
from gensim.models import Word2Vec
import utility
import metrics
    
def extract_topic(topics,num_words):
    tweets_clear=[]
    for topic in topics:
        tweets_clear.append(topic[:num_words])        
    return tweets_clear 
 
def coherence_top2vec(name_top2vec_model,clean_text,start, stop, step,num_words):
    """
    name_top2vec_model: name of the top2Vec model
    clean_text= list of list of token
    start,stop,step: range 
    num_words: top word to consider in evaluation
    return coherence score
    """
    tipo=['c_v','c_npmi']
    dictionary,corpus=utility.prepare_corpus(clean_text)
    stop=stop+step #utile per considerare estremo valido stop
    for val in tipo:
        coherences=[]
        for num_topics in range(start, stop, step):
            model = Top2Vec.load(name_top2vec_model)
            model.hierarchical_topic_reduction(num_topics=num_topics)
            topics=extract_topic( model.topic_words_reduced,num_words)
            if(len(topics)!=num_topics):
                print("warning size topics")
            coherence_per_topic = metrics.coherence(topics,clean_text, val, dictionary)
            coherence_model=sum(coherence_per_topic)/len(coherence_per_topic)
            coherences.append(coherence_model)
        print("topic coherence "+val+": "+str(sum(coherences)/len(coherences)))   

    

def topic_diversity_top2vec(name_top2vec_model,name_w2v_model,start, stop, step,num_words):
    """
    name_top2vec_model: name of the top2Vec model
    name_w2v_model name of the model to evaluate word embedding
    start,stop,step: range 
    num_words: top word to consider in evaluation
    return diversity score
    """
    types=['PUW','JD','SIL_PW','SIL_CB']
    stop=stop+step #utile per considerare estremo valido stop
    wv = Word2Vec.load(name_w2v_model)
    for tipo in types:
        td=[]
        for num_topics in range(start, stop, step):
            model = Top2Vec.load(name_top2vec_model)
            model.hierarchical_topic_reduction(num_topics=num_topics)
            topics=extract_topic( model.topic_words_reduced,num_words)
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
        print("topic diversity "+tipo+" : "+str(sum(td)/len(td))) 
 
      
                           







