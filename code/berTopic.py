from bertopic import BERTopic
from gensim.models import Word2Vec
import os
import metrics
import utility
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def clear_output(topics):
    new_topics=[]
    for i in range (0,len(topics)-1):
        clear_topic=[]
        for topic in topics[i]:
            clear_topic.append(topic[0])
        new_topics.append(clear_topic)
    return new_topics


def coherence_berTopic(name_berTopic_model,text,clean_text,start, stop, step,num_words):
    """
    name_berTopic_model: name of the berTopic model
    text= list of sentences
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
            model = BERTopic.load(name_berTopic_model)
            model.reduce_topics(text, nr_topics=num_topics+1)
            topics=clear_output(model.get_topics())
            if(len(topics)!=num_topics):
                print("warning size topics")
            coherence_per_topic = metrics.coherence(topics,clean_text, val, dictionary)
            coherence_model=sum(coherence_per_topic)/len(coherence_per_topic)
            coherences.append(coherence_model)
        print("topic coherence "+val+": "+str(sum(coherences)/len(coherences))) 





def topic_diversity_berTopic(name_berTopic_model,name_w2v_model,text,start, stop, step,num_words):
    """
    name_berTopic_model: name of the berTopic model
    name_w2v_model name of the model to evaluate word embedding
    text =list of sentences
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
            model = BERTopic.load(name_berTopic_model)
            model.reduce_topics(text, nr_topics=num_topics+1)#+1 outlier
            topics=clear_output(model.get_topics())
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
                      

  
    
    
    
    

