from gensim.models import LdaModel
from gensim.models import Word2Vec
import metrics
import utility

 
    
def clear_output(topics):
    """
    topics= topics in shape realized of the model.show_topics
    return list of topic
    """
    new_topics=[]
    for i in range (0,len(topics)):
        clear_topic=[]
        for topic in topics[i][1]:
            clear_topic.append(topic[0])
        new_topics.append(clear_topic)
    return new_topics

def coherence_lda(clean_text,start, stop, step,num_words):
    """
    clean_text= list of list of token
    start,stop,step: range 
    num_words: top word to consider in evaluation
    return coherence score
    """
    tipo=['c_v','c_npmi']
    dictionary,corpus=utility.prepare_corpus(clean_text)
    stop=stop+step #utile per considerare estremo valido stop
    for val in tipo:
        final_score=[]
        for i in range(0,10):
            coherences=[]
            for num_topics in range(start, stop, step):
                model = LdaModel(corpus, num_topics, id2word = dictionary)  # train model
                topics=clear_output(model.show_topics(num_topics=num_topics, num_words=num_words, log=False, formatted=False))
                if(len(topics)!=num_topics):
                    print("warning size topics")
                coherence_per_topic = metrics.coherence(topics,clean_text, val, dictionary)
                coherence_model=sum(coherence_per_topic)/len(coherence_per_topic)
                coherences.append(coherence_model)                
            final_score.append(sum(coherences)/len(coherences))
        print("topic coherence "+val+": "+str(sum(final_score)/len(final_score)))  



def topic_diversity_lda(name_w2v_model,clean_text,start, stop, step,num_words):
    """
    name_w2v_model name of the model to evaluate word embedding
    clean_text= list of list of token
    start,stop,step: range 
    num_words: top word to consider in evaluation
    return diversity score
    """
    types=['PUW','JD','SIL_PW','SIL_CB']
    dictionary,corpus=utility.prepare_corpus(clean_text)
    stop=stop+step #utile per considerare estremo valido stop
    wv = Word2Vec.load(name_w2v_model)
    for tipo in types:
        final_score=[]
        for i in range(0,10):
            td=[]
            for num_topics in range(start, stop, step):
                model = LdaModel(corpus, num_topics, id2word = dictionary)  # train model
                topics=clear_output(model.show_topics(num_topics=num_topics, num_words=num_words, log=False, formatted=False))
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




