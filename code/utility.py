import os.path
from gensim import corpora

def prepare_corpus(doc_clean):
    dictionary = corpora.Dictionary(doc_clean)
    corpus = [dictionary.doc2bow(doc) for doc in doc_clean]
    return dictionary,corpus

def list_sentence(path,file_name):
    """
    return list of string, each document in one string
    Parameters
    ----------
    path: path of file_name
    file_name: name of the file 
    """
    documents_list = []
    with open( os.path.join(path, file_name) ,"r") as fin:
        for line in fin.readlines():
            text = line.strip()
            documents_list.append(text)
    return documents_list

def list_list_token(path,file_name):
    """
    return list of list of token
    Parameters
    ----------
    path: path of file_name
    file_name: name of the file 
    """ 
    documents_list = []
    with open( os.path.join(path, file_name) ,"r") as fin:
        for line in fin.readlines():
            word_list=[]
            text = line.strip()
            for word in text.split():
                word_list.append(word)
            documents_list.append(word_list)
    return documents_list



