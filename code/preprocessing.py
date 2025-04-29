import os.path
import re
import string   
from string import digits                                              
from nltk.corpus import stopwords  
from nltk.stem import PorterStemmer 
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer 
from collections import defaultdict
import utility


def clear_data(file_path,file_name_input):
    """
    preprocessing file_name_input, writing output on other file
    return name of file whit data preprocessing
    """
    if os.path.isfile(os.path.join(file_path,"clear_"+file_name_input)):
        return os.path.join(file_path,"clear_"+file_name_input)
    documents_list = []
    # instantiate the tokenizer class
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    stopwords_english = stopwords.words('english') 
    # Instantiate stemming class
    stemmer = PorterStemmer() 
    lemmatizer = WordNetLemmatizer()
    # remove words that appear only once
    frequency = defaultdict(int)
    tweets = []
    with open(os.path.join(file_path,"clear_"+file_name_input), 'a') as f:
        with open( os.path.join(file_path, file_name_input) ,"r") as fin:
            for line in fin.readlines():
                tweet = line.strip()
                #rímuove emoji e caratteri speciali
                tweet_clean=tweet.encode('ascii', 'ignore').decode('ascii')
                # toglie solo "RT"
                tweet_clean = re.sub(r'RT', '',tweet_clean)
                # rimuove gli hyperlinks
                tweet_clean= re.sub(r'https?:\/\/.*[\r\n]*', '', tweet_clean)
                # tokenize the tweets
                tweet_tokens = tokenizer.tokenize(tweet_clean)
                tweet_clean = []
                b_list=["..","...",". ...",". ..",". .", ] 
                for word in tweet_tokens:
                    word=word.strip()
                    if (word not in stopwords_english and word not in string.punctuation and word not in b_list and not word.isdigit()):
                        lemm_word=lemmatizer.lemmatize(word)# trasforma feet in foot
                        stem_word=stemmer.stem(lemm_word) #toglie ing,y,etc dalle parole
                        frequency[stem_word] += 1
                        tweet_clean.append(stem_word)               
                tweets.append(tweet_clean)  
            tweets_nf = [[token for token in doc if frequency[token] > 10] for doc in tweets if len(doc)>0]  
            for sent in tweets_nf:
                f.write(' '.join(sent)+'\n')
    f.close()
    return os.path.join(file_path,"clear_"+file_name_input)


def shallow_clear_data(path,file_name_input):
    """
    preprocessing file_name_input, writing output on other file
    return name of file whit data preprocessing
    ------
    used by sentence-level topic models
    """
    if os.path.isfile(os.path.join(path,"clear_2_"+file_name_input)):
        return os.path.join(path,"clear_2_"+file_name_input)
    tweets = []
    with open(os.path.join(path,"clear_2_"+file_name_input), 'a') as f:
        with open( os.path.join(path, file_name_input) ,"r") as fin:
            for line in fin.readlines():
                # Filter
                tweet = line.strip()
                #rímuove emoji e caratteri speciali
                tweet_clean=tweet.encode('ascii', 'ignore').decode('ascii')
                # toglie solo "RT"
                tweet_clean = re.sub(r'RT', '',tweet_clean)
                # cancella solo # dalle parole
                tweet_clean = re.sub(r'#', '', tweet_clean)
                # rimuove gli hyperlinks
                tweet_clean= re.sub(r'https?:\/\/.*[\r\n]*', '', tweet_clean)
                tweet_clean =" ".join(filter(lambda x:x[0]!="@", tweet_clean.split()))
                if(len(tweet_clean)):
                    f.write(tweet_clean+'\n') 
    f.close()
    return os.path.join(path,"clear_2_"+file_name_input)
    










