# coding: utf-8
# Author : Jitendra Upadhyay
# Version 1.0
# Date : 06/006/21

# In[1]:
# Importing the necessary modules

import numpy as np
import pandas as pd
import nltk
import string
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from nltk import word_tokenize
from datetime import timedelta
from scipy.stats.stats import pearsonr
from datetime import datetime
from __future__ import division, unicode_literals

# Creating pandas Data Frame from the input file
create_data_frame=pd.read_csv("2013-07-10.csv",sep=',')
create_data_frame.count()

#help(stopwords)
# Delecting the event where EVENT_TYPE is Delete 
create_data_frame=create_data_frame[create_data_frame.EVENT_TYPE!='DELETE']
create_data_frame=create_data_frame[create_data_frame.PRODUCTS!='TEST']
# Unique rows for UNIQUE_STORY_INDEX 
create_data_frame=create_data_frame.drop_duplicates(['UNIQUE_STORY_INDEX'], keep='last')
create_data_frame=create_data_frame.reset_index(drop=True)


# news per language were released?
Counts = create_data_frame.groupby('LANGUAGE').count()['HEADLINE_ALERT_TEXT'].tolist()
Counts = [(float(x)/float(sum(Counts)))*100 for x in Counts] # counts in percentage
Language_Labels = create_data_frame.groupby('LANGUAGE')['LANGUAGE'].first().tolist()
Dict = dict(zip(Language_Labels, Counts))

get_ipython().run_line_magic('matplotlib', 'inline')

plt.rcParams['figure.figsize'] = 16, 12

plt.bar(range(len(Dict)), Dict.values(), align='center')
plt.xticks(range(len(Dict)), Dict.keys())
plt.title('Distributions as per language',fontsize=24)
plt.xlabel('Language',fontsize=20)
plt.ylabel('Number of rows (in %)',fontsize=20)
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
plt.show()

create_data_frame=create_data_frame[create_data_frame.LANGUAGE=='EN']
create_data_frame=create_data_frame.reset_index(drop=True)

print ("DATE:                ",create_data_frame['DATE'].count() )
print ("TIME:                ",create_data_frame['TIME'].count())
print ("HEADLINE_ALERT_TEXT: ",create_data_frame['HEADLINE_ALERT_TEXT'].count())


create_data_frame['DATE'].unique()

plt.rcParams['figure.figsize'] = 10, 8
perDay = create_data_frame.groupby('DATE').count()['HEADLINE_ALERT_TEXT'].tolist()

plt.plot(range(len(perDay)), perDay,c='red')
plt.scatter(range(len(perDay)), perDay,c='red')
plt.title('Released of news',fontsize=18)
plt.xlabel('Days (month: June 2017)',fontsize=16)
plt.ylabel('# of news',fontsize=16)
plt.show()        




no_of_days=[]
for i in range(len(create_data_frame['DATE'].unique().tolist())):
    no_of_days.append(datetime.strptime(create_data_frame['DATE'].unique().tolist()[i], '%d/%m/%Y').day)   

"""
Creating the date groups
"""

group1=[] 
for j in range(len(no_of_days)):
    listDayI=[]
    for i in range( (create_data_frame['HEADLINE_ALERT_TEXT']).count()):
        if datetime.strptime(create_data_frame['DATE'][i],'%d/%m/%Y').day == no_of_days[j]:
            listDayI.append(str(create_data_frame['HEADLINE_ALERT_TEXT'][i]))
    temp = ' '.join(listDayI)
    group1.append(temp)
print (" news from one specific week [1-4] of the month from group-1")
print ("size of group2: ", len(group1) )



group2=[] 
week1 = week2 = week3 = week4 = []
for i in range( (create_data_frame['HEADLINE_ALERT_TEXT']).count()):
        if datetime.strptime(create_data_frame['DATE'][i],'%d/%m/%Y').day >= 3 and datetime.strptime(create_data_frame['DATE'][i],'%d/%m/%Y').day >= 7:
            week1.append(str(create_data_frame['HEADLINE_ALERT_TEXT'][i]))

        elif datetime.strptime(create_data_frame['DATE'][i],'%d/%m/%Y').day >= 10 and datetime.strptime(create_data_frame['DATE'][i],'%d/%m/%Y').day >= 15:
            week2.append(str(create_data_frame['HEADLINE_ALERT_TEXT'][i]))
        elif datetime.strptime(create_data_frame['DATE'][i],'%d/%m/%Y').day >= 18 and datetime.strptime(create_data_frame['DATE'][i],'%d/%m/%Y').day >= 22:
            week3.append(str(create_data_frame['HEADLINE_ALERT_TEXT'][i]))
        elif datetime.strptime(create_data_frame['DATE'][i],'%d/%m/%Y').day >= 25 and datetime.strptime(create_data_frame['DATE'][i],'%d/%m/%Y').day >= 29:
            week4.append(str(create_data_frame['HEADLINE_ALERT_TEXT'][i]))
         
temp1 = ' '.join(week1)
temp2 = ' '.join(week2)
temp3 = ' '.join(week3)
temp4 = ' '.join(week4)
group2 = [temp1, temp2, temp3, temp4]
print (" news from one specific week [1-4] of the month from group-2")
print ("size of group2: ", len(group2) )


group3=[]
for j in range(7):
    list1=[]
    for i in range( (create_data_frame['HEADLINE_ALERT_TEXT']).count()):
         if datetime.strptime(create_data_frame['DATE'][i],'%d/%m/%Y').weekday() == j:
                list1.append(str(create_data_frame['HEADLINE_ALERT_TEXT'][i]))
    temp = ' '.join(list1)
    group3.append(temp)
print (" news from one specific week [1-4] of the month from group-3")
print ("size of group2: ", len(group3) )
print ("no of unique time stamps: ", len(create_data_frame['TIME'].unique()))


from scipy.stats import kde
from dateutil.parser import parse
times = create_data_frame['TIME'].tolist()
hrs = []
for i in range(len(times)):
    hrs.append(parse(times[i]).hour)

Non_Number=100
Var1=np.asarray(hrs)
distribution_density_1 = kde.gaussian_kde(Var1)
cross_grid_1 = np.linspace(Var1.min(), Var1.max(), Non_Number)

plt.title("Probability distribution of news release during the day",fontsize=14)
plt.xlabel("Time (in hrs)",fontsize=16)
plt.ylabel("Prob. Density Function",fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.plot(cross_grid_1, distribution_density_1(cross_grid_1), 'r-',linewidth=2.0)
plt.plot(Var1, np.zeros(Var1.shape), 'r+', ms=12) #rug plot
plt.show()

group4=[]
for j in range(24):
    list1=[]
    for i in range( (create_data_frame['HEADLINE_ALERT_TEXT']).count()):
         if parse(create_data_frame['TIME'][i]).hour == j:
                list1.append(str(create_data_frame['HEADLINE_ALERT_TEXT'][i]))
    temp = ' '.join(list1)
    group4.append(temp)
print ("An element of group4 contains the words from all the news released at one specific hour of the day [1-24]")
print ("size of group4: ", len(group4))     




import nltk






import re
import unicodedata
import math
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')
nltk.download('wordnet')


newStopWords = ['alert','asia','europe','thompson','update','news','results','service','buzz','page','june','front','see','reuters','top','say']
add_stop_words         = stopwords.words('english') + newStopWords

def define_string(s_data):
    s_data = re.sub("[^a-zA-Z]"," ", s_data)
    wordList = tokenizer.tokenize(s_data.lower())
    fxd = [w for w in wordList if not w in add_stop_words]
    fxd = [wordnet_lemmatizer.lemmatize(w) for w in fxd]
    fxd = [w for w in fxd if len(w)>1]
    fxd = [w.encode('UTF8') for w in fxd]
    final = ' '.join(fxd) 
    return final



def convert_list(inputList):
    cloudList=[]
    for i in range(len(inputList)):
        #cloudList.append(define_string(inputList[i]))
        cloudList.append(inputList[i])
    return cloudList    

dataPoint1 = convert_list(group1)
dataPoint2 = convert_list(group2)
dataPoint3 = convert_list(group3)
dataPoint4 = convert_list(group4)
print (len(dataPoint1))

from wordcloud import WordCloud
wordcloud = WordCloud(max_font_size=40, relative_scaling=.99).generate(dataPoint1[1])
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

def saveWordCloud(dataCloud,rootN):
    for i in range(len(dataCloud)):
        wordcloud = WordCloud(max_font_size=40, relative_scaling=.99).generate(dataCloud[1])
        plt.figure()
        plt.imshow(wordcloud)
        plt.axis("off")
        figName=rootN+str(i+1)+'.png'
        plt.savefig(figName)
        plt.show()
saveWordCloud(dataPoint1,'day')    


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

listX=[]
for i in range(len(dataPoint3)):
    listX.append(str(dataPoint3[i]))


C1 = CountVectorizer()
C2 = C1.fit_transform(listX)
print (C2.shape)

T1 = TfidfTransformer()
T2 = T1.fit_transform(C2)
print ("Dimensions of the data: ", T2.shape)
print ("Trend for each day of the week [Mon-Sun]...")

Number=10
array=[]
for i in range(len(listX)):
    var2 = T2.todense()[i,:].argsort()
    words=[]
    words = [C1.get_feature_names()[var2[0,var2.shape[1]-j]] for j in range(1,Number+1)]
    array.append(words)

array=tuple(zip(*array))

table = [['Monday', 'Tuesday', 'Wednesday','Thusrday','Friday','Saturday','Sunday'], 
         array[1], 
         array[2],
         array[3],
         array[4],
         array[5],
         array[6],
         array[7],
         array[8],
         array[9]]

headers = table.pop(0)
dd = pd.DataFrame(table, columns=headers)
dd


# In[4]:


group1


# In[5]:


group2


# In[7]:


len(group3)


# In[8]:


group4


# In[12]:


import pandas as pd
data = pd.read_csv('data_file.csv', error_bad_lines=False)
data=data[data.LANGUAGE=='EN']
data=data.reset_index(drop=True)
data_text = data[['HEADLINE_ALERT_TEXT']]
data_text['index'] = data_text.index
documents = data_text


# In[13]:


documents


# In[14]:


print(len(documents))
print(documents[:5])


# In[16]:


import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')


# In[19]:


def lem_stem(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lem_stem(token))
    return result


# In[21]:


from nltk.stem.porter import *


# In[23]:


stemmer = PorterStemmer()


# In[25]:


sample_doc = documents[documents['index'] == 3000].values[0][0]
print('original document: ')
words = []
for word in sample_doc.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(sample_doc))


# In[32]:


documents['HEADLINE_ALERT_TEXT'].fillna('').astype(str).map(preprocess)
documents = documents.dropna(subset=['HEADLINE_ALERT_TEXT'])


# In[33]:


processed_docs = documents['HEADLINE_ALERT_TEXT'].map(preprocess)
processed_docs[:10]


# In[34]:


dictionary = gensim.corpora.Dictionary(processed_docs)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break


# In[35]:


dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)


# In[37]:


bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
bow_corpus[3000]


# In[40]:


bow_doc_3000 = bow_corpus[3000]
for i in range(len(bow_doc_3000)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_3000[i][0], 
                                               dictionary[bow_doc_3000[i][0]], 
bow_doc_3000[i][1]))


# In[41]:


from gensim import corpora, models
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
from pprint import pprint
for doc in corpus_tfidf:
    pprint(doc)
    break


# In[42]:


lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id_to_word=dictionary, passes=2, workers=2)


# In[43]:


for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))


# In[44]:


lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id_to_word=dictionary, passes=2, workers=4)
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))


# In[45]:


processed_docs[3000]


# In[46]:


for index, score in sorted(lda_model[bow_corpus[3000]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))


# In[47]:


for index, score in sorted(lda_model_tfidf[bow_corpus[3000]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))


# In[48]:


unseen_document = 'S.KOREA MAY AVG EXPORTS PER WORKING DAY $2.10 BLN VS REVISED $1.93 BLN IN APRIL-REUTERS CALCULATIONS'
bow_vector = dictionary.doc2bow(preprocess(unseen_document))
for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))


# In[50]:


# Run in python console
import nltk; nltk.download('stopwords')


# In[53]:


import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization_function
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


# In[54]:


# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])


# In[60]:


import pandas as pd
data = pd.read_csv('data_file.csv', error_bad_lines=False)
data=data[data.LANGUAGE=='EN']
data=data.reset_index(drop=True)
data_text = data[['HEADLINE_ALERT_TEXT']]
data_text['index'] = data_text.index
documents = data_text


# In[62]:


documents


# In[63]:


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(documents['HEADLINE_ALERT_TEXT']))

print(data_words[:1])


# In[68]:


len(data_words)


# In[70]:


# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data_words[900]]])


# In[71]:


# Define functions for stopwords, bigrams, trigrams and lemmatization_function
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def gen_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def gen_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization_function(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):

    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# In[117]:


import spacy


# In[116]:


nlp=spacy.load('en_core_web_sm')


# In[119]:


# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = gen_bigrams(data_words_nostops)


nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

data_lemmatized = lemmatization_function(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:1])


# In[120]:


# Create Dictionary
id_to_word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id_to_word.doc2bow(text) for text in texts]

# View
print(corpus[:1])


# In[121]:


[[(id_to_word[id], freq) for id, freq in cp] for cp in corpus[:1]]


# In[122]:


# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id_to_word=id_to_word,
                                           num_topics=20, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)


# In[123]:


# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


# In[124]:


# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_lda_model = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id_to_word, coherence='c_v')
coherence_lda = coherence_lda_model.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# In[125]:


# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id_to_word)
vis


# In[ ]:


C:\Users\Jitendra\mallet-2.0.8 (1)\mallet\mallet-2.0.8\bin


# In[148]:


mallet_path = 'C:\\Users\\Jitendra\\mallet' # update this path


# In[149]:


ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=20, id_to_word=id_to_word)


# In[143]:


mallet_path


# In[131]:


import os
os.getcwd()


# In[145]:


os.chdir("C:\\Users\\Jitendra")

# Case 2:

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

doc_a = "ذهب محمد الى المدرسه على دراجته. هذا اول يوم له في المدرسة"
doc_a = doc_a.decode('utf-8')
sw = stopwords.words('arabic')
tokens = nltk.word_tokenize(doc_a)
stopped_tokens = [i for i in tokens if not i in sw]
for item in stopped_tokens:
    print(item)   
	
	['ذهب', 'محمد', 'الى', 'المدرسه', 'على', 'دراجته', '.', 'هذا', 'اول', 'يوم', 'له', 'في', 'المدرسة]
	
	# Cant interpret arabic just for experiment purpsose