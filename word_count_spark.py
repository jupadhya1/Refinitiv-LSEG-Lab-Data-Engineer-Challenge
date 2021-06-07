# -*- coding: utf-8 -*-
"""word_count_spark.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14W0hbR4i1CKOvha8F8OmSCy-c3NPiYaT
"""

import pandas as pd
import numpy as np

"""### 1. Data Understanding"""

import os
import re

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext

"""#### 1.1. Data Cleaning/Preprocessing"""

!head -1 2013-07-09.csv

"""###### Note

- Data in CSV format with comma separator & include HEADER
- This can be load by spark sql (for production) and pandas (for experiement)
"""

!head -40 2013-07-09.csv | tail -23

"""###### Note

- Data contain multiple lines in one column
- Contain multiple double quote "" which not the escape character

###### Asumption

- not escapse double quote "" is meaningless which will not break the text content structure => can be remove
 - After remove meaningless double quote & standadize the text column which only be wraped inside a proper double quote {open} and {close}, for example: "ABCDD    \n \n ADBC"
 - Finally, just need to handle multiLine column

###### Data Flow to clean data

1. Simple Spark job to standadize the text column => store the data to immidiate File
 2. Handle multiple line file
"""

def clean_hdfs_files(sc, path):
    if sc is None:
        print('Spark Context is None')
        return
    
    URI = sc._gateway.jvm.java.net.URI
    Path = sc._gateway.jvm.org.apache.hadoop.fs.Path
    FileSystem = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
 
    fs = FileSystem.get(URI(path), sc._jsc.hadoopConfiguration())
    if fs.exists(Path(path)):
        return fs.delete(Path(path))

""" <b>#1. Filter job</b>"""

conf = SparkConf()\
    .setAppName("Cleaning Job")\
    .setMaster("local[*]")
sc = SparkContext(conf=conf)

raw = sc.textFile('2013-07-09.csv')

raw.count()

raw.take(1)

clean = raw.filter(lambda x: len(x)>0)\
    .map(lambda x: re.sub(pattern=r'""', string=x, repl=''))\
    .map(lambda x: re.sub(pattern=r'^",', string=x, repl='@@@",'))

output_path = 'clean_2013-07-09.csv'
clean_hdfs_files(sc, output_path)
clean.coalesce(1).saveAsTextFile(output_path)

sc.stop()

"""<b>#2. Load data job</b>"""

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("SimpleApp").getOrCreate()

raw = spark.read\
    .option("multiLine", "true")\
    .option("header", "true")\
    .option("inferSchema","true")\
    .option("multiLine","true")\
    .option("quoteMode","ALL")\
    .option("mode","PERMISSIVE")\
    .option("ignoreLeadingWhiteSpace","true")\
    .option("ignoreTrailingWhiteSpace","true")\
    .option("parserLib","UNIVOCITY")\
    .option("escape",'\n')\
    .option("wholeFile","True")\
    .csv('clean_2013-07-09.csv')

spark.stop()

raw.count()

raw.printSchema()

"""- Convert to Pandas DataFrame for Data Analysis"""

df = raw.toPandas()

"""#### 1.2. Data Analysis"""

# Commented out IPython magic to ensure Python compatibility.
import seaborn as sns
import matplotlib.pyplot as plt

# %matplotlib inline

"""- Check data variables"""

df.info(memory_usage=True)

"""###### Conclusion
- We are easy to see that there are three columns: HEADLINE_ALERT_TEXT, ACCUMULATED_STORY_TEXT, TAKE_TEXT which are contain text content
- ACCUMULATED_STORY_TEXT: has very bad quality (just only 376 out of 8181 are non-empty) -> will not consider this colum
- HEADLINE_ALERT_TEXT: seems to be title of article
- TAKE_TEXT: looks like the content body of article
- <b>Will use both of these two columns to extract the trending words</b>
"""

df.ATTRIBUTION.value_counts(dropna=False)

df.LANGUAGE.value_counts(dropna=False).plot(kind = 'barh')

df.PRODUCTS.value_counts().head()

""" ###### Asumption
 - English language is the most popular language here
 - It will make sense to just look into only English text here in order to extract the trending keywords (and skip for the other languages)
 - Also filter out 'TEST' products
"""

df_en = df[(df.LANGUAGE=='EN')&(df.PRODUCTS!='TEST')]
print(df_en.shape)

print("Empty percentage:")
df_en.isnull().sum() * 100.0/len(df_en)

df_en[['DATE', 'TIME', 'HEADLINE_ALERT_TEXT', 'TAKE_TEXT']].sample(10)

from pyspark.sql.types import ArrayType, DoubleType, StringType

raw.registerTempTable('temp')

sql='''
    select *, 
        (CASE
            WHEN LANGUAGE<>'EN' THEN ''
            WHEN PRODUCTS=='TEST' THEN ''
            WHEN TAKE_TEXT IS NULL THEN HEADLINE_ALERT_TEXT
            ELSE TAKE_TEXT
        END) as TEXT_CONTENT
    from temp
'''

# process text
def pre_process(text):
    # lowercase
    if text is None:
        return ''
    text=text.lower()
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    return text.strip()

pre_process_udf = f.udf(pre_process, StringType())

text_df = spark.sql(sqlQuery=sql)\
    .select('DATE', 'TIME', 'LANGUAGE', 'PRODUCTS', pre_process_udf('TEXT_CONTENT').alias('TEXT_CONTENT'))

text_df.show()

"""### 2. Using simple NLP technique to extract keywords"""

from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import CountVectorizer

from pyspark.ml.feature import HashingTF, IDF, Tokenizer

tokenizer = Tokenizer(inputCol="TEXT_CONTENT", outputCol="raw")
remover = StopWordsRemover(inputCol="raw", outputCol="words")
# hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
# idf = IDF(inputCol="rawFeatures", outputCol="features")

cv = CountVectorizer(inputCol="words", outputCol="features")

from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[tokenizer, remover, cv])
model = pipeline.fit(text_df)

results = model.transform(text_df)

results.printSchema()

def extract_values_from_vector(vector):
    return vector.values.tolist()

extract_values_from_vector_udf = f.udf(extract_values_from_vector, ArrayType(DoubleType()))

# And use that UDF to get your values
word_count = results.select('DATE', 'TIME', 'LANGUAGE', extract_values_from_vector_udf('features').alias('count'), 'words')

word_count.show()

"""- Now we can zip two columns (count and words) and sort by count to have better look about word distribution"""

import operator

import operator
x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
sorted_x = sorted(x.items(), key=operator.itemgetter(1))

sorted_x[:2]

import json

def extract_top(count, words, topk=5):
    sorted_x = sorted(zip(words, count), key=operator.itemgetter(1), reverse=True)
    if topk > 0:
        return json.dumps(dict(sorted_x[:topk]))
    else:
        return json.dumps(dict(sorted_x))

extract_top_udf = f.udf(extract_top, StringType())

word_count.select(extract_top_udf('count', 'words')).take(20)

"""#### Conclusion:

- So here, we can easy to see the frequent word from time to time. However we might need to see the word frequency by time window, for example by hour, day, week ...
- We will need to take advantage of DATE, TIME data. 
- One drawback by using SPARK here is: SPARK is not really convenient to deal with timeseries data and espcially to aggregate data by timewindow. So that we will use Pandas to go a little deep dive (This will only usefor experiment & not production)

##### Convert Spark Dataframe to Pandas
"""

wc_df = word_count.select('DATE', 'TIME', extract_top_udf('count', 'words', f.lit(-1)).alias('trending')).toPandas()

wc_df.head()

wc_df.dtypes

wc_df['DATE_TIME'] = wc_df.DATE.dt.strftime('%Y-%m-%d') + ' ' + wc_df.TIME
wc_df['DATE_TIME'] = pd.to_datetime(wc_df['DATE_TIME'])
wc_df = wc_df.set_index('DATE_TIME')

def gen_freq(x, topK=10):
    super_dict = {}
    dicts = x.map(lambda y: json.loads(y)).values
    for d in dicts:
        for k, v in d.iteritems():  # d.items() in Python 3+
            if k:
                super_dict.setdefault(k, []).append(v)
                
    for k, v in super_dict.iteritems():
        super_dict[k] = sum(v)
        
    sorted_x = sorted(super_dict.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_x[:topK]

wc_df.resample('4h').agg({'trending': gen_freq})

wc_df.resample('8h').agg({'trending': gen_freq})

wc_df.resample('1d').agg({'trending': gen_freq})

