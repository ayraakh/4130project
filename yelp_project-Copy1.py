#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install pyspark


# In[3]:


from pyspark.sql import *
from pyspark.sql import SparkSession
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext('local')
spark = SparkSession(sc)


# In[5]:


from pyspark.sql.functions import *
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.types import DoubleType, DateType, StringType, StructType, TimestampType


# In[6]:


reviews_sdf = spark.read.json("/Users/roshni/Downloads/archive/yelp_academic_dataset_review.json")
reviews_sdf = reviews_sdf.withColumnRenamed('stars', 'review_stars')
reviews_sdf = reviews_sdf.withColumnRenamed('date', 'review_date')
reviews_sdf = reviews_sdf.withColumn("review_date", to_date(to_timestamp(col("review_date"), "yyyy-MM-dd HH:mm:ss" )) )
reviews_sdf = reviews_sdf.withColumn("review_age_days", datediff(current_date(),col("review_date")) )
reviews_sdf = reviews_sdf.withColumn("review_age_years", col("review_age_days")/365.0 )
reviews_sdf = reviews_sdf.drop('review_id')


# In[7]:


reviews_sdf.groupBy("review_stars").count().sort("review_stars").show()
reviews_sdf = reviews_sdf.withColumn("goodreview", when(col("review_stars") > 3.0, 1.0).otherwise(0.0))
reviews_sdf.groupBy("goodreview").count().sort("goodreview").show()
reviews_sdf.printSchema()


# In[8]:


business_sdf = spark.read.json("/Users/roshni/Downloads/archive/yelp_academic_dataset_business.json")
business_sdf = business_sdf.withColumnRenamed('stars', 'business_stars')
business_sdf = business_sdf.withColumn("alcohol", business_sdf.attributes.Alcohol)
business_sdf = business_sdf.withColumn("noiselevel", business_sdf.attributes.NoiseLevel)
business_sdf = business_sdf.withColumn("ambience", business_sdf.attributes.Ambience)
business_sdf = business_sdf.drop('city','hours','is_open','latitude','longitude','postal_code','state','address','attributes')
business_sdf.printSchema()


# In[9]:


users_sdf = spark.read.json("/Users/roshni/Downloads/archive/yelp_academic_dataset_user.json")
users_sdf = users_sdf.withColumn("yelping_since", to_date(to_timestamp(col("yelping_since"), "yyyy-MM-dd HH:mm:ss" )) )
users_sdf = users_sdf.withColumn("account_age_days", datediff(current_date(),col("yelping_since")) )
users_sdf = users_sdf.withColumn("account_age_years", col("account_age_days")/365.0 )
users_sdf = users_sdf.drop("cool","funny","useful","elite","fans","friends","compliment_cool","compliment_cute","compliment_funny","compliment_hot","compliment_list","compliment_more","compliment_note","compliment_photos","compliment_plain","compliment_profile","compliment_writer")
users_sdf.printSchema()


# In[11]:


yelp_sdf = reviews_sdf.join(business_sdf, "business_id")

yelp_sdf = yelp_sdf.join(users_sdf, "user_id")

yelp_sdf.printSchema()

yelp_sdf.count()


# In[12]:


yelp_sdf = yelp_sdf.withColumnRenamed('text', 'review_text')

# Tokenizer
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, Word2Vec
tokenizer = RegexTokenizer(inputCol="review_text", outputCol="review_words", pattern="\\w+", gaps=False)
yelp_sdf = tokenizer.transform(yelp_sdf)

yelp_sdf.select("review_text", "review_words").show()

# Remove stopwords. The default is a standard set of English stopwords
stop_words_remover = StopWordsRemover(inputCol="review_words", outputCol="review_words_filtered")
yelp_sdf = stop_words_remover.transform(yelp_sdf)

yelp_sdf.select("review_text", "review_words", "review_words_filtered").show()


# In[13]:


from pyspark.ml.feature import HashingTF, IDF, Tokenizer, RegexTokenizer, StopWordsRemover, Word2Vec

hashtf = HashingTF(numFeatures=8192, inputCol="review_words_filtered", outputCol="review_words_filtered_tf")
yelp_sdf = hashtf.transform(yelp_sdf)

# Calculate the inverse document frequency (IDF)
idf = IDF(inputCol="review_words_filtered_tf", outputCol="review_words_filtered_features", minDocFreq=10)
idfModel = idf.fit(yelp_sdf)
yelp_df = idfModel.transform(yelp_sdf)


# In[41]:


assembler = VectorAssembler(inputCols=["review_age_years","account_age_years","review_words_filtered_tf"], outputCol="features")
output = assembler.transform(yelp_sdf)
model = output.select("review_words_filtered_tf","review_age_years","account_age_years","goodreview","features")


# In[42]:


train, test = model.randomSplit([0.7, 0.3])

lr = LogisticRegression(labelCol="goodreview").fit(train)
results = lr.evaluate(train).predictions

results.show()


# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# In[51]:


from pyspark.sql.functions import year
from pyspark.sql.functions import to_date

yelp_sdf = yelp_sdf.withColumn('review_year',year(col("review_date")))
yelp_sdf = yelp_sdf.withColumn('review_month',month(col("review_date")))
yelp_sdf = yelp_sdf.withColumn('review_yearmonth',date_format(col("review_date"), "yyyy-MM"))


# In[68]:


fig = plt.figure()
plt.bar(star_df['review_stars'],star_df['count'])
plt.title("Review Count by Stars Rating")
plt.show()

