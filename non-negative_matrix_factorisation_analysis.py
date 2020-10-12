# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 15:15:29 2020

@author: Frederick
"""

# IMPORT BASE REQUIREMENTS 
import pandas as pd
from utils.config import config_dict
import utils.db_toolbox as tb

# IMPORT SKLEARN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# CUSTOM SQL CONNECTION FUNCTION
con = tb.db_con(config_dict)


npr = pd.DataFrame(con.read_query("""select *
                                    from articles
                                    where pub_date between '2015-01-01' and '2016-01-01';"""),
                                    columns=['heading','sub_heading','unique_text','author','pub_date','url','website'])
npr['comb_head'] = npr['heading'] + '. ' + npr['sub_heading']

# Term frequency inverse distance function
tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = tfidf.fit_transform(npr['comb_head'])
nmf_model = NMF(n_components=20,random_state=42)
nmf_model.fit(dtm)

# RESULTS
for index, topic in enumerate(nmf_model.components_):
    print(f"THE TOP 15 WORDS FOR TOPIC # {index}")
    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')

topic_results = nmf_model.transform(dtm)
npr['Topic'] = topic_results.argmax(axis=1)
#mapping topic name to column!

#create dictionary
mytopic_dict = {0:'Ops',1:'Trade',2:'??',3:'??',4:'People',5:'Trade',6:'??'}

#assign dictionary to column
npr['Topic label'] = npr['Topic'].map(mytopic_dict)