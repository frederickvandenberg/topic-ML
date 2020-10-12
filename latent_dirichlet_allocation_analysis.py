# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 15:25:44 2020

@author: Frederick
"""

import pandas as pd
from utils.config import config_dict
import utils.db_toolbox as tb

#CONNECTION FUNCTION
con = tb.db_con(config_dict)


nm = pd.DataFrame(con.read_query("""select *
                                    from articles;"""),
                                    columns=['heading','sub_heading','unique_text','author','pub_date','url','website'])

nm['comb_head'] = nm['heading'] + '. ' + nm['sub_heading']
nm['comb_head'][1]
nm.isnull().sum()

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
dtm = cv.fit_transform(nm['comb_head'])

from sklearn.decomposition import LatentDirichletAllocation
component_number = 5
random_state_number = 42

LDA = LatentDirichletAllocation(n_components=component_number, random_state=random_state_number)
LDA.fit(dtm)

single_topic = LDA.components_[2] #for e.g. [0] to max[2]
top_twenty_words = single_topic.argsort()[-20:]
for index in top_twenty_words:
    print(cv.get_feature_names()[index])
    
for i,topic in enumerate(LDA.components_):
    print(f"THE TOP 15 WORDS FOR TOPIC #{i}")
    print([cv.get_feature_names()[index] for index in topic.argsort()[-15:]])
    print('\n')
    print('\n')
    
topic_results = LDA.transform(dtm)
#assigns topic to dataframe!
nm['Topic'] = topic_results.argmax(axis=1) 