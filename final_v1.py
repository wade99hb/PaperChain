import pandas as pd
import numpy as np
import nltk
import re
import time
import os
import math
import psycopg2
import streamlit as st

from psycopg2 import extras
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import clear_output
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

st.set_page_config(layout="wide")

# PostgreSQL 접속 정보
username = 'wade'
password = '0729'
host = 'localhost'
port = '5432'  # PostgreSQL 기본 포트
database = '4test'

def load_table(table_name):
  # PostgreSQL 접속 정보
  username = 'wade'
  password = '0729'
  host = 'localhost'
  port = '5432'  # PostgreSQL 기본 포트
  database = '4test'
  # 데이터베이스 연결
  conn = psycopg2.connect(dbname=database, user=username, password=password, host=host, port=port)
  cursor = conn.cursor()

  # SQL 쿼리 실행: scie_paper 테이블 선택
  cursor.execute(f"SELECT * FROM {table_name}")

  # 결과를 pandas DataFrame으로 변환
  table = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

  # 데이터베이스 연결 종료
  cursor.close()
  conn.close()

  return table

database = load_table('scie_paper')
test_df = load_table('test_df')

test_df['tintro']= test_df['title']+test_df['introduction']

def tokenize(text):
    # 단어 토큰화
    tokens = nltk.word_tokenize(text)

    # NLTK의 pos_tag를 사용하여 단어의 품사 태깅
    tagged_tokens = nltk.pos_tag(tokens)

    # WordNet 품사 태깅으로 변환
    def get_wordnet_pos(tag):
        if tag.startswith('J'):  # 태그가 J로 시작하면 형용사
            return wordnet.ADJ
        elif tag.startswith('V'):  # 태그가 V로 시작하면 동사
            return wordnet.VERB
        elif tag.startswith('N'):  # 태그가 N으로 시작하면 명사
            return wordnet.NOUN
        elif tag.startswith('R'):  # 태그가 R로 시작하면 부사
            return wordnet.ADV
        else:
            return None

    # 레마타이징 -> 단어를 기본 형태로 변환함
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []
    for token, tag in tagged_tokens:
        wordnet_pos = get_wordnet_pos(tag) or wordnet.NOUN
        lemmatized_token = lemmatizer.lemmatize(token, pos=wordnet_pos)
        lemmatized_tokens.append(lemmatized_token.lower())  # 토큰을 소문자로 변환

    # 정규화된 토큰 반환
    return lemmatized_tokens

# db와 tp(test_paper)의 인트로를 각각 토큰화해서 리스트로 저장 (논문별 index 살아있음)

database_tokens = []
for value in database['tintro']:
    token_list = tokenize(value)
    database_tokens.append(token_list)

test_tokens = []
for value in test_df['tintro']:
    token_list = tokenize(value)
    test_tokens.append(token_list)

#리스트를 받아서 리스트 내의 단어들을 걸러주는 함수

def clean_list(dirty_list):
    cleaned_list = []

    for item in dirty_list:
        # None 또는 빈 문자열인 경우 건너뛰기
        if item is None or item == "":
            continue

        # 규칙 1: 한 글자인 단어 제거
        if len(item) == 1:
            continue
        # 규칙 2: 숫자만 있는 단어는 제거
        if re.search(r'\d', item) and not re.search(r'\D', item):
            continue
        # 규칙 3: 영어가 아닌 문자가 포함된 단어 제거
        if re.search(r'[^a-zA-Z]', item):
            continue
        # 규칙 4: "-"를 제외한 특수문자가 포함된 단어 제거
        if re.search(r'[^\w-]', item):
            continue
        # 규칙 5: "-"이나 "-"과 숫자만 있는 단어 제거
        if re.search(r'^-?\d*$', item):
            continue

        cleaned_list.append(item)

    return cleaned_list

#db와 tp의 토큰화 리스트의 set화

set_db_tokens = []
for value in database_tokens:
    for val in value:
        set_db_tokens.append(val)
set_db_tokens = list(set(set_db_tokens))

set_test_tokens = []
for value in test_tokens:
    for val in value:
        set_test_tokens.append(val)
set_test_tokens = list(set(set_test_tokens))

# 문자열 각각 거르기
db_token_list = clean_list(set_db_tokens)
test_token_list = clean_list(set_test_tokens)

# 유효하지 않은 db단어 파일로부터 불러와서 단어 리스트로 저장하기
execute_df = load_table('execute_df')
execute_list = execute_df['word'].tolist()

# 최종적으로 사용할 리스트 생성
Bag_of_Words = list(set(db_token_list) - set(execute_list))

# (index,column)에 index번째 intro의 단어 중 column과 똑같은 단어 갯수 n을 할당함

def word_count_df(index_list, Bag_of_Words):
    # 빈 리스트를 초기화합니다. 이 리스트는 나중에 DataFrame으로 변환될 것입니다.
    data = []
    
    # 값 카운팅
    def count_words(index_list_item):
        return {token: index_list_item.count(token) for token in Bag_of_Words}
    
    # 각 column에 값 할당
    for item in index_list:
        data.append(count_words(item))
    
    # 최종 DataFrame 생성
    bog_df = pd.DataFrame(data, columns=Bag_of_Words)
    
    return bog_df

# db와 test의 word count df를 만듦
db_count_df = word_count_df(database_tokens, Bag_of_Words)
test_count_df = word_count_df(test_tokens, Bag_of_Words)

# test_count_df의 n번째 행을 db_count_df 마지막행에 추가해주는 함수
def add_row_from_b_to_a(db_count_df, test_count_df, n):
    if n < 0 or n >= len(test_count_df):
        raise ValueError("n is out of the range of test_count_df's rows")
    # iloc를 사용하여 n번째 행을 선택하고, 이를 DataFrame으로 변환합니다.
    row_to_add = test_count_df.iloc[[n]]
    # pandas.concat을 사용하여 db_count_df와 row_to_add를 결합합니다.
    added_df = pd.concat([db_count_df, row_to_add], ignore_index=True)
    return added_df

tem_df = add_row_from_b_to_a(db_count_df, test_count_df, 0)

# word count df를 tf-idf score로 바꿔주는 함수
def tf_idf_score(df):
    # TF 계산
    tf = df.div(df.sum(axis=1), axis=0)

    # IDF 계산
    doc_count = (df > 0).sum(axis=0)
    idf = np.log(len(df) / doc_count)
    idf = idf.replace(-np.inf, 0)  # Replacing -inf with 0 if a term does not appear in any document

    # TF-IDF
    tf_idf = tf.mul(idf, axis=1)
    return tf_idf

# 데이터프레임의 마지막 열과 가장 유사한 열과의 인덱스와 점수를 반환한다
def find_most_similar_rows(df, n):
    
    # 마지막 행을 제외한 나머지 행들과의 코사인 유사도를 계산
    cosine_similarities = cosine_similarity(df[:-1], df[-1:])
    
    # 마지막 행과의 유사도가 높은 순으로 인덱스를 얻음
    most_similar_indices = np.argsort(cosine_similarities.flatten())[::-1][:n]
    
    # 해당 인덱스의 유사도를 얻음
    most_similar_scores = cosine_similarities.flatten()[most_similar_indices]
    most_similar_scores = list(zip(most_similar_indices, most_similar_scores))
    
    return most_similar_scores

def print_titles (list):
    j=0
    for i in list:
        j+=1
        st.text(f"Rank : {j:2d}  |  Score : {i[1]:.2f}   |  Title : {database['Title'][i[0]]}")

def calculate_tfidf(db_count_df,test_count_df,low,num):
    tem_df = add_row_from_b_to_a(db_count_df,test_count_df,low)
    tem_df = tf_idf_score(tem_df)
    testtest = find_most_similar_rows(tem_df,num)
    st.text("="*168)
    st.text(f"■ Original Paper : {test_df['title'][low]}")
    st.text("■ We're Searching for : ")
    tem_list = test_df['pnf'][low].split('/\n')
    for i,val in enumerate(tem_list):
        st.text(f"⇒ {val}")
    st.text("="*168)
    print_titles(testtest)

test_title = test_df['title'].tolist()

# 선택 박스
select_title = st.selectbox(
    'ref를 찾을 논문을 선택하시오',
    (test_title)
)

#button = st.button('go')

def num(select_title):
  num=0
  for index, value in enumerate(test_title):
      if select_title == value:
          num = index
  return num

calculate_tfidf(db_count_df,test_count_df,num(select_title),22)

#if button:
#  st.text(num(select_title))
#  calculate_tfidf(db_count_df,test_count_df,num(select_title),22)