import streamlit as st
import torch
from fairseq.models.transformer_lm import TransformerLanguageModel
import os
import datetime
import csv
import time
import pandas as pd


def get_response(text, model_chk, min_len, max_len):
    start_time = time.time()
    dt_now = datetime.datetime.now()

    try: 
        os.chdir('./BioGPT')                    
        m = TransformerLanguageModel.from_pretrained(
            f"checkpoints/{model_chk[0]}", 
            f"{model_chk[1]}", 
            f"data{model_chk[2]}",
            tokenizer='moses', 
            bpe='fastbpe', 
            bpe_codes=f"data{model_chk[2]}/bpecodes",
            min_len=min_len,
            max_len_b=max_len,
            )           
        m.cuda()
        src_tokens = m.encode(text)
        generate = m.generate([src_tokens], beam=5)[0]        
        output = m.decode(generate[0]["tokens"])
        os.chdir('../')
    except:
        output = 'error'
    end_time = time.time()
    elapsed_time = end_time - start_time
    return output, dt_now, elapsed_time



st.header('BioGPT')
os.chdir('/home/python/Documents/biogpt')
st.image('./img/BioGPT in PubMedQA.png')
col1, col2, col3 = st.columns(3)
with col1:
    model_dict = {'BioGPT': ('Pre-trained-BioGPT','checkpoint.pt', '/BioGPT'),
                  'BioGPT-Large': ('Pre-trained-BioGPT-Large', 'checkpoint.pt', '/BioGPT-Large'), 
                  'QA-PubMedQA-BioGPT-Large': ('QA-PubMedQA-BioGPT-Large', 'checkpoint_avg.pt', '/PubMedQA/raw'),
                  'RE-BC5CDR-BioGPT': ('RE-BC5CDR-BioGPT', 'checkpoint_avg.pt', '/BC5CDR/raw')}
    model_name = st.selectbox('学習済みモデルの選択', ['BioGPT', 'BioGPT-Large'])
    # model_name = st.selectbox('学習済みモデルの選択', ['BioGPT', 'QA-PubMedQA-BioGPT-Large', 'RE-BC5CDR-BioGPT'])
    model_chk = model_dict[model_name]
with col2:
    min_len = st.slider('最小文字数', 5, 100, 50)
with col3:
    max_len = st.slider('最大文字数', 500, 1000, 750)

text_input = st.text_area('テキストを入力してください(英文)')
if st.button('次に続くテキストを生成(英文)'):
    with st.spinner('文章生成中...'):
        text_output, dt_now, elapsed_time = get_response(text_input, model_chk, min_len, max_len)
    st.write(text_output)
    with open('prompt_and_results.csv', mode='a') as f:
        writer = csv.writer(f)
        writer.writerow([dt_now, model_name, elapsed_time, text_input, text_output])

link = 'https://arxiv.org/abs/2210.10341'
df_statistics = pd.read_csv('./prompt_and_results.csv', header=None, index_col=None)
df_statistics.columns = ['time', 'model', 'time', 'input', 'output']
search_count = len(df_statistics)
popular_words = df_statistics['input'].value_counts()[:5]
sidebar_message = f"""
***  
version: 0.0.2  
last update: 2023/2/6  
MicorsoftによるBioGPTをwebアプリ化  
BioGPT: {link}  
英文の翻訳はDeepLやGoogle翻訳をお使いください  

文章の例
```
Covid-19 is
```
全検索回数: {search_count}  

"""
st.sidebar.markdown(sidebar_message)
st.sidebar.write(popular_words)
