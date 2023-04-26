import streamlit as st
import pandas as pd
from txtai.embeddings import Embeddings

@st.cache_resource
def get_data():
    embeddings = Embeddings()
    embeddings.load('pa_index')

    pa_sents = pd.read_csv('pa_minutes_sents.csv')
    pa_full = pd.read_csv('pa_minutes_full.csv')
    return embeddings, pa_sents, pa_full
embeddings, pa_sents, pa_full = get_data()

def display_text(tup, context=1):
    selection = pa_full.iloc[pa_sents.org_idx[tup[0]]]

    st.markdown(f"<small style='text-align: right;'>File name: <b>{selection.fname}</b></small>",unsafe_allow_html=True)
    st.markdown(f"<small style='text-align: right;'>Similarity score: <b>{round(tup[1], 3)}</b></small>",unsafe_allow_html=True)

    res = pa_sents.sents[tup[0]]
    res = f"<span style='background-color:#fdd835'>{res}</span>"

    before, after = [], []
    for i in range(context+1):
        if i != 0:
            if pa_sents.org_idx[tup[0]-i] == pa_sents.org_idx[tup[0]]:
                before.append(pa_sents.sents[tup[0]-i])
            if pa_sents.org_idx[tup[0]+i] == pa_sents.org_idx[tup[0]]:
                after.append(pa_sents.sents[tup[0]+i] )
    
    before = '\n'.join(before)
    after  = '\n'.join(after )
    to_display = '\n'.join([before,res,after]).replace('$', '\$').replace('`', '\`')

    st.markdown(to_display,unsafe_allow_html=True)
    st.markdown("<hr style='width: 75%;margin: auto;'>",unsafe_allow_html=True)

st.title('Port Authority Minutes Search Engine MY NAME IS PETER')
st.markdown(
    '''
    This tool is designed to take in a search query and return relevant sentences from the public Port Authority minutes. 
    It uses [txtai's](https://neuml.github.io/txtai/) semantic search API to generate and load embeddings from sBERT's [all-MiniLM-L6-v2](https://huggingface.co/nreimers/MiniLM-L6-H384-uncased) model.
    The data for this search engine can be found [here](https://www.panynj.gov/corporate/en/board-meeting-info/board-minutes-contract-authorizations.html).
    '''
)

query = st.text_input('Search any query')
results = st.number_input(value=5, label='Choose the amount of results you want to see')
context = st.number_input(value=1, label='Choose a context size')

uids = embeddings.search(query, results)

if query != '':
    for tup in uids:
        display_text(tup, context)

st.markdown('<small>This tool is for educational purposes ONLY!</small>', unsafe_allow_html=True)
st.markdown('<small>Assembled by Peter Nadel | TTS Research Technology</small>', unsafe_allow_html=True)
