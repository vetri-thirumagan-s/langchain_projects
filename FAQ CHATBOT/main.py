from langchain.llms import GooglePalm
from dotenv import load_dotenv
import os
import google.generativeai as palm
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders.csv_loader import CSVLoader
import streamlit as st



st.set_page_config(page_title="FAQ BOT",page_icon=':Red Question Mark:',layout="wide")
st.header("FAQ's BOT")
st.sidebar.header("Upload here")


load_dotenv()


def filename():
   
    csv=st.sidebar.file_uploader("here")
    hello=st.write(csv.name)
    return csv



def FAQ_chat(input):
    output=qa(input)
    return output["result"]

st.sidebar.selectbox("Select the platform",["Flipkart","Amazon","Housing"])

llm=GooglePalm(google_api_key=os.environ['GOOGLE_API_KEY'],temperature=0.5)
loader = CSVLoader(file_path="FAQ.csv")
data = loader.load()

embeddings=HuggingFaceEmbeddings()
VB=FAISS.from_documents(data,embeddings)
retrival_data=VB.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    input_key="query",
    retriever=retrival_data,
    return_source_documents=True
)


Query=st.text_input("Enter Your Query...")

if Query:
    final=(FAQ_chat(Query))
    st.header(final)