ui=# import os

# from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from pypdf import PdfReader
# from langchain import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
#from langchain_pinecone import PineconeVectorStore
# from langchain.embeddings import HuggingFaceEmbeddings

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain

from typing import Any, Dict, List

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_pinecone import PineconeVectorStore

from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from pypdf import PdfReader
from langchain.chains import RetrievalQA, ConversationalRetrievalChain

from langchain.chains.retrieval import create_retrieval_chain

from langchain.document_loaders import CSVLoader

from langchain.indexes import VectorstoreIndexCreator
from langchain.prompts import PromptTemplate

#from hugchat import hugchat
from typing import Any, Dict, List
import streamlit as st

if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = ''

if 'llm' not in st.session_state:
    st.session_state['llm'] = ''

if 'db' not in st.session_state:
    st.session_state['db'] = ''

if 'loaded_db' not in st.session_state:
    st.session_state['loaded_db'] =''

if 'run_once' not in st.session_state:
    st.session_state['run_once'] = 0

if 'split_1' not in st.session_state:
    st.session_state['split_1'] = ''


# load_dotenv()
# split_1 = ''

if (st.session_state['run_once'] == 0):
    documents_1 = ''


    # os.environ['CURL_CA_BUNDLE'] = ''
    # if (st.session_state['run_once'] == 0):
    #reader = PdfReader('C:\\Users\\RSPRASAD\\OneDrive - Danaher\\Learning\\Hackathon\\BecLS_Website_v2.pdf')
    reader = PdfReader('website.pdf')
    
    for page in reader.pages:
        documents_1 += page.extract_text()

   
    # Document Splitting
    chunk_size = 200
    chunk_overlap = 10

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    st.session_state['split_1'] = splitter.split_text(documents_1)
    st.session_state['split_1'] = splitter.create_documents(st.session_state['split_1'])
    st.write(st.session_state['split_1'])
    st.session_state['run_once'] = 1
    

def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):

    
    if (st.session_state['embeddings'] ==''):
        embeddings = AzureOpenAIEmbeddings(api_key=st.secrets['AzureOpenAIEmbeddings'],
                                    azure_endpoint = st.secrets['azure_endpoint1'],
                                    model="text-embedding-3-large",
                                    openai_api_version=st.secrets['openai_api_version'])
        st.session_state['embeddings'] = embeddings
    else:
        embeddings = st.session_state['embeddings']
    if(st.session_state['db'] =='' and st.session_state['split_1']):
        
        db = FAISS.from_documents(st.session_state['split_1'], embeddings)
        st.session_state['db'] = db
    else:
        db = st.session_state['db']

    
    # db.save_local('vector store\\becki')
    if(st.session_state['loaded_db'] == ''):
    
        loaded_db = FAISS.load_local('vector store/becki',\
    embeddings, allow_dangerous_deserialization=True)
        st.session_state['loaded_db'] = loaded_db
    
    else:
        loaded_db = st.session_state['loaded_db']
    

    if(st.session_state['llm'] ==''):
        llm = AzureChatOpenAI(api_key = st.secrets['AzureChatOpenAI'],
                        azure_endpoint = st.secrets['azure_endpoint2'],
                        model = "gpt-4o",
                        api_version=st.secrets['api_version'],
                        temperature = 0
                    )
        st.session_state['llm'] = llm
    else:
        llm =  st.session_state['llm']


    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    retrieval_qa_chjat_prompt = PromptTemplate.from_template(
    """
   Answer any use questions based solely on the context below. If you do not find the context, do not use your own kowledge to answer the question even if you know the answer:

<context>

{context}

</context>
    """
)
    
    
    stuff_documents_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    
    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=loaded_db.as_retriever(), prompt=rephrase_prompt
    )
    # qa = create_retrieval_chain(
    #     retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    # )

    # result = qa.invoke(input={"input": query, "chat_history": chat_history})
    # return result
    # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    

    
    retrival_chain = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=combine_docs_chain
    )

    result = retrival_chain.invoke(input={"input": query, "chat_history": chat_history})
    #st.write(result)
    #st.session_state['prompt'] = ''
    return result

