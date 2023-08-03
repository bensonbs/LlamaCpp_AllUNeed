import os
import streamlit as st
from langchain.llms import LlamaCpp
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks import StreamlitCallbackHandler
from langchain.llms import LlamaCpp
model_path = "/home/sung/llm/chinese-alpaca-2-7b/gml-model-q4_0.bin"

from glob import glob 
from langchain.document_loaders import PyPDFLoader


@st.cache_resource
def get_vectorstore(paths):
    data = []
    for path in glob(paths):
        data += PyPDFLoader(path).load()
    # data += WebBaseLoader("https:/xxxx").load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1024, chunk_overlap = 52 ,length_function = len)
    all_splits = text_splitter.split_documents(data)

    # LlamaCppEmbeddings
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=load_emb())

    return vectorstore

def load_emb():
    return LlamaCppEmbeddings(model_path="/home/mefae1/llm/chinese-alpaca-2-7b/ggml-model-q4_0.bin",
            n_gpu_layers=35,
            n_batch=128,
            n_ctx=2048,
            f16_kv=True
        )
    
@st.cache_resource
def load_model():
    llm = LlamaCpp(
            model_path="/home/mefae1/llm/chinese-alpaca-2-7b/ggml-model-q4_0.bin",
            n_gpu_layers=35,
            n_batch=128,
            verbose=True,
            n_ctx=2048,
            input={"temperature": 0.0, "max_length": 2048},
        )
    return llm


template =  """
使用以下文章來回答最後的問題。
如果你不知道答案，就說你不知道，不要試圖編造答案。
文章: {context}
問題:{question}
簡潔的答案:
"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
llama = load_model()
vectorstore = get_vectorstore('/home/mefae1/llm/docs/*')
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
dics = {

}
if prompt := st.chat_input("網家倉庫建物有幾坪?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("ChatGPT is typing ..."):
            st_callback = StreamlitCallbackHandler(st.container())
            qa_chain = RetrievalQA.from_chain_type(
                llama,
                retriever=vectorstore.as_retriever(search_type = "similarity", search_kwargs = {"k" : 5}),
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
                return_source_documents=True,
                callbacks=[st_callback]
            )
            result = qa_chain({"query": prompt})
            st.write(result["result"])
            for i, ref in enumerate(result['source_documents']):
                with st.expander(f'參考內容 {i+1}'):
                    link = ref.metadata['source']
                    basename = os.path.basename(link)
                    name = dics[basename] if basename in dics else basename
                    st.write(f'`{ref.page_content}`')
                    st.write(f'來源: [{name}](http://10.96.212.243:8502/pdf/{basename})')
                    
