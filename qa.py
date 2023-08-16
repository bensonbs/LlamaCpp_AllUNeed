import os
import argparse
import streamlit as st
from langchain.llms import LlamaCpp
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings import LlamaCppEmbeddings, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI

@st.cache_resource
def get_vectorstore(type):
    index_path = f"./faiss/{type}_index"
    return FAISS.load_local(index_path, embeddings=load_emb(type))

@st.cache_resource
def load_emb(type):
    return LlamaCppEmbeddings(model_path=args.model_path,n_ctx=2048, f16_kv=True) if type == "llama" else OpenAIEmbeddings()

@st.cache_resource
def load_model(type):
    return LlamaCpp(model_path=args.model_path, n_gpu_layers=43, n_batch=512, verbose=True, n_ctx=2048) if type == 'llama' else ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
parser.add_argument('--model', type=str, default="llama", help="Specify the model to use for processing (default: 'llama'). Options: 'llama' or 'openai'.")
parser.add_argument('--model-path', type=str, default="./chinese-alpaca-2-7b/ggml-model-q4_0.bin", help="Specify the full path to the model file.")
parser.add_argument('--embedding', type=str, default="llama", help="Choose the embeddings to use (default: 'openai'). Options: 'llama' or 'openai'.")
parser.add_argument('--hyperlink', type=bool, default=False, help="Whether to include hyperlinks in the processed PDFs (default: True). Use 'False' to exclude hyperlinks.")

# Parse the arguments
args = parser.parse_args()

# Parse the arguments
args = parser.parse_args()
template =  """
使用以下文章來回答最後的問題。
如果你不知道答案，就說你不知道，不要試圖編造答案。
文章: {context}
問題:{question}
答案:
"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
llama = load_model(args.model)
vectorstore = get_vectorstore(type=args.embedding)
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
dics = {
    
}
st.write('# 🐪 Alpaca-2：Retrieval QA')
if prompt := st.chat_input("structure of LK-99"):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("ChatGPT is typing ..."):
            st_callback = StreamlitCallbackHandler(st.container())
            qa_chain = RetrievalQA.from_chain_type(
                llama,
                retriever=vectorstore.as_retriever(search_type = "similarity", search_kwargs = {"k" :3}),
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
                    if args.hyperlink:
                        st.write(f'來源: [{name}](http://0.0.0.0:8502/pdf/{basename})')
                    else:
                    
                        st.write(f'來源: {name}')
