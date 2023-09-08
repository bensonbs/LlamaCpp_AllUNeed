import argparse
from glob import glob 
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.embeddings import LlamaCppEmbeddings, OpenAIEmbeddings

def load_data(paths):
    data = []
    for path in glob(paths):
        data += PyPDFLoader(path).load()
    # data += WebBaseLoader("https://zh.wikipedia.org/zh-tw/LK-99").load()
    return data

def split_data(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=52, length_function=len)
    return text_splitter.split_documents(data)

def create_vectorstore(data, type):
    return FAISS.from_documents(documents=data, embedding=load_emb(type))

def load_emb(type):
    return LlamaCppEmbeddings(model_path=args.model_path, n_gpu_layers=43, n_batch=512, n_ctx=2048, f16_kv=True) if type == "llama" else OpenAIEmbeddings()

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf-path', type=str, default="docs/*.pdf", help="Specify the path to the PDF files to process. You can use '*' to process all files in a directory.")
    parser.add_argument('--model-path', type=str, default="./chinese-alpaca-2-7b/ggml-model-q4_0.bin", help="Specify the full path to the model file.")
    parser.add_argument('--embedding', type=str, default="llama", help="Choose the embeddings to use (default: 'openai'). Options: 'llama' or 'openai'.")
    # Parse the arguments
    args = parser.parse_args()
    
    paths=args.pdf_path
    type=args.embedding
    data = load_data(paths)
    all_splits = split_data(data)
    vectorstore = create_vectorstore(all_splits,type)
    vectorstore.save_local(f"./faiss/{type}_index")
    print('finish')
