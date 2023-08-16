# app.py
import langchain
langchain.debug = False
from typing import List, Union
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import streamlit as st
from opencc import OpenCC
from typing import Tuple

cc = OpenCC('s2twp')

def model_check():
    import os
    import gdown

    output_path = './chinese-alpaca-2-7b/ggml-model-q4_0.bin'

    # 只在檔案不存在時下載
    if not os.path.exists(output_path):
        # 確認並創建目錄
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 下載檔案
        gdown.download('https://drive.google.com/uc?id=1bk2-n2fncZ8XSg_G6PIGfhZMqghfn482', output_path, quiet=False)

def init_page() -> None:
    st.set_page_config(
        page_title="🐪 Alpaca-2"
    )
    st.header("🐪 Alpaca-2")


def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content="You are a helpful AI assistant. Reply your answer in mardkown format.")
        ]
        st.session_state.costs = []

@st.cache_resource
def load_llm() -> Union[ChatOpenAI, LlamaCpp]:
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    n_gpu_layers = 43  # Change this value based on your model and your GPU VRAM pool.
    n_batch = 8  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    return LlamaCpp(
        model_path="./chinese-alpaca-2-7b/ggml-model-q4_0.bin",
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        callback_manager=callback_manager,
        verbose=True,
        n_ctx=2048,
    )


def get_answer(llm, messages) -> Tuple[str, float]:
    if isinstance(llm, LlamaCpp):
        if len(messages) > 3:
            messages = [messages[0]] + messages[-3:]
        return cc.convert(llm(llama_v2_prompt(convert_langchainschema_to_dict(messages)))), 0.0


def find_role(message: Union[SystemMessage, HumanMessage, AIMessage]) -> str:
    """
    Identify role name from langchain.schema object.
    """
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    raise TypeError("Unknown message type.")


def convert_langchainschema_to_dict(
        messages: List[Union[SystemMessage, HumanMessage, AIMessage]]) \
        -> List[dict]:
    """
    Convert the chain of chat messages in list of langchain.schema format to
    list of dictionary format.
    """
    return [{"role": find_role(message),
             "content": message.content
             } for message in messages]


def llama_v2_prompt(messages: List[dict]) -> str:
    """
    Convert the messages in list of dictionary format to Llama2 compliant format.
    """
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"
    DEFAULT_SYSTEM_PROMPT = f"""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    if messages[0]["role"] != "system":
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + messages
    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
        }
    ] + messages[2:]

    messages_list = [
        f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    messages_list.append(
        f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")

    return "".join(messages_list)


def main() -> None:
    init_page()
    model_check()
    area = [st.sidebar.empty() for i in range(3)]
    st.sidebar.markdown("## Options")
    llm = select_llm()
    init_messages()
    
    area[0].markdown("""
    # 簡介
     **本專案基於`Llama-2`開發，`Alpaca-2`指令精調大模型**。 
     **這些模型在原版`Llama-2`的基礎上擴充並優化了中文詞表**。
     **提升了中文基礎語義和指令理解能力**。 
    """)
    area[1].markdown("""
    - **Source Code** [Github](https://github.com/bensonbs/LlamaCpp_AllUNeed)
    - **weight 4bit** [Link](https://drive.google.com/file/d/1bk2-n2fncZ8XSg_G6PIGfhZMqghfn482/view?usp=sharing)
    """)
    area[2].markdown("""
    **reference:**
    - [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
    - [llama.cpp](https://github.com/ggerganov/llama.cpp)
    - [streamlit Chat UI](https://medium.com/@daydreamersjp/implementing-locally-hosted-llama2-chat-ui-using-streamlit-53b181651b4e)
    """)
    # Supervise user input
    if user_input := st.chat_input("Input your question!"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("ChatGPT is typing ..."):
            answer, cost = get_answer(llm, st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=answer))
        st.session_state.costs.append(cost)

    # Display chat history
    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)



# streamlit run app.py
if __name__ == "__main__":
    main()
