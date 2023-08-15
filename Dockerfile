# Using the specified image
FROM huggingface/transformers-pytorch-gpu:latest

# Running apt update and installing essential tools
RUN apt update && apt install -y git cmake make

# Clone and build llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp /llama.cpp
WORKDIR /llama.cpp
RUN make && \
    mkdir build && \
    cd build && \
    cmake .. && \
    cmake --build . --config Release

# Install llama-cpp-python
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

# Clone LlamaCpp_AllUNeed
RUN git clone https://github.com/bensonbs/LlamaCpp_AllUNeed /LlamaCpp_AllUNeed
WORKDIR /LlamaCpp_AllUNeed

CMD ["streamlit", "run", "chat.py"]
