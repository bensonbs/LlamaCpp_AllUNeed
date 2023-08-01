# LlamaCpp_AllUNeed

## 系統需求: 
UbuntuOS22.04 
- GPU版本需使用Nvidia顯示卡且vram > 6GB

## 安裝教學

### llama.cpp 環境建置
```
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
python3 -m pip install -r requirements.txt
```

### 模型下載[傳送們](https://huggingface.co/ziqingyang/chinese-alpaca-2-7b)
```
.
├── LlamaCpp_AllUNeed
├── chinese-alpaca-2-7b
└── llama.cpp
```

### 模型精度轉換 pth -> f16
```
python3 convert.py /path_to_model/chinese-alpaca-2-7b/
```

#### 模型精度轉換 f16 -> q4
```
./quantize /path_to_model/chinese-alpaca-2-7b/ggml-model-f16.bin /path_to_model/chinese-alpaca-2-7b/gml-model-q4_0.bin q4_0
```

## 套件安裝
### CPU版本
```
pip install  llama-cpp-python
```

```
pip install langchain
```

### CPU版本
```
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sudo sh cuda_12.2.0_535.54.03_linux.run
sudo apt install nvidia-cuda-toolkit
mkdir build
cd build
cmake .. -DLLAMA_CUBLAS=ON
cmake --build . --config Release
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```

```
pip install langchain
```

## 使用方法
### 依照需求修改`main.py`中的`LlamCpp`

**CPU版本**
```
llm = LlamaCpp(
    model_path="/path_to_model/chinese-alpaca-2-7b/gml-model-q4_0.bin",
    input={"temperature": 0.0, "max_length": 2048},
    callback_manager=callback_manager,
    verbose=True,
)
```

**GPU版本**
```
LlamaCpp(
        model_path="/path_to_model/chinese-alpaca-2-7b/gml-model-q4_0.bin",
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        callback_manager=callback_manager,
        verbose=True,
        input={"temperature": 0.0, "max_length": 2048},
    )
```

### 啟動streamlit UI
```
streamlit run main.py
```

![DEMO](Demo_UI.png)

ref:
- [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [streamlit Chat UI](https://medium.com/@daydreamersjp/implementing-locally-hosted-llama2-chat-ui-using-streamlit-53b181651b4e)
