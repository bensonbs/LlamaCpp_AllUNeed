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

## 使用方法
- 請依照`LlamaCpp_AllUNeed.ipynb`教學建置CPU/GPU環境，與轉換模型權重。
- 使用`streamlit run main.py`啟動你的Llama2中文版UI, and have fun!
![DEMO](Demo_UI.png)

ref:
- [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [streamlit Chat UI](https://medium.com/@daydreamersjp/implementing-locally-hosted-llama2-chat-ui-using-streamlit-53b181651b4e)
