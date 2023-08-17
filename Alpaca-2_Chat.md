# Alpaca-2 Chat

## 簡介
歡迎來到Alpaca-2 專案！此專案利用 Llama-2 和Alpaca-2 作為指令精調的大型模型，基於原始的 Llama-2 擴充並優化了中文詞彙，從而提升了中文的基礎語意理解和指令理解能力。
此專案提供一個 Streamlit 介面，讓你可以即時與Alpaca-2 模型進行對話。

## 特色
- 基於 Llama-2 架構。
- 提供使用 Streamlit 的人工智能對話介面。
- Alpaca-2 指令精調大模型。
- 擴充中文詞彙和理解能力。
- 簡體中文到繁體中文的轉換功能。
- 包含一個處理和管理回調事件的系統。
- 提供一個有效管理 GPU 資源的介面。

## 使用

### 依照需求修改`chat.py`中的`LlamCpp`

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
    )
```

### 啟動LLama Chat UI
```
streamlit run chat.py
```
