# LangChain：Retrieval QA
LangChain是一款強大的工具，使用先進的機器學習模型處理和嵌入PDF文檔。它使用最先進的模型，如LlamaCpp和OpenAI的ChatGPT來提取有價值的信息，並提供智能的、由AI驅動的對問題的回答。它還內建支援多種類型的嵌入，能夠實現高質量的文本數據表示。

此應用程序是用Python構建的，集成了多個庫，包括streamlit用於用戶友好的界面，PyPDFLoader用於文檔處理，FAISS用於向量之間的高效相似性搜索等。LangChain非常適合處理大量的文檔，並根據這些文檔的內容提供即時的AI生成的答案。

## 如何使用
```bash
streamlit run <script_name.py> -- --model <model_name> --model-path <model_path> --pdf-path <pdf_path> --embedding <embedding> --hyperlink <bool> --cache <bool>
```

- `model`：指定用於處理的模型（默認：'llama'）。選項：'llama'或'openai'。
- `model-path`：指定模型文件的完整路徑。
- `pdf-path`：指定要處理的PDF文件的路徑。您可以使用 '*' 處理目錄中的所有文件。
- `embedding`：選擇要使用的嵌入（默認：'openai'）。選項：'llama'或'openai'。
- `hyperlink`：是否在處理的PDF中包含超鏈接（默認：True）。使用'False'排除超鏈接。
- `cache`：指定是否使用緩存。

**注意:選用openai model或embedding 需添加環境變數 `export OPENAI_API_KEY=`**

##　內部運作原理
LangChain首先加載並處理PDF文件，將文本分解成塊。然後使用指定的模型將這些塊嵌入。該應用程序創建所有嵌入的索引，可以保存在本地以供未來使用。

一旦所有數據處理完成，嵌入就緒，應用程序使用語言模型（LlamaCpp或OpenAI的ChatGPT）生成對用戶查詢的回答。這些回答基於處理過的PDF的內容。AI也可以引用提供信息的來源，如果啟用了‵hyperlink‵選項，甚至可以鏈接回原始文檔。

## 使用案例
LangChain的一個完美用例是用戶擁有大量的PDF文件，並在這些文檔中尋找特定的信息。而不是手動閱讀所有文件，用戶可以簡單地問AI獲取信息。AI將處理文檔，找到相關的信息，並提供簡潔的回答，所有這些都只需要幾秒鐘。

## 注意事項
此工具設計用於研究和開發目的，應負責任地使用。
