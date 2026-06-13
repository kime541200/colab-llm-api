# Colab LLM API Server Deployer

這是一個基於 Google Colab CLI 的大模型 API 伺服器一鍵部署方案。本專案支援在 Google Colab 的免費或付費 GPU（如 Tesla T4、L4、A100）上，快速部署相容於 OpenAI API 接口的服務，並透過 Cloudflare Tunnel 提供公網 HTTPS 安全存取。

專案提供兩種主要部署方式：
* **vLLM**：適用於高吞吐量部署，支援 Hugging Face 原生權重（如 Qwen2.5-1.5B/7B-Instruct），已針對 T4 GPU 限制進行了 VRAM 防崩潰（OOM）優化。
* **llama.cpp (llama-cpp-python)**：適用於極致節省顯存的場景，使用 GGUF 格式量化模型，並支援 CUDA GPU 全載入加速（GPU Offloading）。

---

## 📂 目錄結構

```text
colab-user/
├── vllm/                      # vLLM 部署模組
│   ├── vllm_config.yaml       # vLLM 啟動與硬體限制參數
│   ├── serve_vllm.py          # VM 啟動腳本 (安裝相依性 + 執行)
│   └── inspect_vllm.py        # 診斷工具
├── llama_cpp/                 # llama.cpp 部署模組
│   ├── llama_config.yaml      # GGUF 模型、層數加速與伺服器設定
│   └── serve_llama_cpp.py     # VM 啟動腳本 (編譯 CUDA + 下載 GGUF + 執行)
└── README.md                  # 本說明文件
```

---

## 🛠️ 準備工作

在執行任何部署前，請先建立虛擬環境、啟用並安裝 `colab` 命令列工具：

1. **建立 Python 虛擬環境**：
   ```bash
   python3 -m venv .venv
   ```
2. **啟用虛擬環境**：
   ```bash
   source .venv/bin/activate
   ```
3. **安裝 `colab` CLI 工具**：
   ```bash
   # 使用 uv (推薦)：
   uv pip install google-colab-cli

   # 或使用 pip：
   pip install google-colab-cli
   ```
4. **Colab 登入認證**：
   若您是首次在該環境使用 `colab` 指令，請先完成認證：
   ```bash
   colab login
   ```

---

## 🚀 部署教學

### 方案 A：部署 llama.cpp (GGUF 格式)

1. **建立執行階段**：建立遠端 VM 機器（支援 T4 GPU 等）。
   ```bash
   colab new -s llama-api --gpu T4
   ```
2. **上傳設定檔**：將 YAML 設定上傳至遠端 VM。
   ```bash
   colab upload llama_cpp/llama_config.yaml /content/llama_config.yaml
   ```
3. **啟動伺服器**：執行部署腳本。
   ```bash
   colab exec -s llama-api -f llama_cpp/serve_llama_cpp.py
   ```
   *註：首次部署時，腳本會自動在遠端編譯 CUDA 版本的 `llama-cpp-python`（約需 1~2 分鐘），並透過 `huggingface-cli` 下載 GGUF 權重。*

---

### 方案 B：部署 vLLM (Hugging Face 原生格式)

1. **建立執行階段**：建立遠端 VM 機器（支援 T4 GPU 等）。
   ```bash
   colab new -s vllm-api --gpu T4
   ```
2. **上傳設定檔**：將 YAML 設定上傳至遠端 VM。
   ```bash
   colab upload vllm/vllm_config.yaml /content/vllm_config.yaml
   ```
3. **啟動伺服器**：執行部署腳本。
   ```bash
   colab exec -s vllm-api -f vllm/serve_vllm.py
   ```

### 💡 關於 GPU 硬體選擇（付費訂閱用戶）

免費用戶一般僅能使用 `--gpu T4`。若您是 Colab Pro / Pro+ 等付費訂閱用戶，可以將建立機器的指令替換為更強大的 GPU 來大幅提升推論效能：

* **硬體規格選項**：
  * `T4`：免費用戶規格（Turing 架構，16GB VRAM，不原生支援 bfloat16）。
  * `L4`：中階規格（Ampere 架構，24GB VRAM，支援 bfloat16，性價比極佳）。
  * `A100` / `H100`：高階規格（大顯存、超高運算力，適合跑更大參數的模型）。
* **指令範例**：
  ```bash
  # 建立配備 L4 GPU 的機器
  colab new -s llama-api --gpu L4
  ```
* **升級後的設定檔最佳化建議**：
  若您選用 `L4`、`A100` 或 `H100`，建議對 `vllm_config.yaml` 或 `llama_config.yaml` 進行以下調整以最大化效能：
  * 將 `dtype` 欄位從 `"half"` 修改為 `"bfloat16"`，以獲得更好的數值精度。
  * (僅限 vLLM) 可以將 `gpu_memory_utilization` 從 `0.7` 適度上調至 `0.85` 或 `0.9`，釋放更多 VRAM 給 KV cache。

---

## 🔗 連線與測試

部署成功後，終端機會輸出 Cloudflare 產生的公網 URL，格式如下：
```text
🎉 Tunnel established successfully!
🔗 OpenAI Base URL: https://xxxx-xxxx-xxxx.trycloudflare.com/v1
👉 Model name: qwen2.5-1.5b
```

您可以使用 `curl` 或任何 OpenAI SDK 來測試模型推理：

```bash
curl -X POST "https://xxxx-xxxx-xxxx.trycloudflare.com/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "qwen2.5-1.5b",
       "messages": [{"role": "user", "content": "你好，請簡單自我介紹。"}]
     }'
```

#### Python (使用 `openai` SDK 範例)

請先安裝 SDK：
```bash
pip install openai
```

使用以下 Python 腳本進行測試：
```python
from openai import OpenAI

# 初始化客戶端 (將 base_url 換成 Cloudflare Tunnel 輸出的公網網址)
client = OpenAI(
    base_url="https://xxxx-xxxx-xxxx.trycloudflare.com/v1",
    api_key="not-needed"  # 若 YAML 設定中未啟用 api_key，此處填任意字串即可
)

# 進行對話推理
response = client.chat.completions.create(
    model="qwen2.5-1.5b",  # 對應設定檔中的 model_alias 
    messages=[
        {"role": "user", "content": "你好，請簡單自我介紹。"}
    ],
    temperature=0.7
)

print(response.choices[0].message.content)
```

---

## 🧹 資源清理

測試完畢後，**請務必關閉 VM 資源**，以免持續消耗您的 Colab 計算額度。
修補後的 CLI 指令能夠在 VM 已回收的狀態下安全清除本地狀態：

* **停止 llama.cpp 服務**：
  ```bash
  colab stop -s llama-api
  ```
* **停止 vLLM 服務**：
  ```bash
  colab stop -s vllm-api
  ```
* **檢查當前狀態**：
  ```bash
  colab status
  ```
