# llama.cpp API Server Deployer

本模組用於在 Google Colab 上，透過 **llama.cpp (llama-cpp-python)** 引擎一鍵部署支援 CUDA 硬體加速的 OpenAI 相容大模型 API 服務。

---

## 📄 檔案清單與用途

* **`llama_config.yaml`**：
  集中管理 `llama.cpp` 的啟動與編譯參數。可在本地修改此檔來更換模型 Repo、GGUF 檔案與上下文大小。
* **`serve_llama_cpp.py`**：
  遠端 VM 的一鍵啟動腳本。它會為您處理 CUDA 加速編譯、自動下載 HuggingFace 上的 GGUF 檔案、轉換 JSON 設定檔並在背景啟動服務。

---

## 🛠️ 部署指令

請在啟用虛擬環境（`source .venv/bin/activate`）後執行：

```bash
# 1. 建立 T4 GPU VM 執行期
colab new -s llama-api --gpu T4

# 2. 上傳您在本地修改好的設定檔
colab upload llama_cpp/llama_config.yaml /content/llama_config.yaml

# 3. 執行部署腳本
colab exec -s llama-api -f llama_cpp/serve_llama_cpp.py
```

---

## 💡 GGUF 量化與 GPU 加速設定指引

`llama.cpp` 執行的是 GGUF 格式的模型。相比原始精度模型，它能顯著降低顯存需求並保持優異的推論速度。

請在 `llama_config.yaml` 中注意以下核心設定：

1. **`n_gpu_layers: -1` (GPU Offloading)**：
   * 這是最重要的設定！`-1` 代表將模型的所有 Layer 全部載入到 GPU 顯存中運行。
   * 若顯存空間足夠（如 1.5B 或是 7B 的 4-bit 量化模型在 T4 16GB VRAM 內非常寬裕），請務必維持 `-1` 以免回退到 CPU 運算導致推論變慢。
2. **量化規格選擇 (`model_file`)**：
   * 在 T4 的 16GB 顯存限制下，建議優先選擇以 `q4_k_m` (4-bit 中等量化，推論最快且顯存佔用極低) 或是 `q8_0` (8-bit 量化，精度損失極小但顯存佔用較高) 結尾的 GGUF 檔案。
3. **`chat_format` (對話模板)**：
   * 當前預設為 `"qwen"`（適合 Qwen 系列模型）。如果您部署其他模型（如 Llama-3 或 Mistral），請記得將此項改為對應的模板（如 `"llama-3"` 或 `"chatml"`），以確保模型對話的指令遵循能力不會出錯。
