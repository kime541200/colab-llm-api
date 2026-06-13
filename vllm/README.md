# vLLM API Server Deployer

本模組用於在 Google Colab 上，透過 **vLLM** 引擎一鍵部署高效能、高吞吐量的 OpenAI 相容大模型 API 服務。

---

## 📄 檔案清單與用途

* **`vllm_config.yaml`**：
  集中管理 vLLM 的啟動與優化參數。修改此檔即可更換模型、調整埠號或調整顯存配置。
* **`serve_vllm.py`**：
  遠端 VM 的一鍵啟動 daemon 腳本。它會自動使用 `uv` 下載相容 CUDA 12.1 的 vLLM，並在背景運行 API server 與 Cloudflare 隧道。
* **`inspect_vllm.py`**：
  輔助診斷腳本，可用於在 VM 端檢查當前的 GPU 分配與顯存狀態。

---

## 🛠️ 部署指令

請在啟用虛擬環境（`source .venv/bin/activate`）後執行：

```bash
# 1. 建立 T4 GPU VM 執行期
colab new -s vllm-api --gpu T4

# 2. 上傳您在本地修改好的設定檔
colab upload vllm/vllm_config.yaml /content/vllm_config.yaml

# 3. 執行部署腳本
colab exec -s vllm-api -f vllm/serve_vllm.py
```

---

## 💡 Tesla T4 顯存 (VRAM) 調校指引

Google Colab 免費提供的 **Tesla T4 GPU 僅有 16GB 顯存**。vLLM 預設會佔用 90% 以上的顯存來做 KV Cache，這極易導致遠端 VM 的 Jupyter 核心崩潰（OOM）。

為確保 T4 的穩定性，請在 `vllm_config.yaml` 中注意以下配置：

1. **`dtype: "half"`**：
   T4 GPU (Turing 架構) 不支援 `bfloat16` 原生硬體加速，請務必維持 `"half"` (float16) 以免速度大幅下降。
2. **`gpu_memory_utilization: 0.7`**：
   將 vLLM 的顯存佔用率限制在 70%，留出 30% 空間給 PyTorch 與 VM 的系統開銷，以防核心 OOM 崩潰。
3. **`max_model_len: 2048`**：
   適度縮小上下文窗口至 `2048` 標記（Tokens），這會極大地減少 KV Cache 佔用的顯存大小。若要跑 7B 等較大模型，甚至可以下調至 `1024`。
