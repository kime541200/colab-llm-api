# Agent Guidelines & Deployment Manual

## ⚠️ Critical Setup Rule
Before running any `colab` CLI command, the virtual environment MUST be activated first using:
```bash
source .venv/bin/activate
```

---

## 📂 Project Structure & Module Directory
- `llama_cpp/`: Contains `serve_llama_cpp.py` (deployment script) and `llama_config.yaml` (T4 GPU optimized config).
- `vllm/`: Contains `serve_vllm.py` (deployment script) and `vllm_config.yaml` (T4 GPU optimized config).

---

## ⚡ Deployment & Session Management Commands

### 1. Initialize Session
Create a new remote execution runtime with GPU acceleration (supports T4, L4, A100, etc.):
```bash
colab new -s llama-api --gpu T4
colab new -s vllm-api --gpu T4
```

### 2. Upload Configuration File
Upload the T4-optimized configuration to the remote server root:
- **Llama.cpp**:
  ```bash
  colab upload llama_cpp/llama_config.yaml /content/llama_config.yaml
  ```
- **vLLM**:
  ```bash
  colab upload vllm/vllm_config.yaml /content/vllm_config.yaml
  ```

### 3. Launch Serving Script (With Timeout)
Deploy and run the LLM server in the foreground:
- **Llama.cpp**:
  ```bash
  colab exec -s llama-api --timeout 600 -f llama_cpp/serve_llama_cpp.py
  ```
- **vLLM**:
  ```bash
  colab exec -s vllm-api --timeout 600 -f vllm/serve_vllm.py
  ```

### 4. Status Check
Check the status of an active session:
```bash
colab status -s llama-api
colab status -s vllm-api
```

### 5. Terminate and Clean Session
Always clean up and stop the session when finished to prevent resource waste:
```bash
colab stop -s llama-api
colab stop -s vllm-api
```

---

## 🛑 Critical Platform Constraints & Connection Keep-Alive

### KeepAlive 403 Forbidden Issue
- **Root Cause**: The background KeepAlive mechanism of `colab-cli` fails with a `403 Forbidden` error due to service project permission constraints in the user's account environment.

### Prevention of VM Reclamation
- **Foreground Requirement**: To keep the Google Colab VM alive, the serving scripts (`serve_llama_cpp.py` and `serve_vllm.py`) **MUST remain running in the foreground (infinite loop)**.
- **WebSocket Activity**: The scripts output `[Keep-Alive] Server is healthy...` every 15 seconds to generate activity on the WebSocket connection.
- **Constraint**: Do **NOT** exit the foreground script or run it as a background daemon (via daemonization). If the connection is lost, the Colab VM will be reclaimed within 60 seconds.

---

## 📝 Terminal Logs & Output Rules
- Both serving scripts automatically filter terminal output to display only API requests (containing `HTTP/`) and errors.
- Do not modify or disable this filtering behavior to keep the terminal logs readable and prevent connection drops.