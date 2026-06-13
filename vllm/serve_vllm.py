import subprocess
import re
import sys
import time
import os
import json

def main():
    # 0. Cleanup dangling processes to prevent port conflicts
    print("Cleaning up existing vLLM and cloudflared processes...")
    try:
        subprocess.run("pkill -f vllm.entrypoints", shell=True, stderr=subprocess.DEVNULL)
        subprocess.run("pkill -f cloudflared", shell=True, stderr=subprocess.DEVNULL)
        time.sleep(2)
    except Exception:
        pass

    # 1. Download and configure cloudflared
    if not os.path.exists("cloudflared"):
        print("Downloading cloudflared for tunneling...")
        try:
            subprocess.run("curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared", shell=True, check=True)
            subprocess.run("chmod +x cloudflared", shell=True, check=True)
            print("cloudflared downloaded successfully.")
        except Exception as e:
            print(f"Failed to download cloudflared: {e}")
            sys.exit(1)

    # 2. Install dependencies on VM using uv
    print("Installing dependencies (vLLM, nest-asyncio, pyyaml) using uv on VM...")
    try:
        subprocess.run("pip install uv --quiet", shell=True, check=True)
        subprocess.run("uv pip install vllm nest-asyncio pyyaml 'transformers<4.46.0' --system --extra-index-url https://download.pytorch.org/whl/cu121 --quiet", shell=True, check=True)
        print("Dependencies installed successfully.")
    except Exception as e:
        print(f"Failed to install dependencies: {e}")
        sys.exit(1)

    # 2.5 Load YAML configurations
    config = {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "trust_remote_code": False,
        "host": "0.0.0.0",
        "port": 8000,
        "dtype": "half",
        "gpu_memory_utilization": 0.7,
        "max_model_len": 2048,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "api_key": None
    }

    yaml_path = "vllm_config.yaml"
    if os.path.exists(yaml_path):
        try:
            import yaml
            with open(yaml_path, "r", encoding="utf-8") as f:
                user_config = yaml.safe_load(f)
            if user_config and isinstance(user_config, dict):
                config.update(user_config)
                print(f"Loaded configuration from {yaml_path}")
        except Exception as e:
            print(f"Warning: Failed to parse {yaml_path}, using default config: {e}")
    else:
        print(f"No {yaml_path} found. Using default config.")

    # 3. Start vLLM OpenAI API server
    print(f"Starting vLLM server with {config['model']}...")
    cmd_args = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", config["model"],
        "--host", str(config.get("host", "0.0.0.0")),
        "--port", str(config.get("port", 8000))
    ]
    
    if config.get("trust_remote_code"):
        cmd_args.append("--trust-remote-code")
    if config.get("dtype"):
        cmd_args.extend(["--dtype", str(config["dtype"])])
    if config.get("gpu_memory_utilization") is not None:
        cmd_args.extend(["--gpu-memory-utilization", str(config["gpu_memory_utilization"])])
    if config.get("max_model_len") is not None:
        cmd_args.extend(["--max-model-len", str(config["max_model_len"])])
    if config.get("tensor_parallel_size") is not None:
        cmd_args.extend(["--tensor-parallel-size", str(config["tensor_parallel_size"])])
    if config.get("pipeline_parallel_size") is not None:
        cmd_args.extend(["--pipeline-parallel-size", str(config["pipeline_parallel_size"])])
    if config.get("api_key"):
        cmd_args.extend(["--api-key", str(config["api_key"])])

    vllm_proc = subprocess.Popen(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # 4. Start Cloudflare Tunnel
    print("Starting Cloudflare tunnel...")
    cf_cmd = f"./cloudflared tunnel --url http://localhost:{config['port']}"
    cf_proc = subprocess.Popen(cf_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # 5. Capture Tunnel URL
    tunnel_url = None
    start_time = time.time()
    print("Waiting for Cloudflare tunnel URL (timeout: 60s)...")
    while time.time() - start_time < 60:
        if vllm_proc.poll() is not None:
            print("Error: vLLM server crashed on startup.")
            break
        if cf_proc.poll() is not None:
            print("Error: Cloudflare tunnel crashed on startup.")
            break

        line = cf_proc.stderr.readline()
        if line:
            if "trycloudflare.com" in line:
                match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
                if match:
                    tunnel_url = match.group(0)
                    break
        time.sleep(0.1)

    if not tunnel_url:
        print("Error: Could not detect Cloudflare tunnel URL.")
        vllm_proc.terminate()
        cf_proc.terminate()
        sys.exit(1)

    # 6. Wait for vLLM server to initialize
    print("Waiting for vLLM API server to initialize (this may take up to 240s for weights loading and CUDA graph capture)...")
    vllm_started = False
    vllm_start_monitor_time = time.time()
    
    try:
        os.set_blocking(vllm_proc.stdout.fileno(), False)
    except Exception:
        pass

    while time.time() - vllm_start_monitor_time < 240:
        if vllm_proc.poll() is not None:
            print("Error: vLLM server crashed during initialization.")
            break

        try:
            import urllib.request
            with urllib.request.urlopen(f"http://localhost:{config['port']}/v1/models", timeout=2) as response:
                if response.status == 200:
                    vllm_started = True
                    break
        except Exception:
            pass
        time.sleep(2)

    try:
        os.set_blocking(vllm_proc.stdout.fileno(), True)
    except Exception:
        pass

    if vllm_started:
        import threading
        
        # Output reader that filters API requests to keep terminal log neat and tidy
        def pipe_reader(pipe, prefix):
            try:
                for line in iter(pipe.readline, ''):
                    if line:
                        line_str = line.strip()
                        if "HTTP/" in line_str or "error" in line_str.lower() or "ERR" in line_str:
                            print(f"{prefix} {line_str}")
                            sys.stdout.flush()
            except Exception:
                pass
                
        t1 = threading.Thread(target=pipe_reader, args=(vllm_proc.stdout, "[vLLM]"), daemon=True)
        t1.start()
        
        t2 = threading.Thread(target=pipe_reader, args=(cf_proc.stderr, "[cloudflare]"), daemon=True)
        t2.start()

        try:
            print("\n" + "="*60)
            print("🎉 vLLM Server & Cloudflare Tunnel started successfully!")
            print(f"🔗 OpenAI Base URL: {tunnel_url}/v1")
            print(f"👉 Model name: {config['model']}")
            print("💡 Keep this terminal window open to maintain the connection and keep the VM alive.")
            print("💡 Press Ctrl+C to stop the server.")
            print("="*60 + "\n")
            
            while True:
                if vllm_proc.poll() is not None:
                    print(f"\n❌ vLLM server stopped with exit code {vllm_proc.poll()}")
                    break
                if cf_proc.poll() is not None:
                    print(f"\n❌ Cloudflare tunnel stopped with exit code {cf_proc.poll()}")
                    break
                
                print(f"[Keep-Alive] Server is healthy. Base URL: {tunnel_url}/v1")
                sys.stdout.flush()
                time.sleep(15)
        except KeyboardInterrupt:
            print("\nShutting down server and tunnel...")
        finally:
            vllm_proc.terminate()
            cf_proc.terminate()
            vllm_proc.wait()
            cf_proc.wait()
            print("Server and tunnel stopped.")
            sys.exit(0)
    else:
        print("Error: Failed to start server or tunnel.")
        vllm_proc.terminate()
        cf_proc.terminate()
        sys.exit(1)

if __name__ == "__main__":
    main()
