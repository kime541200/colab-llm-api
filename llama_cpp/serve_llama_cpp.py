import subprocess
import re
import sys
import time
import os
import json

def main():
    # 0. Cleanup dangling processes to prevent port conflicts
    print("Cleaning up existing llama.cpp and cloudflared processes...")
    try:
        subprocess.run("pkill -f llama_cpp.server", shell=True, stderr=subprocess.DEVNULL)
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

    # 2. Install dependencies on VM
    print("Installing pre-built CUDA-enabled llama-cpp-python[server] on VM...")
    try:
        env_cmd = (
            'pip install "llama-cpp-python[server]" pyyaml huggingface_hub '
            '--prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121'
        )
        subprocess.run(env_cmd, shell=True, check=True)
        print("Dependencies installed successfully.")
    except Exception as e:
        print(f"Failed to install dependencies: {e}")
        sys.exit(1)

    # 2.5 Load YAML configurations
    config = {
        "repo_id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "model_file": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "model_alias": "qwen2.5-1.5b",
        "host": "0.0.0.0",
        "port": 8000,
        "api_key": None,
        "n_gpu_layers": -1,
        "n_ctx": 2048,
        "n_batch": 512,
        "chat_format": None
    }

    yaml_path = "llama_config.yaml"
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

    # 2.8 Download model file
    print(f"Downloading GGUF model: {config['repo_id']}/{config['model_file']}...")
    try:
        dl_cmd = f"hf download {config['repo_id']} {config['model_file']} --local-dir ."
        subprocess.run(dl_cmd, shell=True, check=True)
        print("Model downloaded successfully.")
    except Exception as e:
        print(f"Failed to download model: {e}")
        sys.exit(1)

    # 2.9 Generate configuration JSON for server startup
    model_settings = {
        "model": config.get("model_file"),
        "model_alias": config.get("model_alias"),
        "n_gpu_layers": config.get("n_gpu_layers", -1),
        "n_ctx": config.get("n_ctx", 2048),
        "n_batch": config.get("n_batch", 512)
    }

    if config.get("clip_model_path"):
        model_settings["clip_model_path"] = config["clip_model_path"]
    if config.get("flash_attn") is not None:
        model_settings["flash_attn"] = config["flash_attn"]
    if config.get("chat_format"):
        model_settings["chat_format"] = config["chat_format"]

    llama_config = {
        "host": config.get("host", "0.0.0.0"),
        "port": config.get("port", 8000),
        "api_key": config.get("api_key"),
        "models": [model_settings]
    }
    
    with open("llama_config.json", "w", encoding="utf-8") as f:
        json.dump(llama_config, f, indent=4)
    print("Generated llama_config.json for server.")

    # 3. Start llama.cpp API server
    print(f"Starting llama.cpp server with {config['model_alias']}...")
    server_cmd = ["python3", "-m", "llama_cpp.server", "--config_file", "llama_config.json"]
    server_proc = subprocess.Popen(server_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # 4. Start Cloudflare Tunnel
    print("Starting Cloudflare tunnel...")
    cf_cmd = f"./cloudflared tunnel --url http://localhost:{config['port']}"
    cf_proc = subprocess.Popen(cf_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # 5. Capture Tunnel URL
    tunnel_url = None
    start_time = time.time()
    print("Waiting for Cloudflare tunnel URL (timeout: 60s)...")
    while time.time() - start_time < 60:
        if server_proc.poll() is not None:
            print("Error: llama.cpp server crashed on startup.")
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
        server_proc.terminate()
        cf_proc.terminate()
        sys.exit(1)

    # 6. Wait for llama.cpp server to initialize
    print("Waiting for llama.cpp API server to initialize...")
    server_started = False
    server_start_monitor_time = time.time()
    
    try:
        os.set_blocking(server_proc.stdout.fileno(), False)
    except Exception:
        pass

    while time.time() - server_start_monitor_time < 120:
        if server_proc.poll() is not None:
            print("Error: llama.cpp server crashed during initialization.")
            break

        try:
            import urllib.request
            with urllib.request.urlopen(f"http://localhost:{config['port']}/v1/models", timeout=2) as response:
                if response.status == 200:
                    server_started = True
                    break
        except Exception:
            pass
        time.sleep(2)

    try:
        os.set_blocking(server_proc.stdout.fileno(), True)
    except Exception:
        pass

    if server_started:
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
                
        t1 = threading.Thread(target=pipe_reader, args=(server_proc.stdout, "[llama.cpp]"), daemon=True)
        t1.start()
        
        t2 = threading.Thread(target=pipe_reader, args=(cf_proc.stderr, "[cloudflare]"), daemon=True)
        t2.start()

        try:
            print("\n" + "="*60)
            print("🎉 Llama.cpp Server & Cloudflare Tunnel started successfully!")
            print(f"🔗 OpenAI Base URL: {tunnel_url}/v1")
            print(f"👉 Model name: {config['model_alias']}")
            print("💡 Keep this terminal window open to maintain the connection and keep the VM alive.")
            print("💡 Press Ctrl+C to stop the server.")
            print("="*60 + "\n")
            
            while True:
                if server_proc.poll() is not None:
                    print(f"\n❌ llama.cpp server stopped with exit code {server_proc.poll()}")
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
            server_proc.terminate()
            cf_proc.terminate()
            server_proc.wait()
            cf_proc.wait()
            print("Server and tunnel stopped.")
            sys.exit(0)
    else:
        print("Error: Failed to start server or tunnel.")
        server_proc.terminate()
        cf_proc.terminate()
        sys.exit(1)

if __name__ == "__main__":
    main()
