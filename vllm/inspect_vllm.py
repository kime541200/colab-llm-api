import os

# 1. 檢查 vllm 內部關鍵路徑的檔案
paths = [
    "/usr/local/lib/python3.12/dist-packages/vllm/utils/__init__.py",
    "/usr/local/lib/python3.12/dist-packages/vllm/inputs/parse.py"
]

for p in paths:
    print(f"=== {p} ===")
    if os.path.exists(p):
        with open(p) as f:
            lines = f.readlines()
            # 印出前 30 行
            for line in lines[:30]:
                print(line, end="")
    else:
        print("File not found")
    print()
