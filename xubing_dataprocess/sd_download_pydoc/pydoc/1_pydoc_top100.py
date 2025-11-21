import subprocess
import sys
import os

# 要处理的库名单（示例少数库，建议扩展至你全部 top100）
libs = [
    "numpy", "pandas", "matplotlib", "scipy", "scikit-learn", "tensorflow", "pytorch", "keras",
    "requests", "flask", "django", "seaborn", "plotly", "beautifulsoup4", "scrapy", "xgboost",
    "lightgbm", "dask", "sqlalchemy", "pillow", "opencv-python", "tensorboard", "plotnine",
    "statsmodels", "gensim", "nltk", "spacy", "fastapi", "uvicorn", "starlette", "jupyter",
    "notebook", "streamlit", "altair", "bokeh", "folium", "dash", "h5py", "pymongo",
    "sqlalchemy-core", "celery", "gunicorn", "pytest", "click", "rich", "typer", "pydantic",
    "loguru", "flask_sqlalchemy", "uvloop", "scikit-image", "scrapy-splash", "torchtext",
    "torchvision", "transformers", "datasets", "accelerate", "openai", "langchain", "ray",
    "fasttext", "mlflow"
]



def run_cmd(cmd):
    proc = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    return proc.returncode, proc.stdout, proc.stderr

def get_installed_version(lib):
    code, out, err = run_cmd(f"{sys.executable} -m pip show {lib}")
    if code != 0:
        return None
    for line in out.splitlines():
        if line.startswith("Version:"):
            return line.split(":",1)[1].strip()
    return None

def uninstall(lib):
    print(f"[→] Uninstalling {lib}")
    run_cmd(f"{sys.executable} -m pip uninstall -y {lib}")

def install_latest(lib):
    print(f"[+] Installing/upgrading {lib}")
    run_cmd(f"{sys.executable} -m pip install --upgrade {lib}")

def export_docs(lib):
    print(f"[★] Generating docs for {lib}")
    # 生成 HTML 文档至当前目录
    doc_lib = lib  # 保存原始库名用于文档命名
    
    # 包名和模块名映射（pip install 的包名 -> import/pydoc 的模块名）
    package_to_module = {
        "beautifulsoup4": "bs4",
        "pytorch": "torch",
        "opencv-python": "cv2",
        "scikit-learn": "sklearn",
        "scikit-image": "skimage",
        "pillow": "PIL",
    }
    
    # 使用模块名而不是包名
    module_name = package_to_module.get(lib, lib)
    
    # 检查目标目录是否已存在文档
    target_path = f"/pfs/training-data/xubingye/data/pydoc/{doc_lib}.html"
    if os.path.exists(target_path):
        print(f"  [Skip] {doc_lib}.html already exists")
        return
    
    code, out, err = run_cmd(f"{sys.executable} -m pydoc -w {module_name}")
    
    # 将生成的 HTML 文档移动到 /pfs/training-data/xubingye/data/pydoc 目录
    os.makedirs("/pfs/training-data/xubingye/data/pydoc", exist_ok=True)
    source_file = f"{module_name}.html"
    if os.path.exists(source_file):
        os.rename(source_file, target_path)
    else:
        print(f"  [WARNING] Failed to generate docs for {lib} (module: {module_name})")
        if err:
            print(f"  [ERROR] {err.strip()}")

def process_lib(lib):
    print(f"--- Processing {lib} ---")
    ver = get_installed_version(lib)
    if ver:
        print(f"Installed version of {lib}: {ver}")
        uninstall(lib)
    install_latest(lib)
    export_docs(lib)

def main():
    os.makedirs("/pfs/training-data/xubingye/data/pydoc", exist_ok=True)
    cwd = os.getcwd()
    for lib in sorted(set(libs)):
        try:
            process_lib(lib)
        except Exception as e:
            print(f"Error processing {lib}: {e}")
    print("Done.")

if __name__ == "__main__":
    main()