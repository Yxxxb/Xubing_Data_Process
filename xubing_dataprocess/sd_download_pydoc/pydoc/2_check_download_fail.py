import os

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

unavailable_libs = []
# 检查/pfs/training-data/xubingye/data/pydoc下不存在于libs的库
for lib in libs:
    if not os.path.exists(f"/pfs/training-data/xubingye/data/pydoc/{lib}.html"):
        unavailable_libs.append(lib)
print(unavailable_libs)

# ['sqlalchemy-core', 'scrapy-splash', 'torchtext', 'transformers']