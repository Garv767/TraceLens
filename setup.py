from setuptools import setup, find_packages

setup(
    name="tracelens",
    version="0.1.0",
    description="TraceLens: Code Retrieval Intelligence Engine",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "mteb",
        "sentence-transformers",
        "faiss-cpu",
        "streamlit",
        "fastapi",
        "uvicorn",
        "rank-bm25",
    ],
)
