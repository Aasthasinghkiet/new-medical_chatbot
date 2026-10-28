
from setuptools import find_packages, setup

setup(
    name="medical_chatbot",
    version="0.1.0",
    author="Aastha Singh",
    author_email="aasthasingh12115@gmail.com",
    description="A medical question-answering chatbot using RAG, LangChain, Gemini, and Pinecone",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10,<3.13",
    install_requires=[
        "langchain==0.2.14",
        "langchain-core==0.2.36",
        "langchain-community==0.2.12",
        "langchain-openai==0.1.8",
        "langchain-pinecone==0.1.3",
        "langchain-groq==0.1.4",
        "langchain-text-splitters==0.2.2",
        "langchain-huggingface==0.0.3",
        "langchain-google-genai==1.0.10",
        "pinecone-client==5.0.1",
        "flask==3.0.3",
        "python-dotenv==1.0.1",
        "sentence-transformers",
        "pypdf",
        "pandas",
        "python-docx",
        "openpyxl",
    ],
)
