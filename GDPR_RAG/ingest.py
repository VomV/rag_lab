"""
This script processes articles from a directory and their summaries from a metadata file,
then creates an automerging index for querying with an LLM.
"""

import re
from typing import Dict, List
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.llms.openai import OpenAI
from utils import build_automerging_index, load_article_summaries
import configparser


# Load settings from config file
config = configparser.ConfigParser()
config.read('config.ini')

# Global settings from config file
OPENAI_MODEL = config['Settings']['OPENAI_MODEL']
METADATA_PATH = config['Settings']['METADATA_PATH']
TEMPERATURE = config.getfloat('Settings', 'TEMPERATURE')
EMBED_MODEL = config['Settings']['EMBED_MODEL']
INDEX_DIR = config['Settings']['INDEX_DIR']
DATA_DIR = config['Settings']['DATA_DIR']

def enrich_documents_with_summaries(documents: List[Document], article_summaries: Dict[str, str]):
    """
    Enrich documents with metadata extracted from article summaries.

    Parameters:
        documents (List[Document]): A list of Document objects.
        article_summaries (Dict[str, str]): Article summaries.
    """
    for doc in documents:
        article_no = doc.metadata['file_name'].split('.')[0]
        pattern = re.compile(f'{article_no} -')
        extracted_values = [(key, value) for key, value in article_summaries.items() if pattern.search(key)]
        if extracted_values:
            article_number, article_summary = extracted_values[0]
            doc.metadata = {"article_number": article_number, "article_summary": article_summary}

def main(folder_path: str, meta_file_path: str, model: str, embed_model: str, save_dir: str):
    """
    Main function to process documents and create an automerging index.

    Parameters:
        folder_path (str): Directory containing article files.
        meta_file_path (str): Metadata file path.
        model (str): Model name for OpenAI GPT.
        embed_model (str): Embedding model name.
        save_dir (str): Directory to save the automerging index.
    """
    documents = SimpleDirectoryReader(input_dir=folder_path).load_data()
    article_summaries = load_article_summaries(meta_file_path)
    llm = OpenAI(model=model, temperature=TEMPERATURE)
    enrich_documents_with_summaries(documents, article_summaries)
    build_automerging_index(documents, llm, embed_model=embed_model, save_dir=save_dir)

if __name__ == "__main__":

    main(DATA_DIR, METADATA_PATH, OPENAI_MODEL, EMBED_MODEL, INDEX_DIR)
