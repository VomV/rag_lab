import os
import io
import sys
import warnings
from dotenv import load_dotenv, find_dotenv
from typing import Dict, List

from llama_index.core import ServiceContext, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.vector_stores import MetadataFilter

from trulens_eval import Feedback, TruLlama, OpenAI
from trulens_eval.feedback import Groundedness

import numpy as np
import configparser

warnings.filterwarnings("ignore")

# Load settings from config file
config = configparser.ConfigParser()
config.read('config.ini')

# Global settings from config file
RERANK_MODEL = config['Settings']['RERANK_MODEL']
SIMILARITY_TOP_K = config.getint('Settings', 'SIMILARITY_TOP_K')
RERANK_TOP_N = config.getint('Settings', 'RERANK_TOP_N')
EMBED_MODEL = config['Settings']['EMBED_MODEL']
INDEX_DIR = config['Settings']['INDEX_DIR']


def suppress_print(func):
    """
    Decorator to suppress printing to the console within a function.
    
    Args:
        func: The function to wrap.
        
    Returns:
        Wrapped function that suppresses print statements.
    """
    def wrapper(*args, **kwargs):
        # Redirect stdout to an in-memory stream
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            # Execute the function with printing suppressed
            result = func(*args, **kwargs)
        finally:
            # Restore the original stdout
            sys.stdout = original_stdout
        return result
    return wrapper



def load_article_summaries(meta_file_path: str) -> Dict[str, str]:
    """
    Loads article summaries from a specified metadata text file.

    Parameters:
        meta_file_path: The path to the metadata file.

    Returns:
        A dictionary mapping article identifiers to summaries.
    """
    article_summaries = {}
    with open(meta_file_path, 'r') as f:
        for line in f:
            if line.strip():
                key, value = line.strip().split(':', 1)
                article_summaries[key.split('**', 2)[1]] = value
    return article_summaries

def get_openai_api_key() -> str:
    """
    Loads the OpenAI API key from the environment, prompting for it if not found.

    Returns:
        The OpenAI API key.
    """
    _ = load_dotenv(find_dotenv())
    return os.getenv("OPENAI_API_KEY")

def build_automerging_index(documents: List[Dict], llm: OpenAI, embed_model: str = "local:BAAI/bge-small-en-v1.5", 
                            save_dir: str = "merging_index", chunk_sizes: List[int] = None) -> VectorStoreIndex:
    """
    Builds or loads an automerging index for the provided documents.

    Parameters:
        documents: A list of document dictionaries to index.
        llm: An instance of the OpenAI model.
        embed_model: The embedding model identifier.
        save_dir: The directory where the index should be saved or loaded from.
        chunk_sizes: A list defining chunk sizes for hierarchical parsing.

    Returns:
        An automerging index.
    """
    chunk_sizes = chunk_sizes or [2048, 512, 128]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    
    merging_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
    storage_context = StorageContext.from_defaults()
    
    storage_context.docstore.add_documents(nodes)

    if not os.path.exists(save_dir):
        automerging_index = VectorStoreIndex(leaf_nodes, storage_context=storage_context, service_context=merging_context)
        automerging_index.storage_context.persist(persist_dir=save_dir)
    else:
        automerging_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=save_dir), service_context=merging_context)

    return automerging_index

def load_automerging_index(llm: OpenAI = None, embed_model: str = EMBED_MODEL, save_dir: str = "merging_index") -> VectorStoreIndex:
    """
    Loads an existing automerging index from storage.

    Parameters:
        llm: An instance of the OpenAI model.
        embed_model: The embedding model identifier.
        save_dir: The directory where the index is saved.

    Returns:
        An automerging index.
    """
    merging_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
    return load_index_from_storage(StorageContext.from_defaults(persist_dir=save_dir), service_context=merging_context)

def get_automerging_query_engine(automerging_index: VectorStoreIndex, relevant_articles: List[str], similarity_top_k: int = SIMILARITY_TOP_K, 
                                 rerank_top_n: int = RERANK_TOP_N) -> RetrieverQueryEngine:
    """
    Configures and returns a query engine for an automerging index.

    Parameters:
        automerging_index: The automerging index to query against.
        relevant_articles: A list of titles of relevant articles to filter by.
        similarity_top_k: The number of top similar results to retrieve.
        rerank_top_n: The number of top results to rerank.

    Returns:
        A configured query engine.
    """

    #Filter based on metadata
    node_ids = [n.node_id for n in automerging_index.docstore.docs.values() if n.metadata['article_number'] in relevant_articles]

    base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(base_retriever, automerging_index.storage_context, verbose=True)

    rerank = SentenceTransformerRerank(top_n=rerank_top_n, model=RERANK_MODEL)
    
    auto_merging_engine = RetrieverQueryEngine.from_args(retriever, node_ids=node_ids, node_postprocessors=[rerank])

    return auto_merging_engine

@suppress_print
def configure_feedback(openai_model: OpenAI) -> list:
    """
    Configure feedback mechanisms for evaluating the relevance and groundedness
    of answers generated by a model.

    Parameters:
        openai_model: An instance of the OpenAI model class.

    Returns:
        A list of configured feedback mechanisms.
    """
    # Configuring feedback for answer relevance using Chain of Thought (CoT) reasons.
    qa_relevance = Feedback(
        openai_model.relevance_with_cot_reasons,
        name="Answer Relevance"
    ).on_input_output()

    # Configuring feedback for context relevance with aggregation.
    qs_relevance = Feedback(
        openai_model.relevance_with_cot_reasons,
        name="Context Relevance"
    ).on_input().on(TruLlama.select_source_nodes().node.text).aggregate(np.mean)

    # Configuring feedback for groundedness measurement.
    grounded = Groundedness(groundedness_provider=openai_model)
    groundedness = Feedback(
        grounded.groundedness_measure_with_cot_reasons,
        name="Groundedness"
    ).on(TruLlama.select_source_nodes().node.text).on_output().aggregate(grounded.grounded_statements_aggregator)

    # Collecting all feedback mechanisms into a list.
    feedbacks = [qa_relevance, qs_relevance, groundedness]
    
    return feedbacks

@suppress_print
def get_prebuilt_trulens_recorder(query_engine, app_id: str) -> TruLlama:
    """
    Create a TruLlama recorder instance with the specified query engine,
    feedback mechanisms, and application ID.

    Parameters:
        query_engine: The query engine to be used with TruLlama.
        app_id: A unique identifier for the application using TruLlama.

    Returns:
        An instance of TruLlama configured with the given parameters.
    """

    openai_model = OpenAI()  
    feedbacks = configure_feedback(openai_model)
    tru_recorder = TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=feedbacks
    )
    return tru_recorder




if __name__ == "__main__":

    pass