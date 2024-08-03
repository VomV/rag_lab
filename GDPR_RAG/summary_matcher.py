"""
This module defines a class for matching article summaries to a query using semantic similarity.
It utilizes the sentence transformer model to encode texts and compute similarities.
"""

from transformers import AutoTokenizer, AutoModel
import torch
from typing import Dict, List, Tuple
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SummaryMatcher:
    """
    A class for finding the top document names that match a query based on semantic similarity of their summaries.
    """

    def __init__(self, k: int  = 5, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", summaries_dict: Dict[str, str] = None):
        """
        Initializes the SummaryMatcher with a specific transformer model and preloads summaries.

        :param k: no of similar documents to be returned
        :param model_name: Name of the transformer model to use.
        :param summaries_dict: A dictionary mapping document names to their summaries.
        """
        self.k = k
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        if summaries_dict is None:
            raise ValueError("summaries_dict is required and cannot be None")
        self.doc_names, self.summaries, self.embeddings = self.load_and_encode_summaries(summaries_dict)

    def load_and_encode_summaries(self, summaries_dict: Dict[str, str]) -> Tuple[List[str], List[str], torch.Tensor]:
        """
        Encodes the summaries using the model and tokenizer.

        :param summaries_dict: A dictionary mapping document names to summaries.
        :return: A tuple of document names, summaries, and their corresponding embeddings.
        """
        doc_names = list(summaries_dict.keys())
        summaries = list(summaries_dict.values())
        with torch.no_grad():
            embeddings = self.encode_texts(summaries)
        return doc_names, summaries, embeddings

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Encodes a list of texts into embeddings.

        :param texts: A list of texts to encode.
        :return: Embeddings of the texts.
        """
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
        model_output = self.model(**encoded_input)
        return model_output.last_hidden_state.mean(dim=1)

    def compute_similarity(self, query: str) -> List[str]:
        """
        Finds the top k document names most similar to the query.

        :param query: The query text.
        :return: Names of the top k matching documents.
        """
        query_embedding = self.encode_texts([query])
        cos_sims = torch.nn.functional.cosine_similarity(query_embedding, self.embeddings)
        top_matches_indices = cos_sims.topk(self.k).indices
        return [self.doc_names[i] for i in top_matches_indices]



if __name__ == "__main__":


    meta_file_path = 'metadata.txt'
    article_summaries = {}

    with open(meta_file_path, 'r') as f:

        for line in f:
            if line.strip():
                key, value = line.strip().split(':', 1)
                key = key.split('**', 2)[1]
                article_summaries[key] = value

    matcher = SummaryMatcher(summaries_dict=article_summaries)
    top_doc_names = matcher.compute_similarity("What is the main purpose of GDPR?")
    print(top_doc_names)
