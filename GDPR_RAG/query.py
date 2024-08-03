"""
This script allows a user to input their queries through console, compares it to article summaries to find
relevant articles, retrieves the appropirate chunks from filtered articles and synthesizes a response
"""


import os
import warnings
from summary_matcher import SummaryMatcher
from llama_index.llms.openai import OpenAI
from utils import get_automerging_query_engine, load_automerging_index, get_prebuilt_trulens_recorder
import configparser
from trulens_eval import Tru

tru = Tru()
tru.reset_database()

warnings.filterwarnings("ignore")

# Load settings from config file
config = configparser.ConfigParser()
config.read('config.ini')

# Global settings from config file
K = config.getint('Settings', 'K')
OPENAI_MODEL = config['Settings']['OPENAI_MODEL']
METADATA_PATH = config['Settings']['METADATA_PATH']
TEMPERATURE = config.getfloat('Settings', 'TEMPERATURE')
EMBED_MODEL = config['Settings']['EMBED_MODEL']
INDEX_DIR = config['Settings']['INDEX_DIR']

# Ensures the OpenAI API key is set in the environment variables.
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = input("\n\nPlease enter your OpenAI API Key: ")

def load_article_summaries(meta_file_path: str) -> dict:
    """
    Load article summaries from the specified metadata text file.

    Args:
        meta_file_path: Path to the metadata file containing article summaries.

    Returns:
        A dictionary mapping article keys to their summaries.
    """
    article_summaries = {}
    with open(meta_file_path, 'r') as file:
        for line in file:
            if line.strip():  # Check for non-empty lines
                key, value = line.strip().split(':', 1)
                # Extract the key after '**' and assign its value
                article_summaries[key.split('**', 2)[1]] = value
    return article_summaries


def main():
    """Main function to process user queries and display similar articles."""
    # Load article summaries from a file.
    article_summaries = load_article_summaries(METADATA_PATH)
    
    # Initialize the summary matcher with the loaded summaries.
    matcher = SummaryMatcher(k=K, summaries_dict=article_summaries)

    # Continuously process user queries until 'exit' is entered.
    while True:
        user_query = input("\n\nEnter your query (or type 'exit' to quit): ").lower()
        if user_query == 'exit':
            break

        # Find top matching document names based on the user query.
        top_doc_names = matcher.compute_similarity(user_query)
        print("\nTop similar articles:")
        for doc_name in top_doc_names:
            print(doc_name)

        # Load the automerging index and query engine for advanced processing.
        llm = OpenAI(model=OPENAI_MODEL, temperature=TEMPERATURE)
        automerging_index = load_automerging_index(llm=llm, embed_model=EMBED_MODEL, save_dir=INDEX_DIR)
        automerging_query_engine = get_automerging_query_engine(automerging_index, top_doc_names)
        
        # Query the engine and display the response.
        tru_recorder_automerging = get_prebuilt_trulens_recorder(automerging_query_engine,
                                                                app_id="Automerging Query Engine")
        with tru_recorder_automerging as recording:
            auto_merging_response = automerging_query_engine.query(user_query)

        print("\n"+"#"*100)
        print('\nSource Article:::')
        print('Title:', auto_merging_response.source_nodes[0].metadata['article_number'])
        print('Summary:', auto_merging_response.source_nodes[0].metadata['article_summary'])

        print("\n"+"#"*100)
        print("\nResponse:", str(auto_merging_response))
        
        print("\n"+"#"*100)
        rec = recording.get()
        print("\nEvaluation:::")
        for feedback, feedback_result in rec.wait_for_feedback_results().items():
            print(feedback.name, feedback_result.result)

if __name__ == "__main__":

    main()