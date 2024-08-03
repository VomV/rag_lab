"""
This script reads the GDPR pdf and splits it into separate txt files for each Article in the pdf
"""

import fitz  # PyMuPDF
import re

pdf_path = "data/GDPR Art 1-21.pdf"

def save_article(article_number: str, article_text: str) -> str:
    """
    Saves the extracted article text to a text file.

    Args:
        article_number: The article number as a string.
        article_text: The text content of the article.

    Returns:
        The filename where the article text was saved.
    """
    filename = f"data/articles/Article {article_number}.txt"
    with open(filename, 'w') as file:
        file.write(article_text)
    return filename

def split_gdpr_articles(pdf_path: str) -> list:
    """
    Splits the GDPR PDF document into separate articles and saves them as text files.

    Args:
        pdf_path: The path to the GDPR PDF file.

    Returns:
        A list of filenames for the saved article text files.
    """
    doc = fitz.open(pdf_path)
    article_pattern = re.compile(r'EN\nArticle \d+\.')

    current_article = None
    current_article_number = None
    saved_files = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()

        for match in re.finditer(article_pattern, text):
            if current_article is not None:
                filename = save_article(current_article_number, current_article)
                saved_files.append(filename)

            # Prepare for the next article
            current_article = ""
            current_article_number = text[match.start():match.end()].split()[2].replace('.', '')

        if current_article is not None:
            current_article += text

    if current_article is not None:
        filename = save_article(current_article_number, current_article)
        saved_files.append(filename)

    return saved_files

if __name__ == "__main__":

    saved_article_files = split_gdpr_articles(pdf_path)
    print(saved_article_files)
