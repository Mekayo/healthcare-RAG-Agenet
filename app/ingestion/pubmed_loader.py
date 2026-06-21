import requests
from bs4 import BeautifulSoup
from typing import List
from langchain_core.documents import Document
url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

def search_pubmed(query:str,max_results:int=100):
    preams = {
        "db":"pubmed",
        "term":f"{query}",
        "retmax":max_results,
        "retmode":"json"
    }
    response = requests.get(url, params=preams)
    
    data=response.json()
    return data["esearchresult"]["idlist"]


def fetch_article_details(pmids: list[str]):

    ids = ",".join(pmids)
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    params = {
        "db": "pubmed",
        "id": ids,
        "retmode": "xml"
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
        )
    
    response.raise_for_status()  # Raises exception for HTTP errors

    soup = BeautifulSoup(response.content, "xml")

    articles = soup.find_all("PubmedArticle")

    return articles

def parse_article(articles, query):

    docs = []

    for article in articles:

        pmid_tag = article.find("PMID")
        title_tag = article.find("ArticleTitle")
        abstract_tag = article.find("AbstractText")

        pmid = pmid_tag.text if pmid_tag else ""
        title = title_tag.text if title_tag else ""
        abstract = abstract_tag.text if abstract_tag else ""

        docs.append(
            Document(
                page_content=f"""
                Title: {title}
                Abstract: 
                {abstract}
                """,
                metadata={
                    "pmid": pmid,
                    "title": title,
                    "query": query,
                    "source": "pubmed",
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                }
            )
        )

    return docs

def load_pubmed_documents(query:str, max_results:int=100):
    
    pmids = search_pubmed(
        query,
        max_results
    )

    articles = fetch_article_details(
        pmids
    )

    docs = parse_article(
        articles,
        query
    )

    return docs
