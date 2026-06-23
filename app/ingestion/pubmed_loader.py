import json
from pathlib import Path
from typing import List

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document


# ==========================================================
# CONFIG
# ==========================================================

PUBMED_SEARCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
)

PUBMED_FETCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
)


# ==========================================================
# SEARCH
# ==========================================================

def search_pubmed(query: str, max_results: int = 200) -> List[str]:

    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json"
    }

    response = requests.get(
        PUBMED_SEARCH_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    return data["esearchresult"]["idlist"]


# ==========================================================
# FETCH
# ==========================================================

def fetch_article_details(pmids: list[str]):

    ids = ",".join(pmids)

    params = {
        "db": "pubmed",
        "id": ids,
        "retmode": "xml"
    }

    response = requests.get(
        PUBMED_FETCH_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.content,
        "xml"
    )

    return soup.find_all("PubmedArticle")


# ==========================================================
# PARSE
# ==========================================================

def parse_articles(articles, query: str ) -> list[Document]:

    docs = []
    
    for article in articles:

        pmid_tag = article.find("PMID")
        title_tag = article.find("ArticleTitle")
        abstract_tag = article.find("AbstractText")

        pmid = pmid_tag.text if pmid_tag else ""
        title = title_tag.text if title_tag else ""
        abstract = abstract_tag.text if abstract_tag else ""

        if not abstract:
            continue

        docs.append(
            Document(
            page_content=f"""
        Title: {title}

        Abstract:{abstract} """,    
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


# ==========================================================
# SAVE
# ==========================================================

def save_documents(docs: list[Document], filepath: str):

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    data = []

    for doc in docs:

        data.append(
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
        )

    with open(filepath,"w",
        encoding="utf-8" ) as f:

        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"Saved {len(data)} documents to {filepath}"
    )


# ==========================================================
# ORCHESTRATOR
# ==========================================================

def load_pubmed_documents(query: str, max_results: int = 300):

    pmids = search_pubmed(
        query=query,
        max_results=max_results
    )

    articles = fetch_article_details(
        pmids
    )

    docs = parse_articles(
        articles,
        query
    )

    return docs
