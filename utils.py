# -*- coding: utf-8 -*-
import json
from gensim.corpora import Dictionary


def load_arxiv_metadata(path):
    """Load arXiv metadata."""
    with open(path, "r") as fp:
        for line in fp:
            yield line


def extract_abstracts(metadata,
                      categories=["cs.AI", "cs.GT", "cs.CV", "cs.IR",
                                  "cs.LG", "cs.MA", "cs.NE", "stat.ML",
                                  "stat.ME", "stat.CO", "stat.TH"]):
    """Extract paper abstracts from arXiv metadata by category."""
    abstracts = []
    for item in metadata:
        paper = json.loads(item)
        for category in categories:
            if category in paper["categories"]:
                abstracts.append(str(paper["abstract"]))
                break
    return abstracts


def export_documents(documents, filepath="documents.txt"):
    """Export documents to txt."""
    with open(filepath, "w") as fp:
        fp.write(json.dumps(documents))


def load_documents(filepath="documents.txt"):
    """Load documents from txt."""
    with open(filepath, "r") as fp:
        documents = json.loads(fp.read())
        return documents


def build_corpus(documents, idx2word):
    """Build term-document frequency corpus."""
    return [idx2word.doc2bow(doc) for doc in documents]
