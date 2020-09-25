# -*- coding: utf-8 -*-
import json
import pickle
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


def export_object(obj, filepath):
    """Export Python object."""
    pickle.dump(obj, open(filepath, "wb"))


def load_object(filepath):
    """Load saved Python object."""
    obj = pickle.load(open(filepath, "rb"))
    return obj


def create_directory_if_not_exists(dir_path):
    """Create directory if it does not exist."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
