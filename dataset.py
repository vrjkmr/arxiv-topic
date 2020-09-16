# -*- coding: utf-8 -*-
from gensim.corpora import Dictionary
from preprocess import ArXivPreprocessor
from utils import load_arxiv_metadata, extract_abstracts, load_documents


class ArXivDataset:

    """A dataset for arXiv abstracts.

    Attributes
    ----------
    documents : array_like
        Tokenized and pre-processed documents.

    idx2word : dict
        Dictionary of words in the corpus.

    corpus : array_like
        List of term-document frequencies for each document.

    """

    def __init__(self):
        pass

    def from_tokenized(self, filepath):
        """Load tokenized documents."""
        self.documents = load_documents(filepath)
        self.build_corpus()

    def from_metadata(self, filepath):
        """Load and process documents from metadata."""
        metadata = load_arxiv_metadata(metadata_filepath)
        abstracts = extract_abstracts(metadata)
        self.processor = ArXivPreprocessor()
        self.documents = self.processor.fit_transform(abstracts)
        self.build_corpus()

    def build_corpus(self):
        """Build word dictionary and corpus."""
        self.idx2word = Dictionary(self.documents)
        self.corpus = [self.idx2word.doc2bow(doc) for doc in self.documents]
