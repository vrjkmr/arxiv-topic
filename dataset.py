# -*- coding: utf-8 -*-
from gensim.corpora import Dictionary
from preprocess import ArXivPreprocessor
from utils import (load_arxiv_metadata, extract_abstracts,
                   load_documents, load_object, export_object)


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

    processor : ArXivPreprocessor
        Document pre-processor.

    """

    def __init__(self):
        pass

    @staticmethod
    def from_tokenized(tokenized_filepath, preprocessor_filepath):
        """Load tokenized documents and preprocessor."""
        dataset = ArXivDataset()
        dataset.documents = load_documents(tokenized_filepath)
        dataset.processor = load_object(preprocessor_filepath)
        dataset.build_corpus()
        return dataset

    @staticmethod
    def from_metadata(filepath,
                      additional_stopwords=[],
                      max_n=3,
                      n_gram_threshold=100,
                      pos_tags=["NOUN", "ADJ", "PROPN"]):
        """Load and process documents from metadata."""
        metadata = load_arxiv_metadata(filepath)
        abstracts = extract_abstracts(metadata)
        dataset = ArXivDataset()
        dataset.processor = ArXivPreprocessor()
        documents = dataset.processor.fit_transform(abstracts,
                                                    additional_stopwords,
                                                    max_n, n_gram_threshold,
                                                    pos_tags)
        dataset.documents = documents
        dataset.build_corpus()
        return dataset

    @staticmethod
    def load(filepath):
        """Load ArXivDataset object."""
        dataset = load_object(filepath)
        return dataset

    def build_corpus(self):
        """Build word dictionary and corpus."""
        self.idx2word = Dictionary(self.documents)
        self.corpus = self.to_bow(self.documents)

    def transform(self, documents):
        """Transform documents into term-document frequencies."""
        processed = self.processor.transform(documents)
        return self.to_bow(processed)

    def to_bow(self, documents):
        return [self.idx2word.doc2bow(doc) for doc in documents]

    def save(self, filepath):
        """Export dataset."""
        export_object(self, filepath)

    def __len__(self):
        return len(self.documents)
