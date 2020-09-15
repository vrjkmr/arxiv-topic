# -*- coding: utf-8 -*-
import re
import gensim
import spacy
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser


class Preprocessor:

    """A text pre-processor.

    Attributes
    ----------
    stopwords : array_like
        List of stopwords.

    nlp : spacy.lang.en.English
        The SpaCy English model used for lemmatization.

    n_gram_models : array_like
        List containing n-gram models.

    """

    def __init__(self):
        pass

    def fit_transform(self, documents, additional_stopwords=[], max_n=3,
                      n_gram_threshold=100, pos_tags=["NOUN", "ADJ", "PROPN"]):
        """Pre-process documents after fitting.

        Parameters
        ----------
        documents : array_like
            Sequence of document strings.

        additional_stopwords : array_like
            List of stopwords (in addition to gensim stopwords).
            Defaults to [].

        max_n : int
            Maximum n value for n-gram phrase learning. Enables phrases up
            to n words in length. Defaults to 3.

        n_gram_threshold : int
            Minimum n-gram frequency threshold. All n-grams with a frequency
            lower than the threshold will be ignored. Defaults to 100.

        pos_tags : array_like
            Part-of-speech tags extracted from distinct tokens. Defaults to
            ["NOUN", "ADJ", "PROPN"].

        Returns
        -------
        documents : array_like
            Tokenized and pre-processed documents.

        """

        # set instance attributes
        self.max_n = max_n
        self.n_gram_threshold = n_gram_threshold
        self.pos_tags = pos_tags
        self.stopwords = stopwords.words("english")
        self.stopwords.extend(additional_stopwords)
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

        # fit-transform documents
        documents = self.remove_latex_equations(documents)
        documents = self.remove_newlines(documents)
        documents = self.tokenize(documents)
        documents = self.remove_stopwords(documents, self.stopwords)
        documents = self.identify_phrases(documents, max_n=self.max_n,
                                          threshold=self.n_gram_threshold)
        documents = self.lemmatize(documents, self.pos_tags)

        return documents

    def transform(self, documents):
        """Pre-process documents.

        Parameters
        ----------
        documents : array_like
            Sequence of document strings.

        Returns
        -------
        documents : array_like
            Tokenized and pre-processed documents.
        """

        documents = self.remove_latex_equations(documents)
        documents = self.remove_newlines(documents)
        documents = self.tokenize(documents)
        documents = self.remove_stopwords(documents, self.stopwords)
        documents = self.identify_phrases(documents, max_n=self.max_n,
                                          threshold=self.n_gram_threshold,
                                          fit=False)
        documents = self.lemmatize(documents, self.pos_tags)

        return documents

    def remove_latex_equations(self, documents):
        """Remove LaTex equations."""
        processed = []
        for document in documents:
            # remove text between every two consecutive occurences of "$"
            indices = [match.start() for match in re.finditer("\$", document)]
            parsed = document
            for idx in range(0, len(indices), 2):
                parsed = parsed.replace(document[indices[idx]:indices[idx+1]+1], "")
            processed.append(parsed)

        return processed

    def remove_newlines(self, documents):
        """Remove newline characters and extra spaces."""
        return [re.sub("\s+", " ", doc) for doc in documents]

    def tokenize(self, documents):
        """Tokenize a document using Gensim pre-processing."""
        return [simple_preprocess(str(doc)) for doc in documents]

    def remove_stopwords(self, documents, stop_words):
        """Remove stopwords."""
        return [[word for word in doc if word not in stop_words] 
                for doc in documents]

    def identify_phrases(self, documents, max_n, threshold, fit=True):
        """Identify and transform phrases using n-grams."""
        processed = documents
        if fit:
            self.n_gram_models = []
            for n in range(2, self.max_n):
                n_grams = Phrases(processed, threshold=threshold)
                n_gram_model = Phraser(n_grams)
                self.n_gram_models.append(n_gram_model)
                processed = [n_gram_model[doc] for doc in processed]
        else:
            for model in self.n_gram_models:
                processed = [model[doc] for doc in processed]
        return processed

    def lemmatize(self, documents, pos_tags):
        """Lemmatize documents."""
        lemmatized = []
        for doc in documents:
            tokens = self.nlp(" ".join(doc))
            lemmatized.append([token.lemma_ for token in tokens
                               if token.pos_ in pos_tags])
        return lemmatized
