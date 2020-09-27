# -*- coding: utf-8 -*-
import numpy as np
from gensim.models import LdaModel
from dataset import ArXivDataset


class TopicModel:

    """The topic model class.

    Attributes
    ----------
    model : gensim.models.LdaModel
        Trained Linear Dirichlet Allocation model.

    dataset : ArXivDataset
        Dataset used to pre-process and train the LDA model.

    num_topics : int
        Number of topics for the topic model.

    topic_names : array_like
        List of topic names (of length num_topics).

    """

    def __init__(self, model_path, dataset_path):
        """Instantiate a TopicModel object."""
        self.model = LdaModel.load(model_path)
        self.dataset = ArXivDataset.load(dataset_path)
        self.num_topics = self.model.num_topics
        self.topic_names = list(range(self.num_topics))

    def set_topic_names(self, names):
        """Assign topic names."""
        self.topic_names = names

    def print_topics(self, terms_per_topic=7):
        """Display terms for each topic."""
        for n in range(self.num_topics):
            topic_terms = self._get_top_terms_by_topic(n, terms_per_topic)
            topic_text = "Topic #{topic_num}".format(topic_num=n+1)
            if not isinstance(self.topic_names[0], int):
                topic_text += " ({name})".format(name=self.topic_names[n])
            print(topic_text)
            print("=" * len(topic_text))
            if n < self.num_topics - 1:
                print(topic_terms)
                print("\n")
            else:
                print(topic_terms)

    def _get_top_terms_by_topic(self, topic_idx, num_terms=7):
        """Extract terms by topic id."""
        terms = self.model.get_topic_terms(topic_idx)
        sorted_terms = sorted(terms, key=lambda x: x[1], reverse=True)
        topic_terms = []
        for i, term in enumerate(sorted_terms):
            if i == num_terms:
                break
            topic_terms.append(self.dataset.idx2word[term[0]])
        return topic_terms

    def predict(self, text):
        """Predict topics for a piece of text."""
        bow_transformed = self.dataset.transform([text])[0]
        topic_predictions = self.model.get_document_topics(bow_transformed)
        sorted_predictions = sorted(topic_predictions, key=lambda x: x[1],
                                    reverse=True)
        sorted_predictions = [(self.topic_names[topic_idx], prob)
                              for (topic_idx, prob) in sorted_predictions]
        return sorted_predictions
