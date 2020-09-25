# -*- coding: utf-8 -*-
import os
from dataset import ArXivDataset
from gensim.models import LdaModel, CoherenceModel
from utils import create_directory_if_not_exists
from pprint import pprint
import logging

# load and preprocess texts
metadata_filepath = "./data/arxiv-metadata-oai-snapshot.json"
dataset = ArXivDataset().from_metadata(metadata_filepath)
print("Size of dataset: {s}".format(s=len(dataset)))

# export dataset
dataset_filepath = "./data/dataset.obj"
dataset.export(dataset_filepath)
print("Exported dataset to {path}.".format(path=dataset_filepath))

# set model hyperparameters
num_topics = 12
num_passes = 5
random_state = 100

# set up logs
model_filename = "lda_model_n{n}_p{p}_r{r}".format(n=num_topics, p=num_passes,
                                                   r=random_state)
logging.basicConfig(filename=model_filename,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

# build and train model
print("Training model...", end="")
model = LdaModel(corpus=dataset.corpus, id2word=dataset.idx2word,
                 num_topics=num_topics, passes=num_passes,
                 random_state=random_state, per_word_topics=True)
print("done.")

# calculate coherence score
coherence_model = CoherenceModel(model=model, texts=dataset.documents,
                                 dictionary=dataset.idx2word, coherence="c_v")
score = coherence_model.get_coherence()
print("Coherence: {:.3f}".format(score))

# print model topics
pprint(model.print_topics())

# export model
create_directory_if_not_exists("./models")
model_filepath = "./models/{name}_c{score}".format(name=model_filename,
                                                   score=score)
model.save(model_filepath)
