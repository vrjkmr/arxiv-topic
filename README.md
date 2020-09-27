# ArXiv Topic Modeling

This repository contains the code for a [Latent Dirichlet Allocation (LDA)](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) topic model built and trained on the abstracts of ~160,000 ML-related research papers from the [ArXiv.org dataset](https://www.kaggle.com/Cornell-University/arxiv) on Kaggle.

```
TODO
----

1. [ ] Add dataset creation code
2. [ ] Clean up code
    - [ ] Training + hyperparameter tuning
    - [ ] Model inference
3. [ ] Build a good model
4. [ ] Update final README.md
    - [ ] Update examples
    - [ ] Update "Results: Topics" section
```

To illustrate, shown below is an example of the model's ability to predict topics present in the paper ["Why Molière most likely did write his plays"](https://arxiv.org/abs/2001.01595) by Cafiero and Camps (2020).

```
Paper
-----
"Why Molière most likely did write his plays" (Cafiero & Camps, 2020)

Abstract
--------
  As for Shakespeare, a hard-fought debate has emerged about Molière, a
supposedly uneducated actor who, according to some, could not have written the
masterpieces attributed to him. In the past decades, the century-old thesis
according to which Pierre Corneille would be their actual author has become
popular, mostly because of new works in computational linguistics. These
results are reassessed here through state-of-the-art attribution methods. We
study a corpus of comedies in verse by major authors of Molière and
Corneille's time. Analysis of lexicon, rhymes, word forms, affixes,
morphosyntactic sequences, and function words do not give any clue that another
author among the major playwrights of the time would have written the plays
signed under the name Molière.

Predicted topics
----------------
[('Natural language processing', 0.38158375),
 ('Paper-related terms?', 0.298497),
 ('ML-related terms?', 0.091592446)]
```

### Motivation

You know how when reading research papers, the first thing we read is the abstract? The abstract helps us (as humans) get a general sense of what different topics are explored in any given paper. But what if we can train an unsupervised model to automatically "categorize" papers for us?

In this project, my ultimate goal was to build an clustering model that can:

1. Identify salient trends and sub-topics in machine learning research today, and
2. Automatically predict the topic(s) explored in any given paper simply by looking at its abstract.

### Project structure

This project is organized as follows.

```
.
├── dataset.py                          # script containing the dataset class
├── model.py                            # script containing the topic model class
├── preprocess.py                       # script containing the text preprocessor class
├── utils.py                            # script containing helper functions
├── Inference.ipynb                     # notebook illustrating how to predict topics of papers
├── Training.ipynb                      # notebook to train and tune LDA models
└── README.md
```

### Results: Topics

The final model achieved a coherence score of 49.9%. While this score is quite low (an ideal coherence score tends to be around 60-80%), the model was able to detect some interesting topic clusters, a few of which are listed below. Note that while the topic terms were generated by the LDA model, the topic titles themselves are subjective, since they were added by me after looking at the term distribution for each of the topics.

1. **Algorithms and optimization:** "algorithm", "problem", "search", "optimization", "gradient"
2. **Probabilistic modeling and inference:** "distribution", "bayesian", "inference", "process", "variable"
3. **Natural language processing (NLP):** "language", "text", "semantic", "knowledge", "word"
4. **Computer vision:** "image", "object", "detection", "segmentation", "convolution"
5. **Reinforcement learning:** "agent", "policy", "action", "state", "environment"
6. **Deep learning architectures:** "network", "neural", "deep", "architecture", "layer"
7. **Graph theory:** "graph", "structure", "node", "tree", "edge"
8. **Medicine and healthcare applications** "patient", "covid", "causal", "treatment", "population"

### Model inference

To predict which topics might be related to any paper on arXiv, simply build a `TopicModel` object, scrape the abstract section, and pass in the raw text into the model's `predict()` method. The output is an ordered list of tuples, with each tuple holding the topic name and the likelihood of the topic's presence in the paper.

```python
from model import TopicModel
from utils import scrape_arxiv_abstract

lda_filepath = "./models/model_001"
dataset_filepath = "./data/dataset.obj"
topic_model = TopicModel(lda_filepath, dataset_filepath)

# Paper: "Future Frame Prediction of a Video Sequence" (Kaur & Das, 2020)
paper_url = "https://arxiv.org/abs/2009.08825"
abstract = scrape_arxiv_abstract(paper_url)
predictions = topic_model.predict(abstract)
print(predictions)

'''
Output
------
[('Paper-related terms?', 0.25155145),
 ('Computer vision', 0.18688545),
 ('ML-related terms?', 0.17415679)]
'''
```
### Acknowledgements

- Radim Řehůřek's [tips](https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html) on building Gensim LDA models
- Cornell University's [arXiv.org dataset](https://www.kaggle.com/Cornell-University/arxiv) hosted on Kaggle
