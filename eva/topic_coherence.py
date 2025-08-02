import numpy as np
from tqdm import tqdm
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from typing import List
from data.file_utils import split_text_word


def get_top_words(beta, vocab, num_top_words, verbose=False):
    """Extract top words for each topic from beta distribution"""
    topic_str_list = list()
    for i, topic_dist in enumerate(beta):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(num_top_words + 1):-1]
        topic_str = ' '.join(topic_words)
        topic_str_list.append(topic_str)
        if verbose:
            print('Topic {}: {}'.format(i, topic_str))
    return topic_str_list

def _coherence(
        reference_corpus: List[str],
        vocab: List[str],
        top_words: List[str],
        coherence_type='c_v',
        topn=20
    ):
    """Calculate coherence for a set of topics"""
    split_top_words = split_text_word(top_words)
    split_reference_corpus = split_text_word(reference_corpus)
    dictionary = Dictionary(split_text_word(vocab))

    cm = CoherenceModel(
        texts=split_reference_corpus,
        dictionary=dictionary,
        topics=split_top_words,
        topn=topn,
        coherence=coherence_type,
    )
    cv_per_topic = cm.get_coherence_per_topic()
    score = np.mean(cv_per_topic)

    return score


def dynamic_coherence(train_texts, train_times, vocab, top_words_list, coherence_type='c_v', verbose=False):
    """Calculate dynamic coherence across time slices"""
    cv_score_list = list()

    for time, top_words in tqdm(enumerate(top_words_list)):
        # use the texts of each time slice as the reference corpus.
        idx = np.where(train_times == time)[0]
        reference_corpus = [train_texts[i] for i in idx]

        # use the topics at a time slice
        cv_score = _coherence(reference_corpus, vocab, top_words, coherence_type)
        cv_score_list.append(cv_score)

    if verbose:
        print(f"dynamic TC list: {cv_score_list}")

    return np.mean(cv_score_list)


def _coherence1(
        reference_corpus: List[str],
        vocab: List[str],
        top_words: List[str],
        coherence_type='c_v',
        topn=20
    ):
    """Calculate coherence for individual topics (returns per-topic scores)"""
    split_top_words = split_text_word(top_words)
    split_reference_corpus = split_text_word(reference_corpus)
    dictionary = Dictionary(split_text_word(vocab))

    cm = CoherenceModel(
        texts=split_reference_corpus,
        dictionary=dictionary,
        topics=split_top_words,
        topn=topn,
        coherence=coherence_type,
    )
    cv_per_topic = cm.get_coherence_per_topic()

    return cv_per_topic