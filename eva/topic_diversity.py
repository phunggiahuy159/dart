import numpy as np
from collections import Counter
from tqdm import tqdm
from typing import List


def _diversity(top_words: List[str]):
    """Calculate topic diversity for a set of topics"""
    num_words = 0.
    word_set = set()
    for words in top_words:
        ws = words.split()
        num_words += len(ws)
        word_set.update(ws)

    TD = len(word_set) / num_words
    return TD


def multiaspect_diversity(top_words: List[str], _type="TD"):
    """Calculate diversity across multiple aspects/time slices"""
    TD_list = list()
    for level_top_words in top_words:
        TD = _diversity(level_top_words, _type)
        TD_list.append(TD)

    return np.mean(TD_list)


def _time_slice_diversity(topics, time_vocab):
    """Calculate diversity for a specific time slice"""
    num_associated_words = 0.
    T = len(topics[0].split())
    flatten_topic_words = [word for topic_words in topics for word in topic_words.split()]
    counter = Counter(flatten_topic_words)

    for word in np.sort(flatten_topic_words):
        if (counter[word] == 1) and word in time_vocab:
            num_associated_words += 1

    return num_associated_words / (len(topics) * T)


def dynamic_diversity(
        top_words: List[str],
        train_bow: np.ndarray,
        train_times: List[int],
        vocab: List[str],
        verbose=False
    ):
    """Calculate dynamic diversity across time slices"""
    TD_list = list()

    time_idx = np.sort(np.unique(train_times))

    for time in tqdm(time_idx):
        doc_idx = np.where(train_times == time)[0]
        time_vocab_idx = np.nonzero(train_bow[doc_idx].sum(0))[0]
        time_vocab = np.asarray(vocab)[time_vocab_idx]

        topics = top_words[time]
        TD_list.append(_time_slice_diversity(topics, time_vocab))

    if verbose:
        print(f"dynamic TD list: {TD_list}")

    return np.mean(TD_list)


def compute_topic_diversity_optimized(topic_words_str: str, all_topic_words_at_timeslice: List[str]) -> float:
    """Optimized topic diversity calculation"""
    current_topic_word_list = topic_words_str.split()
    if not current_topic_word_list:
        return 0.0

    other_topic_word_sets = []
    for other_topic_str in all_topic_words_at_timeslice:
        if other_topic_str != topic_words_str:
            other_words = other_topic_str.split()
            if other_words:
                 other_topic_word_sets.append(set(other_words))

    if not other_topic_word_sets:
        return 1.0

    redundancy_count = 0
    for word in current_topic_word_list:
        for other_set in other_topic_word_sets:
            if word in other_set:
                redundancy_count += 1

    total_possible_pairings = len(current_topic_word_list) * len(other_topic_word_sets)

    if total_possible_pairings == 0:
        return 1.0

    normalized_redundancy = redundancy_count / total_possible_pairings
    return 1.0 - normalized_redundancy