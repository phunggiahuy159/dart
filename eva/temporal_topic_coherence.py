import numpy as np
from typing import List, Dict, Set, Tuple
from collections import defaultdict


def _calculate_npmi(p_w1: float, p_w2: float, p_w1_w2: float, epsilon: float = 1e-10) -> float:
    """Calculates NPMI with robust handling of edge cases."""
    if p_w1 == 0 or p_w2 == 0:
        if p_w1_w2 == 0:
            return 0.0

    # Numerator of PMI: p_w1_w2 + epsilon
    pmi_num = p_w1_w2 + epsilon
    # Denominator of PMI: p_w1 * p_w2 + epsilon
    pmi_den = p_w1 * p_w2 + epsilon

    if pmi_den <= 0: 
        return 0.0

    pmi = np.log(pmi_num / pmi_den)

    # Normalization term for NPMI: -log(p_w1_w2 + epsilon)
    norm_term_arg = p_w1_w2 + epsilon
    if norm_term_arg <= 0: 
        return 0.0
    if norm_term_arg == 1.0:
        return 0.0

    norm_val = -np.log(norm_term_arg)
    if norm_val == 0: 
        return 0.0

    npmi = pmi / norm_val

    if np.isnan(npmi) or np.isinf(npmi):
        return 0.0
    return npmi


def precompute_word_occurrence_data(reference_corpus: List[str]) -> Tuple[Dict[str, int], Dict[str, Set[int]], int]:
    """
    Precomputes word document frequencies and an inverted index.
    Returns: (word_doc_freq, inverted_index, N_docs)
    """
    word_doc_freq: Dict[str, int] = {}
    inverted_index: Dict[str, Set[int]] = defaultdict(set)

    if not reference_corpus:
        return word_doc_freq, inverted_index, 0

    for i, doc_text in enumerate(reference_corpus):
        words_in_doc = doc_text.split()
        unique_words_in_doc = set(words_in_doc)

        for word in unique_words_in_doc:
            word_doc_freq[word] = word_doc_freq.get(word, 0) + 1
            inverted_index[word].add(i)

    N_docs = len(reference_corpus)
    return word_doc_freq, inverted_index, N_docs


def temporal_topic_coherence_optimized(topic_words_over_time: List[str],
                                      word_doc_freq: Dict[str, int],
                                      inverted_index: Dict[str, Set[int]],
                                      N: int,
                                      window_size: int = 2) -> Tuple[float, List[float]]:
    """Calculate temporal topic coherence with optimized computation"""
    ttc_scores = []
    split_topic_words = [words.split() for words in topic_words_over_time]

    if N == 0:
        num_windows = max(0, len(split_topic_words) - window_size + 1)
        return 0.0, [0.0] * num_windows

    for t in range(len(split_topic_words) - window_size + 1):
        words_t = split_topic_words[t]
        words_t_plus = split_topic_words[t + window_size - 1]

        pair_scores = []
        if not words_t or not words_t_plus:
            ttc_scores.append(0.0)
            continue

        for w1 in words_t:
            w1_docs_count = word_doc_freq.get(w1, 0)
            for w2 in words_t_plus:
                w2_docs_count = word_doc_freq.get(w2, 0)

                if w1_docs_count == 0 or w2_docs_count == 0:
                    both_docs_count = 0
                elif w1 in inverted_index and w2 in inverted_index:
                    both_docs_count = len(inverted_index[w1].intersection(inverted_index[w2]))
                else:
                    both_docs_count = 0

                p_w1 = w1_docs_count / N
                p_w2 = w2_docs_count / N
                p_w1_w2 = both_docs_count / N

                npmi = _calculate_npmi(p_w1, p_w2, p_w1_w2)
                pair_scores.append(npmi)

        current_ttc = np.mean(pair_scores) if pair_scores else 0.0
        ttc_scores.append(current_ttc if not np.isnan(current_ttc) else 0.0)

    avg_ttc = np.mean(ttc_scores) if ttc_scores else 0.0
    return avg_ttc if not np.isnan(avg_ttc) else 0.0, ttc_scores


def compute_static_topic_coherence(topic_words_str: str,
                                   word_doc_freq: Dict[str, int],
                                   inverted_index: Dict[str, Set[int]],
                                   N: int) -> float:
    """Calculate static topic coherence for a single time slice"""
    words = topic_words_str.split()
    if not words or N == 0:
        return 0.0

    pair_scores = []
    for i in range(len(words)):
        w1 = words[i]
        w1_docs_count = word_doc_freq.get(w1, 0)

        for j in range(i + 1, len(words)):
            w2 = words[j]
            w2_docs_count = word_doc_freq.get(w2, 0)

            if w1_docs_count == 0 or w2_docs_count == 0:
                 both_docs_count = 0
            elif w1 in inverted_index and w2 in inverted_index:
                both_docs_count = len(inverted_index[w1].intersection(inverted_index[w2]))
            else:
                both_docs_count = 0

            p_w1 = w1_docs_count / N
            p_w2 = w2_docs_count / N
            p_w1_w2 = both_docs_count / N

            npmi = _calculate_npmi(p_w1, p_w2, p_w1_w2)
            pair_scores.append(npmi)

    mean_score = np.mean(pair_scores) if pair_scores else 0.0
    return mean_score if not np.isnan(mean_score) else 0.0