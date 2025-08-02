import numpy as np
from typing import List, Dict, Union, Tuple, Set
from functools import lru_cache
from collections import defaultdict
from .topic_diversity import compute_topic_diversity_optimized
from .temporal_topic_coherence import _calculate_npmi
from .temporal_topic_smoothness import cached_temporal_topic_smoothness
from .temporal_topic_quality import temporal_topic_quality_optimized
from .dynamic_topic_quality import dynamic_topic_quality
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from data.file_utils import split_text_word
from data.dynamic_dataset import DynamicDataset
def _coherence(
        reference_corpus: List[str],
        vocab: List[str],
        top_words: List[str],
        coherence_type='c_v',
        topn=20
    ):
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
    score = (cv_per_topic)

    return score

def precompute_word_occurrence_data(reference_corpus: List[str]) \
        -> Tuple[Dict[str, int], Dict[str, Set[int]], int]:
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
                # if w1 == w2: 
                #     continue

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

def evaluate_dynamic_topic_model(top_words_all_topics: List[List[str]], 
                               train_texts: List[str],
                               train_times: Union[List[int], np.ndarray],
                               dataset: DynamicDataset,
                               window_size: int = 2,
                               ) -> Dict[str, Union[float, List[float]]]:

    default_empty_results = {'TQ': [], 'TQ_avg': 0.0, 'TTC': [], 'TTS': [], 'TTQ': [], 'TTQ_avg': 0.0, 'DTQ': 0.0}
    if not train_texts:
        print("Warning: train_texts is empty. Metrics will be zero or empty.")
        return default_empty_results

    unique_times = sorted(list(set(train_times)))
    if not unique_times:
        print("Warning: No unique time slices. Metrics will be zero or empty.")
        return default_empty_results
    num_times = len(unique_times)

    if not top_words_all_topics or len(top_words_all_topics) != num_times:
        raise ValueError(f"top_words_all_topics length ({len(top_words_all_topics) if top_words_all_topics else 0}) "
                         f"must match number of unique time slices ({num_times}).")
    if not top_words_all_topics[0]:
        print("Warning: No topics found (top_words_all_topics[0] is empty). Metrics will be zero or empty.")
        default_empty_results['TQ'] = [0.0] * num_times
        return default_empty_results
    num_topics = len(top_words_all_topics[0])
    if num_topics == 0:
        print("Warning: num_topics is 0. Metrics will be zero or empty.")
        default_empty_results['TQ'] = [0.0] * num_times
        return default_empty_results


    global_word_doc_freq, global_inverted_index, global_N = \
        precompute_word_occurrence_data(train_texts)

    docs_by_time_slice: Dict[Union[int,np.integer], List[str]] = defaultdict(list)
    for i, time_val in enumerate(train_times):
        docs_by_time_slice[time_val].append(train_texts[i])

    tq_scores_per_slice = []  # List of TQ_avg for each time slice
    ttc_avg_per_topic = []    # List of TTC_avg for each topic
    tts_avg_per_topic = []    # List of TTS_avg for each topic
    ttq_avg_per_topic = []    # List of TTQ_avg for each topic

    topics_k_words_over_time_list: List[List[str]] = [[] for _ in range(num_topics)]
    for t_idx in range(num_times):
        if len(top_words_all_topics[t_idx]) != num_topics:
            raise ValueError(f"Inconsistent number of topics at time_idx {t_idx}. "
                             f"Expected {num_topics}, got {len(top_words_all_topics[t_idx])}.")
        for k in range(num_topics):
            topics_k_words_over_time_list[k].append(top_words_all_topics[t_idx][k])

    for k in range(num_topics):
        topic_k_words_str_list = topics_k_words_over_time_list[k]

        ttq_avg_for_topic_k, _ = temporal_topic_quality_optimized(
            topic_k_words_str_list, global_word_doc_freq, global_inverted_index, global_N, window_size
        )
        ttq_avg_per_topic.append(ttq_avg_for_topic_k)

        ttc_avg_topic_k, _ = temporal_topic_coherence_optimized(
             topic_k_words_str_list, global_word_doc_freq, global_inverted_index, global_N, window_size
        )
        ttc_avg_per_topic.append(ttc_avg_topic_k)

        tts_avg_topic_k, _ = cached_temporal_topic_smoothness(tuple(topic_k_words_str_list), window_size)
        tts_avg_per_topic.append(tts_avg_topic_k)


    for t_idx, current_time_val in enumerate(unique_times):
        docs_for_slice = docs_by_time_slice[current_time_val]

        if not docs_for_slice:
            tq_scores_per_slice.append(0.0)
            continue

        slice_word_doc_freq, slice_inverted_index, slice_N = \
            precompute_word_occurrence_data(docs_for_slice)

        if slice_N == 0:
            tq_scores_per_slice.append(0.0)
            continue

        qualities_for_topics_in_slice = []
        current_slice_topic_words_list = top_words_all_topics[t_idx]
        coh  = _coherence(dataset.train_texts,dataset.vocab,current_slice_topic_words_list)
        for k in range(num_topics):
            topic_k_words_str = current_slice_topic_words_list[k]

            coherence = coh[k]
            diversity = compute_topic_diversity_optimized( 
                topic_k_words_str, current_slice_topic_words_list
            )
            qualities_for_topics_in_slice.append(coherence * diversity)

        avg_tq_for_slice = np.mean(qualities_for_topics_in_slice) if qualities_for_topics_in_slice else 0.0
        tq_scores_per_slice.append(avg_tq_for_slice if not np.isnan(avg_tq_for_slice) else 0.0)

    final_tq_avg = np.mean(tq_scores_per_slice) if tq_scores_per_slice else 0.0
    final_ttq_avg = np.mean(ttq_avg_per_topic) if ttq_avg_per_topic else 0.0

    dtq_score = dynamic_topic_quality(tq_scores_per_slice, ttq_avg_per_topic)

    return {
        'TQ': tq_scores_per_slice,
        'TQ_avg': final_tq_avg if not np.isnan(final_tq_avg) else 0.0,
        'TTC': np.mean(ttc_avg_per_topic), # List of avg TTC per topic
        'TTS': np.mean(tts_avg_per_topic), # List of avg TTS per topic
        'TTQ': np.mean(ttq_avg_per_topic), # List of avg TTQ per topic
        'TTQ_avg': final_ttq_avg if not np.isnan(final_ttq_avg) else 0.0,
        'DTQ': dtq_score
    }
