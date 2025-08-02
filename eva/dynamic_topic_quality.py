import numpy as np
from typing import List, Dict, Union
from collections import defaultdict
from .temporal_topic_coherence import precompute_word_occurrence_data, compute_static_topic_coherence
from .temporal_topic_quality import temporal_topic_quality_optimized
from .topic_coherence import _coherence1
from .topic_diversity import compute_topic_diversity_optimized


def dynamic_topic_quality(tq_list: List[float], ttq_list: List[float]) -> float:
    """Aggregates static TQ (per time slice) and temporal TTQ (per topic)."""
    tq_avg = np.mean(tq_list) if tq_list else 0.0
    ttq_avg = np.mean(ttq_list) if ttq_list else 0.0

    if np.isnan(tq_avg): 
        tq_avg = 0.0
    if np.isnan(ttq_avg): 
        ttq_avg = 0.0

    return 0.5 * (tq_avg + ttq_avg)


def evaluate_dynamic_topic_model(top_words_all_topics: List[List[str]],
                               train_texts: List[str],
                               train_times: Union[List[int], np.ndarray],
                               vocab: List[str],
                               window_size: int = 2) -> Dict[str, Union[float, List[float]]]:
    """Main evaluation function for dynamic topic models"""

    # Initial Checks and Setup
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

    # Global Precomputation (for TTC, TTQ which use full corpus)
    global_word_doc_freq, global_inverted_index, global_N = \
        precompute_word_occurrence_data(train_texts)

    # Pre-group documents by time slice (for TQ)
    docs_by_time_slice: Dict[Union[int,np.integer], List[str]] = defaultdict(list)
    for i, time_val in enumerate(train_times):
        docs_by_time_slice[time_val].append(train_texts[i])

    # Result accumulators
    tq_scores_per_slice = []  
    ttc_avg_per_topic = []    
    tts_avg_per_topic = []    
    ttq_avg_per_topic = []    

    # Calculate TTC, TTS, TTQ per topic
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

        # Individual TTC and TTS averages per topic
        from .temporal_topic_coherence import temporal_topic_coherence_optimized
        from .temporal_topic_smoothness import cached_temporal_topic_smoothness
        
        ttc_avg_topic_k, _ = temporal_topic_coherence_optimized(
             topic_k_words_str_list, global_word_doc_freq, global_inverted_index, global_N, window_size
        )
        ttc_avg_per_topic.append(ttc_avg_topic_k)

        tts_avg_topic_k, _ = cached_temporal_topic_smoothness(tuple(topic_k_words_str_list), window_size)
        tts_avg_per_topic.append(tts_avg_topic_k)

    # Calculate static TQ for each time slice
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
        
        # Get coherence scores using gensim
        coh = _coherence1(docs_for_slice, vocab, current_slice_topic_words_list)
        
        for k in range(num_topics):
            topic_k_words_str = current_slice_topic_words_list[k]

            coherence = coh[k]
            diversity = compute_topic_diversity_optimized(
                topic_k_words_str, current_slice_topic_words_list
            )
            qualities_for_topics_in_slice.append(coherence * diversity)

        avg_tq_for_slice = np.mean(qualities_for_topics_in_slice) if qualities_for_topics_in_slice else 0.0
        tq_scores_per_slice.append(avg_tq_for_slice if not np.isnan(avg_tq_for_slice) else 0.0)

    # Aggregate final metrics
    final_tq_avg = np.mean(tq_scores_per_slice) if tq_scores_per_slice else 0.0
    final_ttq_avg = np.mean(ttq_avg_per_topic) if ttq_avg_per_topic else 0.0

    dtq_score = dynamic_topic_quality(tq_scores_per_slice, ttq_avg_per_topic)

    return {
        'TQ': tq_scores_per_slice,
        'TQ_avg': final_tq_avg if not np.isnan(final_tq_avg) else 0.0,
        'TTC': np.mean(ttc_avg_per_topic),
        'TTS': np.mean(tts_avg_per_topic),
        'TTQ': np.mean(ttq_avg_per_topic),
        'TTQ_avg': final_ttq_avg if not np.isnan(final_ttq_avg) else 0.0,
        'DTQ': dtq_score
    }