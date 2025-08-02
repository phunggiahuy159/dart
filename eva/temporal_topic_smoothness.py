import numpy as np
from typing import List, Tuple
from functools import lru_cache


def temporal_topic_smoothness(topic_words_over_time: List[str],
                             window_size: int = 2) -> Tuple[float, List[float]]:
    """Calculate temporal topic smoothness"""
    tts_scores = []
    split_topic_words = [words.split() for words in topic_words_over_time]

    if window_size <= 1:
        num_windows = max(0, len(split_topic_words) - window_size + 1)
        return 0.0, [0.0] * num_windows

    for t in range(len(split_topic_words) - window_size + 1):
        words_t_set = set(split_topic_words[t])

        if not words_t_set:
            tts_scores.append(0.0)
            continue

        words_later_sets = [set(split_topic_words[t+i]) for i in range(1, window_size)]

        redundancy = 0
        total_possible_comparisons = len(words_t_set) * (window_size - 1)

        if total_possible_comparisons == 0:
            tts_scores.append(0.0)
            continue

        for word in words_t_set:
            for later_set in words_later_sets:
                if word in later_set:
                    redundancy += 1

        current_tts = redundancy / total_possible_comparisons
        tts_scores.append(current_tts)

    avg_tts = np.mean(tts_scores) if tts_scores else 0.0
    return avg_tts if not np.isnan(avg_tts) else 0.0, tts_scores


@lru_cache(maxsize=128)
def cached_temporal_topic_smoothness(topic_words_tuple: Tuple[str, ...], window_size: int = 2) -> Tuple[float, List[float]]:
    """Cached version of temporal_topic_smoothness for performance"""
    return temporal_topic_smoothness(list(topic_words_tuple), window_size)