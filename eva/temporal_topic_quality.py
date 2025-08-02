import numpy as np
from typing import List, Dict, Set, Tuple
from .temporal_topic_coherence import temporal_topic_coherence_optimized
from .temporal_topic_smoothness import cached_temporal_topic_smoothness


def temporal_topic_quality_optimized(topic_words_over_time: List[str],
                                    word_doc_freq: Dict[str, int],
                                    inverted_index: Dict[str, Set[int]],
                                    N: int,
                                    window_size: int = 2) -> Tuple[float, List[float]]:
    """Calculates TTQ for a single topic using precomputed corpus data."""
    _, ttc_per_window = temporal_topic_coherence_optimized(
        topic_words_over_time, word_doc_freq, inverted_index, N, window_size
    )
    _, tts_per_window = cached_temporal_topic_smoothness(tuple(topic_words_over_time), window_size)

    ttq_per_window_products = []
    if len(ttc_per_window) == len(tts_per_window):
        ttq_per_window_products = [ttc * tts for ttc, tts in zip(ttc_per_window, tts_per_window)]
    else:
        print(f"Warning: TTC/TTS per-window list length mismatch during TTQ calculation. "
              f"TTC: {len(ttc_per_window)}, TTS: {len(tts_per_window)}. TTQ may be partial.")
        min_len = min(len(ttc_per_window), len(tts_per_window))
        ttq_per_window_products = [ttc * tts for ttc, tts in zip(ttc_per_window[:min_len], tts_per_window[:min_len])]

    ttq_avg = np.mean(ttq_per_window_products) if ttq_per_window_products else 0.0
    return ttq_avg if not np.isnan(ttq_avg) else 0.0, ttq_per_window_products