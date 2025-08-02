import numpy as np
import torch
import itertools
from data.dynamic_dataset import DynamicDataset
from data.download import download_dataset
from trainer.trainer import DynamicTrainer
from model.Dart import DART
from eva.topic_coherence import dynamic_coherence
from eva.topic_diversity import dynamic_diversity
from eva.clustering import _clustering, purity_score
from eva.classification import f1_score, accuracy_score, _cls
from eva.evaluate_dynamic_topic_model import evaluate_dynamic_topic_model
from eva.custom_coherence import apply_custom_coherence_patch
apply_custom_coherence_patch()



download_dataset('NYT', cache_path='./datasets')

# Define parameter values to search
device = 'cuda'
dataset_dir = "./datasets/NYT"

dataset = DynamicDataset(dataset_dir, batch_size=200, read_labels=True, device=device)

weight_neg_values = [7e+7]
weight_beta_align_values = [1,1,1,1,1]
weight_alpha_values = [300,350,400,450]

# Create all possible combinations
param_combinations = list(itertools.product(weight_neg_values, weight_beta_align_values, weight_alpha_values))

# Store results
results = []

# Iterate through parameter combinations
for weight_neg, weight_beta_align, weight_alpha in param_combinations:
    # Set seeds for reproducibility

    print(f"\nTraining with parameters: weight_neg={weight_neg}, weight_beta_align={weight_beta_align}, weight_alpha={weight_alpha}")

    model = DART(
        vocab_size=dataset.vocab_size,
        num_times=dataset.num_times,
        pretrained_WE=dataset.pretrained_WE,
        doc_tfidf=dataset.doc_tfidf,
        train_time_wordfreq=dataset.train_time_wordfreq,
        num_topics=50,
        en_units=200,
        weight_neg=weight_neg,
        weight_pos=1.0,
        weight_beta_align=100,
        weight_alpha=1,
        beta_temp=1,
        dropout=0.01,

    )

    model = model.to(device)

    # Create trainer with two-phase settings
    trainer = DynamicTrainer(
        model,
        dataset,
        epochs=weight_alpha,
        learning_rate=0.002,
        batch_size=200,
        log_interval=5,
        verbose=True
    )

    # Run training
    top_words, train_theta = trainer.train()

    # get theta (doc-topic distributions)
    train_theta, test_theta = trainer.export_theta()

    train_times = dataset.train_times.cpu().numpy()
    # Save top words to a file with a name that reflects the parameters
    filename = f"top_words_neg_{weight_neg}_beta_{weight_beta_align}_alpha_{weight_alpha}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        for t, topics in enumerate(top_words):
            f.write(f"--------------Time {t + 1}:--------------\n")
            for i, word in enumerate(topics):
                f.write(f"  Topic {i + 1}: {word}\n")
            f.write("\n")

    print(f"Top words saved to {filename}")

    # compute topic coherence
    dynamic_TC = dynamic_coherence(dataset.train_texts, train_times, dataset.vocab, top_words)
    print("dynamic_TC: ", dynamic_TC)

    # compute topic diversity
    dynamic_TD = dynamic_diversity(top_words, dataset.train_bow.cpu().numpy(), train_times, dataset.vocab)
    print("dynamic_TD: ", dynamic_TD)
    # evaluate clustering
    cluster = _clustering(test_theta, dataset.test_labels)
    purity = cluster['Purity']
    nmi = cluster['NMI']

    # evaluate classification
    clf = _cls(train_theta, test_theta, dataset.train_labels, dataset.test_labels)
    acc = clf['acc']
    f1 = clf['macro-F1']
    # compute TTQ_avg and DTQ using the evaluate_dynamic_topic_model function
    evaluation_results = evaluate_dynamic_topic_model(
        top_words_all_topics=top_words,
        dataset = dataset,
        train_texts=dataset.train_texts,
        train_times=train_times,
        window_size=2
    )
    ttq_avg = evaluation_results['TTQ_avg']
    dtq = evaluation_results['DTQ']
    tq = evaluation_results['TQ']
    ttc = evaluation_results['TTC']
    tts = evaluation_results['TTS']
    ttq = evaluation_results['TTQ']
    tq_avg = evaluation_results['TQ_avg']

    print(f"TTQ_avg: {ttq_avg:.4f}")
    print(f"Dynamic Topic Quality (DTQ): {dtq:.4f}")
    print(f"Temporal Topic Coherence (TTC): {ttc:.4f}")
    print(f"Temporal Topic Smoothness (TTS): {tts:.4f}")
    print(f"Temporal Topic Quality (TTQ): {ttq:.4f}")
    print(f"Topic Quality (TQ_avg): {tq_avg:.4f}")

    # Store results
    results.append({
        'weight_neg': weight_neg,
        'weight_beta_align': weight_beta_align,
        'weight_alpha': weight_alpha,

        'dynamic_TC': dynamic_TC,
        'dynamic_TD': dynamic_TD,
        'purity': purity,
        'nmi': nmi,
        'acc':acc,
        'f1':f1,
        'TTQ_avg': ttq_avg,
        'DTQ': dtq,
        'TQ_avg': tq_avg,
        'TTC': ttc,
        'TTS': tts,
        'TTQ': ttq
    })

# Find and display best parameters
print(results)
best_dtq_result = max(results, key=lambda x: x['DTQ'])
print("\nBest parameters based on Dynamic Topic Quality (DTQ):")
print(f"weight_neg={best_dtq_result['weight_neg']}, weight_beta_align={best_dtq_result['weight_beta_align']}, weight_alpha={best_dtq_result['weight_alpha']}")
print(f"dynamic_TC={best_dtq_result['dynamic_TC']}, dynamic_TD={best_dtq_result['dynamic_TD']}, TTQ_avg={best_dtq_result['TTQ_avg']}, DTQ={best_dtq_result['DTQ']}")
import pandas as pd
data = pd.DataFrame(results)
print(data)
data.to_csv('output.csv')