# import numpy as np
# import torch
# import itertools
# from data.dynamic_dataset import DynamicDataset
# from data.download import download_dataset
# from trainer.trainer import DynamicTrainer
# from model.Dart import DART
# from eva.topic_coherence import dynamic_coherence
# from eva.topic_diversity import dynamic_diversity
# from eva.clustering import _clustering, purity_score
# from eva.classification import f1_score, accuracy_score, _cls
# from eva.evaluate_dynamic_topic_model import evaluate_dynamic_topic_model
# from eva.custom_coherence import apply_custom_coherence_patch
# apply_custom_coherence_patch()



# download_dataset('NYT', cache_path='./datasets')

# # Define parameter values to search
# device = 'cuda'
# dataset_dir = "./datasets/NYT"

# dataset = DynamicDataset(dataset_dir, batch_size=200, read_labels=True, device=device)


# model = DART(
#     vocab_size=dataset.vocab_size,
#     num_times=dataset.num_times,
#     pretrained_WE=dataset.pretrained_WE,
#     doc_tfidf=dataset.doc_tfidf,
#     train_time_wordfreq=dataset.train_time_wordfreq,
#     num_topics=50,
#     en_units=200,
#     weight_neg=7e+7,
#     weight_pos=1.0,
#     weight_beta_align=100,
#     beta_temp=1,
#     dropout=0.01,

# )

# model = model.to(device)

# # Create trainer with two-phase settings
# trainer = DynamicTrainer(
#     model,
#     dataset,
#     epochs=300,
#     learning_rate=0.002,
#     batch_size=200,
#     log_interval=5,
#     verbose=True
# )

# # Run training
# top_words, train_theta = trainer.train()

# # get theta (doc-topic distributions)
# train_theta, test_theta = trainer.export_theta()

# train_times = dataset.train_times.cpu().numpy()
# # Save top words to a file with a name that reflects the parameters
# filename = f"top_words.txt"
# with open(filename, "w", encoding="utf-8") as f:
#     for t, topics in enumerate(top_words):
#         f.write(f"--------------Time {t + 1}:--------------\n")
#         for i, word in enumerate(topics):
#             f.write(f"  Topic {i + 1}: {word}\n")
#         f.write("\n")

# print(f"Top words saved to {filename}")

# # compute topic coherence
# dynamic_TC = dynamic_coherence(dataset.train_texts, train_times, dataset.vocab, top_words)
# print("dynamic_TC: ", dynamic_TC)

# # compute topic diversity
# dynamic_TD = dynamic_diversity(top_words, dataset.train_bow.cpu().numpy(), train_times, dataset.vocab)
# print("dynamic_TD: ", dynamic_TD)
# # evaluate clustering
# cluster = _clustering(test_theta, dataset.test_labels)
# purity = cluster['Purity']
# nmi = cluster['NMI']

# # evaluate classification
# clf = _cls(train_theta, test_theta, dataset.train_labels, dataset.test_labels)
# acc = clf['acc']
# f1 = clf['macro-F1']
# # compute TTQ_avg and DTQ using the evaluate_dynamic_topic_model function
# evaluation_results = evaluate_dynamic_topic_model(
#     top_words_all_topics=top_words,
#     dataset = dataset,
#     train_texts=dataset.train_texts,
#     train_times=train_times,
#     window_size=2
# )
# ttq_avg = evaluation_results['TTQ_avg']
# dtq = evaluation_results['DTQ']
# tq = evaluation_results['TQ']
# ttc = evaluation_results['TTC']
# tts = evaluation_results['TTS']
# ttq = evaluation_results['TTQ']
# tq_avg = evaluation_results['TQ_avg']

# print(f"TTQ_avg: {ttq_avg:.4f}")
# print(f"Dynamic Topic Quality (DTQ): {dtq:.4f}")
# print(f"Temporal Topic Coherence (TTC): {ttc:.4f}")
# print(f"Temporal Topic Smoothness (TTS): {tts:.4f}")
# print(f"Temporal Topic Quality (TTQ): {ttq:.4f}")
# print(f"Topic Quality (TQ_avg): {tq_avg:.4f}")


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


def main():
    """
    Main function to train DART model and evaluate its performance.
    """
    # Apply custom coherence patch
    apply_custom_coherence_patch()
    
    # Download dataset
    print("Downloading NYT dataset...")
    download_dataset('NYT', cache_path='./datasets')
    
    # Configuration
    device = 'cuda'
    dataset_dir = "./datasets/NYT"
    
    # Load dataset
    print("Loading dataset...")
    dataset = DynamicDataset(dataset_dir, batch_size=200, read_labels=True, device=device)
    
    # Initialize model
    print("Initializing DART model...")
    model = DART(
        vocab_size=dataset.vocab_size,
        num_times=dataset.num_times,
        pretrained_WE=dataset.pretrained_WE,
        doc_tfidf=dataset.doc_tfidf,
        train_time_wordfreq=dataset.train_time_wordfreq,
        num_topics=50,
        en_units=200,
        weight_neg=7e+7,
        weight_pos=1.0,
        weight_beta_align=100,
        beta_temp=1,
        dropout=0.01,
    )
    
    model = model.to(device)
    
    # Create trainer
    print("Creating trainer...")
    trainer = DynamicTrainer(
        model,
        dataset,
        epochs=300,
        learning_rate=0.002,
        batch_size=200,
        log_interval=5,
        verbose=True
    )
    
    # Run training
    print("Starting training...")
    top_words, train_theta = trainer.train()
    
    # Get theta (doc-topic distributions)
    print("Exporting theta distributions...")
    train_theta, test_theta = trainer.export_theta()
    
    train_times = dataset.train_times.cpu().numpy()
    
    # Save top words to file
    print("Saving top words...")
    save_top_words(top_words, "top_words.txt")
    
    # Evaluate model performance
    print("Evaluating model performance...")
    evaluate_model(top_words, train_theta, test_theta, dataset, train_times)


def save_top_words(top_words, filename):
    """
    Save top words to a text file.
    
    Args:
        top_words: List of top words for each time period and topic
        filename: Output filename
    """
    with open(filename, "w", encoding="utf-8") as f:
        for t, topics in enumerate(top_words):
            f.write(f"--------------Time {t + 1}:--------------\n")
            for i, word in enumerate(topics):
                f.write(f"  Topic {i + 1}: {word}\n")
            f.write("\n")
    
    print(f"Top words saved to {filename}")


def evaluate_model(top_words, train_theta, test_theta, dataset, train_times):
    """
    Evaluate the trained model using various metrics.
    
    Args:
        top_words: Top words for each time period and topic
        train_theta: Training document-topic distributions
        test_theta: Test document-topic distributions
        dataset: Dataset object
        train_times: Training time indices
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    
    # Compute topic coherence
    print("Computing dynamic topic coherence...")
    dynamic_TC = dynamic_coherence(dataset.train_texts, train_times, dataset.vocab, top_words)
    print(f"Dynamic Topic Coherence (TC): {dynamic_TC:.4f}")
    
    # Compute topic diversity
    print("Computing dynamic topic diversity...")
    dynamic_TD = dynamic_diversity(top_words, dataset.train_bow.cpu().numpy(), train_times, dataset.vocab)
    print(f"Dynamic Topic Diversity (TD): {dynamic_TD:.4f}")
    
    # Evaluate clustering
    print("Evaluating clustering performance...")
    cluster = _clustering(test_theta, dataset.test_labels)
    purity = cluster['Purity']
    nmi = cluster['NMI']
    print(f"Clustering Purity: {purity:.4f}")
    print(f"Clustering NMI: {nmi:.4f}")
    
    # Evaluate classification
    print("Evaluating classification performance...")
    clf = _cls(train_theta, test_theta, dataset.train_labels, dataset.test_labels)
    acc = clf['acc']
    f1 = clf['macro-F1']
    print(f"Classification Accuracy: {acc:.4f}")
    print(f"Classification F1-Score: {f1:.4f}")
    
    # Compute comprehensive evaluation metrics
    print("Computing comprehensive temporal metrics...")
    evaluation_results = evaluate_dynamic_topic_model(
        top_words_all_topics=top_words,
        dataset=dataset,
        train_texts=dataset.train_texts,
        train_times=train_times,
        window_size=2
    )
    
    # Extract and display results
    ttq_avg = evaluation_results['TTQ_avg']
    dtq = evaluation_results['DTQ']
    tq = evaluation_results['TQ']
    ttc = evaluation_results['TTC']
    tts = evaluation_results['TTS']
    ttq = evaluation_results['TTQ']
    tq_avg = evaluation_results['TQ_avg']
    
    print(f"Temporal Topic Quality Average (TTQ_avg): {ttq_avg:.4f}")
    print(f"Dynamic Topic Quality (DTQ): {dtq:.4f}")
    print(f"Temporal Topic Coherence (TTC): {ttc:.4f}")
    print(f"Temporal Topic Smoothness (TTS): {tts:.4f}")
    print(f"Temporal Topic Quality (TTQ): {ttq:.4f}")
    print(f"Topic Quality Average (TQ_avg): {tq_avg:.4f}")
    
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)
    
    # Return all metrics for potential further use
    return {
        'dynamic_TC': dynamic_TC,
        'dynamic_TD': dynamic_TD,
        'purity': purity,
        'nmi': nmi,
        'accuracy': acc,
        'f1_score': f1,
        'ttq_avg': ttq_avg,
        'dtq': dtq,
        'ttc': ttc,
        'tts': tts,
        'ttq': ttq,
        'tq_avg': tq_avg
    }


if __name__ == "__main__":
    main()