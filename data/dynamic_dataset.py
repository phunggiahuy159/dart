import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.sparse
from .file_utils import read_text


class _SequentialDataset(Dataset):
    def __init__(self, bow, times, time_wordfreq, doc_tfidf):
        super().__init__()
        self.bow = bow
        self.times = times
        self.time_wordfreq = time_wordfreq
        self.doc_tfidf = doc_tfidf

    def __len__(self):
        return len(self.bow)

    def __getitem__(self, index):
        time_idx = self.times[index]
        return_dict = {
            'bow': self.bow[index],
            'times': time_idx,
            'time_wordfreq': self.time_wordfreq[time_idx],
            'doc_tfidf': self.doc_tfidf[time_idx]
        }
        return return_dict


class DynamicDataset:
    def __init__(self, dataset_dir, batch_size=200, read_labels=False, device='cpu', as_tensor=True):
        self.load_data(dataset_dir, read_labels)
        
        # Sort data by timestamps in ascending order
        self.sort_by_time()

        self.vocab_size = len(self.vocab)
        self.train_size = len(self.train_bow)
        self.num_times = len(np.unique(self.train_times))
        self.train_time_wordfreq = self.get_time_wordfreq(self.train_bow, self.train_times)
        
        # Precompute document-level TF-IDF
        self.doc_tfidf = self.precompute_document_tfidf(self.train_bow, self.train_times)

        print('train size: ', len(self.train_bow))
        print('test size: ', len(self.test_bow))
        print('vocab size: ', len(self.vocab))
        print('average length: {:.3f}'.format(self.train_bow.sum(1).mean().item()))
        print('num of each time slice: ', self.num_times, np.bincount(self.train_times))

        if as_tensor:
            self.doc_tfidf = torch.from_numpy(self.doc_tfidf).float().to(device)
            self.train_bow = torch.from_numpy(self.train_bow).float().to(device)
            self.test_bow = torch.from_numpy(self.test_bow).float().to(device)
            self.train_times = torch.from_numpy(self.train_times).long().to(device)
            self.test_times = torch.from_numpy(self.test_times).long().to(device)
            self.train_time_wordfreq = torch.from_numpy(self.train_time_wordfreq).float().to(device)

            self.train_dataset = _SequentialDataset(self.train_bow, self.train_times, self.train_time_wordfreq, self.doc_tfidf)
            self.test_dataset = _SequentialDataset(self.test_bow, self.test_times, self.train_time_wordfreq, self.doc_tfidf)

            # Set shuffle=False to maintain time order
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False)

    def load_data(self, path, read_labels):
        self.train_bow = scipy.sparse.load_npz(f'{path}/train_bow.npz').toarray().astype('float32')
        self.test_bow = scipy.sparse.load_npz(f'{path}/test_bow.npz').toarray().astype('float32')
        self.word_embeddings = scipy.sparse.load_npz(f'{path}/word_embeddings.npz').toarray().astype('float32')

        self.train_texts = read_text(f'{path}/train_texts.txt')
        self.test_texts = read_text(f'{path}/test_texts.txt')

        self.train_times = np.loadtxt(f'{path}/train_times.txt').astype('int32')
        self.test_times = np.loadtxt(f'{path}/test_times.txt').astype('int32')

        self.vocab = read_text(f'{path}/vocab.txt')

        self.pretrained_WE = scipy.sparse.load_npz(f'{path}/word_embeddings.npz').toarray().astype('float32')

        if read_labels:
            self.train_labels = np.loadtxt(f'{path}/train_labels.txt').astype('int32')
            self.test_labels = np.loadtxt(f'{path}/test_labels.txt').astype('int32')

    def precompute_document_tfidf(self, bow, times):
        time_tfidf = np.zeros((self.num_times, self.vocab_size))
        print("BOW shape:", bow.shape)
        print("Number of documents:", len(bow))
        print("Vocabulary size:", bow.shape[1])
        print("Number of time slices:", len(np.unique(times)))

        for t in range(self.num_times):
            # Get documents from this time slice
            idx = np.where(times == t)[0]
            docs_in_time = bow[idx]
            
            # Print how many documents are in this time slice
            print(f"Time slice {t}: {len(docs_in_time)} documents")
            print("doc_in_time shape", docs_in_time.shape)
            
            # Skip if no documents
            if len(docs_in_time) == 0:
                continue

            # Calculate TF for each document
            doc_lengths = np.sum(docs_in_time, axis=1, keepdims=True) + 1e-10
            tf = docs_in_time / doc_lengths

            # Calculate IDF using true document frequency
            df = np.sum(docs_in_time > 0, axis=0)  # In how many docs each word appears
            idf = np.log(len(docs_in_time) / (df + 1e-10))

            # Calculate TF-IDF for each document
            tfidf = tf * idf

            # Average TF-IDF across documents for this time slice
            time_tfidf[t] = np.mean(tfidf, axis=0)

        return time_tfidf

    def sort_by_time(self):
        """Sort all data arrays by timestamps in ascending order"""
        # Sort training data
        train_indices = np.argsort(self.train_times)
        self.train_times = self.train_times[train_indices]
        self.train_bow = self.train_bow[train_indices]

        # Update train_texts if available
        if hasattr(self, 'train_texts') and self.train_texts:
            self.train_texts = [self.train_texts[i] for i in train_indices]

        # Update train_labels if available
        if hasattr(self, 'train_labels'):
            self.train_labels = self.train_labels[train_indices]

        # Sort test data
        test_indices = np.argsort(self.test_times)
        self.test_times = self.test_times[test_indices]
        self.test_bow = self.test_bow[test_indices]

        # Update test_texts if available
        if hasattr(self, 'test_texts') and self.test_texts:
            self.test_texts = [self.test_texts[i] for i in test_indices]

        # Update test_labels if available
        if hasattr(self, 'test_labels'):
            self.test_labels = self.test_labels[test_indices]
        print("Data sorted by timestamp in ascending order")
    # word frequency at each time slice.
    def get_time_wordfreq(self, bow, times):
        train_time_wordfreq = np.zeros((self.num_times, self.vocab_size))
        for time in range(self.num_times):
            idx = np.where(times == time)[0]
            train_time_wordfreq[time] += bow[idx].sum(0)
        cnt_times = np.bincount(times)
        train_time_wordfreq = train_time_wordfreq / cnt_times[:, np.newaxis]
        return train_time_wordfreq