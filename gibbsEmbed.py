import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from test_dataset import sports_corpus,politics_corpus,tech_war_science_corpus,car_corpus,movies_corpus
from load_20_news_group import process_20newsgroups
class EmbeddingGibbsSampler:
    def __init__(self, embedding_dimension, alpha=0.1, beta=0.01, iters=1000):
        self.embedding_dimension = embedding_dimension
        self.alpha = alpha
        self.beta = beta
        self.iters = iters

    def _initialize(self, pairs):
        self.vocab = list(set(word for pair in pairs for word in pair))
        self.word2id = {word: idx for idx, word in enumerate(self.vocab)}
        self.pair_context_counts = np.zeros((len(pairs), self.embedding_dimension)) + self.alpha
        self.context_word_counts = np.zeros((self.embedding_dimension, len(self.vocab))) + self.beta
        self.context_counts = np.zeros(self.embedding_dimension) + self.beta * len(self.vocab)
        self.contexts = [random.randint(0, self.embedding_dimension - 1) for _ in range(len(pairs))]
        self.word_embeddings = np.zeros((len(self.vocab), self.embedding_dimension))

        for p, pair in enumerate(pairs):
            word1, word2 = pair
            context = self.contexts[p]
            self.pair_context_counts[p, context] += 1
            self.context_word_counts[context, self.word2id[word1]] += 1
            self.context_word_counts[context, self.word2id[word2]] += 1
            self.context_counts[context] += 2
            self.word_embeddings[self.word2id[word1], context] += 1
            self.word_embeddings[self.word2id[word2], context] += 1

    def _sample_new_context(self, d, word1_id, word2_id, current_context):
        # Decrement counts
        self.pair_context_counts[d, current_context] -= 1
        self.context_word_counts[current_context, word1_id] -= 1
        self.context_word_counts[current_context, word2_id] -= 1
        self.context_counts[current_context] -= 2
        self.word_embeddings[word1_id, current_context] -= 1
        self.word_embeddings[word2_id, current_context] -= 1

        # Calculate new context distribution
        prob_dist = (self.pair_context_counts[d] *
                     self.context_word_counts[:, word1_id] *
                     self.context_word_counts[:, word2_id] /
                     self.context_counts)
        prob_dist /= np.sum(prob_dist)
        new_context = np.random.choice(np.arange(self.embedding_dimension), p=prob_dist)

        # Increment counts
        self.pair_context_counts[d, new_context] += 1
        self.context_word_counts[new_context, word1_id] += 1
        self.context_word_counts[new_context, word2_id] += 1
        self.context_counts[new_context] += 2
        self.word_embeddings[word1_id, new_context] += 1
        self.word_embeddings[word2_id, new_context] += 1

        return new_context
    def fit(self, pairs, target_words, top_n=10, print_every=100):
        self._initialize(pairs)
        for it in tqdm(range(self.iters), desc="Iteration"):
            if it % print_every == 0:
                print(f"\nIteration {it}:")
                self.find_closest_words_for_targets(target_words, top_n)
            for p, pair in tqdm(enumerate(pairs), total=len(pairs), desc="Pair", leave=False):
                word1, word2 = pair
                current_context = self.contexts[p]
                word1_id = self.word2id[word1]
                word2_id = self.word2id[word2]
                new_context = self._sample_new_context(p, word1_id, word2_id, current_context)
                self.contexts[p] = new_context

    def find_closest_words(self, word, top_n):
        if word not in self.word2id:
            raise ValueError(f"Word '{word}' not found in the vocabulary.")
        word_id = self.word2id[word]
        word_embedding = self.word_embeddings[word_id]
        similarities = cosine_similarity([word_embedding], self.word_embeddings)[0]
        most_similar_indices = similarities.argsort()[::-1][:top_n+1]
        most_similar_words = [(self.vocab[idx], similarities[idx]) for idx in most_similar_indices if idx != word_id]
        return most_similar_words[:top_n]

    def find_closest_words_for_targets(self, target_words, top_n=10):
        for target_word in target_words:
            try:
                closest_words = self.find_closest_words(target_word, top_n)
                closest_words_str = ", ".join(f"{word} ({similarity:.4f})" for word, similarity in closest_words)
                print(f"{target_word}: {closest_words_str}")
                print("")
            except ValueError as e:
                print(f"{target_word}: {e}")


def preprocess_documents(documents):
    preprocessed_docs = []
    for doc in documents:
        words = doc.split()  # Split the document string into words
        # Remove empty strings and whitespace from words
        words = [word.strip() for word in words if word.strip()]
        preprocessed_docs.append(words)
    return preprocessed_docs

def create_word_pairs(corpus, window_size=1):
    pairs = []
    for doc in corpus:
        words = doc.split()
        for i in range(len(words)):
            for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
                if i != j:  # Avoid pairing the word with itself
                    pairs.append((words[i], words[j]))
                    #print(words[i],words[j])
    return pairs


_20_news_group = process_20newsgroups()[:1000]
corpus = sports_corpus + politics_corpus + tech_war_science_corpus + car_corpus + movies_corpus + _20_news_group #+ nytimes

random.shuffle(corpus)
pairs = create_word_pairs(corpus, window_size=4)

model = EmbeddingGibbsSampler(embedding_dimension=100, alpha=0.1, beta=0.01, iters=400)

target_words = ["political", "soccer", "computer", "military","film","cpu","car","war"]

model.fit(pairs, target_words, top_n=10, print_every=10)