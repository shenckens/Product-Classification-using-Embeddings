import json
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pickle
import re
from sklearn.metrics.pairwise import cosine_similarity


def load_embeddings(path):
    pkl = path[:-3] + "pkl"
    if os.path.exists(pkl):
        print(f"[*] Loading {pkl}")
        with open(pkl, 'rb') as f:
            embeddings = pickle.load(f)
        return embeddings

    embeddings = {}
    print(f"[*] Loading {path}")
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            embedding = np.array(values[1:], dtype='float32')
            embeddings[word] = embedding

    with open(pkl, "wb") as f:
        print(f"[*] Creating {pkl} for quicker re-accessing.")
        pickle.dump(embeddings, f)
    return embeddings


def sentence_embedding(sentence, embeddings):
    sentence = sentence.lower()
    regex = r'(?<![0-9])[\.,](?![0-9])|[^\w\s,.]'
    sentence = re.sub(regex, ' ', sentence)
    tokens = nltk.tokenize.word_tokenize(sentence)
    sentence_vector = np.mean([embeddings[token] if token in embeddings else np.zeros(100) for token in tokens], axis=0)
    return sentence_vector


def load_product_data(path):
    with open(path, 'r', encoding="utf-8") as f:
        print(f"[*] Loading {path}")
        data = json.load(f)
    return data


def category_embeddings(product_data, embeddings):
    cat_embeddings = dict()
    print("[*] Computing product embedding list for each category.")
    for product in product_data:
        if product["name"] is not None:
            for cat in product["category"]:
                if cat not in cat_embeddings:
                    cat_embeddings[cat] = []
                product_embedding = sentence_embedding(product["name"], embeddings)
                cat_embeddings[cat].append(product_embedding)
    print("[*] Averaging product embeddings for each category.")
    for cat, emb in cat_embeddings.items():
        cat_embeddings[cat] = np.mean(emb, axis=0)
    return cat_embeddings


if __name__ == "__main__":
    
    glove = "inputs/glove.6B.100d.txt"
    embeddings = load_embeddings(glove)
    products = load_product_data("inputs/products.json")
    products_test = load_product_data("inputs/products.test.json")
    cat_emb = category_embeddings(products, embeddings)

    correct = 0
    total = 0
    x = []
    y = []

    print("[*] Calculating similarity scores for products.test")
    for product in products_test:
        product_embedding = sentence_embedding(product["name"], embeddings)
        if np.any(product_embedding):
            similarities = {}
            for cat, emb in cat_emb.items():
                similarities[cat] = cosine_similarity(
                    [product_embedding], [emb])[0][0]
            predicted_category = max(similarities, key=similarities.get)
            total += 1
            if predicted_category in product["category"]:
                correct += 1
                print("Correct.")
            else:
                print("Incorrect.")
            y.append(correct/total)
            x.append(total)

    accuracy = correct / total if total > 0 else 0
    print("Accuracy:", accuracy)

    plt.plot(x, y)
    plt.title(f'Prediction Accuracy ({round(accuracy * 100, 3)}% after all products)')
    plt.xlabel('Product count')
    plt.ylabel('Accuracy')
    plt.show()
