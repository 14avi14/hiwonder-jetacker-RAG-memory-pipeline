"""vector_store.py - a RAG based memory store

This script contains the VectorStore class used in the pipeline to store
both embedded textual data in addition to location and temporal data.

The external library used for the embedding is `sentence_transformers`,
while the default embedding model is the all-MiniLM-L6-v2.
"""

import numpy as np
from sentence_transformers import SentenceTransformer # For light-weight text embeddings

class VectorStore:
    """RAG databased for storing textual data along with numerical data.

    The VectorStore class is used for storing primarily embedded text data.
    It features the ability to store that data(see `insert` method), and
    search for text semantically similar to a query based on vector
    embeddings(`text_query` method)

    It can also store other information, such as location or time. Searches
    can also be done based on these, through the
    `euclidian_similarity_query` method.

    Attributes:
        model: model for embedding text
        docs: stores all information, e.g. location, time, text
        embeddings: NumPy array storing the embedded text
    """
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.docs = []
        self.embeddings = None

    def insert(self, docs):
        """Takes in data and stores it in vector DB

        Args:
            docs: a list of dictionaries, each containing attributes.
                The "text" attribute will always be embedded, while the
                others will simply be stored in the docs attribute.
        """
        # docs should contain dictionaries of data(e.g. {"text": ..., "subject": ...}
        texts = [d["text"] for d in docs]

        # Embeddings
        embeddings = self.model.encode(texts)

        # Optimization: normalize so that the dot product is the cosine
        embedding_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        new_embeddings = embeddings / embedding_norms

        # Add to embeddings array
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        self.docs.extend(docs)

    def text_query(self, query, top_k=5):
        """"Takes query and searches for 5 most semantically similar"""
        q_embedding = self.model.encode(query)

        scores = q_embedding @ self.embeddings.T

        indices_gl = np.argsort(scores)[::-1] # Indicies of greatest to least similar

        results = []
        
        for i in indices_gl[:top_k]:
            results.append({"data": self.docs[i], "similarity": scores[i]})

        return results

    def euclidian_similarity_query(self, arr, field, top_k=5):
        """Takes `arr`(a float array) and searches for 5 closest in DB"""
        dists = [np.linalg.norm(np.subtract(d[field], arr)) for d in self.docs]

        indices_lg = np.argsort(dists)

        results = []

        for i in indices_lg[:top_k]:
            results.append({"data": self.docs[i], "distance": dists[i]})

        return results
        
