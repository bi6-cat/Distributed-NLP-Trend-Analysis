"""
models/bertopic_model.py - BERTopic với PhoBERT cho tiếng Việt

Kiến trúc:
    Document → PhoBERT (768-dim) → UMAP (5-dim) → HDBSCAN → Topics
    
Components:
    1. PhoBERT: Contextual embeddings (vinai/phobert-base)
    2. UMAP: Dimensionality reduction (768 → 5 dim)
    3. HDBSCAN: Density-based clustering
    4. c-TF-IDF: Topic word extraction

Usage:
    # Train
    model = VietnameseBERTopicModel(n_neighbors=15, min_cluster_size=15)
    topics, probs = model.fit(documents)
    
    # Analysis
    info = model.get_topic_info()
    coherence = model.calculate_coherence(documents)
    
    # Save/Load
    model.save("output/bertopic_model/")
    model = VietnameseBERTopicModel.load("output/bertopic_model/")
    
    # Inference
    new_topics, new_probs = model.transform(new_documents)

Author: Member 3 (ML Engineer)
Date: 2026-04-04
"""

import os
import pickle
import warnings
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd

# BERTopic components
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

# Coherence calculation
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

# GPU detection
import torch

warnings.filterwarnings('ignore')


class VietnameseBERTopicModel:
    """
    Wrapper cho BERTopic với PhoBERT embeddings
    
    Attributes:
        embedding_model (SentenceTransformer): PhoBERT model
        umap_model (UMAP): Dimensionality reducer
        hdbscan_model (HDBSCAN): Clusterer
        topic_model (BERTopic): Main BERTopic instance
        topics_ (List[int]): Topic assignments
        probs_ (np.ndarray): Topic probabilities
        device (str): 'cuda' or 'cpu'
    """
    
    def __init__(
        self,
        embedding_model: str = "vinai/phobert-base",
        n_neighbors: int = 15,
        n_components: int = 5,
        min_dist: float = 0.0,
        min_cluster_size: int = 15,
        min_samples: int = 10,
        top_n_words: int = 10,
        verbose: bool = True,
        use_gpu: bool = True
    ):
        """
        Khởi tạo VietnameseBERTopicModel
        
        Args:
            embedding_model: Model name hoặc path (default: PhoBERT)
            n_neighbors: UMAP n_neighbors (5-30)
                - Nhỏ: local structure, nhiều clusters nhỏ
                - Lớn: global structure, ít clusters lớn
            n_components: UMAP output dimensions (3-10)
                - Nhỏ: compress nhiều, mất info
                - Lớn: giữ info, chậm hơn
            min_dist: UMAP min_dist (0.0-0.5)
                - 0.0: clusters chặt
                - 0.5: clusters loose
            min_cluster_size: HDBSCAN min cluster size (10-50)
                - Nhỏ: nhiều topics nhỏ
                - Lớn: ít topics lớn
            min_samples: HDBSCAN min samples (5-20)
                - Nhỏ: ít noise points
                - Lớn: nhiều noise points, conservative
            top_n_words: Số words per topic (5-15)
            verbose: Print progress
            use_gpu: Sử dụng GPU nếu có (default: True)
        """
        self.verbose = verbose
        
        # GPU detection
        if use_gpu and torch.cuda.is_available():
            self.device = 'cuda'
            if self.verbose:
                gpu_name = torch.cuda.get_device_name(0)
                print(f"🚀 GPU detected: {gpu_name}")
        else:
            self.device = 'cpu'
            if self.verbose and use_gpu:
                print("⚠️  GPU not available, using CPU")
        
        # Save hyperparameters
        self.embedding_model_name = embedding_model
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_dist = min_dist
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.top_n_words = top_n_words
        
        # [1] Initialize PhoBERT embedding model
        if self.verbose:
            print(f"\n[1/4] Loading embedding model: {embedding_model}")
        
        self.embedding_model = SentenceTransformer(
            embedding_model,
            device=self.device
        )
        
        if self.verbose:
            print(f"✅ Embedding model loaded (device: {self.device})")
        
        # [2] Initialize UMAP
        if self.verbose:
            print(f"\n[2/4] Configuring UMAP")
            print(f"  - n_neighbors: {n_neighbors}")
            print(f"  - n_components: {n_components}")
            print(f"  - min_dist: {min_dist}")
        
        self.umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            metric='cosine',
            random_state=42
        )
        
        if self.verbose:
            print("✅ UMAP configured")
        
        # [3] Initialize HDBSCAN
        if self.verbose:
            print(f"\n[3/4] Configuring HDBSCAN")
            print(f"  - min_cluster_size: {min_cluster_size}")
            print(f"  - min_samples: {min_samples}")
        
        self.hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        if self.verbose:
            print("✅ HDBSCAN configured")
        
        # [4] Initialize BERTopic
        if self.verbose:
            print(f"\n[4/4] Building BERTopic pipeline")
        
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            top_n_words=top_n_words,
            verbose=verbose,
            calculate_probabilities=True
        )
        
        if self.verbose:
            print("✅ BERTopic pipeline ready")
            print(f"\n{'='*60}")
            print("Model initialized successfully!")
            print(f"{'='*60}\n")
        
        # Placeholders for training results
        self.topics_ = None
        self.probs_ = None
    
    def fit(self, documents: List[str]) -> Tuple[List[int], np.ndarray]:
        """
        Train BERTopic trên documents
        
        Pipeline:
            Input → PhoBERT → UMAP → HDBSCAN → c-TF-IDF → Output
        
        Args:
            documents: List of text documents (preprocessed clean_text)
        
        Returns:
            topics: Topic assignments cho mỗi document (-1 = outlier)
            probs: Topic probabilities (shape: n_docs × n_topics)
        
        Example:
            >>> model = VietnameseBERTopicModel()
            >>> topics, probs = model.fit(documents)
            >>> print(f"Found {len(set(topics))} topics")
            >>> print(f"Outliers: {sum(t == -1 for t in topics)}")
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print("TRAINING BERTOPIC MODEL")
            print(f"{'='*60}")
            print(f"Documents: {len(documents)}")
            print(f"Device: {self.device}")
        
        # Validate input
        if not documents:
            raise ValueError("documents cannot be empty")
        
        if not all(isinstance(doc, str) for doc in documents):
            raise ValueError("All documents must be strings")
        
        # Train BERTopic
        if self.verbose:
            print("\nStarting training...")
            print("  [1/4] Encoding documents (PhoBERT)...")
        
        topics, probs = self.topic_model.fit_transform(documents)
        
        # Store results
        self.topics_ = topics
        self.probs_ = probs
        
        # Summary
        if self.verbose:
            n_topics = len(set(topics)) - (1 if -1 in topics else 0)
            n_outliers = sum(t == -1 for t in topics)
            
            print(f"\n{'='*60}")
            print("TRAINING COMPLETED")
            print(f"{'='*60}")
            print(f"✅ Topics found: {n_topics}")
            print(f"✅ Outliers: {n_outliers} ({n_outliers/len(topics)*100:.1f}%)")
            print(f"✅ Documents assigned: {len(topics) - n_outliers}")
            print(f"{'='*60}\n")
        
        return topics, probs
    
    def transform(self, documents: List[str]) -> Tuple[List[int], np.ndarray]:
        """
        Predict topics cho new documents (inference)
        
        Args:
            documents: New documents to predict
        
        Returns:
            topics: Topic assignments
            probs: Topic probabilities
        
        Raises:
            ValueError: Nếu model chưa được train
        
        Example:
            >>> new_topics, new_probs = model.transform(new_docs)
        """
        if self.topics_ is None:
            raise ValueError(
                "Model chưa được train. Hãy gọi fit() trước hoặc load() model đã train."
            )
        
        if self.verbose:
            print(f"\nPredicting topics for {len(documents)} documents...")
        
        topics, probs = self.topic_model.transform(documents)
        
        if self.verbose:
            print("✅ Prediction completed")
        
        return topics, probs
    
    def get_topic_info(self) -> pd.DataFrame:
        """
        Lấy thông tin tất cả topics
        
        Returns:
            DataFrame với columns:
              - Topic: Topic ID (-1, 0, 1, 2, ...)
              - Count: Số documents trong topic
              - Name: Tên topic (auto-generated từ top words)
              - Representation: Top words với scores
        
        Example:
            >>> info = model.get_topic_info()
            >>> print(info.head())
               Topic  Count                    Name
            0     -1     15              -1_outlier
            1      0     45  0_iphone_pro_max_camera
            2      1     32     1_samsung_galaxy_note
        """
        if self.topics_ is None:
            raise ValueError("Model chưa được train")
        
        return self.topic_model.get_topic_info()
    
    def get_topics(self, topic_id: Optional[int] = None) -> Dict[int, List[Tuple[str, float]]]:
        """
        Lấy top words cho topic(s)
        
        Args:
            topic_id: ID của topic cần lấy. Nếu None, lấy tất cả.
        
        Returns:
            Dict mapping topic_id → List[(word, score)]
        
        Example:
            >>> # Lấy topic 0
            >>> topic_0 = model.get_topics(0)
            >>> print(topic_0[0][:3])  # Top 3 words
            [('iphone', 0.45), ('pro', 0.38), ('max', 0.35)]
            
            >>> # Lấy tất cả
            >>> all_topics = model.get_topics()
            >>> print(f"Total topics: {len(all_topics)}")
        """
        if self.topics_ is None:
            raise ValueError("Model chưa được train")
        
        if topic_id is not None:
            topic = self.topic_model.get_topic(topic_id)
            if topic:
                return {topic_id: topic}
            else:
                return {}
        else:
            # Return all topics
            all_topics = {}
            for tid in set(self.topics_):
                if tid != -1:  # Skip outliers
                    topic = self.topic_model.get_topic(tid)
                    if topic:
                        all_topics[tid] = topic
            return all_topics
    
    def calculate_coherence(
        self,
        documents: List[str],
        coherence_type: str = 'c_v'
    ) -> float:
        """
        Tính Coherence score để so sánh với LDA
        
        Coherence C_V đo "chất lượng" của topics:
        - Dựa trên co-occurrence của top words
        - Range: 0.0-1.0 (càng cao càng tốt)
        - Typical good score: > 0.35
        
        Args:
            documents: Original documents (để tính co-occurrence)
            coherence_type: 'c_v' (default), 'u_mass', 'c_uci', 'c_npmi'
        
        Returns:
            Coherence score (0.0-1.0)
        
        Example:
            >>> coherence = model.calculate_coherence(documents)
            >>> print(f"Coherence C_V: {coherence:.4f}")
            Coherence C_V: 0.4123
        """
        if self.topics_ is None:
            raise ValueError("Model chưa được train")
        
        if self.verbose:
            print(f"\nCalculating coherence ({coherence_type})...")
        
        # Get topics (exclude outliers)
        topics_dict = self.get_topics()
        
        if not topics_dict:
            if self.verbose:
                print("⚠️  No topics found (all outliers)")
            return 0.0
        
        # Extract top words for each topic
        topics_words = []
        for topic_id in sorted(topics_dict.keys()):
            words = [word for word, score in topics_dict[topic_id]]
            topics_words.append(words)
        
        # Tokenize documents
        texts = [doc.split() for doc in documents]
        
        # Create Gensim dictionary
        dictionary = Dictionary(texts)
        
        # Calculate coherence
        coherence_model = CoherenceModel(
            topics=topics_words,
            texts=texts,
            dictionary=dictionary,
            coherence=coherence_type
        )
        
        coherence_score = coherence_model.get_coherence()
        
        if self.verbose:
            print(f"✅ Coherence {coherence_type.upper()}: {coherence_score:.4f}")
        
        return coherence_score
    
    def save(self, path: str) -> None:
        """
        Lưu trained model vào folder
        
        Saves:
            - bertopic_model/       (BERTopic model files)
            - config.pkl            (Hyperparameters)
            - topics.pkl            (Topic assignments & probs)
        
        Args:
            path: Folder path (e.g., "output/bertopic_model/")
        
        Example:
            >>> model.save("output/bertopic_model/")
            >>> # Can reload later with: model.load("output/bertopic_model/")
        """
        if self.verbose:
            print(f"\nSaving model to: {path}")
        
        # Create folder
        os.makedirs(path, exist_ok=True)
        
        # [1] Save BERTopic model
        bertopic_path = os.path.join(path, "bertopic_model")
        self.topic_model.save(bertopic_path, serialization="pickle")
        
        if self.verbose:
            print(f"✅ BERTopic model saved to {bertopic_path}")
        
        # [2] Save config
        config = {
            'embedding_model': self.embedding_model_name,
            'n_neighbors': self.n_neighbors,
            'n_components': self.n_components,
            'min_dist': self.min_dist,
            'min_cluster_size': self.min_cluster_size,
            'min_samples': self.min_samples,
            'top_n_words': self.top_n_words,
            'device': self.device
        }
        
        config_path = os.path.join(path, "config.pkl")
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)
        
        if self.verbose:
            print(f"✅ Config saved to {config_path}")
        
        # [3] Save topics/probs if available
        if self.topics_ is not None:
            results = {
                'topics': self.topics_,
                'probs': self.probs_
            }
            
            results_path = os.path.join(path, "topics.pkl")
            with open(results_path, 'wb') as f:
                pickle.dump(results, f)
            
            if self.verbose:
                print(f"✅ Topics/probs saved to {results_path}")
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("Model saved successfully!")
            print(f"{'='*60}\n")
    
    @classmethod
    def load(cls, path: str, verbose: bool = True) -> 'VietnameseBERTopicModel':
        """
        Load saved model từ folder
        
        Args:
            path: Folder path
            verbose: Print progress
        
        Returns:
            VietnameseBERTopicModel instance
        
        Example:
            >>> model = VietnameseBERTopicModel.load("output/bertopic_model/")
            >>> new_topics = model.transform(new_docs)
        """
        if verbose:
            print(f"\nLoading model from: {path}")
        
        # [1] Load config
        config_path = os.path.join(path, "config.pkl")
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        
        if verbose:
            print(f"✅ Config loaded")
        
        # [2] Create instance
        instance = cls(
            embedding_model=config['embedding_model'],
            n_neighbors=config['n_neighbors'],
            n_components=config['n_components'],
            min_dist=config['min_dist'],
            min_cluster_size=config['min_cluster_size'],
            min_samples=config['min_samples'],
            top_n_words=config['top_n_words'],
            verbose=verbose
        )
        
        # [3] Load BERTopic model
        bertopic_path = os.path.join(path, "bertopic_model")
        instance.topic_model = BERTopic.load(bertopic_path)
        
        if verbose:
            print(f"✅ BERTopic model loaded")
        
        # [4] Load topics/probs if available
        results_path = os.path.join(path, "topics.pkl")
        if os.path.exists(results_path):
            with open(results_path, 'rb') as f:
                results = pickle.load(f)
            
            instance.topics_ = results['topics']
            instance.probs_ = results['probs']
            
            if verbose:
                print(f"✅ Topics/probs loaded")
        
        if verbose:
            print(f"\n{'='*60}")
            print("Model loaded successfully!")
            print(f"{'='*60}\n")
        
        return instance
    
    def visualize_topics(self):
        """
        Visualize topics trong 2D space (interactive Plotly)
        
        Returns:
            Plotly figure
        
        Example:
            >>> fig = model.visualize_topics()
            >>> fig.show()
            >>> # Or save: fig.write_html("topics_viz.html")
        """
        if self.topics_ is None:
            raise ValueError("Model chưa được train")
        
        return self.topic_model.visualize_topics()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Lấy summary của model (để logging/reporting)
        
        Returns:
            Dict với thông tin: n_topics, n_outliers, hyperparameters, etc.
        """
        if self.topics_ is None:
            return {
                'status': 'not_trained',
                'device': self.device,
                'hyperparameters': {
                    'n_neighbors': self.n_neighbors,
                    'n_components': self.n_components,
                    'min_dist': self.min_dist,
                    'min_cluster_size': self.min_cluster_size,
                    'min_samples': self.min_samples,
                    'top_n_words': self.top_n_words
                }
            }
        
        n_topics = len(set(self.topics_)) - (1 if -1 in self.topics_ else 0)
        n_outliers = sum(t == -1 for t in self.topics_)
        
        return {
            'status': 'trained',
            'n_documents': len(self.topics_),
            'n_topics': n_topics,
            'n_outliers': n_outliers,
            'outlier_ratio': n_outliers / len(self.topics_),
            'hyperparameters': {
                'embedding_model': self.embedding_model_name,
                'n_neighbors': self.n_neighbors,
                'n_components': self.n_components,
                'min_dist': self.min_dist,
                'min_cluster_size': self.min_cluster_size,
                'min_samples': self.min_samples,
                'top_n_words': self.top_n_words
            },
            'device': self.device
        }


# Helper function cho batch processing (nếu dataset lớn)
def train_bertopic_batch(
    documents: List[str],
    batch_size: int = 1000,
    **kwargs
) -> VietnameseBERTopicModel:
    """
    Train BERTopic với batch processing (cho dataset lớn)
    
    Args:
        documents: All documents
        batch_size: Number of docs per batch (default: 1000)
        **kwargs: Arguments cho VietnameseBERTopicModel
    
    Returns:
        Trained model
    """
    if len(documents) <= batch_size:
        # No need for batching
        model = VietnameseBERTopicModel(**kwargs)
        model.fit(documents)
        return model
    
    print(f"⚠️  Large dataset ({len(documents)} docs)")
    print(f"   Consider using full dataset (BERTopic handles it well)")
    print(f"   Or subsample for faster training\n")
    
    # For now, just train on full dataset
    model = VietnameseBERTopicModel(**kwargs)
    model.fit(documents)
    return model


if __name__ == "__main__":
    # Quick test
    print("VietnameseBERTopicModel module loaded successfully!")
    print(f"GPU available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
