"""
models/bertopic_model.py - BERTopic với PhoBERT cho tiếng Việt

Kiến trúc:
    Document → PhoBERT (768-dim) → UMAP (5-dim) → HDBSCAN → Topics
    
Components:
    1. PhoBERT: Contextual embeddings (vinai/phobert-base)
    2. UMAP: Dimensionality reduction (768 → 5 dim)
    3. HDBSCAN: Density-based clustering
    4. c-TF-IDF: Topic word extraction

Output Schema (theo data_flow_schema_evolution.md — Stage 4 & 5):
    /silver/post_topics/  → export_post_topics():
        post_id           String   FK → posts_core.post_id
        topic_id          Int32    BERTopic assignment
        topic_probability Float32  Assignment confidence 0.0–1.0
        model_type        String   'bertopic'
        predicted_at      DateTime Inference timestamp

    /silver/topics/       → export_stg_topics():
        topic_id          Int32
        label             String   Auto-generated từ top keywords
        top_keywords      Array(String)  Top 10 words by c-TF-IDF
        coherence_score   Float32 (nullable)
        model_version     String   'bertopic_v2'
        created_at        DateTime

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

import csv
import os
import pickle
import warnings
from datetime import datetime, timezone
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
        nr_topics: Optional[Any] = None,
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
        self.nr_topics = nr_topics
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
            nr_topics=nr_topics,
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
    
    def fit(
        self,
        documents: List[str],
        embeddings: Optional[np.ndarray] = None,
    ) -> Tuple[List[int], np.ndarray]:
        """
        Train BERTopic trên documents.

        Pipeline (khi không truyền embeddings):
            Input → PhoBERT → UMAP → HDBSCAN → c-TF-IDF → Output

        Pipeline (khi truyền pre-computed embeddings — RECOMMENDED cho tuning):
            Pre-computed embeddings → UMAP → HDBSCAN → c-TF-IDF → Output
            (Bỏ qua bước encode PhoBERT, tiết kiệm ~90% thời gian)

        Args:
            documents: List of text documents (preprocessed clean_text).
            embeddings: Pre-computed PhoBERT embeddings, shape (n_docs, 768).
                Nếu None, model sẽ tự encode bằng SentenceTransformer.
                Phải được tạo bởi cùng SentenceTransformer("vinai/phobert-base")
                để đảm bảo vector space nhất quán.

        Returns:
            topics: Topic assignments cho mỗi document (-1 = outlier)
            probs:  Topic probabilities (shape: n_docs × n_topics)

        Raises:
            ValueError: Nếu documents rỗng, có phần tử không phải str,
                        hoặc embeddings shape không khớp với documents.

        Example:
            >>> # Cách 1: Không cache (tự encode)
            >>> model = VietnameseBERTopicModel()
            >>> topics, probs = model.fit(documents)

            >>> # Cách 2: Dùng pre-computed embeddings (KHUYẾN NGHỊ khi tuning)
            >>> emb = model.encode(documents)          # cache 1 lần
            >>> topics, probs = model.fit(documents, embeddings=emb)
            >>> print(f"Found {len(set(topics))} topics")
            >>> print(f"Outliers: {sum(t == -1 for t in topics)}")
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print("TRAINING BERTOPIC MODEL")
            print(f"{'='*60}")
            print(f"Documents: {len(documents)}")
            print(f"Device: {self.device}")

        # --- Validate documents ---
        if not documents:
            raise ValueError("documents cannot be empty")

        if not all(isinstance(doc, str) for doc in documents):
            raise ValueError("All documents must be strings")

        # --- Validate embeddings nếu được truyền vào ---
        if embeddings is not None:
            if not isinstance(embeddings, np.ndarray):
                raise ValueError(
                    f"embeddings phải là np.ndarray, nhận được {type(embeddings)}"
                )
            if embeddings.shape[0] != len(documents):
                raise ValueError(
                    f"embeddings.shape[0] ({embeddings.shape[0]}) "
                    f"!= len(documents) ({len(documents)}). "
                    "Đảm bảo embeddings được tạo từ cùng list documents."
                )
            if self.verbose:
                print(f"\nStarting training...")
                print(f"  [1/4] Dùng pre-computed embeddings {embeddings.shape} — bỏ qua encode PhoBERT ✅")
        else:
            if self.verbose:
                print("\nStarting training...")
                print("  [1/4] Encoding documents (PhoBERT)...")

        # --- Train BERTopic ---
        # Truyền embeddings=None khi không có (BERTopic tự encode),
        # hoặc truyền ma trận đã cache để skip bước encode.
        topics, probs = self.topic_model.fit_transform(documents, embeddings=embeddings)

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

    def reduce_topics(
        self,
        documents: List[str],
        nr_topics: int
    ) -> None:
        """
        [MAGIC STEP] Gộp các topics hiện tại xuống một số lượng cụ thể.
        Dùng sau khi fit() để tối ưu hóa Dashboard mà không mất đi độ chính xác 
        của việc phân cụm ban đầu.

        Args:
            documents: List documents ban đầu.
            nr_topics: Số lượng topics mục tiêu (ví dụ 70).

        Example:
            >>> model.fit(documents) # Tìm ra 145 topics
            >>> model.reduce_topics(documents, nr_topics=70) # Gộp về 70
        """
        if self.topics_ is None:
            raise ValueError("Model chưa được train. Hãy gọi fit() trước.")

        if self.verbose:
            print(f"\n🪄  Reducing topics from {len(set(self.topics_))-1} to {nr_topics}...")

        self.topic_model.reduce_topics(documents, nr_topics=nr_topics)
        
        # Cập nhật lại topics_ và nr_topics nội bộ
        self.topics_ = self.topic_model.topics_
        self.nr_topics = nr_topics

        if self.verbose:
            print(f"✅ Topics reduced successfully to {len(set(self.topics_))-1}")

    def encode(self, documents: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode documents thành PhoBERT embeddings (dùng để cache trước khi tuning).

        Dùng cùng SentenceTransformer đã khởi tạo trong __init__ để đảm bảo
        vector space nhất quán với production model.

        Args:
            documents:  List of text documents.
            batch_size: Số documents mỗi batch GPU (default 32, T4 GPU ~OK).
                        Giảm xuống 16 nếu gặp CUDA OOM.

        Returns:
            embeddings: np.ndarray shape (n_docs, 768), dtype float32.

        Example:
            >>> # Cache embeddings 1 lần, reuse cho nhiều experiments
            >>> model = VietnameseBERTopicModel()
            >>> embeddings = model.encode(documents, batch_size=32)
            >>> np.save("embeddings_cache.npy", embeddings)

            >>> # Sau đó load và dùng lại:
            >>> embeddings = np.load("embeddings_cache.npy")
            >>> topics, probs = model.fit(documents, embeddings=embeddings)
        """
        if self.verbose:
            print(f"\nEncoding {len(documents):,} documents với PhoBERT...")
            print(f"  batch_size={batch_size}, device={self.device}")

        embeddings = self.embedding_model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=self.verbose,
            convert_to_numpy=True,
            normalize_embeddings=False,  # BERTopic xử lý normalization qua UMAP cosine
        )

        if self.verbose:
            print(f"✅ Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")

        return embeddings
    
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
        # Lưu ý: PhoBERT đã word-segment tiếng Việt (dấu "_" nối từ ghép),
        # nên split() là đúng. Filter token độ dài <= 1 để loại ký tự đơn lẻ
        # (dấu câu, số đơn, ...) không có nghĩa trong co-occurrence.
        texts = [
            [token for token in doc.split() if len(token) > 1]
            for doc in documents
        ]

        # Bỏ qua documents rỗng sau khi filter
        texts = [t for t in texts if t]
        if not texts:
            if self.verbose:
                print("⚠️  Tất cả documents trống sau khi tokenize")
            return 0.0

        # Create Gensim dictionary — filter extremes để coherence ổn định hơn
        dictionary = Dictionary(texts)
        dictionary.filter_extremes(no_below=2, no_above=0.95)
        
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
    
    # =========================================================================
    # SCHEMA-COMPLIANT EXPORT METHODS (data_flow_schema_evolution.md Stage 4/5)
    # =========================================================================

    def export_post_topics(
        self,
        post_ids: List[str],
        output_path: str,
        coherence_score: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Xuất kết quả gán topic theo schema ``stg_post_topics`` (ClickHouse).

        Schema output (data_flow_schema_evolution.md — Stage 5):
            post_id           String   FK → stg_posts_core.post_id
            topic_id          Int32    BERTopic assignment (-1 = outlier)
            topic_probability Float32  Assignment confidence 0.0–1.0
            model_type        String   'bertopic'
            predicted_at      DateTime Inference timestamp (ISO 8601 UTC)

        Args:
            post_ids: Danh sách post_id tương ứng với thứ tự documents lúc fit().
            output_path: Đường dẫn file CSV đầu ra
                (ví dụ: "output/bertopic/stg_post_topics.csv").
            coherence_score: Coherence score tổng thể (không ghi vào bảng này,
                chỉ log để tham khảo).

        Returns:
            pd.DataFrame với 5 cột theo schema stg_post_topics.

        Raises:
            ValueError: Nếu model chưa train hoặc len(post_ids) != len(topics_).

        Example:
            >>> model.fit(documents)
            >>> df = model.export_post_topics(post_ids, "output/stg_post_topics.csv")
            >>> df.columns.tolist()
            ['post_id', 'topic_id', 'topic_probability', 'model_type', 'predicted_at']
        """
        if self.topics_ is None:
            raise ValueError("Model chưa được train. Hãy gọi fit() trước.")

        if len(post_ids) != len(self.topics_):
            raise ValueError(
                f"Số lượng post_ids ({len(post_ids)}) "
                f"không khớp với số topics ({len(self.topics_)})."
            )

        predicted_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Lấy xác suất cho topic được gán (topic_id tương ứng)
        # self.probs_ shape: (n_docs,) hoặc (n_docs, n_topics)
        probs_array = np.array(self.probs_) if self.probs_ is not None else None

        rows = []
        for i, (pid, tid) in enumerate(zip(post_ids, self.topics_)):
            # Tính topic_probability
            if probs_array is not None and probs_array.ndim == 2:
                # BERTopic trả về ma trận (n_docs, n_topics)
                if tid >= 0 and tid < probs_array.shape[1]:
                    prob = float(probs_array[i, tid])
                else:
                    prob = 0.0  # outlier (tid == -1)
            elif probs_array is not None and probs_array.ndim == 1:
                prob = float(probs_array[i])
            else:
                prob = 0.0

            rows.append({
                "post_id": str(pid),
                "topic_id": int(tid),
                "topic_probability": round(prob, 6),
                "model_type": "bertopic",
                "predicted_at": predicted_at,
            })

        df = pd.DataFrame(
            rows,
            columns=["post_id", "topic_id", "topic_probability", "model_type", "predicted_at"],
        )

        # Ép kiểu đúng dtype
        df["topic_id"] = df["topic_id"].astype("int32")
        df["topic_probability"] = df["topic_probability"].astype("float32")

        # Ghi ra CSV
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )
        df.to_csv(output_path, index=False, encoding="utf-8")

        if self.verbose:
            print(f"✅ stg_post_topics → {output_path} ({len(df):,} rows)")
            if coherence_score is not None:
                print(f"   coherence_score (info only): {coherence_score:.4f}")

        return df

    def export_stg_topics(
        self,
        output_path: str,
        coherence_score: Optional[float] = None,
        model_version: str = "bertopic_v2",
    ) -> pd.DataFrame:
        """
        Xuất bảng tra cứu topics theo schema ``stg_topics`` (ClickHouse).

        Schema output (data_flow_schema_evolution.md — Stage 5):
            topic_id        Int32
            label           String    Auto-generated từ top-3 keywords
            top_keywords    Array(String)  Top 10 words by c-TF-IDF
            coherence_score Float32 (nullable)
            model_version   String    e.g. 'bertopic_v2'
            created_at      DateTime  ISO 8601 UTC

        Lưu ý về ``top_keywords``:
            - Trong file CSV: được serialize thành chuỗi JSON ("[\"iphone\",\"pin\"]")
              để dbt/ClickHouse có thể parse bằng JSONExtract.
            - Trong DataFrame trả về: là List[str] thực sự (Array).

        Args:
            output_path: Đường dẫn file CSV đầu ra
                (ví dụ: "output/bertopic/stg_topics.csv").
            coherence_score: Coherence C_V score của model (nullable).
            model_version: Nhãn phiên bản model (mặc định 'bertopic_v2').

        Returns:
            pd.DataFrame với 6 cột theo schema stg_topics.

        Raises:
            ValueError: Nếu model chưa train.

        Example:
            >>> cv = model.calculate_coherence(documents)
            >>> df = model.export_stg_topics("output/stg_topics.csv", coherence_score=cv)
            >>> df.columns.tolist()
            ['topic_id', 'label', 'top_keywords', 'coherence_score', 'model_version', 'created_at']
        """
        if self.topics_ is None:
            raise ValueError("Model chưa được train. Hãy gọi fit() trước.")

        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        topics_dict = self.get_topics()  # {topic_id: [(word, score), ...]}

        rows = []
        for tid in sorted(topics_dict.keys()):
            word_score_list = topics_dict[tid]  # [(word, score), ...]

            # top_keywords: lấy tối đa 10 từ theo c-TF-IDF weight
            top_keywords: List[str] = [
                word for word, _score in word_score_list[:10]
            ]

            # label: ghép 3 từ đầu bằng " | "
            label = " | ".join(top_keywords[:3]) if top_keywords else f"topic_{tid}"

            rows.append({
                "topic_id": int(tid),
                "label": label,
                "top_keywords": top_keywords,  # List[str] — Array(String) trong CH
                "coherence_score": float(coherence_score) if coherence_score is not None else None,
                "model_version": model_version,
                "created_at": created_at,
            })

        df = pd.DataFrame(
            rows,
            columns=["topic_id", "label", "top_keywords", "coherence_score", "model_version", "created_at"],
        )
        df["topic_id"] = df["topic_id"].astype("int32")

        # Dùng pandas nullable Float32 (chữ hoa) để hỗ trợ pd.NA an toàn
        # thay vì float32 thường sẽ convert None → NaN không nhất quán
        df["coherence_score"] = pd.array(
            [float(coherence_score) if coherence_score is not None else pd.NA] * len(df),
            dtype="Float32",  # Nullable — hỗ trợ pd.NA, tương thích ClickHouse Nullable(Float32)
        )

        # Ghi CSV: top_keywords serialize thành JSON array string
        # Ví dụ: ["iphone","pin","camera"] — ClickHouse ARRAY JOIN or JSONExtract
        import json as _json
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )
        csv_rows = df.copy()
        csv_rows["top_keywords"] = csv_rows["top_keywords"].apply(
            lambda kws: _json.dumps(kws, ensure_ascii=False)
        )
        csv_rows.to_csv(output_path, index=False, encoding="utf-8")

        if self.verbose:
            print(f"✅ stg_topics → {output_path} ({len(df):,} topics)")

        return df

    def save(self, path: str) -> None:
        """
        Lưu trained model vào folder.

        Saves:
            - bertopic_model/            (BERTopic model files — pickle)
            - config.pkl                 (Hyperparameters)
            - topics.pkl                 (Topic assignments & probs)
            - stg_post_topics.csv        (Schema-compliant — stg_post_topics)
            - stg_topics.csv             (Schema-compliant — stg_topics)

        Lưu ý: ``stg_post_topics.csv`` yêu cầu post_ids. Nếu chưa có (chỉ dùng
        index giả), hãy gọi export_post_topics() riêng sau khi có post_ids thực.

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

            # [4] Export schema-compliant CSVs
            # stg_topics: không cần post_ids
            self.export_stg_topics(
                output_path=os.path.join(path, "stg_topics.csv"),
                model_version="bertopic_v2",
            )

            # stg_post_topics: dùng index giả nếu chưa có post_ids thực
            # Người dùng nên gọi export_post_topics(post_ids, ...) với post_ids thực.
            fake_ids = [f"doc_{i}" for i in range(len(self.topics_))]
            self.export_post_topics(
                post_ids=fake_ids,
                output_path=os.path.join(path, "stg_post_topics.csv"),
            )
            if self.verbose:
                print(
                    "⚠️  stg_post_topics.csv sử dụng index giả (doc_0, doc_1, ...). "
                    "Gọi export_post_topics(real_post_ids, ...) để ghi lại với post_ids thực."
                )
        
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
