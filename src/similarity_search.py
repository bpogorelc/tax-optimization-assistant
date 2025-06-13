"""Similarity search engine using embeddings and vector database."""
import pandas as pd
import numpy as np
import faiss
import logging
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import weaviate
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class SimilaritySearchEngine:
    """Similarity search engine for finding related transactions."""
    
    def __init__(self, config):
        """Initialize similarity search engine."""
        self.config = config
        self.model = SentenceTransformer(config.embedding_model)
        self.index = None
        self.transaction_embeddings = None
        self.transactions_df = None
        
        # Initialize Weaviate client
        try:
            self.weaviate_client = weaviate.Client(
                url=config.weaviate_url,
                timeout_config=(5, 15)
            )
            self._setup_weaviate_schema()
            logger.info("Weaviate client initialized successfully")
        except Exception as e:
            logger.warning(f"Weaviate initialization failed: {e}. Using FAISS only.")
            self.weaviate_client = None
    
    def _setup_weaviate_schema(self):
        """Setup Weaviate schema for transactions."""
        if not self.weaviate_client:
            return
            
        schema = {
            "classes": [{
                "class": "Transaction",
                "description": "Financial transaction",
                "properties": [
                    {"name": "user_id", "dataType": ["string"]},
                    {"name": "amount", "dataType": ["number"]},
                    {"name": "category", "dataType": ["string"]},
                    {"name": "subcategory", "dataType": ["string"]},
                    {"name": "vendor", "dataType": ["string"]},
                    {"name": "transaction_date", "dataType": ["date"]},
                    {"name": "description", "dataType": ["text"]},
                ]
            }]
        }
        
        try:
            if not self.weaviate_client.schema.contains(schema):
                self.weaviate_client.schema.create(schema)
        except Exception as e:
            logger.error(f"Failed to create Weaviate schema: {e}")
    
    def build_index(self, transactions_df: pd.DataFrame):
        """Build similarity search index from transactions."""
        self.transactions_df = transactions_df.copy()
        
        # Create text descriptions for embedding
        descriptions = []
        for _, row in transactions_df.iterrows():
            desc = f"{row['category']} {row.get('subcategory', '')} {row['vendor']} {row['amount']:.2f}€"
            descriptions.append(desc)
        
        # Generate embeddings
        logger.info("Generating embeddings for transactions...")
        self.transaction_embeddings = self.model.encode(descriptions)
        
        # Build FAISS index
        dimension = self.transaction_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.transaction_embeddings)
        self.index.add(self.transaction_embeddings.astype(np.float32))
        
        # Store in Weaviate if available
        self._store_in_weaviate()
        
        logger.info(f"Built similarity index with {len(descriptions)} transactions")
    
    def _store_in_weaviate(self):
        """Store transactions in Weaviate vector database."""
        if not self.weaviate_client:
            return
            
        try:
            with self.weaviate_client.batch as batch:
                for idx, (_, row) in enumerate(self.transactions_df.iterrows()):
                    data_object = {
                        "user_id": row['user_id'],
                        "amount": float(row['amount']),
                        "category": row['category'],
                        "subcategory": row.get('subcategory', ''),
                        "vendor": row['vendor'],
                        "transaction_date": row['transaction_date'].isoformat(),
                        "description": f"{row['category']} {row.get('subcategory', '')} {row['vendor']}"
                    }
                    
                    batch.add_data_object(
                        data_object,
                        "Transaction",
                        vector=self.transaction_embeddings[idx].tolist()
                    )
            
            logger.info("Stored transactions in Weaviate")
        except Exception as e:
            logger.error(f"Failed to store in Weaviate: {e}")
    
    def find_similar_transactions(self, query_transaction: Dict[str, Any], 
                                top_k: int = 10, 
                                exclude_user: str = None) -> List[Dict[str, Any]]:
        """Find similar transactions to a query transaction."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Create query description
        query_desc = f"{query_transaction['category']} {query_transaction.get('subcategory', '')} {query_transaction['vendor']} {query_transaction['amount']:.2f}€"
        
        # Generate query embedding
        query_embedding = self.model.encode([query_desc])
        faiss.normalize_L2(query_embedding)
        
        # Search similar transactions
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k * 2)  # Get more to filter
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # No more results
                break
                
            transaction = self.transactions_df.iloc[idx]
            
            # Skip if excluding user
            if exclude_user and transaction['user_id'] == exclude_user:
                continue
                
            results.append({
                'transaction': transaction.to_dict(),
                'similarity_score': float(score),
                'index': int(idx)
            })
            
            if len(results) >= top_k:
                break
        
        return results
    
    def find_similar_by_category(self, category: str, subcategory: str = None, 
                               top_k: int = 10) -> List[Dict[str, Any]]:
        """Find similar transactions by category and subcategory."""
        # Filter transactions by category
        filtered_df = self.transactions_df[self.transactions_df['category'] == category]
        
        if subcategory:
            filtered_df = filtered_df[filtered_df.get('subcategory', '') == subcategory]
        
        if filtered_df.empty:
            return []
        
        # Get indices of filtered transactions
        indices = filtered_df.index.tolist()
        
        # Get embeddings for these transactions
        filtered_embeddings = self.transaction_embeddings[indices]
        
        # Calculate average embedding
        avg_embedding = np.mean(filtered_embeddings, axis=0, keepdims=True)
        faiss.normalize_L2(avg_embedding)
        
        # Search for similar transactions
        scores, result_indices = self.index.search(avg_embedding.astype(np.float32), top_k)
        
        results = []
        for score, idx in zip(scores[0], result_indices[0]):
            if idx == -1:
                break
                
            transaction = self.transactions_df.iloc[idx]
            results.append({
                'transaction': transaction.to_dict(),
                'similarity_score': float(score),
                'index': int(idx)
            })
        
        return results
    
    def find_user_similar_patterns(self, user_id: str, top_k: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """Find similar spending patterns for a user."""
        user_transactions = self.transactions_df[self.transactions_df['user_id'] == user_id]
        
        if user_transactions.empty:
            return {}
        
        # Group by category
        patterns = {}
        for category in user_transactions['category'].unique():
            category_transactions = user_transactions[user_transactions['category'] == category]
            
            # Find similar transactions in this category from other users
            similar_transactions = []
            for _, transaction in category_transactions.iterrows():
                similar = self.find_similar_transactions(
                    transaction.to_dict(), 
                    top_k=5, 
                    exclude_user=user_id
                )
                similar_transactions.extend(similar)
            
            # Remove duplicates and sort by similarity
            seen_indices = set()
            unique_similar = []
            for item in similar_transactions:
                if item['index'] not in seen_indices:
                    seen_indices.add(item['index'])
                    unique_similar.append(item)
            
            patterns[category] = sorted(unique_similar, key=lambda x: x['similarity_score'], reverse=True)[:top_k]
        
        return patterns
    
    def search_by_text(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search transactions by text query."""
        if self.weaviate_client:
            return self._search_weaviate_by_text(query_text, top_k)
        else:
            return self._search_faiss_by_text(query_text, top_k)
    
    def _search_weaviate_by_text(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        """Search using Weaviate's text capabilities."""
        try:
            result = (
                self.weaviate_client.query
                .get("Transaction", ["user_id", "amount", "category", "subcategory", "vendor", "transaction_date", "description"])
                .with_near_text({"concepts": [query_text]})
                .with_limit(top_k)
                .with_additional(["certainty"])
                .do()
            )
            
            transactions = result["data"]["Get"]["Transaction"]
            return [
                {
                    'transaction': tx,
                    'similarity_score': tx['_additional']['certainty']
                }
                for tx in transactions
            ]
        except Exception as e:
            logger.error(f"Weaviate text search failed: {e}")
            return self._search_faiss_by_text(query_text, top_k)
    
    def _search_faiss_by_text(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        """Search using FAISS with text embedding."""
        if self.index is None:
            return []
        
        query_embedding = self.model.encode([query_text])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                break
                
            transaction = self.transactions_df.iloc[idx]
            results.append({
                'transaction': transaction.to_dict(),
                'similarity_score': float(score),
                'index': int(idx)
            })
        
        return results
    
    def get_transaction_clusters(self, user_id: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get transaction clusters for analysis."""
        if self.transactions_df is None:
            return {}
        
        df = self.transactions_df
        if user_id:
            df = df[df['user_id'] == user_id]
        
        clusters = {}
        for category in df['category'].unique():
            category_transactions = df[df['category'] == category]
            clusters[category] = [tx.to_dict() for _, tx in category_transactions.iterrows()]
        
        return clusters
    
    def save_index(self, filepath: str):
        """Save the FAISS index to disk."""
        if self.index is not None:
            faiss.write_index(self.index, filepath)
            logger.info(f"Index saved to {filepath}")
    
    def load_index(self, filepath: str):
        """Load FAISS index from disk."""
        if Path(filepath).exists():
            self.index = faiss.read_index(filepath)
            logger.info(f"Index loaded from {filepath}")
        else:
            logger.warning(f"Index file not found: {filepath}")