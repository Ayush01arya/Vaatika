import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
import re
import json
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VATIKARetriever:
    """
    Advanced retrieval system for VATIKA dataset
    Implements multiple retrieval strategies for better context matching
    """

    def __init__(self, embedding_model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.contexts = []
        self.context_embeddings = None
        self.domain_embeddings = {}
        self.keyword_index = defaultdict(set)
        self.question_embeddings = []
        self.qa_pairs = []

    def load_and_index_data(self, contexts_data: List[Dict[str, Any]]):
        """
        Load data and create multiple indexes for efficient retrieval
        """
        logger.info("Loading and indexing VATIKA data...")

        self.contexts = contexts_data
        self._create_context_embeddings()
        self._create_domain_embeddings()
        self._create_keyword_index()
        self._create_qa_embeddings()

        logger.info(f"Indexed {len(self.contexts)} contexts successfully")

    def _create_context_embeddings(self):
        """Create embeddings for all contexts"""
        context_texts = [ctx['context'] for ctx in self.contexts]
        self.context_embeddings = self.embedding_model.encode(
            context_texts,
            show_progress_bar=True,
            batch_size=32
        )
        logger.info(f"Created embeddings for {len(context_texts)} contexts")

    def _create_domain_embeddings(self):
        """Create domain-specific embeddings"""
        domain_contexts = defaultdict(list)

        for ctx in self.contexts:
            domain_contexts[ctx['domain']].append(ctx['context'])

        for domain, contexts in domain_contexts.items():
            # Combine all contexts for a domain
            combined_context = " ".join(contexts)
            domain_embedding = self.embedding_model.encode([combined_context])
            self.domain_embeddings[domain] = domain_embedding[0]

        logger.info(f"Created domain embeddings for {len(self.domain_embeddings)} domains")

    def _create_keyword_index(self):
        """Create keyword-based index for fast lookups"""
        hindi_keywords = {
            'घाट': ['ghat', 'घाट', 'तट'],
            'मंदिर': ['temple', 'मंदिर', 'देवालय'],
            'आरती': ['aarti', 'आरती', 'पूजा'],
            'भोजन': ['food', 'भोजन', 'खाना'],
            'होटल': ['hotel', 'होटल', 'आवास'],
            'यात्रा': ['travel', 'यात्रा', 'सफर'],
            'समय': ['time', 'समय', 'टाइम'],
            'दूरी': ['distance', 'दूरी', 'फासला'],
            'कैसे': ['how', 'कैसे', 'कैसे'],
            'कहां': ['where', 'कहां', 'कहाँ'],
            'क्या': ['what', 'क्या'],
            'कब': ['when', 'कब'],
            'कितना': ['how much', 'कितना', 'कितनी'],
        }

        for idx, ctx in enumerate(self.contexts):
            context_text = ctx['context'].lower()
            domain = ctx['domain']

            # Index by keywords
            for keyword, variants in hindi_keywords.items():
                for variant in variants:
                    if variant in context_text:
                        self.keyword_index[keyword].add(idx)

            # Index by domain
            self.keyword_index[domain].add(idx)

            # Index QA pairs
            for qa in ctx['qas']:
                question_text = qa['question'].lower()
                for keyword, variants in hindi_keywords.items():
                    for variant in variants:
                        if variant in question_text:
                            self.keyword_index[keyword].add(idx)

    def _create_qa_embeddings(self):
        """Create embeddings for all Q&A pairs for direct matching"""
        for ctx_idx, ctx in enumerate(self.contexts):
            for qa in ctx['qas']:
                qa_text = qa['question'] + " " + qa['answer']
                self.qa_pairs.append({
                    'context_idx': ctx_idx,
                    'qa': qa,
                    'combined_text': qa_text,
                    'domain': ctx['domain']
                })

        if self.qa_pairs:
            qa_texts = [qa['combined_text'] for qa in self.qa_pairs]
            self.question_embeddings = self.embedding_model.encode(
                qa_texts,
                show_progress_bar=True,
                batch_size=32
            )
            logger.info(f"Created embeddings for {len(self.qa_pairs)} Q&A pairs")

    def retrieve_contexts(self, query: str, top_k: int = 5, strategy: str = 'hybrid') -> List[Dict[str, Any]]:
        """
        Retrieve relevant contexts using different strategies

        Args:
            query: User query
            top_k: Number of contexts to retrieve
            strategy: 'semantic', 'keyword', 'hybrid', 'domain_aware'
        """
        if strategy == 'semantic':
            return self._semantic_retrieval(query, top_k)
        elif strategy == 'keyword':
            return self._keyword_retrieval(query, top_k)
        elif strategy == 'domain_aware':
            return self._domain_aware_retrieval(query, top_k)
        else:  # hybrid
            return self._hybrid_retrieval(query, top_k)

    def _semantic_retrieval(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Pure semantic similarity retrieval"""
        if self.context_embeddings is None:
            return []

        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.context_embeddings)[0]

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0.2:  # Minimum similarity threshold
                results.append({
                    'context': self.contexts[idx],
                    'similarity': float(similarities[idx]),
                    'method': 'semantic'
                })

        return results

    def _keyword_retrieval(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Keyword-based retrieval"""
        query_lower = query.lower()
        candidate_indices = set()

        # Find contexts containing query keywords
        for keyword, indices in self.keyword_index.items():
            if keyword in query_lower:
                candidate_indices.update(indices)

        # Score candidates based on keyword frequency
        scored_candidates = []
        for idx in candidate_indices:
            score = self._calculate_keyword_score(query_lower, self.contexts[idx])
            scored_candidates.append((idx, score))

        # Sort by score and return top_k
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scored_candidates[:top_k]:
            results.append({
                'context': self.contexts[idx],
                'similarity': score,
                'method': 'keyword'
            })

        return results

    def _domain_aware_retrieval(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Domain-aware retrieval"""
        # First, identify the most relevant domain
        query_embedding = self.embedding_model.encode([query])

        domain_similarities = {}
        for domain, domain_embedding in self.domain_embeddings.items():
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                domain_embedding.reshape(1, -1)
            )[0][0]
            domain_similarities[domain] = similarity

        # Get top 2 domains
        top_domains = sorted(domain_similarities.items(), key=lambda x: x[1], reverse=True)[:2]

        # Filter contexts by top domains
        domain_filtered_contexts = []
        for i, ctx in enumerate(self.contexts):
            if ctx['domain'] in [d[0] for d in top_domains]:
                domain_filtered_contexts.append((i, ctx))

        if not domain_filtered_contexts:
            return self._semantic_retrieval(query, top_k)

        # Perform semantic retrieval within filtered contexts
        filtered_indices = [i for i, _ in domain_filtered_contexts]
        filtered_embeddings = self.context_embeddings[filtered_indices]

        similarities = cosine_similarity(query_embedding, filtered_embeddings)[0]

        # Get top results
        top_local_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for local_idx in top_local_indices:
            global_idx = filtered_indices[local_idx]
            if similarities[local_idx] > 0.2:
                results.append({
                    'context': self.contexts[global_idx],
                    'similarity': float(similarities[local_idx]),
                    'method': 'domain_aware',
                    'domain': self.contexts[global_idx]['domain']
                })

        return results

    def _hybrid_retrieval(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Combine semantic and keyword retrieval"""
        # Get results from both methods
        semantic_results = self._semantic_retrieval(query, top_k)
        keyword_results = self._keyword_retrieval(query, top_k)

        # Combine and re-rank
        combined_results = {}

        # Add semantic results with weight
        for result in semantic_results:
            ctx_id = id(result['context'])
            combined_results[ctx_id] = {
                'context': result['context'],
                'semantic_score': result['similarity'],
                'keyword_score': 0.0,
                'method': 'hybrid'
            }

        # Add keyword results
        for result in keyword_results:
            ctx_id = id(result['context'])
            if ctx_id in combined_results:
                combined_results[ctx_id]['keyword_score'] = result['similarity']
            else:
                combined_results[ctx_id] = {
                    'context': result['context'],
                    'semantic_score': 0.0,
                    'keyword_score': result['similarity'],
                    'method': 'hybrid'
                }

        # Calculate combined score
        final_results = []
        for ctx_id, result in combined_results.items():
            # Weighted combination: 70% semantic, 30% keyword
            combined_score = (0.7 * result['semantic_score'] +
                              0.3 * result['keyword_score'])

            final_results.append({
                'context': result['context'],
                'similarity': combined_score,
                'method': result['method'],
                'semantic_score': result['semantic_score'],
                'keyword_score': result['keyword_score']
            })

        # Sort by combined score
        final_results.sort(key=lambda x: x['similarity'], reverse=True)

        return final_results[:top_k]

    def _calculate_keyword_score(self, query: str, context: Dict[str, Any]) -> float:
        """Calculate keyword-based similarity score"""
        context_text = (context['context'] + " " +
                        " ".join([qa['question'] + " " + qa['answer']
                                  for qa in context['qas']])).lower()

        query_words = set(query.split())
        context_words = set(context_text.split())

        # Jaccard similarity
        intersection = len(query_words.intersection(context_words))
        union = len(query_words.union(context_words))

        if union == 0:
            return 0.0

        return intersection / union

    def find_exact_qa_match(self, query: str, threshold: float = 0.8) -> Dict[str, Any]:
        """Find exact Q&A matches for the query"""
        if not self.question_embeddings.size:
            return None

        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.question_embeddings)[0]

        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]

        if best_similarity > threshold:
            return {
                'qa': self.qa_pairs[best_match_idx]['qa'],
                'context': self.contexts[self.qa_pairs[best_match_idx]['context_idx']],
                'similarity': float(best_similarity),
                'method': 'exact_qa_match'
            }

        return None

    def get_domain_statistics(self) -> Dict[str, int]:
        """Get statistics about domains in the dataset"""
        domain_counts = defaultdict(int)
        for ctx in self.contexts:
            domain_counts[ctx['domain']] += 1
        return dict(domain_counts)


class AdvancedVATIKARetriever(VATIKARetriever):
    """
    Extended retriever with additional features for better performance
    """

    def __init__(self, embedding_model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        super().__init__(embedding_model_name)
        self.query_cache = {}
        self.max_cache_size = 1000

    def retrieve_with_caching(self, query: str, top_k: int = 5, strategy: str = 'hybrid') -> List[Dict[str, Any]]:
        """Retrieve with caching for better performance"""
        cache_key = f"{query}_{top_k}_{strategy}"

        if cache_key in self.query_cache:
            logger.info(f"Cache hit for query: {query[:50]}...")
            return self.query_cache[cache_key]

        results = self.retrieve_contexts(query, top_k, strategy)

        # Cache management
        if len(self.query_cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]

        self.query_cache[cache_key] = results
        return results

    def retrieve_with_reranking(self, query: str, top_k: int = 5, rerank_top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Two-stage retrieval: first retrieve more candidates, then rerank
        """
        # First stage: retrieve more candidates
        candidates = self.retrieve_contexts(query, rerank_top_k, 'hybrid')

        if len(candidates) <= top_k:
            return candidates

        # Second stage: rerank using more sophisticated scoring
        reranked_candidates = []

        for candidate in candidates:
            # Calculate additional features
            domain_relevance = self._calculate_domain_relevance(query, candidate['context']['domain'])
            qa_relevance = self._calculate_qa_relevance(query, candidate['context']['qas'])

            # Combined score
            final_score = (0.5 * candidate['similarity'] +
                           0.3 * domain_relevance +
                           0.2 * qa_relevance)

            reranked_candidates.append({
                **candidate,
                'final_score': final_score,
                'domain_relevance': domain_relevance,
                'qa_relevance': qa_relevance
            })

        # Sort by final score
        reranked_candidates.sort(key=lambda x: x['final_score'], reverse=True)

        return reranked_candidates[:top_k]

    def _calculate_domain_relevance(self, query: str, domain: str) -> float:
        """Calculate domain relevance score"""
        domain_keywords = {
            'temple': ['मंदिर', 'देवालय', 'temple', 'पूजा', 'दर्शन'],
            'ghat': ['घाट', 'ghat', 'तट', 'गंगा'],
            'aarti': ['आरती', 'aarti', 'पूजा', 'गंगा'],
            'food': ['भोजन', 'खाना', 'food', 'खाने'],
            'travel': ['यात्रा', 'travel', 'जाना', 'पहुंचना'],
            'museum': ['संग्रहालय', 'museum', 'म्यूजियम'],
            'ashram': ['आश्रम', 'ashram'],
            'kund': ['कुंड', 'kund', 'तालाब'],
            'cruise': ['क्रूज़', 'cruise', 'नाव'],
            'toilet': ['शौचालय', 'toilet', 'टॉयलेट']
        }

        query_lower = query.lower()
        domain_lower = domain.lower()

        # Direct domain match
        if domain_lower in query_lower:
            return 1.0

        # Keyword match
        if domain_lower in domain_keywords:
            keywords = domain_keywords[domain_lower]
            for keyword in keywords:
                if keyword in query_lower:
                    return 0.8

        return 0.1

    def _calculate_qa_relevance(self, query: str, qas: List[Dict[str, Any]]) -> float:
        """Calculate Q&A relevance score"""
        if not qas:
            return 0.0

        max_relevance = 0.0

        for qa in qas:
            question_similarity = self._text_similarity(query, qa['question'])
            max_relevance = max(max_relevance, question_similarity)

        return max_relevance

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using embeddings"""
        try:
            embeddings = self.embedding_model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except:
            return 0.0