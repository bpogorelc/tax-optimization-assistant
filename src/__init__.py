"""
Tax Document Processing and Pattern Recognition System

A comprehensive AI-powered solution for processing tax documents and generating
personalized tax optimization recommendations.

Author: Tax AI Engineering Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Tax AI Engineering Team"

# Import main components for easy access
from .config import Config
from .data_loader import DataLoader
from .document_processor import DocumentProcessor
from .pattern_analyzer import PatternAnalyzer
from .similarity_search import SimilaritySearchEngine
from .tip_generator import TipGenerator

__all__ = [
    "Config",
    "DataLoader", 
    "DocumentProcessor",
    "PatternAnalyzer",
    "SimilaritySearchEngine",
    "TipGenerator"
]
