"""
Tax Document Processing and Pattern Recognition System
A comprehensive AI-powered solution for processing tax documents and generating optimization tips.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path

def clean_data_for_json(obj):
    """Recursively clean data to ensure JSON serialization compatibility."""
    if isinstance(obj, dict):
        # Convert tuple keys to strings
        cleaned = {}
        for key, value in obj.items():
            if isinstance(key, tuple):
                key = '_'.join(str(k) for k in key)
            elif not isinstance(key, (str, int, float, bool, type(None))):
                key = str(key)
            cleaned[key] = clean_data_for_json(value)
        return cleaned
    elif isinstance(obj, list):
        return [clean_data_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return [clean_data_for_json(item) for item in obj]
    elif pd.isna(obj):
        return None
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif hasattr(obj, 'tolist'):  # numpy array
        return obj.tolist()
    else:
        return obj

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from src.document_processor import DocumentProcessor
from src.pattern_analyzer import PatternAnalyzer
from src.similarity_search import SimilaritySearchEngine
from src.tip_generator import TipGenerator
from src.config import Config
from src.data_loader import DataLoader

def main():
    """Main execution function."""
    print("🚀 Starting Tax Document Processing System")
    
    # Initialize configuration
    config = Config()
    
    # Initialize components
    data_loader = DataLoader(config)
    document_processor = DocumentProcessor(config)
    pattern_analyzer = PatternAnalyzer(config)
    similarity_engine = SimilaritySearchEngine(config)
    tip_generator = TipGenerator(config)
    
    # Load CSV data
    print("📊 Loading CSV data...")
    transactions_df, users_df, tax_filings_df = data_loader.load_csv_data()
    
    # Process documents
    print("📄 Processing documents...")
    receipt_data = document_processor.process_receipts()
    payslip_data = document_processor.process_payslips()
    
    # Analyze patterns
    print("🔍 Analyzing patterns...")
    patterns = pattern_analyzer.analyze_patterns(
        transactions_df, users_df, tax_filings_df, receipt_data, payslip_data
    )
    
    # Initialize similarity search
    print("🔗 Setting up similarity search...")
    similarity_engine.build_index(transactions_df)
    
    # Generate tips for all users
    print("💡 Generating tax optimization tips...")
    all_tips = {}
    for user_id in users_df['user_id'].unique():
        user_tips = tip_generator.generate_tips_for_user(
            user_id, transactions_df, users_df, tax_filings_df, patterns
        )
        all_tips[user_id] = user_tips
        print(f"Generated {len(user_tips)} tips for user {user_id}")
      # Save results
    print("💾 Saving results...")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Clean data for JSON serialization
    cleaned_receipt_data = clean_data_for_json(receipt_data)
    cleaned_payslip_data = clean_data_for_json(payslip_data)
    cleaned_patterns = clean_data_for_json(patterns)
    cleaned_tips = clean_data_for_json(all_tips)
    
    # Save extracted document data
    with open(results_dir / "receipt_data.json", "w") as f:
        json.dump(cleaned_receipt_data, f, indent=2, default=str)
    
    with open(results_dir / "payslip_data.json", "w") as f:
        json.dump(cleaned_payslip_data, f, indent=2, default=str)
    
    with open(results_dir / "patterns.json", "w") as f:
        json.dump(cleaned_patterns, f, indent=2, default=str)
    
    with open(results_dir / "all_tips.json", "w") as f:
        json.dump(cleaned_tips, f, indent=2, default=str)
    
    print("✅ Processing complete! Check the results directory for outputs.")
    print("📊 To view visualizations, run the Pattern Recognition Analysis notebook.")
    print("🌐 To start the web interface, run: streamlit run app.py")

if __name__ == "__main__":
    main()
