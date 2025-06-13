"""Data loading utilities for the tax processing system."""

import pandas as pd
from pathlib import Path
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Data loader for CSV files and document processing."""
    
    def __init__(self, config):
        """Initialize data loader with configuration."""
        self.config = config
    
    def load_csv_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all CSV data files.
        
        Returns:
            Tuple of (transactions_df, users_df, tax_filings_df)
        """
        try:
            # Load transactions data
            transactions_df = pd.read_csv(self.config.csv_dir / 'transactions.csv')
            transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])
            
            # Load users data
            users_df = pd.read_csv(self.config.csv_dir / 'users.csv')
            
            # Load tax filings data
            tax_filings_df = pd.read_csv(self.config.csv_dir / 'tax_filings.csv')
            tax_filings_df['filing_date'] = pd.to_datetime(tax_filings_df['filing_date'])
            
            logger.info(f"Loaded {len(transactions_df)} transactions, {len(users_df)} users, {len(tax_filings_df)} tax filings")
            
            return transactions_df, users_df, tax_filings_df
            
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            raise
    
    def get_document_files(self, document_type: str) -> list:
        """
        Get list of document files for processing.
        
        Args:
            document_type: 'receipt' or 'payslip'
            
        Returns:
            List of file paths
        """
        if document_type == 'receipt':
            dir_path = self.config.receipt_dir
            extensions = ['.png', '.jpg', '.jpeg']
        elif document_type == 'payslip':
            dir_path = self.config.payslip_dir
            extensions = ['.pdf']
        else:
            raise ValueError(f"Unknown document type: {document_type}")
        
        files = []
        for ext in extensions:
            files.extend(list(dir_path.glob(f'*{ext}')))
        
        return sorted(files)
    
    def save_processed_data(self, data: dict, filename: str):
        """
        Save processed data to JSON file.
        
        Args:
            data: Data to save
            filename: Output filename
        """
        import json
        
        output_path = self.config.results_dir / filename
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Saved processed data to {output_path}")
    
    def load_processed_data(self, filename: str) -> dict:
        """
        Load processed data from JSON file.
        
        Args:
            filename: Input filename
            
        Returns:
            Loaded data
        """
        import json
        
        input_path = self.config.results_dir / filename
        if not input_path.exists():
            return {}
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded processed data from {input_path}")
        return data
