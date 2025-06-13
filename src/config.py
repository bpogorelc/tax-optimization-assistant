"""Configuration module for the tax processing system."""

import os
from dotenv import load_dotenv
from pathlib import Path

class Config:
    """Configuration class for the tax processing system."""
    
    def __init__(self):
        """Initialize configuration by loading environment variables."""
        # Load environment variables
        load_dotenv()
        
        # Google Cloud Configuration
        self.google_credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', './credentials/google-credentials.json')
        self.google_project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'tax-demo-461209')
        self.document_ai_location = os.getenv('DOCUMENT_AI_LOCATION', 'us')
        
        # Document AI Processor IDs
        self.receipt_processor_id = os.getenv('RECEIPT_PROCESSOR_ID')
        self.income_statement_processor_id = os.getenv('INCOME_STATEMENT_PROCESSOR_ID')
        self.occupation_category_processor_id = os.getenv('OCCUPATION_CATEGORY_PROCESSOR_ID')
          # Weaviate Configuration
        self.weaviate_url = os.getenv('WEAVIATE_URL', 'http://localhost:8081')        # Vertex AI Configuration
        self.vertex_ai_project = os.getenv('VERTEX_AI_PROJECT', self.google_project_id)
        self.vertex_ai_location = os.getenv('VERTEX_AI_LOCATION', 'global')
        self.vertex_ai_model = os.getenv('VERTEX_AI_MODEL', 'gemini-2.0-flash-001')
        
        # Data paths
        self.data_dir = Path('data')
        self.csv_dir = self.data_dir / 'csv'
        self.receipt_dir = self.data_dir / 'receipt'
        self.payslip_dir = self.data_dir / 'payslip'
        
        # Output paths
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        
        # Model configuration
        self.embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
        
        # Validate required configurations
        self._validate_config()
    
    def _validate_config(self):
        """Validate that required configuration is present."""
        required_vars = [
            'receipt_processor_id',
            'income_statement_processor_id',
            'occupation_category_processor_id'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not getattr(self, var):
                missing_vars.append(var.upper())
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Check if credentials file exists
        if not Path(self.google_credentials_path).exists():
            raise FileNotFoundError(f"Google credentials file not found: {self.google_credentials_path}")
        
        # Check if data directories exist
        for dir_path in [self.csv_dir, self.receipt_dir, self.payslip_dir]:
            if not dir_path.exists():
                raise FileNotFoundError(f"Data directory not found: {dir_path}")
