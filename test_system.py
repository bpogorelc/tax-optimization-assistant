"""
Test script to verify the tax processing system functionality.
Run this to ensure all components are working correctly.
"""

import sys
from pathlib import Path
import logging

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported successfully."""
    try:
        from src.config import Config
        from src.data_loader import DataLoader
        from src.document_processor import DocumentProcessor
        from src.pattern_analyzer import PatternAnalyzer
        from src.similarity_search import SimilaritySearchEngine
        from src.tip_generator import TipGenerator
        
        logger.info("✅ All modules imported successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Import error: {e}")
        return False

def test_configuration():
    """Test configuration loading."""
    try:
        from src.config import Config
        config = Config()
        
        # Check required attributes
        required_attrs = [
            'google_project_id', 'document_ai_location',
            'receipt_processor_id', 'income_statement_processor_id',
            'occupation_category_processor_id'
        ]
        
        for attr in required_attrs:
            if not hasattr(config, attr) or not getattr(config, attr):
                logger.error(f"❌ Missing configuration: {attr}")
                return False
        
        logger.info("✅ Configuration loaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Configuration error: {e}")
        return False

def test_data_loading():
    """Test data loading functionality."""
    try:
        from src.config import Config
        from src.data_loader import DataLoader
        
        config = Config()
        data_loader = DataLoader(config)
        
        # Test CSV loading
        transactions_df, users_df, tax_filings_df = data_loader.load_csv_data()
        
        if len(transactions_df) == 0:
            logger.error("❌ No transactions loaded")
            return False
        
        if len(users_df) == 0:
            logger.error("❌ No users loaded")
            return False
            
        if len(tax_filings_df) == 0:
            logger.error("❌ No tax filings loaded")
            return False
        
        logger.info(f"✅ Data loaded: {len(transactions_df)} transactions, {len(users_df)} users, {len(tax_filings_df)} tax filings")
        return True
    except Exception as e:
        logger.error(f"❌ Data loading error: {e}")
        return False

def test_pattern_analysis():
    """Test pattern analysis functionality."""
    try:
        from src.config import Config
        from src.data_loader import DataLoader
        from src.pattern_analyzer import PatternAnalyzer
        
        config = Config()
        data_loader = DataLoader(config)
        pattern_analyzer = PatternAnalyzer(config)
        
        # Load data
        transactions_df, users_df, tax_filings_df = data_loader.load_csv_data()
        
        # Test basic pattern analysis (without document data)
        patterns = pattern_analyzer.analyze_patterns(
            transactions_df, users_df, tax_filings_df, [], []
        )
        
        if not patterns:
            logger.error("❌ No patterns generated")
            return False
        
        logger.info("✅ Pattern analysis completed successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Pattern analysis error: {e}")
        return False

def test_tip_generation():
    """Test tax tip generation functionality."""
    try:
        from src.config import Config
        from src.data_loader import DataLoader
        from src.pattern_analyzer import PatternAnalyzer
        from src.tip_generator import TipGenerator
        
        config = Config()
        data_loader = DataLoader(config)
        pattern_analyzer = PatternAnalyzer(config)
        tip_generator = TipGenerator(config)
        
        # Load data
        transactions_df, users_df, tax_filings_df = data_loader.load_csv_data()
        
        # Generate patterns
        patterns = pattern_analyzer.analyze_patterns(
            transactions_df, users_df, tax_filings_df, [], []
        )
        
        # Test tip generation for first user
        first_user = users_df['user_id'].iloc[0]
        tips = tip_generator.generate_tips_for_user(
            first_user, transactions_df, users_df, tax_filings_df, patterns
        )
        
        logger.info(f"✅ Generated {len(tips)} tips for user {first_user}")
        return True
    except Exception as e:
        logger.error(f"❌ Tip generation error: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting Tax Processing System Tests")
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_configuration),
        ("Data Loading Test", test_data_loading),
        ("Pattern Analysis Test", test_pattern_analysis),
        ("Tip Generation Test", test_tip_generation)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name}...")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    logger.info("\n📊 Test Results Summary:")
    logger.info("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info("=" * 50)
    logger.info(f"Tests Passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        logger.info("🎉 All tests passed! System is ready to use.")
        return True
    else:
        logger.error("⚠️ Some tests failed. Please check the configuration and data files.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
