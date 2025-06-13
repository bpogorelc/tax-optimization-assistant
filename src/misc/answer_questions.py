#!/usr/bin/env python3
"""
LLM-based Question Answering Script for Tax Documents

This script extracts questions from Questions.pdf using Document AI OCR and answers them
using Vertex AI based on processed payslip and receipt data (LLM approach).
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config import Config
from src.question_answerer import QuestionAnswerer

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main function to run LLM-based question answering."""
    print("🤖 Starting LLM-based Question Answering System")
    print("📖 Using Vertex AI + Document AI OCR for intelligent Q&A")
    
    try:
        # Setup logging
        setup_logging()
        
        # Load configuration
        config = Config()
        print("✅ Configuration loaded")
        
        # Initialize question answerer
        qa_system = QuestionAnswerer(config)
        print("✅ LLM Question Answerer initialized (Vertex AI + Document AI)")
        
        # Path to questions PDF
        questions_pdf = Path("data/Questions.pdf")
        
        if not questions_pdf.exists():
            raise FileNotFoundError(f"Questions PDF not found: {questions_pdf}")
        
        print(f"📄 Processing questions from: {questions_pdf}")
        print("🔍 Using Document AI OCR to extract questions...")
        
        # Answer all questions using LLM
        results = qa_system.answer_all_questions(questions_pdf)
        
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return
        
        # Save results
        output_path = qa_system.save_results(results)
        print(f"💾 LLM results saved to: {output_path}")
        
        # Create and save summary report
        summary = qa_system.create_summary_report(results)
        summary_path = Path("results/llm_qa_summary.md")
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"📊 Summary report saved to: {summary_path}")
        
        # Print summary statistics
        metadata = results.get('metadata', {})
        qa_results = results.get('qa_results', {})
        
        print("\n📈 LLM System Summary:")
        print(f"  - Questions extracted: {len(metadata.get('parsed_questions', []))}")
        print(f"  - Questions answered: {len(qa_results)}")
        print(f"  - Payslips analyzed: {metadata.get('data_summary', {}).get('payslips_count', 0)}")
        print(f"  - Receipts analyzed: {metadata.get('data_summary', {}).get('receipts_count', 0)}")
        print(f"  - LLM System: {metadata.get('llm_system', 'Unknown')}")

        # Print first question and answer as preview
        if qa_results:
            print("\n🔍 Preview of first Q&A:")
            first_key = list(qa_results.keys())[0]
            first_qa = qa_results[first_key]
            print(f"  Q: {first_qa['question'][:100]}...")
            print(f"  A: {first_qa['answer'][:200]}...")
        
        print("\n✅ LLM-based question answering completed successfully!")
        print("🎯 Check results/llm_qa_results.json for full results")
        print("📖 Check results/llm_qa_summary.md for readable report")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logging.error(f"LLM question answering failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()