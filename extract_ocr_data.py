#!/usr/bin/env python3
# filepath: c:\Users\Bogdan\Documents\git\tax-cp\extract_ocr_data.py

import os
import json
from pathlib import Path
from google.cloud import documentai

from src.config import Config

class OCRDataExtractor:
    def __init__(self):
        self.config = Config()
        self.client = documentai.DocumentProcessorServiceClient()
        self.processor_id = os.getenv('OCCUPATION_CATEGORY_PROCESSOR_ID')
        self.location = os.getenv('DOCUMENT_AI_LOCATION', 'us')
        
        # Where to write results
        self.results_dir = Path(self.config.results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Input data folders for receipts & payslips
        self.receipts_folder = Path('./data/receipt')
        self.payslips_folder = Path('./data/payslip')        # Output data folders for receipts & payslips
        self.results_folder = Path('./results')
        
    def process_files_line_by_line(self):
        print("=" * 50)
        print("EXTRACTING RECEIPT OCR DATA")
        print("=" * 50)
        
        # Process all receipt files, store in receipt_ocr.json
        receipt_output = self._process_and_read_lines(
            self.receipts_folder,
            'receipt_ocr.json'
        )
        
        print("\n" + "=" * 50)
        print("EXTRACTING PAYSLIP OCR DATA")
        print("=" * 50)
        
        # Process all payslip files, store in payslip_ocr.json
        payslip_output = self._process_and_read_lines(
            self.payslips_folder,
            'payslip_ocr.json'        )
        
        print(f"\n✅ Receipts OCR saved to: {receipt_output}")
        print(f"✅ Payslips OCR saved to: {payslip_output}")

    def _process_and_read_lines(self, input_folder: Path, output_filename: str):
        """Runs Document AI on each file in input_folder, splits text into lines, and writes JSON."""
        ocr_results = []
        if not input_folder.exists():
            print(f"Folder not found: {input_folder}")
            return "Error: Folder not found"

        print(f"Processing files from: {input_folder}")
        files_to_process = []
        
        # Get appropriate file types based on folder
        if 'receipt' in str(input_folder):
            files_to_process = list(input_folder.glob('*.png')) + list(input_folder.glob('*.jpg')) + list(input_folder.glob('*.jpeg'))
        elif 'payslip' in str(input_folder):
            files_to_process = list(input_folder.glob('*.pdf'))
        else:
            files_to_process = list(input_folder.glob('*.*'))
        
        if not files_to_process:
            print(f"No files found in {input_folder}")
            return "Error: No files found"
            
        print(f"Found {len(files_to_process)} files to process")

        for file_path in sorted(files_to_process):
            print(f"  Processing: {file_path.name}")
            try:
                document_text = self._run_ocr(file_path)
                lines = document_text.split('\n')

                # Build a dict with "file_name" and line0..lineN
                file_info = {"file_name": file_path.name}
                for i, line in enumerate(lines):
                    if line.strip():  # Only include non-empty lines
                        file_info[f"line{i}"] = line.strip()

                ocr_results.append(file_info)
                print(f"    ✅ Extracted {len([k for k in file_info.keys() if k.startswith('line')])} lines")
                
            except Exception as e:
                print(f"    ❌ Error processing {file_path.name}: {e}")
                # Still add an entry with error info
                ocr_results.append({
                    "file_name": file_path.name,
                    "error": str(e)
                })

        output_path = self.results_dir / output_filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ocr_results, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(ocr_results)} processed files to {output_path}")
        return str(output_path)

    def _run_ocr(self, file_path: Path) -> str:
        """Calls Document AI OCR with the specified processor_id."""
        with open(file_path, 'rb') as f:
            file_content = f.read()

        name = self.client.processor_path(
            project=self.config.google_project_id,
            location=self.location,
            processor=self.processor_id,
        )

        request = documentai.ProcessRequest(
            name=name,
            raw_document=documentai.RawDocument(
                content=file_content,
                mime_type=self._guess_mime_type(file_path)
            )
        )

        result = self.client.process_document(request=request)
        if not result.document.text:
            return ""
        return result.document.text

    def _guess_mime_type(self, file_path: Path) -> str:
        # Basic mime-type guessing
        ext = file_path.suffix.lower()
        if ext in ('.png', '.jpg', '.jpeg'):
            return 'image/png' if ext == '.png' else 'image/jpeg'
        return 'application/pdf'

def main():
    extractor = OCRDataExtractor()
    extractor.process_files_line_by_line()

if __name__ == "__main__":
    main()