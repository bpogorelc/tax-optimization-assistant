"""Document processing module using Google Document AI."""

import os
from google.cloud import documentai_v1 as documentai
from google.api_core.client_options import ClientOptions
import logging
from pathlib import Path
from typing import Dict, List, Any
import json
import datetime
import mimetypes
from google.auth import default
from google.protobuf.json_format import MessageToDict

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Document processor using Google Document AI."""
    
    def __init__(self, config):
        """Initialize document processor with configuration."""
        self.config = config
        
        # Set up Google Cloud credentials
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.config.google_credentials_path
        
        # Use explicit authentication with default credentials
        credentials, project_id = default()
        
        # Use project_id from environment if not provided by credentials
        self.project_id = project_id or self.config.google_project_id
        
        # Set quota project via client options
        client_options = ClientOptions(
            api_endpoint=f"{self.config.document_ai_location}-documentai.googleapis.com",
            quota_project_id=self.project_id
        )
        
        # Initialize Document AI client
        self.client = documentai.DocumentProcessorServiceClient(
            client_options=client_options,
            credentials=credentials
        )        
        # Build processor names using the proper method
        self.receipt_processor_name = self.client.processor_path(
            self.project_id,
            self.config.document_ai_location,
            self.config.receipt_processor_id
        )
        
        self.payslip_processor_name = self.client.processor_path(
            self.project_id,
            self.config.document_ai_location,
            self.config.income_statement_processor_id
        )
        
        self.occupation_processor_name = self.client.processor_path(
            self.project_id,
            self.config.document_ai_location,
            self.config.occupation_category_processor_id
        )
    
    def _get_mime_type(self, file_path: Path) -> str:
        """Determine MIME type based on file extension."""
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type is None:
            # Default based on file extension
            extension = file_path.suffix.lower()
            if extension in ['.jpg', '.jpeg']:
                mime_type = 'image/jpeg'
            elif extension == '.png':
                mime_type = 'image/png'
            elif extension in ['.tif', '.tiff']:
                mime_type = 'image/tiff'
            else:
                mime_type = 'application/pdf'
        return mime_type
    
    def _serialize_document_entities(self, document: documentai.Document) -> Dict[str, Any]:
        """Convert Document AI entities to a serializable dictionary."""
        result = {}
        
        for entity in document.entities:
            entity_data = {
                "confidence": entity.confidence,
                "mention_text": entity.mention_text,
                "normalized_value": entity.normalized_value.text if entity.normalized_value else None
            }
            
            # Handle nested entities if they exist
            if entity.properties:
                entity_data["properties"] = {}
                for prop in entity.properties:
                    entity_data["properties"][prop.type_] = {
                        "confidence": prop.confidence,
                        "mention_text": prop.mention_text,
                        "normalized_value": prop.normalized_value.text if prop.normalized_value else None
                    }
            
            # Use entity type as key or append to list if multiple entities of same type
            if entity.type_ in result:
                if not isinstance(result[entity.type_], list):
                    result[entity.type_] = [result[entity.type_]]
                result[entity.type_].append(entity_data)
            else:
                result[entity.type_] = entity_data                
        return result
    
    def process_document(self, file_path: Path, processor_name: str) -> Dict[str, Any]:
        """
        Process a single document using Document AI.
        
        Args:
            file_path: Path to the document file
            processor_name: Document AI processor name
            
        Returns:
            Extracted document data
        """
        try:
            # Read the file
            with open(file_path, "rb") as file:
                file_content = file.read()
            
            # Determine MIME type
            mime_type = self._get_mime_type(file_path)
            
            # Configure the process request
            request = documentai.ProcessRequest(
                name=processor_name,
                raw_document=documentai.RawDocument(
                    content=file_content,
                    mime_type=mime_type
                )
            )
            
            # Process the document
            logger.info(f"Calling Document AI endpoint: {processor_name}")
            response = self.client.process_document(request=request, timeout=30)
            document = response.document
            
            # Extract structured data using the proven method
            extracted_data = self._serialize_document_entities(document)
            
            # Add metadata
            extracted_data["_metadata"] = {
                "processed_at": datetime.datetime.now().isoformat(),
                "file_name": file_path.name,
                "mime_type": mime_type,
                "page_count": len(document.pages),
                "full_text": document.text
            }
            
            logger.info(f"Processed document {file_path.name}: {len(extracted_data)} entity types extracted")
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            return {
                '_metadata': {
                    'file_name': file_path.name,
                    'error': str(e),
                    'processed_at': datetime.datetime.now().isoformat()
                }
            }
    
    def process_for_occupation(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a document specifically for occupation category information.
        Simply extract Department and Position from document text.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing the extracted occupation data
        """
        try:
            # Read the file
            with open(file_path, "rb") as file:
                file_content = file.read()
            
            # Determine MIME type
            mime_type = self._get_mime_type(file_path)
            
            # Get processor name
            processor_name = self.occupation_processor_name
            logger.info(f"Calling Occupation Category processor: {processor_name}")
            
            # Configure the process request
            request = documentai.ProcessRequest(
                name=processor_name,
                raw_document=documentai.RawDocument(
                    content=file_content, mime_type=mime_type
                )
            )
            
            # Process the document
            response = self.client.process_document(request=request, timeout=30)
            
            # Convert the entire protobuf response to a dictionary
            full_response = MessageToDict(
                response._pb,
                preserving_proto_field_name=True,
                use_integers_for_enums=False
            )
            
            logger.info("Full occupation processor response received")
            
            # Simply extract from the document text content - KISS approach
            department = None
            position = None
            
            # Get the full text from the document
            if 'document' in full_response and 'text' in full_response['document']:
                text = full_response['document']['text']
                logger.info(f"First 200 chars of document text: {text[:200]}")
                
                # Look for Department and Position in the text
                lines = text.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('Department:'):
                        department = line.replace('Department:', '').strip()
                        logger.info(f"Found department: {department}")
                    elif line.startswith('Position:'):
                        position = line.replace('Position:', '').strip()
                        logger.info(f"Found position: {position}")
            
            # Create the occupation category
            occupation_category = None
            if department and position:
                occupation_category = f"{department}-{position}"
            elif department:
                occupation_category = department
            elif position:
                occupation_category = position
                
            # Keep a simplified approach without confidence scores
            return {
                "occupation_category": occupation_category,
                "occupation_category_confidence": 1.0 if occupation_category else 0.0,
                "department": department,
                "department_confidence": 1.0 if department else 0.0,
                "position": position,
                "position_confidence": 1.0 if position else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error processing document for occupation: {str(e)}")
            return {}
    
    def process_receipts(self) -> List[Dict[str, Any]]:
        """
        Process all receipt images.
        
        Returns:
            List of extracted receipt data
        """
        receipt_files = list(self.config.receipt_dir.glob('*.png')) + \
                       list(self.config.receipt_dir.glob('*.jpg')) + \
                       list(self.config.receipt_dir.glob('*.jpeg'))
        
        receipt_data = []
        for file_path in sorted(receipt_files):
            logger.info(f"Processing receipt: {file_path.name}")
            data = self.process_document(file_path, self.receipt_processor_name)
            
            # Parse receipt-specific data
            parsed_data = self._parse_receipt_data(data)
            receipt_data.append(parsed_data)        
        return receipt_data
    
    def process_payslips(self) -> List[Dict[str, Any]]:
        """
        Process all payslip PDFs.
        
        Returns:
            List of extracted payslip data
        """
        payslip_files = list(self.config.payslip_dir.glob('*.pdf'))
        
        payslip_data = []
        for file_path in sorted(payslip_files):
            logger.info(f"Processing payslip: {file_path.name}")
            
            # Process with income statement processor
            logger.info(f"Processing with income statement processor...")
            income_data = self.process_document(file_path, self.payslip_processor_name)
            
            # Process with occupation category processor for department/position
            logger.info(f"Processing with occupation category processor...")
            occupation_data = self.process_for_occupation(file_path)
            
            # Log processor results
            logger.info(f"Income processor returned {len(income_data)} entity types")
            logger.info(f"Occupation processor returned: {occupation_data}")
            
            # Check for errors
            if '_metadata' in income_data and 'error' in income_data['_metadata']:
                logger.error(f"Income processor error: {income_data['_metadata']['error']}")
            
            # Combine and parse data
            parsed_data = self._parse_payslip_data(income_data, occupation_data)
            payslip_data.append(parsed_data)        
        return payslip_data
    
    def _parse_receipt_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse receipt data from Document AI response.
        
        Args:
            raw_data: Raw Document AI response
            
        Returns:
            Parsed receipt data
        """
        parsed = {
            'file_name': raw_data.get('_metadata', {}).get('file_name', 'unknown'),
            'receipt_date': None,
            'total_amount': None,
            'vendor_name': None,
            'vendor_address': None,
            'line_items': [],
            'raw_entities': self._convert_entities_to_list(raw_data)
        }
        
        # Extract key information from serialized entities
        for entity_type, entity_data in raw_data.items():
            if entity_type.startswith('_'):  # Skip metadata
                continue
                
            # Handle both single entity and list of entities
            entities = entity_data if isinstance(entity_data, list) else [entity_data]
            
            for entity in entities:
                if not isinstance(entity, dict):
                    continue
                    
                mention_text = entity.get('mention_text', '')
                entity_type_lower = entity_type.lower()
                
                if 'receipt_date' in entity_type_lower or 'date' in entity_type_lower:
                    parsed['receipt_date'] = mention_text
                elif 'total_amount' in entity_type_lower or 'total' in entity_type_lower:
                    parsed['total_amount'] = self._extract_amount(mention_text)
                elif 'supplier_name' in entity_type_lower or 'vendor' in entity_type_lower:
                    parsed['vendor_name'] = mention_text
                elif 'supplier_address' in entity_type_lower or 'address' in entity_type_lower:
                    parsed['vendor_address'] = mention_text
                elif 'line_item' in entity_type_lower:
                    parsed['line_items'].append(mention_text)        
        return parsed
    
    def _parse_payslip_data(self, income_data: Dict[str, Any], 
                           occupation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse payslip data from Document AI responses.
        
        Args:
            income_data: Income statement processor response
            occupation_data: Occupation category processor response
            
        Returns:
            Parsed payslip data
        """
        file_name = income_data.get('_metadata', {}).get('file_name', 'unknown')
        
        # Debug logging
        logger.info(f"Processing payslip {file_name}")
        logger.info(f"Income entity types: {list(income_data.keys())}")
        logger.info(f"Occupation data: {occupation_data}")
        
        parsed = {
            'file_name': file_name,
            'employee_name': None,
            'employer_name': None,
            'pay_period': None,
            'gross_pay': None,
            'net_pay': None,
            'deductions': [],
            'position': occupation_data.get('position'),
            'department': occupation_data.get('department'),
            'raw_income_entities': self._convert_entities_to_list(income_data),
            'raw_occupation_entities': [occupation_data] if occupation_data else []
        }
        
        # Extract income information from serialized entities
        for entity_type, entity_data in income_data.items():
            if entity_type.startswith('_'):  # Skip metadata
                continue
                
            # Handle both single entity and list of entities
            entities = entity_data if isinstance(entity_data, list) else [entity_data]
            
            for entity in entities:
                if not isinstance(entity, dict):
                    continue
                    
                mention_text = entity.get('mention_text', '')
                entity_type_lower = entity_type.lower()
                
                if 'employee_name' in entity_type_lower or 'name' in entity_type_lower:
                    parsed['employee_name'] = mention_text
                elif 'employer_name' in entity_type_lower or 'company' in entity_type_lower:
                    parsed['employer_name'] = mention_text
                elif 'pay_period' in entity_type_lower or 'period' in entity_type_lower:
                    parsed['pay_period'] = mention_text
                elif 'gross_pay' in entity_type_lower or 'gross' in entity_type_lower:
                    parsed['gross_pay'] = self._extract_amount(mention_text)
                elif 'net_pay' in entity_type_lower or 'net' in entity_type_lower:
                    parsed['net_pay'] = self._extract_amount(mention_text)
                elif 'deduction' in entity_type_lower:
                    parsed['deductions'].append(mention_text)
        
        logger.info(f"Final parsed position: {parsed['position']}, department: {parsed['department']}")
        return parsed
    
    def _convert_entities_to_list(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert the serialized entities back to a list format for compatibility."""
        entities = []
        for entity_type, entity_data in data.items():
            if entity_type.startswith('_'):  # Skip metadata
                continue
                
            # Handle both single entity and list of entities
            entity_list = entity_data if isinstance(entity_data, list) else [entity_data]
            
            for entity in entity_list:
                if isinstance(entity, dict):
                    entities.append({
                        'type': entity_type,
                        'mention_text': entity.get('mention_text', ''),
                        'confidence': entity.get('confidence', 0.0),
                        'normalized_value': entity.get('normalized_value')
                    })
        
        return entities
    
    def _extract_amount(self, text: str) -> float:
        """
        Extract numeric amount from text.
        
        Args:
            text: Text containing amount
            
        Returns:
            Extracted amount as float
        """
        import re
        
        # Remove currency symbols and extract numbers
        amount_match = re.search(r'[\d,]+\.?\d*', text.replace(',', ''))
        if amount_match:
            try:
                return float(amount_match.group())
            except ValueError:
                pass
        
        return 0.0
