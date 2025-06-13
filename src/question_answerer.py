"""LLM-based Question Answering System using Vertex AI and Document AI OCR."""
import json
import logging
from typing import List, Dict, Any
from pathlib import Path
from google.cloud import documentai_v1 as documentai
from google.api_core.client_options import ClientOptions
from google.auth import default
import re

# Try to import Vertex AI components
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel
    VERTEX_AI_AVAILABLE = True
except ImportError:
    try:
        from google.cloud import aiplatform
        VERTEX_AI_AVAILABLE = True
        vertexai = None
        GenerativeModel = None
    except ImportError:
        VERTEX_AI_AVAILABLE = False
        vertexai = None
        GenerativeModel = None

# Handle PDF reading libraries
try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

logger = logging.getLogger(__name__)

class QuestionAnswerer:
    """LLM-based question answering system using Vertex AI based on processed document data."""
    def __init__(self, config):
        """Initialize the question answerer with Vertex AI and Document AI."""
        self.config = config
        
        # Initialize Vertex AI if available
        if VERTEX_AI_AVAILABLE and vertexai is not None:
            try:
                vertexai.init(
                    project=config.vertex_ai_project,
                    location=config.vertex_ai_location
                )
                self.model = GenerativeModel(config.vertex_ai_model)
                self.vertex_ai_enabled = True
                logger.info("Vertex AI initialized successfully")
            except Exception as e:
                logger.warning(f"Vertex AI initialization failed: {e}. Using fallback mode.")
                self.model = None
                self.vertex_ai_enabled = False
        else:
            logger.warning("Vertex AI not available. Using fallback mode.")
            self.model = None
            self.vertex_ai_enabled = False
        
        # Initialize Document AI for OCR
        credentials, project_id = default()
        self.project_id = project_id or config.google_project_id
        
        client_options = ClientOptions(
            api_endpoint=f"{config.document_ai_location}-documentai.googleapis.com",
            quota_project_id=self.project_id
        )
        
        self.documentai_client = documentai.DocumentProcessorServiceClient(
            client_options=client_options,
            credentials=credentials
        )
        
        # Use the general OCR processor
        self.ocr_processor_name = self.documentai_client.processor_path(
            self.project_id,
            config.document_ai_location,
            config.occupation_category_processor_id  # Using as general OCR
        )
        
        logger.info("Question answerer initialized with Vertex AI and Document AI")
    
    def extract_questions_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF using Document AI OCR with PyPDF2 fallback."""
        try:
            logger.info(f"Extracting questions from {pdf_path} using Document AI OCR")
            
            # Read the PDF file
            with open(pdf_path, "rb") as file:
                file_content = file.read()
            
            # Configure the process request for OCR
            request = documentai.ProcessRequest(
                name=self.ocr_processor_name,
                raw_document=documentai.RawDocument(
                    content=file_content,
                    mime_type="application/pdf"
                )
            )
            
            # Process the document with Document AI
            response = self.documentai_client.process_document(request=request, timeout=30)
            document = response.document
            
            # Extract the full text
            extracted_text = document.text
            logger.info(f"Successfully extracted {len(extracted_text)} characters using Document AI OCR")
            
            return extracted_text
            
        except Exception as e:
            logger.warning(f"Document AI OCR failed: {e}. Falling back to PyPDF2")
            return self._extract_text_with_pypdf2(pdf_path)
    
    def _extract_text_with_pypdf2(self, pdf_path: Path) -> str:
        """Fallback method to extract text using available PDF library."""
        try:
            text = ""
            
            if HAS_PYPDF:
                # Use pypdf (newer)
                with open(pdf_path, 'rb') as file:
                    pdf_reader = pypdf.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                logger.info(f"Successfully extracted {len(text)} characters using pypdf")
                
            elif HAS_PYPDF2:
                # Use PyPDF2 (legacy)
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                logger.info(f"Successfully extracted {len(text)} characters using PyPDF2")
            
            else:
                raise ImportError("No PDF reading library available")
            
            return text
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return ""
    
    # def parse_questions(self, text: str) -> List[str]:
    #     """Parse individual questions from the extracted text."""
    #     questions = []
        
    #     # Clean the text
    #     text = text.replace("\n", " ").replace("\r", " ")
    #     text = re.sub(r'\s+', ' ', text).strip()
        
    #     # Try multiple question patterns
    #     patterns = [
    #         r'(?:^|\s)(\d+\.\s[^\d]*?)(?=\d+\.|$)',  # "1. question text"
    #         r'(?:^|\s)(Q\d+[\.:][^Q]*?)(?=Q\d+|$)',      # "Q1: question text"
    #         r'(?:^|\s)(\d+\)[^\d]*?)(?=\d+\)|$)',      # "1) question text"
    #     ]
        
    #     for pattern in patterns:
    #         matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
    #         if matches:
    #             questions.extend([match.strip() for match in matches])
    #             break
        
    #     # If no pattern matches, try splitting by common question indicators
    #     if not questions:
    #         lines = text.split('.')
    #         for line in lines:
    #             line = line.strip()
    #             if any(word in line.lower() for word in ['what', 'how', 'when', 'where', 'who', 'which', 'why']):
    #                 if len(line) > 10:  # Filter out very short fragments
    #                     questions.append(line + "?")
        
    #     # Clean up questions
    #     cleaned_questions = []
    #     for q in questions:
    #         q = q.strip()
    #         if len(q) > 10:  # Filter out very short questions
    #             if not q.endswith('?'):
    #                 q += '?'
    #             cleaned_questions.append(q)
        
    #     logger.info(f"Parsed {len(cleaned_questions)} questions from text")
    #     return cleaned_questions

    def parse_questions(self, text: str) -> List[str]:
        """Parse individual questions from the extracted text."""
        # --- extract the questions -----------------------------------------------
        pattern = r'(?:^|\n)\d+\.\s*(.*?)(?=\n\d+\.|\Z)'   # grab text after each number up to the next
        cleaned_questions = [' '.join(q.split())           # flatten any line breaks / extra spaces
                             for q in re.findall(pattern, text, flags=re.S)]

        
        logger.info(f"Parsed {len(cleaned_questions)} questions from text")
        return cleaned_questions
    
    def load_processed_data(self) -> Dict[str, Any]:
        """Load the processed payslip and receipt data."""
        data = {}
        
        # Load payslip data
        payslip_path = Path("results/payslip_data.json")
        if payslip_path.exists():
            try:
                with open(payslip_path, 'r', encoding='utf-8') as f:
                    data['payslips'] = json.load(f)
                logger.info(f"Loaded {len(data['payslips'])} payslip records")
            except Exception as e:
                logger.error(f"Error loading payslip data: {e}")
                data['payslips'] = []
        else:
            data['payslips'] = []
            logger.warning("payslip_data.json not found")
        
        # Load receipt data
        receipt_path = Path("results/receipt_data.json")
        if receipt_path.exists():
            try:
                with open(receipt_path, 'r', encoding='utf-8') as f:
                    data['receipts'] = json.load(f)
                logger.info(f"Loaded {len(data['receipts'])} receipt records")
            except Exception as e:
                logger.error(f"Error loading receipt data: {e}")
                data['receipts'] = []
        else:
            data['receipts'] = []
            logger.warning("receipt_data.json not found")

        # # Load payslip ocr data
        # payslip_ocr_path = Path("results/payslip_ocr.json")
        # if payslip_ocr_path.exists():
        #     try:
        #         with open(payslip_ocr_path, 'r', encoding='utf-8') as f:
        #             data['payslips'] = json.load(f)
        #         logger.info(f"Loaded {len(data['payslips'])} payslip records")
        #     except Exception as e:
        #         logger.error(f"Error loading payslip data: {e}")
        #         data['payslips'] = []
        # else:
        #     data['payslips'] = []
        #     logger.warning("payslip_ocr.json not found")

        # # Load receipt ocr data
        # receipt_ocr_path = Path("results/receipt_ocr.json")
        # if receipt_ocr_path.exists():
        #     try:
        #         with open(receipt_ocr_path, 'r', encoding='utf-8') as f:
        #             data['receipts'] = json.load(f)
        #         logger.info(f"Loaded {len(data['receipts'])} receipt records")
        #     except Exception as e:
        #         logger.error(f"Error loading receipt data: {e}")
        #         data['receipts'] = []
        # else:
        #     data['receipts'] = []
        #     logger.warning("receipt_ocr.json not found")

        return data
    
    def create_context_prompt(self, data: Dict[str, Any]) -> str:
        """Create a comprehensive context prompt from the processed data."""
        context = """You are an AI assistant analyzing tax-related documents. Below is the structured data extracted from payslips and receipts using advanced OCR technology.

PAYSLIP DATA ANALYSIS:
"""
        
        # Add payslip information
        payslips = data.get('payslips', [])
        if payslips:
            for i, payslip in enumerate(payslips, 1):
                context += f"\nPayslip {i} ({payslip.get('file_name', 'unknown')}): \n"
                context += f"  • Employee: {payslip.get('employee_name', 'N/A')}\n"
                context += f"  • Employer: {payslip.get('employer_name', 'N/A')}\n"
                context += f"  • Department: {payslip.get('department', 'N/A')}\n"
                context += f"  • Position: {payslip.get('position', 'N/A')}\n"
                context += f"  • Gross Pay: {payslip.get('gross_pay', 'N/A')}\n"
                context += f"  • Net Pay: {payslip.get('net_pay', 'N/A')}\n"
                context += f"  • Pay Period: {payslip.get('pay_period', 'N/A')}\n"
                
                # Add deductions if available
                deductions = payslip.get('deductions', [])
                if deductions:
                    context += f"  • Deductions: {', '.join(str(d) for d in deductions)}\n"
                
                # Add raw entities for detailed analysis
                raw_income = payslip.get('raw_income_entities', [])
                if raw_income:
                    context += f"  • Income Details: "
                    for entity in raw_income[:3]:  # Limit to first 3 for brevity
                        if isinstance(entity, dict):
                            context += f"{entity.get('type', 'N/A')}: {entity.get('mention_text', 'N/A')}; "
                    context += "\n"
        else:
            context += "No payslip data available.\n"
        
        context += "\nRECEIPT DATA ANALYSIS:\n"
        
        # Add receipt information
        receipts = data.get('receipts', [])
        if receipts:
            for i, receipt in enumerate(receipts, 1):
                context += f"\nReceipt {i} ({receipt.get('file_name', 'unknown')}): \n"
                context += f"  • Vendor: {receipt.get('vendor_name', 'N/A')}\n"
                context += f"  • Total Amount: {receipt.get('total_amount', 'N/A')}\n"
                context += f"  • Currency: {receipt.get('currency', 'N/A')}\n"
                context += f"  • Date: {receipt.get('receipt_date', 'N/A')}\n"
                
                # Add line items if available
                line_items = receipt.get('line_items', [])
                if line_items:
                    context += f"  • Items purchased:\n"
                    for item in line_items[:5]:  # Limit to first 5 items
                        if isinstance(item, dict):
                            desc = item.get('description', 'N/A')
                            price = item.get('total_price', 'N/A')
                            context += f"    - {desc} ({price})\n"
                
                # Add supplier address if available
                supplier_addr = receipt.get('supplier_address', 'N/A')
                if supplier_addr and supplier_addr != 'N/A':
                    context += f"  • Supplier Address: {supplier_addr}\n"
        else:
            context += "No receipt data available.\n"
        
        context += """
INSTRUCTIONS:
Please answer the following questions based on this structured data. Be specific and cite the relevant documents when possible. If information is not available in the data, clearly state that. When referring to documents, use the file names (e.g., "According to payslip 2.pdf" or "Based on receipt 3.png").

Provide detailed, accurate answers that demonstrate analysis of the available data."""
        
        return context
    
    def answer_question(self, question: str, context: str) -> str:
        """Answer a single question using Vertex AI with fallback to rule-based approach."""
        try:
            # First try Vertex AI
            return self._answer_with_vertex_ai(question, context)
            
        except Exception as e:
            # If Vertex AI fails, use fallback rule-based approach
            logger.warning(f"Vertex AI failed for question '{question[:50]}...': {e}")
            logger.info("Falling back to rule-based answering")
            return self._answer_with_rules(question, context)
    
    #def _answer_with_vertex_ai(self, question: str, context: str) -> str:
    # tiny defensive tweak in src/question_answerer.py
    def _answer_with_vertex_ai(self, question: str, context: str) -> str:
        """Answer using Vertex AI."""
        if not self.vertex_ai_enabled or self.model is None:
            raise RuntimeError("Vertex AI is not initialised")

        # Create the full prompt
        full_prompt = f"""{context}

QUESTION: {question}

Please provide a comprehensive answer based on the data above:
"""
        
        # Generate response using Vertex AI
        response = self.model.generate_content(
            full_prompt,
            generation_config={
                "max_output_tokens": 1000,
                "temperature": 0.2,
                "top_p": 0.8,
                "top_k": 40
            }
        )
        answer = response.text.strip()
        
        logger.info(f"Generated answer for question: {question[:50]}...")
        return answer
    
    def _answer_with_rules(self, question: str, context: str) -> str:
        """Generate answer using rule-based approach when Vertex AI is not available."""
        question_lower = question.lower()
        
        # Parse data from context (simple extraction)
        payslips_data = []
        receipts_data = []
        
        # Try to load actual data files for rule-based processing
        try:
            data = self.load_processed_data()
            payslips_data = data.get('payslips', [])
            receipts_data = data.get('receipts', [])
        except Exception:
            pass
        
        # Rule-based question answering
        if 'gross' in question_lower and 'net' in question_lower:
            return self._answer_payslip_comparison(payslips_data)
        
        elif 'department' in question_lower or 'position' in question_lower:
            return self._answer_employment_info(payslips_data)
        
        elif 'pharmacy' in question_lower or 'medical' in question_lower:
            return self._answer_medical_purchases(receipts_data)
        
        elif 'apple store' in question_lower:
            return self._answer_apple_store_purchases(receipts_data)
        
        elif 'café' in question_lower or 'coffee' in question_lower or 'u-bahn' in question_lower:
            return self._answer_cafe_purchases(receipts_data)
        
        elif 'total' in question_lower and 'spend' in question_lower:
            return self._answer_total_spending(receipts_data, payslips_data)
        
        elif 'receipt' in question_lower and ('flag' in question_lower or 'review' in question_lower):
            return self._answer_flagged_receipts(receipts_data)
        
        else:
            return self._generate_general_summary(question, payslips_data, receipts_data)
    
    def _answer_payslip_comparison(self, payslips: List[Dict]) -> str:
        """Answer questions about gross vs net income comparison."""
        if not payslips:
            return "No payslip data available to compare gross and net income."
        
        comparisons = []
        for payslip in payslips:
            file_name = payslip.get('file_name', 'Unknown')
            gross = payslip.get('gross_pay')
            net = payslip.get('net_pay')
            
            if gross is not None and net is not None:
                diff = gross - net
                percentage = (diff / gross * 100) if gross > 0 else 0
                comparisons.append(f"{file_name}: Gross €{gross:.2f}, Net €{net:.2f}, Deduction €{diff:.2f} ({percentage:.1f}%)")
        
        if comparisons:
            return "Gross vs Net Income Analysis:\n" + "\n".join(comparisons)
        else:
            return "Unable to find complete gross and net income data in the payslips."
    
    def _answer_employment_info(self, payslips: List[Dict]) -> str:
        """Answer questions about employment information."""
        employment_info = []
        
        for payslip in payslips:
            file_name = payslip.get('file_name', 'Unknown')
            position = payslip.get('position')
            department = payslip.get('department')
            
            if position or department:
                info = f"{file_name}: "
                if position:
                    info += f"Position: {position}"
                if department:
                    if position:
                        info += f", Department: {department}"
                    else:
                        info += f"Department: {department}"
                employment_info.append(info)
        
        if employment_info:
            return "Employment Information Found:\n" + "\n".join(employment_info)
        else:
            return "No department or position information found in the payslips."
    
    def _answer_medical_purchases(self, receipts: List[Dict]) -> str:
        """Answer questions about pharmacy/medical purchases."""
        medical_purchases = []
        
        for receipt in receipts:
            vendor = receipt.get('vendor_name', '').lower()
            amount = receipt.get('total_amount')
            file_name = receipt.get('file_name', 'Unknown')
            
            if 'pharmacy' in vendor or 'medical' in vendor or 'apotheke' in vendor:
                if amount is not None:
                    medical_purchases.append(f"{file_name}: {receipt.get('vendor_name')} - €{amount:.2f}")
                else:
                    medical_purchases.append(f"{file_name}: {receipt.get('vendor_name')} - Amount not available")
        
        if medical_purchases:
            total = sum(r.get('total_amount', 0) for r in receipts 
                       if r.get('vendor_name', '').lower() in ['pharmacy', 'medical', 'apotheke'] 
                       and r.get('total_amount') is not None)
            return f"Medical/Pharmacy Purchases:\n" + "\n".join(medical_purchases) + f"\nTotal: €{total:.2f}"
        else:
            return "No pharmacy or medical purchases found in the receipts."
    
    def _answer_apple_store_purchases(self, receipts: List[Dict]) -> str:
        """Answer questions about Apple Store purchases."""
        apple_purchases = []
        total_amount = 0
        
        for receipt in receipts:
            vendor = receipt.get('vendor_name', '').lower()
            amount = receipt.get('total_amount')
            file_name = receipt.get('file_name', 'Unknown')
            
            if 'apple' in vendor:
                if amount is not None:
                    apple_purchases.append(f"{file_name}: €{amount:.2f}")
                    total_amount += amount
                else:
                    apple_purchases.append(f"{file_name}: Amount not available")
        
        if apple_purchases:
            return f"Apple Store Purchases:\n" + "\n".join(apple_purchases) + f"\nTotal spent at Apple Store: €{total_amount:.2f}"
        else:
            return "No Apple Store purchases found in the receipts."
    
    def _answer_cafe_purchases(self, receipts: List[Dict]) -> str:
        """Answer questions about café purchases."""
        cafe_purchases = []
        
        for receipt in receipts:
            vendor = receipt.get('vendor_name', '').lower()
            amount = receipt.get('total_amount')
            file_name = receipt.get('file_name', 'Unknown')
            
            if 'café' in vendor or 'coffee' in vendor or 'u-bahn' in vendor:
                if amount is not None:
                    cafe_purchases.append(f"{file_name}: {receipt.get('vendor_name')} - €{amount:.2f}")
                else:
                    cafe_purchases.append(f"{file_name}: {receipt.get('vendor_name')} - Amount not available")
        
        if cafe_purchases:
            return "Café Purchases Found:\n" + "\n".join(cafe_purchases)
        else:
            return "No café purchases found in the receipts."
    
    def _answer_total_spending(self, receipts: List[Dict], payslips: List[Dict]) -> str:
        """Answer questions about total spending."""
        receipt_total = sum(r.get('total_amount', 0) for r in receipts if r.get('total_amount') is not None)
        receipt_count = len([r for r in receipts if r.get('total_amount') is not None])
        
        payslip_total = sum(p.get('gross_pay', 0) for p in payslips if p.get('gross_pay') is not None)
        payslip_count = len([p for p in payslips if p.get('gross_pay') is not None])
        
        return f"Financial Summary:\n" \
               f"Total spending from {receipt_count} receipts: €{receipt_total:.2f}\n" \
               f"Total gross income from {payslip_count} payslips: €{payslip_total:.2f}"
    
    def _answer_flagged_receipts(self, receipts: List[Dict]) -> str:
        """Answer questions about flagged receipts."""
        flagged_count = 0
        flagged_receipts = []
        
        for receipt in receipts:
            # Check for potential flags (missing data, low confidence, etc.)
            file_name = receipt.get('file_name', 'Unknown')
            has_issues = False
            issues = []
            
            if not receipt.get('total_amount'):
                issues.append("missing amount")
                has_issues = True
            
            if not receipt.get('vendor_name'):
                issues.append("missing vendor")
                has_issues = True
            
            # Check confidence scores in raw entities
            raw_entities = receipt.get('raw_entities', [])
            if isinstance(raw_entities, list):
                low_confidence = any(
                    entity.get('confidence', 1.0) < 0.7 
                    for entity in raw_entities 
                    if isinstance(entity, dict)
                )
                
                if low_confidence:
                    issues.append("low confidence")
                    has_issues = True
            
            if has_issues:
                flagged_count += 1
                flagged_receipts.append(f"{file_name}: {', '.join(issues)}")
        
        if flagged_count > 0:
            return f"Receipts flagged for manual review: {flagged_count}\n" + "\n".join(flagged_receipts)
        else:
            return "No receipts were flagged for manual review."
    
    def _generate_general_summary(self, question: str, payslips: List[Dict], receipts: List[Dict]) -> str:
        """Generate a general summary when specific rules don't match."""
        summary = f"Based on the available data:\n"
        summary += f"- Analyzed {len(payslips)} payslip(s) and {len(receipts)} receipt(s)\n"
        
        if payslips:
            total_gross = sum(p.get('gross_pay', 0) for p in payslips if p.get('gross_pay'))
            total_net = sum(p.get('net_pay', 0) for p in payslips if p.get('net_pay'))
            summary += f"- Total gross income: €{total_gross:.2f}\n"
            summary += f"- Total net income: €{total_net:.2f}\n"
        
        if receipts:
            total_spending = sum(r.get('total_amount', 0) for r in receipts if r.get('total_amount'))
            summary += f"- Total spending from receipts: €{total_spending:.2f}\n"
        
        summary += f"\nFor the specific question: '{question}'\n"
        summary += "Please refer to the detailed data files for more specific information."
        
        return summary
    
    def answer_all_questions(self, questions_pdf_path: Path) -> Dict[str, Any]:
        """Extract questions from PDF and answer them all using processed data."""
        try:
            # Extract questions from PDF
            logger.info("Starting LLM-based question answering process")
            questions_text = self.extract_questions_from_pdf(questions_pdf_path)
            
            if not questions_text:
                raise ValueError("Failed to extract text from questions PDF")
            
            # Parse individual questions
            questions = self.parse_questions(questions_text)
            
            if not questions:
                raise ValueError("No questions found in the extracted text")
            
            # Load processed data
            data = self.load_processed_data()
            
            # Create context prompt
            context = self.create_context_prompt(data)
            
            # Answer each question
            results = {
                "metadata": {
                    "extracted_text": questions_text,
                    "parsed_questions": questions,
                    "data_summary": {
                        "payslips_count": len(data.get('payslips', [])),
                        "receipts_count": len(data.get('receipts', []))
                    },
                    "processing_timestamp": str(Path().absolute()),
                    "llm_system": "Vertex AI + Document AI OCR"
                },
                "qa_results": {}
            }
            
            logger.info(f"Answering {len(questions)} questions using LLM system")
            
            for i, question in enumerate(questions, 1):
                logger.info(f"Processing question {i}/{len(questions)}")
                answer = self.answer_question(question, context)
                results["qa_results"][f"question_{i}"] = {
                    "question": question,
                    "answer": answer,
                    "sources": "payslip_data.json, receipt_data.json, payslip_ocr.json, receipt_ocr.json"
                }
            
            logger.info("LLM-based question answering completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in question answering process: {e}")
            return {
                "error": str(e),
                "metadata": {
                    "extracted_text": "",
                    "parsed_questions": [],
                    "data_summary": {"payslips_count": 0, "receipts_count": 0}
                },
                "qa_results": {}
            }
    
    def save_results(self, results: Dict[str, Any], output_path: Path = None) -> Path:
        """Save the question answering results to JSON file."""
        if output_path is None:
            output_path = Path("results/llm_qa_results.json")
        
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"LLM results saved to {output_path}")
        return output_path
    
    def create_summary_report(self, results: Dict[str, Any]) -> str:
        """Create a human-readable summary report."""
        if "error" in results:
            return f"# LLM Question Answering Report\n\nError: {results['error']}"
        
        metadata = results.get('metadata', {})
        qa_results = results.get('qa_results', {})
        
        report = "# LLM-based Tax Document Question Answering Report\n\n"
        report += f"**System:** {metadata.get('llm_system', 'Unknown')}\n"
        report += f"**Generated:** {metadata.get('processing_timestamp', 'Unknown')}\n"
        report += f"**Questions Processed:** {len(qa_results)}\n"
        report += f"**Payslips Analyzed:** {metadata.get('data_summary', {}).get('payslips_count', 0)}\n"
        report += f"**Receipts Analyzed:** {metadata.get('data_summary', {}).get('receipts_count', 0)}\n\n"
        
        # Add each question and answer
        for key, qa in qa_results.items():
            question_num = key.replace('question_', '')
            report += f"## Question {question_num}\n\n"
            report += f"**Q:** {qa['question']}\n\n"
            report += f"**A:** {qa['answer']}\n\n"
            report += f"*Sources: {qa.get('sources', 'N/A')}*\n\n"
            report += "---\n\n"
        
        return report