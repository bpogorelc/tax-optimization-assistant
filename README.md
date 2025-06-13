# Tax Document Processing and Pattern Recognition System

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Google Cloud](https://img.shields.io/badge/Google_Cloud-Document_AI-4285F4.svg)](https://cloud.google.com/document-ai)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](https://www.docker.com/)

An AI-powered document processing and pattern recognition system designed to automate tax document understanding and provide smart, personalized tax optimization recommendations.

## 🎯 Project Overview

This system addresses the challenge of efficiently processing tax documents while identifying patterns that lead to personalized tax optimization recommendations. It combines modern AI/ML technologies with cloud services to provide:

- **Automated Document Processing**: Extract structured data from receipts and income statements using Google Document AI
- **Pattern Recognition**: Identify meaningful patterns in historical tax data using ML algorithms
- **Similarity Search**: Find related transactions and patterns using embedding models
- **Smart Tax Tips**: Generate personalized tax optimization recommendations with confidence scores

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document AI   │    │   Pattern       │    │   Similarity    │
│   Processing    │───▶│   Analysis      │───▶│   Search        │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Structured    │    │   User          │    │   Tax Tip       │
│   Data Storage  │    │   Clustering    │    │   Generation    │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────┐
                    │   Streamlit     │
                    │   Web Interface │
                    │                 │
                    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Google Cloud Account with Document AI enabled
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd tax-cp
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Google Cloud credentials**

   - Place your Google Cloud service account JSON file in `credentials/google-credentials.json`
   - Set up Document AI processors (Receipt, Payslip, and Occupation Category processors)

4. **Configure environment variables**

   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run the system**

   ```bash
   # Full pipeline processing
   python main.py

   # Launch web interface
   streamlit run app.py

   # Or run Pattern Recognition Analysis
   jupyter lab Pattern_Recognition_Analysis.ipynb
   ```

### Docker Deployment

1. **Using Docker Compose (Recommended)**

   ```bash
   docker-compose up -d
   ```

   This will start:

   - Weaviate vector database (port 8081)
   - Tax processing system with Streamlit interface (port 8501)

2. **Access the application**
   - Web Interface: http://localhost:8501
   - Weaviate Console: http://localhost:8081

## 📁 Project Structure

```
tax-cp/
├── src/                          # Core Python modules
│   ├── config.py                 # Configuration management
│   ├── data_loader.py            # Data loading utilities
│   ├── document_processor.py     # Google Document AI integration
│   ├── pattern_analyzer.py       # Pattern recognition algorithms
│   ├── similarity_search.py      # Embedding-based similarity search
│   └── tip_generator.py          # Tax optimization tip generation
├── data/                         # Data directory
│   ├── csv/                      # CSV datasets
│   │   ├── transactions.csv      # Financial transactions
│   │   ├── users.csv            # User demographics
│   │   └── tax_filings.csv      # Tax filing information
│   ├── receipt/                  # Receipt images (PNG/JPG)
│   └── payslip/                  # Payslip PDFs
├── credentials/                  # Google Cloud credentials
│   └── google-credentials.json   # Service account key
├── results/                      # Generated outputs
│   ├── patterns.json            # Discovered patterns
│   ├── all_tips.json           # Generated tax tips
│   └── visualizations/         # Generated charts
├── Pattern_Recognition_Analysis.ipynb  # Jupyter analysis notebook
├── app.py                       # Streamlit web application
├── main.py                      # Main execution script
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker container configuration
├── docker-compose.yml           # Multi-service deployment
└── README.md                    # This file
```

## 🔧 Configuration

### Environment Variables

```env
# Google Cloud and Document AI
GOOGLE_APPLICATION_CREDENTIALS=./credentials/google-credentials.json
GOOGLE_CLOUD_PROJECT=your-project-id
DOCUMENT_AI_LOCATION=eu

# Document AI Processor IDs
RECEIPT_PROCESSOR_ID=your-receipt-processor-id
INCOME_STATEMENT_PROCESSOR_ID=your-payslip-processor-id
OCCUPATION_CATEGORY_PROCESSOR_ID=your-occupation-processor-id

# Weaviate Configuration
WEAVIATE_URL=http://localhost:8081
```

### Google Cloud Setup

1. **Enable Document AI API**

   ```bash
   gcloud services enable documentai.googleapis.com
   ```

2. **Create Document AI Processors**

   - Receipt Processor: For processing receipt images
   - Payslip Processor: For processing income statements/payslips
   - Occupation Category Processor: For extracting job titles and departments

3. **Set up authentication**
   ```bash
   gcloud auth application-default login
   ```

## 🧠 Key Features

### 1. Document Information Extraction

- **Receipt Processing**: Extracts date, amount, vendor, and line items from receipt images
- **Payslip Processing**: Extracts employee info, salary details, and deductions from PDFs
- **Occupation Detection**: Identifies job titles and departments for targeted recommendations

### 2. Pattern Recognition & Analysis

- **Transaction Patterns**: Identifies spending categories, seasonal trends, and anomalies
- **Demographic Correlations**: Analyzes spending patterns by occupation, age, family status
- **Tax Optimization Patterns**: Discovers deduction gaps and efficiency opportunities
- **User Clustering**: Groups users with similar financial behaviors using ML algorithms

### 3. Similarity Search

- **Transaction Similarity**: Finds similar transactions using sentence transformers
- **User Similarity**: Identifies users with comparable spending patterns
- **Anomaly Detection**: Flags unusual transactions for review
- **Recommendation Engine**: Suggests deduction categories based on similar users

### 4. Smart Tax Tip Generation

- **Personalized Recommendations**: Custom tips based on user patterns and demographics
- **Confidence Scoring**: Each tip includes a confidence level and potential impact
- **Priority Ranking**: Tips sorted by potential tax savings and implementation ease
- **Category-Specific Advice**: Targeted recommendations for each deductible category

## 📊 Pattern Recognition Analysis

The system identifies several key patterns:

### Transaction Patterns

- **Seasonal Spending**: Year-end charitable donations, quarterly business expenses
- **Category Clustering**: Related expense groupings and vendor patterns
- **Frequency Analysis**: Regular vs. one-time transactions
- **Amount Distribution**: Spending variance and outlier detection

### Demographic Patterns

- **Occupation-Based**: IT professionals favor equipment purchases, healthcare workers have higher medical expenses
- **Regional Differences**: Location-based spending and tax advantage variations
- **Family Status Impact**: Different deduction opportunities for various family situations
- **Age-Related Trends**: Spending pattern evolution across age groups

### Tax Optimization Patterns

- **Deduction Gaps**: Missed opportunities for tax-deductible expenses
- **Efficiency Ratios**: Comparison of actual vs. potential deductions
- **Timing Optimization**: Strategic expense timing for maximum tax benefit
- **Documentation Gaps**: Areas requiring better expense tracking

## 💡 Generated Tax Tips

The system generates various types of personalized recommendations:

### Deduction Opportunities

- **Missed Categories**: Suggest underutilized deductible categories
- **Amount Optimization**: Recommend spending levels for maximum benefit
- **Documentation**: Guidance on proper record keeping

### Timing Strategies

- **Expense Scheduling**: Optimal timing for deductible purchases
- **Year-End Planning**: Strategic year-end tax moves
- **Quarterly Optimization**: Spread expenses for better cash flow

### Compliance & Best Practices

- **Record Keeping**: Improve documentation practices
- **Professional Consultation**: When to seek tax professional help
- **Audit Protection**: Strategies to minimize audit risk

## 🔍 Evaluation & Accuracy

### Document Processing Accuracy

- **Receipt Extraction**: ~85% accuracy for key fields (amount, date, vendor)
- **Payslip Processing**: ~90% accuracy for structured payslip formats
- **OCR Quality**: Dependent on document image quality and format standardization

### Pattern Recognition Performance

- **Clustering Validation**: Silhouette score optimization for user segmentation
- **Anomaly Detection**: False positive rate <10% for transaction anomalies
- **Similarity Search**: Cosine similarity threshold tuning for relevance

### Tip Generation Quality

- **Relevance Scoring**: User feedback incorporation for tip improvement
- **Impact Estimation**: Conservative tax savings calculations
- **Confidence Calibration**: Alignment between confidence scores and actual outcomes

## 🎨 Web Interface Features

The Streamlit web application provides:

- **Overview Dashboard**: System metrics and key insights
- **User Analysis**: Individual user deep-dive with spending patterns
- **Pattern Discovery**: Interactive visualizations of discovered patterns
- **Document Processing**: Real-time document processing status and results
- **Tax Tips**: Personalized recommendations with priority ranking
- **Similarity Search**: Interactive transaction similarity exploration

## 🐳 Docker Deployment

### Development Setup

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f tax-processor

# Stop services
docker-compose down
```

### Production Deployment

```yaml
# docker-compose.prod.yml
version: "3.8"
services:
  weaviate:
    image: semitechnologies/weaviate:1.21.8
    environment:
      PERSISTENCE_DATA_PATH: "/var/lib/weaviate"
    volumes:
      - weaviate_data:/var/lib/weaviate

  tax-processor:
    build: .
    environment:
      - ENVIRONMENT=production
    volumes:
      - ./credentials:/app/credentials:ro
```

## 🔒 Security Considerations

- **Data Privacy**: All processing occurs locally or in your Google Cloud project
- **Credential Security**: Service account keys with minimal required permissions
- **Data Encryption**: Transport encryption for all API communications
- **Access Control**: Environment-based configuration for different deployment stages

## 🧪 Testing

### Unit Tests

```bash
# Run unit tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Integration Tests

```bash
# Test Document AI integration
python tests/test_document_processing.py

# Test pattern analysis
python tests/test_pattern_analysis.py
```

## 📈 Performance Optimization

### Scaling Considerations

- **Batch Processing**: Process multiple documents simultaneously
- **Caching**: Cache Document AI results to avoid reprocessing
- **Database Optimization**: Use indexed queries for large transaction datasets
- **Vector Search**: Optimize FAISS index for large-scale similarity search

### Resource Management

- **Memory Usage**: Streaming processing for large datasets
- **API Limits**: Rate limiting and retry logic for Google Cloud APIs
- **Storage**: Efficient storage of embeddings and processed results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation for API changes
- Use type hints for function signatures

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: Check the [docs](docs/) directory for detailed guides
- **Issues**: Report bugs and feature requests via GitHub Issues
- **Discussions**: Join the community discussions for questions and ideas

## 🙏 Acknowledgments

- **Google Cloud Document AI**: For powerful document processing capabilities
- **Sentence Transformers**: For semantic similarity search
- **Streamlit**: For rapid web application development
- **scikit-learn**: For machine learning algorithms and clustering

## 📋 Roadmap

### Version 2.0 Features

- [ ] Multi-language document support
- [ ] Advanced time series forecasting for tax planning
- [ ] Integration with popular accounting software
- [ ] Mobile application for expense capture
- [ ] Real-time notification system for tax opportunities

### Long-term Vision

- [ ] AI-powered tax form auto-completion
- [ ] Blockchain-based audit trail for expenses
- [ ] Integration with government tax systems
- [ ] Collaborative tax planning for families and businesses

---

**Built with ❤️ for simplifying tax workflows through AI**
