"""
Setup script for the Tax Document Processing System.
This script helps with initial system configuration and dependency installation.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def install_dependencies():
    """Install Python dependencies."""
    print("📦 Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("📁 Creating directory structure...")
    
    directories = [
        "results",
        "results/visualizations",
        "results/similarity_index",
        "credentials"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {directory}")
    
    return True

def check_environment_file():
    """Check if .env file exists and has required variables."""
    print("🔧 Checking environment configuration...")
    
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️ .env file not found. Creating template...")
        
        env_template = """# Google Cloud and Document AI
GOOGLE_APPLICATION_CREDENTIALS=./credentials/google-credentials.json
GOOGLE_CLOUD_PROJECT=your-project-id
DOCUMENT_AI_LOCATION=eu

# Document AI Processor IDs
RECEIPT_PROCESSOR_ID=your-receipt-processor-id
INCOME_STATEMENT_PROCESSOR_ID=your-payslip-processor-id
OCCUPATION_CATEGORY_PROCESSOR_ID=your-occupation-processor-id

# Weaviate Configuration
WEAVIATE_URL=http://localhost:8081
"""
        
        with open(env_file, "w") as f:
            f.write(env_template)
        
        print("📝 Created .env template. Please update with your configuration.")
        return False
    else:
        print("✅ .env file exists")
        return True

def check_google_credentials():
    """Check if Google Cloud credentials exist."""
    print("🔑 Checking Google Cloud credentials...")
    
    creds_file = Path("credentials/google-credentials.json")
    if not creds_file.exists():
        print("⚠️ Google Cloud credentials not found.")
        print("   Please place your service account JSON file at:")
        print(f"   {creds_file.absolute()}")
        return False
    else:
        # Validate JSON format
        try:
            with open(creds_file, 'r') as f:
                json.load(f)
            print("✅ Google Cloud credentials found and valid")
            return True
        except json.JSONDecodeError:
            print("❌ Google Cloud credentials file is not valid JSON")
            return False

def check_data_files():
    """Check if required data files exist."""
    print("📊 Checking data files...")
    
    required_files = [
        "data/csv/transactions.csv",
        "data/csv/users.csv", 
        "data/csv/tax_filings.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"  ✅ Found: {file_path}")
    
    if missing_files:
        print("⚠️ Missing required data files:")
        for file_path in missing_files:
            print(f"  ❌ Missing: {file_path}")
        return False
    else:
        print("✅ All required data files found")
        return True

def start_weaviate():
    """Start Weaviate using Docker Compose."""
    print("🚀 Starting Weaviate vector database...")
    
    try:
        # Check if Docker is available
        subprocess.check_call(["docker", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Start only Weaviate service
        subprocess.check_call(["docker-compose", "up", "-d", "weaviate"])
        print("✅ Weaviate started successfully")
        print("   Access Weaviate console at: http://localhost:8081")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ Docker not available or failed to start Weaviate")
        print("   You can run Weaviate manually or use an external instance")
        return False

def run_system_test():
    """Run the system test to verify everything works."""
    print("🧪 Running system tests...")
    
    try:
        result = subprocess.run([sys.executable, "test_system.py"], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ System tests passed!")
            return True
        else:
            print("❌ System tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Failed to run system tests: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Tax Document Processing System Setup")
    print("=" * 50)
    
    steps = [
        ("Installing dependencies", install_dependencies),
        ("Creating directories", create_directories),
        ("Checking environment file", check_environment_file),
        ("Checking Google credentials", check_google_credentials),
        ("Checking data files", check_data_files),
    ]
    
    # Run setup steps
    all_passed = True
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        if not step_func():
            all_passed = False
    
    # Optional steps
    print(f"\n🔧 Optional setup steps:")
    
    if input("Start Weaviate with Docker? (y/n): ").lower() == 'y':
        start_weaviate()
    
    if all_passed and input("Run system tests? (y/n): ").lower() == 'y':
        run_system_test()
    
    # Final instructions
    print("\n" + "=" * 50)
    print("🎯 Setup Summary")
    print("=" * 50)
    
    if all_passed:
        print("✅ Basic setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Update .env file with your Google Cloud configuration")
        print("2. Place Google Cloud credentials in credentials/google-credentials.json")
        print("3. Run: python main.py (for full processing)")
        print("4. Run: streamlit run app.py (for web interface)")
        print("5. Open: Pattern_Recognition_Analysis.ipynb (for analysis)")
    else:
        print("⚠️ Setup completed with some issues.")
        print("Please resolve the above issues before running the system.")
    
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    main()
