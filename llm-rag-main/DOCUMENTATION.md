# 📚 Legal LLM Fine-tuning and RAG System - Complete Documentation

## 📋 **Table of Contents**

1. [System Overview](#system-overview)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Installation Guide](#installation-guide)
5. [Data Preparation](#data-preparation)
6. [Fine-tuning Process](#fine-tuning-process)
7. [RAG System Setup](#rag-system-setup)
8. [Web Interface Launch](#web-interface-launch)
9. [Testing and Validation](#testing-and-validation)
10. [Troubleshooting](#troubleshooting)
11. [Deployment Options](#deployment-options)

---

## 🎯 **System Overview**

This project implements a comprehensive Legal AI system that combines:

- **Fine-tuned Llama 3.2-1B** model for legal document summarization
- **RAG (Retrieval-Augmented Generation)** system with 7,823+ legal documents
- **Interactive web interface** with chat, search, and analytics capabilities
- **Multi-device compatibility** (CPU, MPS, CUDA)

### **Key Features**

- ✅ Legal document summarization with fine-tuned LLM
- ✅ Semantic search through legal precedents and case law
- ✅ AI-powered legal chatbot with conversation memory
- ✅ Web interface with multiple interaction modes
- ✅ Support for both training and inference on different hardware

---

## 🖥️ **Prerequisites**

### **Hardware Requirements**

#### **Minimum (CPU/Mac)**

- **RAM**: 8GB+
- **Storage**: 20GB free space
- **CPU**: Multi-core processor
- **Time**: 1-2 hours for training

#### **Recommended (GPU)**

- **GPU**: 8GB+ VRAM (RTX 3070, 4060, or better)
- **RAM**: 16GB+
- **Storage**: 50GB free space
- **CUDA**: 11.8+ or 12.x
- **Time**: 15-30 minutes for training

### **Software Requirements**

- **Python**: 3.8+ (3.9-3.11 recommended)
- **Git**: For version control
- **Internet**: For downloading models and dependencies

---

## 🔧 **Environment Setup**

### **Step 2: Create Virtual Environment**

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Verify activation (should show (venv) in prompt)
which python
python --version
```

### **Step 3: Upgrade pip**

```bash
# Upgrade pip to latest version
pip install --upgrade pip
```

---

## 📦 **Installation Guide**

### **Step 1: Install Core Dependencies**

```bash
# Install PyTorch (CPU version for Mac/CPU systems)
pip install torch torchvision torchaudio

# For CUDA systems, use:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### **Step 2: Install All Requirements**

```bash
# Install all project dependencies
pip install -r requirements.txt

# If you encounter issues, install key packages individually:
pip install transformers>=4.44.0
pip install datasets>=2.14.0
pip install accelerate>=0.24.0
pip install peft>=0.7.0
pip install sentence-transformers>=2.2.2
pip install faiss-cpu>=1.7.4
pip install langchain>=0.0.350
pip install streamlit>=1.28.0
pip install nltk>=3.8.1
pip install pandas>=2.0.0
pip install numpy>=1.24.0
```

### **Step 3: Download NLTK Data**

```bash
# Download required NLTK datasets
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### **Step 4: Verify Installation**

```bash
# Test key imports
python -c "
import torch
import transformers
import faiss
import streamlit
import nltk
print('✅ All core packages installed successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'Transformers version: {transformers.__version__}')
"
```

---

## 📊 **Data Preparation**

### **Step 1: Verify Dataset Structure**

```bash
# Check dataset directory structure
ls -la dataset/
# Expected structure:
# dataset/
# ├── IN-Abs/
# │   ├── train-data/
# │   │   ├── judgement/
# │   │   └── summary/
# │   └── test-data/
# │       ├── judgement/
# │       └── summary/
# ├── IN-Ext/
# └── UK-Abs/
```

### **Step 2: Preprocess Legal Documents**

```bash
# Run data preprocessing
python scripts/preprocess_data.py \
    --dataset_path dataset \
    --output_path data/processed \
    --clean_text

# Expected output:
# - Loading legal documents...
# - Dataset Statistics: 7,823 total documents
# - Saving processed data to data/processed
# - Train: 6,178, Validation: 1,545, Test: 100
```

### **Step 3: Verify Processed Data**

```bash
# Check processed data files
ls -la data/processed/
# Should show:
# - train_split.json (6,178 samples)
# - validation.json (1,545 samples)
# - test.json (100 samples)

# Check data sample
python -c "
import json
with open('data/processed/train_split.json', 'r') as f:
    data = json.load(f)
print(f'Training samples: {len(data)}')
print(f'Sample keys: {list(data[0].keys())}')
"
```

---

## 🎯 **Fine-tuning Process**

### **Step 1: Configure Training Settings**

#### **For CPU/Mac Systems:**

```bash
# Use default config (already optimized for CPU/MPS)
cat config/training_config.yaml
# Key settings:
# - per_device_train_batch_size: 1
# - gradient_accumulation_steps: 8
# - learning_rate: 0.0003
# - num_train_epochs: 3
```

#### **For GPU Systems:**

```bash
# Use GPU-optimized config
cat config/training_config_gpu.yaml
# Key settings:
# - per_device_train_batch_size: 8
# - gradient_accumulation_steps: 2
# - fp16: true
```

### **Step 2: Start Fine-tuning**

#### **CPU/Mac Training:**

```bash
# Start training with CPU/MPS optimization
python scripts/train_summarization.py

# Expected process:
# 1. Device detection (MPS/CPU)
# 2. Model loading (~2.5GB download)
# 3. LoRA setup (5.6M trainable parameters)
# 4. Training progress (45-90 minutes)
# 5. Model saving to ./results/
```

#### **GPU Training:**

```bash
# Start training with GPU optimization
python scripts/train_summarization.py --config config/training_config_gpu.yaml

# Expected process:
# 1. Device detection (CUDA)
# 2. Model loading with fp16 precision
# 3. Faster training (10-20 minutes)
# 4. Model saving to ./results/
```

### **Step 3: Monitor Training Progress**

```bash
# Training will show:
# - Loading model: meta-llama/Llama-3.2-1B-Instruct
# - LoRA configuration applied
# - Training device: MPS/CUDA/CPU
# - Progress bars with loss values
# - Periodic evaluation results
# - Model saving messages

# For wandb tracking (optional):
# Choose option 3: "Don't visualize my results" when prompted
```

### **Step 4: Verify Training Completion**

```bash
# Check trained model files
ls -la results/
# Should contain:
# - adapter_config.json
# - adapter_model.safetensors
# - training_args.bin
# - tokenizer files

# Test trained model
python -c "
from src.models.summarizer import LegalSummarizer
try:
    summarizer = LegalSummarizer.from_pretrained('./results')
    print('✅ Fine-tuned model loaded successfully')
except Exception as e:
    print(f'❌ Model loading failed: {e}')
"
```

---

## 🔍 **RAG System Setup**

### 1. Fine-tune the Model

```bash
python scripts/train_summarization.py --config config/training_config.yaml
```

### **Step 1: Build Vector Index**

```bash
# Build RAG index from legal documents
python scripts/build_rag_index.py \
    --documents_path dataset \
    --index_path data/rag_index

# Expected process:
# 1. Loading embeddings model (sentence-transformers)
# 2. Processing 7,823+ documents
# 3. Generating embeddings in batches
# 4. Building FAISS vector index
# 5. Saving index to data/rag_index/
```

### **Step 2: Verify RAG Index**

```bash
# Check RAG index files
ls -la data/rag_index/
# Should contain:
# - faiss_index.bin
# - metadata.pkl
# - documents.pkl
# - config.json

# Test RAG system
python -c "
from src.rag.legal_rag import LegalRAG
try:
    rag = LegalRAG.load('data/rag_index')
    results = rag.search('property tax', top_k=3)
    print(f'✅ RAG system working: {len(results)} results found')
except Exception as e:
    print(f'❌ RAG system failed: {e}')
"
```

### **Step 3: Test Search Functionality**

```bash
# Test various search queries
python -c "
from src.rag.legal_rag import LegalRAG
rag = LegalRAG.load('data/rag_index')

queries = [
    'property tax deduction',
    'income tax appeal',
    'municipal tax liability',
    'Supreme Court judgment'
]

for query in queries:
    results = rag.search(query, top_k=3)
    print(f'Query: \"{query}\" - Found {len(results)} results')
"
```

---

## 🌐 **Web Interface Launch**

### **Step 1: Launch Streamlit Application**

```bash
# Start the web interface
streamlit run web_app/app.py

# Expected output:
# Local URL: http://localhost:8501
# Network URL: http://192.168.x.x:8501
#
# You can now view your Streamlit app in your browser.
```

### **Step 2: Access Web Interface**

```bash
# Open browser and navigate to:
# http://localhost:8501

# Or use custom port if needed:
streamlit run web_app/app.py --server.port 8502
```

### **Step 3: Verify System Status**

The web interface will show system status:

- **With Fine-tuned Model**: "Full system loaded (Fine-tuned model + RAG)"
- **Without Fine-tuned Model**: "Base model loaded (Limited functionality)"
- **RAG System**: Shows number of indexed documents

### **Step 4: Test Web Interface Features**

#### **💬 Chat Tab**

```
Test queries:
- "What are the key principles of property tax law?"
- "Find cases related to income tax deductions"
- "Explain the legal framework for municipal taxation"
```

#### **📄 Document Summarization Tab**

```
Upload or paste legal document text:
- Court judgments
- Legal contracts
- Statutory provisions
- Case briefs
```

#### **🔍 Legal Search Tab**

```
Search queries:
- "property tax deduction cases"
- "Supreme Court income tax judgments"
- "municipal corporation tax powers"
```

#### **📊 Analytics Tab**

```
View system statistics:
- Total indexed documents
- Chat session count
- System performance metrics
- Export conversation history
```

---

## ✅ **Testing and Validation**

### **Step 1: System Health Check**

```bash
# Run comprehensive system test
python -c "
import sys
sys.path.append('.')

print('🔍 Testing Legal LLM System...')
print('=' * 50)

# Test 1: Data Loading
try:
    from src.data.data_loader import LegalDataLoader
    loader = LegalDataLoader('dataset')
    docs = loader.load_all_documents()
    print(f'✅ Data Loading: {len(docs)} documents loaded')
except Exception as e:
    print(f'❌ Data Loading: {e}')

# Test 2: Model Loading
try:
    from src.models.summarizer import LegalSummarizer
    summarizer = LegalSummarizer()
    summarizer.load_base_model()
    print('✅ Base Model: Loaded successfully')
except Exception as e:
    print(f'❌ Base Model: {e}')

# Test 3: Fine-tuned Model (if available)
try:
    summarizer = LegalSummarizer.from_pretrained('./results')
    print('✅ Fine-tuned Model: Loaded successfully')
except Exception as e:
    print(f'⚠️  Fine-tuned Model: {e}')

# Test 4: RAG System
try:
    from src.rag.legal_rag import LegalRAG
    rag = LegalRAG.load('data/rag_index')
    results = rag.search('test query', top_k=1)
    print(f'✅ RAG System: {len(results)} results found')
except Exception as e:
    print(f'❌ RAG System: {e}')

# Test 5: Chatbot
try:
    from src.chatbot.legal_chatbot import LegalChatbot
    chatbot = LegalChatbot()
    print('✅ Chatbot: Initialized successfully')
except Exception as e:
    print(f'❌ Chatbot: {e}')

print('=' * 50)
print('🎉 System test completed!')
"
```

### **Step 2: Performance Validation**

```bash
# Test summarization performance
python -c "
from src.models.summarizer import LegalSummarizer
import time

# Load model
try:
    summarizer = LegalSummarizer.from_pretrained('./results')
    model_type = 'Fine-tuned'
except:
    summarizer = LegalSummarizer()
    summarizer.load_base_model()
    model_type = 'Base'

# Test summarization
test_text = '''
Appeal No. LXVI of 1949. This is an appeal against a judgment of the High Court
of Judicature at Bombay in an income tax matter and it raises the question whether
municipal property tax and urban immoveable property tax payable under the relevant
Bombay Acts are allowable deductions under section 9 (1) (iv) of the Indian Income tax Act.
'''

start_time = time.time()
summary = summarizer.summarize(test_text)
end_time = time.time()

print(f'Model Type: {model_type}')
print(f'Processing Time: {end_time - start_time:.2f} seconds')
print(f'Summary Length: {len(summary.split())} words')
print(f'Summary: {summary[:200]}...')
"
```

### **Step 3: Web Interface Testing**

```bash
# Test web interface accessibility
python -c "
import requests
import time

# Start streamlit in background (you need to do this manually)
print('Please start: streamlit run web_app/app.py')
print('Then press Enter to test...')
input()

try:
    response = requests.get('http://localhost:8501')
    if response.status_code == 200:
        print('✅ Web interface is accessible')
    else:
        print(f'⚠️  Web interface returned status: {response.status_code}')
except Exception as e:
    print(f'❌ Web interface test failed: {e}')
"
```

---

## 🛠️ **Troubleshooting**

### **Common Installation Issues**

#### **Issue 1: PyTorch Installation**

```bash
# Error: "No module named 'torch'"
# Solution:
pip uninstall torch
pip install torch torchvision torchaudio

# For Apple Silicon Mac:
pip install torch torchvision torchaudio

# For CUDA systems:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### **Issue 2: FAISS Installation**

```bash
# Error: "No module named 'faiss'"
# Solution:
pip uninstall faiss-cpu
pip install faiss-cpu==1.7.4

# Alternative for Mac:
conda install -c conda-forge faiss-cpu
```

#### **Issue 3: Transformers Version**

```bash
# Error: "Llama model not supported"
# Solution:
pip install transformers>=4.44.0

# Or upgrade:
pip install --upgrade transformers
```

### **Common Training Issues**

#### **Issue 1: Memory Errors**

```bash
# Error: "CUDA out of memory" or "MPS out of memory"
# Solution: Reduce batch size in config/training_config.yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 16

# Or use CPU training:
export CUDA_VISIBLE_DEVICES=""
python scripts/train_summarization.py
```

#### **Issue 2: Model Loading Errors**

```bash
# Error: "Model not found"
# Solution: Check model path and permissions
ls -la results/
# Ensure adapter_model.safetensors exists

# Or retrain:
python scripts/train_summarization.py
```

#### **Issue 3: Device Compatibility**

```bash
# Error: "Device not supported"
# Solution: Check device detection
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'Device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')
"
```

### **Common Web Interface Issues**

#### **Issue 1: Port Already in Use**

```bash
# Error: "Port 8501 is already in use"
# Solution: Use different port
streamlit run web_app/app.py --server.port 8502

# Or kill existing process:
lsof -ti:8501 | xargs kill -9
```

#### **Issue 2: Model Loading in Web App**

```bash
# Error: "Model not found in web interface"
# Solution: Check model paths in web_app/app.py
# Ensure results/ directory exists with trained model
```

#### **Issue 3: RAG System Not Working**

```bash
# Error: "RAG system not available"
# Solution: Rebuild RAG index
python scripts/build_rag_index.py --documents_path dataset --index_path data/rag_index
```

---

## 🚀 **Deployment Options**

### **Option 1: Local Development**

```bash
# For development and testing
source venv/bin/activate
streamlit run web_app/app.py
# Access: http://localhost:8501
```

### **Option 2: Local Network Access**

```bash
# For access from other devices on network
streamlit run web_app/app.py --server.address 0.0.0.0
# Access: http://[your-ip]:8501
```

### **Option 3: Production Server**

```bash
# For production deployment
pip install gunicorn
# Configure reverse proxy (nginx/apache)
# Set up SSL certificates
# Configure firewall rules
```

### **Option 4: Cloud Deployment**

```bash
# For cloud platforms (AWS, GCP, Azure)
# 1. Package application
tar -czf legal-llm-system.tar.gz \
  --exclude='dataset' \
  --exclude='venv' \
  --exclude='__pycache__' \
  .

# 2. Upload to cloud instance
# 3. Install dependencies
# 4. Configure security groups
# 5. Set up domain and SSL
```

---

## 📈 **Performance Optimization**

### **For CPU/Mac Systems**

```yaml
# Optimized settings in config/training_config.yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 0.0003
  num_train_epochs: 1 # Reduce for faster training

data:
  max_source_length: 512 # Reduce for memory
  max_target_length: 128
```

### **For GPU Systems**

```yaml
# Optimized settings in config/training_config_gpu.yaml
training:
  per_device_train_batch_size: 16
  gradient_accumulation_steps: 1
  learning_rate: 0.0003
  num_train_epochs: 3
  fp16: true

data:
  max_source_length: 1024
  max_target_length: 256
```

---

## 🎯 **Quick Start Commands**

### **Complete Setup (First Time)**

```bash
# 1. Environment setup
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# 2. Install dependencies
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# 3. Prepare data
python scripts/preprocess_data.py --dataset_path dataset --output_path data/processed --clean_text

# 4. Build RAG index
python scripts/build_rag_index.py --documents_path dataset --index_path data/rag_index

# 5. Train model (optional)
python scripts/train_summarization.py

# 6. Launch web interface
streamlit run web_app/app.py
```

### **Quick Launch (After Setup)**

```bash
# Activate environment and launch
source venv/bin/activate
streamlit run web_app/app.py
```

### **Development Mode**

```bash
# For development with auto-reload
source venv/bin/activate
streamlit run web_app/app.py --server.runOnSave true
```

---

## 📞 **Support and Validation**

### **Final Verification Checklist**

- [ ] ✅ Virtual environment created and activated
- [ ] ✅ All dependencies installed without errors
- [ ] ✅ NLTK data downloaded
- [ ] ✅ Dataset preprocessed (7,823 documents)
- [ ] ✅ RAG index built successfully
- [ ] ✅ Model training completed (optional)
- [ ] ✅ Web interface accessible at http://localhost:8501
- [ ] ✅ Chat functionality working
- [ ] ✅ Document summarization working
- [ ] ✅ Legal search returning results
- [ ] ✅ System analytics displaying correctly

### **Performance Benchmarks**

| Component           | CPU/Mac   | GPU       |
| ------------------- | --------- | --------- |
| **Training Time**   | 45-90 min | 10-20 min |
| **Inference Speed** | 2-5 sec   | 0.5-1 sec |
| **Memory Usage**    | 4-8 GB    | 8-16 GB   |
| **RAG Search**      | <1 sec    | <0.5 sec  |
| **Web Response**    | 1-3 sec   | 0.5-1 sec |

### **Success Indicators**

1. **Training**: Loss decreases consistently, model saves without errors
2. **RAG**: Search returns relevant legal documents with good scores
3. **Web Interface**: All tabs functional, no error messages
4. **Performance**: Response times within expected ranges

---

## 🏆 **Congratulations!**

You have successfully set up and deployed a complete Legal LLM system with:

- ✅ **Fine-tuned Llama 3.2-1B** for legal document summarization
- ✅ **RAG System** with 7,823+ legal documents indexed
- ✅ **Interactive Web Interface** with multiple interaction modes
- ✅ **Production-ready Architecture** scalable to different hardware

**Your Legal AI Assistant is now ready at: http://localhost:8501** 🎉

For questions, issues, or contributions, refer to the troubleshooting section or create an issue in the project repository.
