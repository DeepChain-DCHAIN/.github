# DeepChain

<a id="user-content-pinned" href="#pinned">
  <h3>📌 Pinned</h3>
</a>

<div align="center">
  <img src="https://raw.githubusercontent.com/DeepChain-DCHAIN/assets/09b82a9bbdbb123a9dd7ef043d5aa76503cec93a/logo.svg" alt="DeepChain Logo" width="400px">
</div>

<div align="center">

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![DeepSeek-R1](https://img.shields.io/badge/DeepSeek--R1-671B-red.svg)](https://huggingface.co/deepseek-ai/DeepSeek-R1)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68%2B-009688.svg)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/🦜_LangChain-0.1.0-blue.svg)](https://github.com/hwchase17/langchain)
[![Anchor](https://img.shields.io/badge/Anchor-0.28.0-black.svg)](https://www.anchor-lang.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

DeepChain is an AI-powered blockchain automation layer designed to enhance smart contract security, token generation, and on-chain intelligence on Solana. By integrating DeepSeek-R1 (671B parameters), DeepChain enables faster, more efficient, and automated blockchain interactions.

## 🚀 Core Products

### [DeepScan™](https://github.com/DeepChain-DCHAIN/DeepScan)
AI-powered smart contract security scanner that analyzes Solana tokens and smart contracts for security risks and provides detailed insights.

### [DeepCode™](https://github.com/DeepChain-DCHAIN/DeepCode)
AI-powered smart contract generator that creates optimized Solana smart contracts with built-in security features.

### [DeepChat™](https://github.com/DeepChain-DCHAIN/DeepChat)
AI blockchain development assistant that helps with smart contract development, debugging, and blockchain concepts explanation.

## Key Features

### DeepScan™ - AI-Powered Smart Contract Auditing
- Smart contract security scoring
- Risk assessment and key findings
- Token details analysis
- Supply and market cap verification
- Wallet concentration analysis
- Deployment information verification

### DeepCode™ - Smart Contract Generation
- Automated SPL token contract generation
- NFT collection smart contract templates
- Security-optimized deployment
- Rust contract optimization
- Custom token economics
- Metadata management
- Royalty configuration
- Collection attributes

### DeepChat™ - AI Blockchain Assistant
- Smart contract code explanation
- Blockchain concepts clarification
- Development guidance and best practices
- Error troubleshooting assistance
- Documentation help and examples

## Prerequisites

| Software | Version |
|----------|---------|
| Python | 3.9+ |
| CUDA | 11.8+ |
| Enterprise Linux | RHEL/CentOS 8+ |
| Rust | 1.70.0+ |
| Solana CLI Tools | Latest |
| Node.js | 16+ |

## DeepSeek-R1 Setup Guide

### Method 1: HuggingFace Transformers (Recommended)

```bash
# Install dependencies
pip install torch transformers accelerate bitsandbytes

# Download and setup DeepSeek-R1
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = 'deepseek-ai/DeepSeek-R1'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True,
    use_safetensors=True
)
"
```

### Method 2: Using Ollama

1. Install Ollama:
```bash
# Linux/WSL
curl -fsSL https://ollama.com/install.sh | sh

# MacOS
brew install ollama
```

2. Pull and run DeepSeek-R1:
```bash
ollama pull deepseek-r1
ollama run deepseek-r1
```

3. Use the Ollama API:
```python
import requests

response = requests.post('http://localhost:11434/api/generate', 
    json={
        'model': 'deepseek-r1',
        'prompt': 'Analyze this smart contract...'
    }
)
```

### Method 3: Docker Deployment

```bash
# Pull our pre-configured Docker image
docker pull deepchain/deepseek-r1:latest

# Run the container with GPU support
docker run --gpus all -p 8000:8000 \
    -v ${PWD}/data:/app/data \
    -e CUDA_VISIBLE_DEVICES=0,1 \
    deepchain/deepseek-r1:latest
```

## 🚀 Quick Start

1. Clone the repository and set up Python environment:
```bash
# Clone the repository
git clone https://github.com/DeepChain-DCHAIN/DeepChain.git
cd DeepChain

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

2. Install base dependencies:
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other required packages
pip install transformers accelerate bitsandbytes fastapi uvicorn langchain
```

3. Verify DeepSeek-R1 installation:
```bash
# Create a simple test script
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Try loading the model to verify installation
try:
    model_name = 'deepseek-ai/DeepSeek-R1'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True
    )
    print('✅ DeepSeek-R1 is correctly installed and loaded')
except Exception as e:
    print('❌ Error loading DeepSeek-R1:', str(e))
"
```

4. Set up environment variables:
```bash
# Copy example env file
cp .env.example .env

# Edit .env file with your configuration
# Required variables:
# - DEEPSEEK_MODEL_PATH: Path to your downloaded model
# - SOLANA_RPC_URL: Your Solana RPC endpoint
```

5. Start the API server:
```bash
# Start the FastAPI server
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

6. Test DeepChain features:
```bash
# Test DeepScan (Smart Contract Audit)
curl -X POST "http://localhost:8000/scan" \
     -H "Content-Type: application/json" \
     -d '{
       "contract": "// Your Solana contract here\npub mod token {\n    use anchor_lang::prelude::*;\n    // ...\n}"
     }'

# Test DeepCode (Contract Generation)
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "type": "token",
       "params": {
         "name": "MyToken",
         "symbol": "MTK",
         "decimals": 9
       }
     }'

# Test DeepChat (AI Assistant)
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "How do I implement token vesting in Solana?"
     }'
```

Each endpoint will return detailed JSON responses with the requested information. For example, DeepScan will return security findings, DeepCode will return generated contract code, and DeepChat will return AI-powered guidance.

## 🏗️ Architecture

DeepChain leverages a modular architecture combining Solana's high-performance blockchain with DeepSeek-R1's AI capabilities:

| Component | Technology Stack | Purpose |
|-----------|-----------------|----------|
| Smart Contracts | Rust + Anchor | Core blockchain logic |
| AI Layer | DeepSeek-R1 (671B) | Contract generation & analysis |
| API Layer | FastAPI + LangChain | Service integration |
| Frontend | TypeScript + React | User interfaces |

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Contract Generation | ~2-3s |
| Security Analysis | ~30s |
| Chat Response Time | ~500ms |
| Max Context Window | 128K tokens |
| Concurrent Users | 100+ |

## 🔒 Security Features

- Enterprise-grade encryption
- Real-time vulnerability detection
- Access control patterns
- Rate limiting
- Input validation
- Error handling
- Automated auditing

## 🔍 Model Details

DeepChain uses DeepSeek-R1, a state-of-the-art language model:

- Architecture: Mixture of Experts (MoE)
- Parameters: 671B total (37B activated)
- Context Length: 128K tokens
- Training: Reinforcement Learning + SFT
- Specialization: Blockchain & Smart Contracts

## 🤝 Contributing

We welcome contributions! Please read our contributing guidelines before submitting PRs.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 


## 🔗 Links

- [Website](https://deepchain.cloud/)
- [Documentation](https://docs.deepchain.cloud/)
- [Twitter](https://x.com/deepchaincloud)
