# Synthetic Dataset Generator for LLM Finetuning

This web application creates high-quality question-answer pairs from documents for fine-tuning large language models (LLMs). It utilizes Ollama to interact with local LLM models and offers a user-friendly interface for generating datasets. The application stores documents in a vector database (ChromaDB) and retrieves content based on the specified keywords.

## Features

- Generate Q&A pairs with customizable parameters
- Interactive results display
- Export datasets in JSON format
- Customizable instruction prompts
- Multiple model support through Ollama

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd dataset-generator
```

### 2. Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install fastapi uvicorn httpx python-multipart langchain langchain-ollama
```

### 3. Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install
```

### 4. Install and Setup Ollama
1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull a compatible model:
```bash
ollama pull llama3.2
# or
ollama pull mistral
```

## Starting the Application

### 1. Start Ollama Server
```bash
ollama serve
```

### 2. Start Backend Server
```bash
# Make sure you're in the backend directory and virtual environment is activated
cd backend
uvicorn main:app --reload --port 8000
```
The backend will be available at `http://localhost:8000`

### 3. Start Frontend Development Server
```bash
# In a new terminal, navigate to frontend directory
cd frontend
npm start
```
The application will open automatically at `http://localhost:3000`

## Usage

1. Open your browser and go to `http://localhost:3000`
2. Upload a pdf file
3. Configure generation parameters:
   - Number of Q&A pairs to generate
   - Temperature (0.1-1.0)
   - Select LLM model
   - Customize instruction prompt if needed
4. Click "Generate Dataset" to start generation
5. Review generated pairs in the interface
6. Download the dataset using the "Save" button

## Development Notes

- Backend runs on FastAPI with async support
- Frontend built with React
- Real-time streaming of generated pairs
- Automatic retry mechanism for failed generations

## Output Format

Generated datasets are saved in JSON format:
```json
{
    "conversations": [
        {
            "from": "human",
            "value": "Generated question?"
        },
        {
            "from": "assistant",
            "value": "Generated answer."
        }
    ],
    "source": "filename.pdf"
}
```

## Demo
![image](https://github.com/user-attachments/assets/cebb43d6-0cb0-40db-a2e6-d95ec45ece63)

