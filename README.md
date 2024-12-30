# Dataset Generator for LLM Finetuning

A web application that generates high-quality question-answer pairs from text documents for LLM finetuning. The application uses Ollama to interact with local LLM models and provides a user-friendly interface for dataset generation.

## Features

- Upload text files for processing
- Generate Q&A pairs with customizable parameters
- Real-time generation feedback
- Interactive results display
- Export datasets in JSON format
- Customizable instruction prompts
- Multiple model support through Ollama
- Adjustable temperature settings
- Error tracking and validation

## Prerequisites

- Node.js (v14 or higher)
- Python (3.8 or higher)
- Ollama installed and running locally
- A compatible LLM model pulled in Ollama (e.g., llama3.2, mistral)

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
pip install fastapi uvicorn httpx python-multipart
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
2. Upload a text file (UTF-8 encoded)
3. Configure generation parameters:
   - Number of Q&A pairs to generate
   - Temperature (0.1-1.0)
   - Select LLM model
   - Customize instruction prompt if needed
4. Click "Generate Dataset" to start generation
5. Review generated pairs in the interface
6. Download the dataset using the "Save" button

## Troubleshooting

### Common Issues

1. **Backend Connection Error**
   - Ensure backend server is running on port 8000
   - Check if virtual environment is activated
   - Verify all Python dependencies are installed

2. **Ollama Connection Error**
   - Verify Ollama is running (`ollama serve`)
   - Check if selected model is installed
   - Ensure no firewall blocking port 11434

3. **Frontend Issues**
   - Clear browser cache
   - Verify Node.js version
   - Check console for error messages

### Error Messages

- "Failed to fetch models": Ollama service not running or unreachable
- "Model not available": Selected model not installed in Ollama
- "File too large": Text file exceeds size limit
- "Generation failed": Error during Q&A pair generation

## Project Structure

```
dataset-generator/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py
│   │   └── services/
│   │       └── ollama_service.py
│   └── main.py
└── frontend/
    ├── src/
    │   ├── components/
    │   │   ├── InstructDataset.js
    │   │   └── InstructDataset.css
    │   └── index.js
    └── public/
        └── index.html
```

## Development Notes

- Backend runs on FastAPI with async support
- Frontend built with React
- Real-time streaming of generated pairs
- Automatic retry mechanism for failed generations
- Comprehensive error tracking and reporting

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
    "source": "filename.txt"
}
```
