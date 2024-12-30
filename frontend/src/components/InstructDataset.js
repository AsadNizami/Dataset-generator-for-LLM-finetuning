import React, { useState, useEffect } from 'react';
import './InstructDataset.css';

function InstructDataset() {
    const [file, setFile] = useState(null);
    const [numPairs, setNumPairs] = useState(5);
    const [temperature, setTemperature] = useState(0.7);
    const [generatedPairs, setGeneratedPairs] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [expandedRows, setExpandedRows] = useState(new Set());
    const [models, setModels] = useState([]);
    const [selectedModel, setSelectedModel] = useState('');
    const [prompt, setPrompt] = useState(`Analyze the given text and create exactly one question-answer pair.

You must:
1. Return only a JSON array containing one object
2. Use exactly this format, no extra text:
[
    {
        "question": "Clear, specific question from the text?",
        "answer": "Direct, factual answer from the text."
    }
]

Important:
- Keep answers concise and factual
- Questions should be specific and answerable from the text
- Do not add any explanations or additional text
- Do not create multiple pairs
- Ensure valid JSON syntax with double quotes`);

    useEffect(() => {
        const fetchModels = async () => {
            try {
                const response = await fetch('http://localhost:8000/api/models');
                if (!response.ok) throw new Error('Failed to fetch models');
                const data = await response.json();
                setModels(data.models);
                if (data.models.length > 0) {
                    setSelectedModel(data.models[0].name);
                }
            } catch (error) {
                console.error('Error fetching models:', error);
            }
        };
        
        fetchModels();
    }, []);

    const handleFileChange = (event) => {
        setFile(event.target.files[0]);
        setError(null);
    };

    const handleNumPairsChange = (event) => {
        setNumPairs(parseInt(event.target.value) || 5);
    };

    const handleTemperatureChange = (event) => {
        setTemperature(parseFloat(event.target.value));
    };

    const handleModelChange = (event) => {
        setSelectedModel(event.target.value);
    };

    const handlePromptChange = (event) => {
        setPrompt(event.target.value);
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        if (!file) {
            setError('Please select a file');
            return;
        }

        setIsLoading(true);
        setError(null);

        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('num_pairs', numPairs);
            formData.append('temperature', temperature);
            formData.append('model', selectedModel);
            formData.append('prompt', prompt);

            const response = await fetch('http://localhost:8000/api/generate-dataset', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error('Failed to generate dataset');
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n').filter(line => line.trim());

                for (const line of lines) {
                    try {
                        const data = JSON.parse(line);
                        if (data.error) {
                            setError(data.error);
                            continue;
                        }
                        setGeneratedPairs(prev => [...prev, data]);
                    } catch (e) {
                        console.error('Error parsing JSON:', e);
                    }
                }
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    };

    const handleClear = () => {
        setGeneratedPairs([]);
        setExpandedRows(new Set());
        setError(null);
    };

    const toggleRowExpansion = (index) => {
        const newExpandedRows = new Set(expandedRows);
        if (newExpandedRows.has(index)) {
            newExpandedRows.delete(index);
        } else {
            newExpandedRows.add(index);
        }
        setExpandedRows(newExpandedRows);
    };

    const downloadJSON = () => {
        const dataStr = JSON.stringify(generatedPairs, null, 2);
        const blob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'conversations.json';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    };

    const handleDeleteRow = (index, event) => {
        event.stopPropagation();
        setGeneratedPairs(prev => prev.filter((_, i) => i !== index));
    };

    return (
        <div className="instruct-dataset">
            <form onSubmit={handleSubmit}>
                <div className="form-controls">
                    <div className="form-inputs">
                        <div className="form-group file-upload">
                            <label>
                                Upload Text File:
                                {!file ? (
                                    <input
                                        type="file"
                                        accept=".txt"
                                        onChange={handleFileChange}
                                    />
                                ) : (
                                    <div className="file-selected">
                                        <span className="filename">{file.name}</span>
                                        <button 
                                            type="button" 
                                            className="change-file" 
                                            onClick={() => setFile(null)}
                                        >
                                            ×
                                        </button>
                                    </div>
                                )}
                            </label>
                        </div>
                        <div className="form-group pairs-input">
                            <label>
                                Number of Pairs:
                                <input
                                    type="number"
                                    min="1"
                                    value={numPairs}
                                    onChange={handleNumPairsChange}
                                />
                            </label>
                        </div>
                        <div className="form-group temperature-control">
                            <label htmlFor="temperature">
                                <div className="temperature-label">
                                    Temperature
                                    <span className="temperature-value">{temperature}</span>
                                </div>
                                <div className="slider-container">
                                    <input
                                        type="range"
                                        id="temperature"
                                        min="0"
                                        max="1"
                                        step="0.1"
                                        value={temperature}
                                        onChange={handleTemperatureChange}
                                    />
                                    <div className="slider-labels">
                                        <span>Focused</span>
                                        <span>Creative</span>
                                    </div>
                                </div>
                            </label>
                        </div>
                        <div className="form-group model-select">
                            <label>
                                Model:
                                <select
                                    value={selectedModel}
                                    onChange={handleModelChange}
                                    disabled={!models.length}
                                >
                                    {models.length === 0 ? (
                                        <option value="">Loading models...</option>
                                    ) : (
                                        models.map(model => (
                                            <option key={model.name} value={model.name}>
                                                {model.name}
                                            </option>
                                        ))
                                    )}
                                </select>
                            </label>
                        </div>
                    </div>
                    <div className="form-actions">
                        <button type="submit" disabled={isLoading || !file}>
                            {isLoading ? 'Generating...' : 'Generate Dataset'}
                        </button>
                        {generatedPairs.length > 0 && (
                            <>
                                <button type="button" onClick={downloadJSON} className="save-button">
                                    Save
                                </button>
                                <button type="button" onClick={handleClear} className="clear-button">
                                    Clear All
                                </button>
                            </>
                        )}
                    </div>
                </div>
            </form>

            <div className="content-container">
                <div className="form-group prompt-input">
                    <label>
                        Instruction Prompt:
                        <textarea
                            value={prompt}
                            onChange={handlePromptChange}
                            placeholder="Enter your instruction prompt..."
                            rows={4}
                        />
                    </label>
                </div>
                <div className="generated-pairs">
                    <div className="dataset-header">
                        <div className="header-cell number">#</div>
                        <div className="header-cell conversations">conversations</div>
                        <div className="header-cell source">source</div>
                    </div>
                    <div className="pairs-list">
                        {generatedPairs.map((item, index) => (
                            <div 
                                key={index} 
                                className={`dataset-row ${expandedRows.has(index) ? 'expanded' : ''}`}
                                onClick={() => toggleRowExpansion(index)}
                            >
                                <div className="cell number">{index + 1}</div>
                                <div className="cell conversations">
                                    <pre>{JSON.stringify(item.conversations, null, expandedRows.has(index) ? 2 : 0)}</pre>
                                </div>
                                <div className="cell source">
                                    {item.source}
                                    <button 
                                        className="delete-row"
                                        onClick={(e) => handleDeleteRow(index, e)}
                                        title="Delete row"
                                    >
                                        ×
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {error && (
                <div className="error-message">
                    Error: {error}
                </div>
            )}
        </div>
    );
}

export default InstructDataset; 