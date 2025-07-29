#!/usr/bin/env python3

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import pipeline
import uvicorn
import os
import json

# Create FastAPI app
app = FastAPI(title="Custom LLM API")

# Global variables
classifier = None
model_path = None
label_mapping = None

class TextInput(BaseModel):
    text: str

def find_and_load_model():
    """Find and load any available model"""
    global classifier, model_path, label_mapping
    
    model_paths = [
        "models/custom_support_model_WORKING"
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                print(f"Loading model: {path}")
                classifier = pipeline("text-classification", model=path, tokenizer=path)
                model_path = path
                
                # Load label mapping
                label_file = os.path.join(path, "label_mapping.json")
                if os.path.exists(label_file):
                    with open(label_file, 'r') as f:
                        mapping_data = json.load(f)
                        if 'label_mapping' in mapping_data:
                            label_mapping = mapping_data['label_mapping']
                        else:
                            label_mapping = mapping_data
                
                print("✅ Model loaded successfully!")
                return True
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                continue
    
    print("⚠️  No model found")
    return False

@app.on_event("startup")
async def startup():
    find_and_load_model()

# Beautiful Web Interface
@app.get("/", response_class=HTMLResponse)
async def web_interface():
    model_status = "Model Loaded" if classifier else "No Model"
    model_info = f"Path: {model_path}" if model_path else "Please train a model first"
    categories = list(label_mapping.keys()) if label_mapping else ["billing", "technical_support", "product_inquiry", "complaint", "compliment"]
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Custom LLM - Customer Support Classifier</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }}
            
            .container {{
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                padding: 40px;
                max-width: 800px;
                width: 100%;
                animation: fadeIn 0.5s ease-in;
            }}
            
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(20px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            
            .header {{
                text-align: center;
                margin-bottom: 30px;
            }}
            
            .header h1 {{
                color: #333;
                margin-bottom: 10px;
                font-size: 2.5em;
                background: linear-gradient(45deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}
            
            .header p {{
                color: #666;
                font-size: 1.1em;
            }}
            
            .status {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                background: #f8f9fa;
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 30px;
                border-left: 4px solid #667eea;
            }}
            
            .status-item {{
                display: flex;
                flex-direction: column;
            }}
            
            .status-label {{
                font-size: 0.9em;
                color: #666;
                margin-bottom: 5px;
            }}
            
            .status-value {{
                font-weight: bold;
                color: #333;
            }}
            
            .input-section {{
                margin-bottom: 30px;
            }}
            
            .input-label {{
                display: block;
                margin-bottom: 10px;
                font-weight: bold;
                color: #333;
            }}
            
            .text-input {{
                width: 100%;
                padding: 15px;
                border: 2px solid #e9ecef;
                border-radius: 10px;
                font-size: 1em;
                resize: vertical;
                min-height: 120px;
                transition: border-color 0.3s ease;
            }}
            
            .text-input:focus {{
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }}
            
            .button-group {{
                display: flex;
                gap: 15px;
                margin-bottom: 30px;
            }}
            
            .btn {{
                padding: 12px 24px;
                border: none;
                border-radius: 8px;
                font-size: 1em;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.3s ease;
                flex: 1;
            }}
            
            .btn-primary {{
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
            }}
            
            .btn-primary:hover {{
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }}
            
            .btn-secondary {{
                background: #6c757d;
                color: white;
            }}
            
            .btn-secondary:hover {{
                background: #5a6268;
                transform: translateY(-2px);
            }}
            
            .result {{
                background: #f8f9fa;
                border-radius: 10px;
                padding: 20px;
                margin-top: 20px;
                display: none;
                border-left: 4px solid #28a745;
            }}
            
            .result.show {{
                display: block;
                animation: slideIn 0.3s ease-out;
            }}
            
            @keyframes slideIn {{
                from {{ opacity: 0; transform: translateX(-20px); }}
                to {{ opacity: 1; transform: translateX(0); }}
            }}
            
            .result-header {{
                font-weight: bold;
                color: #333;
                margin-bottom: 15px;
                font-size: 1.1em;
            }}
            
            .result-item {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 10px;
                padding: 10px;
                background: white;
                border-radius: 5px;
            }}
            
            .result-label {{
                font-weight: bold;
                color: #555;
            }}
            
            .result-value {{
                color: #333;
            }}
            
            .confidence-bar {{
                width: 100px;
                height: 8px;
                background: #e9ecef;
                border-radius: 4px;
                overflow: hidden;
                margin-left: 10px;
            }}
            
            .confidence-fill {{
                height: 100%;
                background: linear-gradient(45deg, #28a745, #20c997);
                transition: width 0.5s ease;
            }}
            
            .examples {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 10px;
                margin-top: 15px;
            }}
            
            .example {{
                background: #e9ecef;
                padding: 10px;
                border-radius: 5px;
                cursor: pointer;
                transition: all 0.2s ease;
                font-size: 0.9em;
            }}
            
            .example:hover {{
                background: #dee2e6;
                transform: translateY(-1px);
            }}
            
            .loading {{
                text-align: center;
                color: #667eea;
                font-weight: bold;
            }}
            
            .error {{
                background: #f8d7da;
                color: #721c24;
                border-left: 4px solid #dc3545;
            }}
            
            .categories {{
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-top: 15px;
            }}
            
            .category-tag {{
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                padding: 5px 12px;
                border-radius: 15px;
                font-size: 0.8em;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 Custom LLM Classifier</h1>
                <p>Customer Support Text Classification powered by Hugging Face</p>
            </div>
            
            <div class="status">
                <div class="status-item">
                    <span class="status-label">Model Status</span>
                    <span class="status-value">{model_status}</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Model Info</span>
                    <span class="status-value">{model_info}</span>
                </div>
            </div>
            
            <div class="input-section">
                <label class="input-label">Enter your customer support message:</label>
                <textarea class="text-input" id="textInput" placeholder="Type your message here... For example: 'I need help with my billing statement' or 'The app keeps crashing'"></textarea>
            </div>
            
            <div class="button-group">
                <button class="btn btn-primary" onclick="classifyText()">🚀 Classify Text</button>
                <button class="btn btn-secondary" onclick="clearAll()">🗑️ Clear</button>
            </div>
            
            <div style="margin-bottom: 15px;">
                <strong>Try these examples:</strong>
                <div class="examples">
                    <div class="example" onclick="setExample('I was charged twice for my subscription')">💰 Billing Issue</div>
                    <div class="example" onclick="setExample('The mobile app keeps crashing on startup')">🔧 Technical Problem</div>
                    <div class="example" onclick="setExample('Your customer service team is amazing!')">⭐ Compliment</div>
                    <div class="example" onclick="setExample('What features are included in the premium plan?')">❓ Product Question</div>
                    <div class="example" onclick="setExample('I am very disappointed with this service')">😞 Complaint</div>
                </div>
            </div>
            
            <div>
                <strong>Available Categories:</strong>
                <div class="categories">
                    {''.join([f'<span class="category-tag">{cat.replace("_", " ").title()}</span>' for cat in categories])}
                </div>
            </div>
            
            <div id="result" class="result">
                <div class="result-header">Classification Result:</div>
                <div id="resultContent"></div>
            </div>
        </div>
        
        <script>
            function setExample(text) {{
                document.getElementById('textInput').value = text;
            }}
            
            function clearAll() {{
                document.getElementById('textInput').value = '';
                document.getElementById('result').classList.remove('show');
            }}
            
            async function classifyText() {{
                const text = document.getElementById('textInput').value.trim();
                const resultDiv = document.getElementById('result');
                const resultContent = document.getElementById('resultContent');
                
                if (!text) {{
                    alert('Please enter some text to classify!');
                    return;
                }}
                
                resultDiv.classList.add('show');
                resultContent.innerHTML = '<div class="loading">🤖 Analyzing your text...</div>';
                
                try {{
                    const response = await fetch('/classify', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json'
                        }},
                        body: JSON.stringify({{ text: text }})
                    }});
                    
                    const data = await response.json();
                    
                    if (response.ok) {{
                        const confidence = Math.round(data.confidence * 100);
                        const category = data.category.replace('_', ' ').toUpperCase();
                        
                        resultContent.innerHTML = `
                            <div class="result-item">
                                <span class="result-label">Input Text:</span>
                                <span class="result-value">"${{text}}"</span>
                            </div>
                            <div class="result-item">
                                <span class="result-label">Predicted Category:</span>
                                <span class="result-value">${{category}}</span>
                            </div>
                            <div class="result-item">
                                <span class="result-label">Confidence:</span>
                                <div style="display: flex; align-items: center;">
                                    <span class="result-value">${{confidence}}%</span>
                                    <div class="confidence-bar">
                                        <div class="confidence-fill" style="width: ${{confidence}}%"></div>
                                    </div>
                                </div>
                            </div>
                        `;
                    }} else {{
                        throw new Error(data.detail || 'Classification failed');
                    }}
                }} catch (error) {{
                    resultDiv.classList.remove('show');
                    resultDiv.classList.add('error');
                    resultContent.innerHTML = `<div>❌ Error: ${{error.message}}</div>`;
                    setTimeout(() => {{
                        resultDiv.classList.remove('error');
                    }}, 3000);
                }}
            }}
            
            // Allow Enter key to submit
            document.getElementById('textInput').addEventListener('keydown', function(event) {{
                if (event.ctrlKey && event.key === 'Enter') {{
                    classifyText();
                }}
            }});
        </script>
    </body>
    </html>
    """
    return html_content

# API Endpoints
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": classifier is not None,
        "model_path": model_path
    }

@app.post("/classify")
async def classify(input_data: TextInput):
    if not classifier:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train a model first.")
    
    try:
        result = classifier(input_data.text)
        if isinstance(result, list):
            result = result[0]
        
        return {
            "text": input_data.text,
            "category": result.get("label", "unknown"),
            "confidence": result.get("score", 0.0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

@app.get("/test")
async def test():
    if not classifier:
        return {"error": "No model loaded"}
    
    test_cases = [
        "I need help with my billing statement",
        "The app keeps crashing on startup", 
        "Your customer service is excellent",
        "What features are in the premium plan",
        "I am disappointed with this service"
    ]
    
    results = []
    for text in test_cases:
        try:
            result = classifier(text)
            if isinstance(result, list):
                result = result[0]
            results.append({
                "text": text,
                "prediction": result.get("label", "unknown"),
                "confidence": result.get("score", 0.0)
            })
        except Exception as e:
            results.append({
                "text": text,
                "prediction": "error",
                "confidence": 0.0,
                "error": str(e)
            })
    
    return {"test_results": results}

if __name__ == "__main__":
    print("🚀 Starting Custom LLM API with Beautiful Web Interface...")
    print("🌐 Web Interface: http://localhost:8000")
    print("📖 API Docs: http://localhost:8000/docs")
    print("🧪 Test Endpoint: http://localhost:8000/test")
    print("="*60)
    
    uvicorn.run(app, host="127.0.0.1", port=8000)