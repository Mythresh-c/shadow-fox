import time
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import google.generativeai as genai
from config import GEMINI_API_KEY, MODEL_NAME, MAX_OUTPUT_TOKENS, TEMPERATURE

def setup_gemini_api():
    """Initialize Gemini API with authentication"""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
    return model

def generate_text(model, prompt: str, max_tokens: int = MAX_OUTPUT_TOKENS, temperature: float = TEMPERATURE) -> str:
    """Generate text using Gemini API with error handling"""
    try:
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature
        )
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        return response.text if response.text else "No response generated"
    
    except Exception as e:
        print(f"Error generating text: {str(e)}")
        return f"Error: {str(e)}"

def batch_generate_text(model, prompts: List[str], delay: float = 1.0) -> List[str]:
    """Generate text for multiple prompts with rate limiting"""
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}")
        result = generate_text(model, prompt)
        results.append(result)
        
        # Rate limiting
        if i < len(prompts) - 1:
            time.sleep(delay)
    
    return results

def analyze_text_metrics(text: str) -> Dict[str, Any]:
    """Analyze various metrics of generated text"""
    import textstat
    
    metrics = {
        'word_count': len(text.split()),
        'character_count': len(text),
        'sentence_count': textstat.sentence_count(text),
        'flesch_reading_ease': textstat.flesch_reading_ease(text),
        'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
        'gunning_fog': textstat.gunning_fog(text),
        'automated_readability_index': textstat.automated_readability_index(text),
        'coleman_liau_index': textstat.coleman_liau_index(text)
    }
    
    return metrics

def save_results(data: Dict[str, Any], filename: str):
    """Save analysis results to JSON file"""
    filepath = f"results/data/{filename}"
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {filepath}")

def load_results(filename: str) -> Dict[str, Any]:
    """Load analysis results from JSON file"""
    filepath = f"results/data/{filename}"
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

print("Utility functions loaded successfully!")
