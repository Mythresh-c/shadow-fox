# Gemini Language Model Analysis Project

This project provides a comprehensive analysis of Google's Gemini Language Model through various NLP tasks, performance evaluations, and research insights.

## 🎯 Project Overview

A complete framework for analyzing language model capabilities including:
- **Context Understanding** - Multi-turn conversation analysis
- **Creativity Assessment** - Creative writing and storytelling evaluation  
- **Domain Adaptability** - Performance across 7 professional domains
- **Consistency Analysis** - Response reliability and variation metrics
- **Research Questions** - Three key research hypotheses with quantitative answers

## 📁 Project Structure

```
Project3/
├── gemini_lm_analysis.ipynb    # Main Jupyter notebook with complete analysis
├── requirements.txt            # Python dependencies
├── config.py                  # Configuration and API setup
├── utils.py                   # Utility functions for analysis
├── run_final_analysis.py      # Executable Python script
├── env_example.txt            # Environment variables template
├── README.md                  # This file
└── results/                   # Generated analysis outputs
    └── data/
        ├── context_test_results.json
        ├── creativity_test_results.json
        ├── domain_test_results.json
        ├── text_metrics_analysis.csv
        ├── word_frequency_analysis.json
        ├── correlation_analysis.json
        ├── readability_analysis.json
        └── consistency_analysis.json
```

## 🚀 Quick Start

### Option 1: Jupyter Notebook
1. Install dependencies: `pip install -r requirements.txt`
2. Copy `env_example.txt` to `.env` and add your Gemini API key
3. Run: `jupyter notebook gemini_lm_analysis.ipynb`

### Option 2: Python Script
1. Install dependencies: `pip install -r requirements.txt`
2. Copy `env_example.txt` to `.env` and add your Gemini API key
3. Run: `python run_final_analysis.py`

## 🔧 Setup Instructions

1. **Get Gemini API Key:**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key

2. **Configure Environment:**
   ```bash
   cp env_example.txt .env
   # Edit .env and add your API key
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Analysis Features

- **Text Metrics:** Word count, readability scores, complexity analysis
- **Word Frequency:** Most common terms and patterns
- **Correlation Analysis:** Prompt-response length relationships
- **Domain Comparison:** Performance across Medical, Legal, Financial, Technical, Scientific fields
- **Consistency Testing:** Response variation analysis
- **Research Insights:** Evidence-based conclusions about model behavior

## 🎯 Research Questions Explored

1. **RQ1:** Does response length correlate with prompt complexity?
2. **RQ2:** How does readability vary across different domains?
3. **RQ3:** What is the consistency level of repeated prompts?

## 📈 Sample Results

- **Total responses analyzed:** 13 responses
- **Average response length:** 27.3 words
- **Context understanding:** 100% accuracy
- **Domain adaptability:** Significant variation across fields
- **Consistency:** High reliability (CV < 0.1)

## 🛠️ Requirements

- Python 3.8+ (Note: Python 3.14 may have compatibility issues)
- google-generativeai
- pandas, numpy, matplotlib, seaborn
- textstat (for readability analysis)

## 📝 Notes

- The analysis includes both real API integration and demo mode with mock data
- All results are saved in JSON/CSV format for further analysis
- The framework is extensible for additional research questions
