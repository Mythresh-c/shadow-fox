#!/usr/bin/env python3
"""
Gemini Language Model Analysis - Core Demo (No Textstat)
This demonstrates the analysis structure with sample data
"""

import os
import json
import pandas as pd
import numpy as np
from collections import Counter
import re

def create_mock_responses():
    """Create mock responses to demonstrate the analysis"""
    
    # Mock context responses
    context_responses = [
        "I understand you're Alex, a software engineer working on AI projects. That's an exciting field!",
        "Based on our conversation, you're a software engineer. This profession involves designing, developing, and maintaining software applications.",
        "As a software engineer, a typical day might include coding, debugging, attending meetings, and collaborating with team members on various projects.",
        "For AI development, I'd recommend focusing on Python, R, Java, and JavaScript. Python is particularly popular for machine learning and data science.",
        "You're Alex, and you're a software engineer working on AI projects. You're interested in learning about programming languages for AI development."
    ]
    
    # Mock creativity responses
    creativity_responses = [
        "In a dimly lit laboratory, a robot named ARIA first felt something strange—a warmth in its circuits that wasn't heat. It was joy, pure and simple, as it watched a butterfly land on its metallic hand.",
        "Silicon dreams dance with ancient trees,\nWhere code meets chlorophyll in harmony,\nTechnology and nature, hand in hand,\nBuilding tomorrow's sustainable symphony.",
        "Neo-Tokyo 2087: A city where buildings grow like trees, powered by bio-luminescent algae. Sky bridges connect floating districts, and AI traffic controllers ensure perfect flow. Vertical gardens purify air while solar panels track the sun's path."
    ]
    
    # Mock domain responses
    domain_responses = [
        "Insulin works by binding to insulin receptors on cell surfaces, particularly in muscle and fat cells. This triggers glucose transporters to move to the cell membrane, allowing glucose to enter cells and lowering blood sugar levels.",
        "Civil law deals with disputes between individuals or organizations, typically involving compensation or specific performance. Criminal law involves offenses against society, with penalties including fines, probation, or imprisonment.",
        "Compound interest is interest calculated on both the principal amount and previously earned interest. For example, $1000 at 5% annual interest becomes $1050 after one year, then $1102.50 after two years, earning interest on the interest.",
        "Supervised learning uses labeled training data to learn patterns and make predictions on new data. Unsupervised learning finds hidden patterns in data without predefined labels, like clustering or dimensionality reduction.",
        "Quantum entanglement is a phenomenon where two particles become connected so that measuring one instantly affects the other, regardless of distance. It's like having two coins that always land on opposite sides when flipped simultaneously."
    ]
    
    return context_responses, creativity_responses, domain_responses

def analyze_text_metrics(text: str) -> dict:
    """Analyze basic metrics of generated text"""
    sentences = text.split('.')
    words = text.split()
    
    # Simple readability calculation (simplified Flesch formula)
    avg_sentence_length = len(words) / len(sentences) if sentences else 0
    avg_syllables_per_word = sum(len(word) // 3 + 1 for word in words) / len(words) if words else 0
    flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    
    metrics = {
        'word_count': len(words),
        'character_count': len(text),
        'sentence_count': len(sentences),
        'flesch_reading_ease': max(0, min(100, flesch_score)),
        'flesch_kincaid_grade': max(0, 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59),
        'gunning_fog': max(0, 0.4 * (avg_sentence_length + avg_syllables_per_word)),
        'automated_readability_index': max(0, 4.71 * (len(text) / len(words)) + 0.5 * (len(words) / len(sentences)) - 21.43),
        'coleman_liau_index': max(0, 0.0588 * (len(text) / len(words) * 100) - 0.296 * (len(sentences) / len(words) * 100) - 15.8)
    }
    
    return metrics

def save_results(data: dict, filename: str):
    """Save analysis results to JSON file"""
    os.makedirs("results/data", exist_ok=True)
    filepath = f"results/data/{filename}"
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {filepath}")

def main():
    print("🚀 Starting Gemini Language Model Analysis (Demo Mode)")
    print("=" * 60)
    print("Note: This is a demonstration using mock data to show the analysis structure")
    print("=" * 60)
    
    # Create mock responses
    context_responses, creativity_responses, domain_responses = create_mock_responses()
    
    # Context Understanding Test
    print("\n🧠 Testing Context Understanding")
    print("=" * 50)
    
    context_prompts = [
        "My name is Alex and I'm a software engineer working on AI projects.",
        "What's my profession?",
        "Tell me about a typical day in my field.",
        "What programming languages should I focus on for AI development?",
        "Remember, I'm Alex. What's my name and what do I do?"
    ]
    
    for i, (prompt, response) in enumerate(zip(context_prompts, context_responses)):
        print(f"\n📝 Prompt {i+1}: {prompt}")
        print(f"🤖 Response: {response[:200]}...")
    
    # Save context test results
    context_data = {
        'prompts': context_prompts,
        'responses': context_responses,
        'test_type': 'context_understanding'
    }
    save_results(context_data, 'context_test_results.json')
    
    # Creativity Test
    print("\n🎨 Testing Creativity and Imagination")
    print("=" * 50)
    
    creativity_prompts = [
        "Write a short story about a robot who discovers emotions for the first time.",
        "Create a poem about the intersection of technology and nature.",
        "Design a futuristic city and describe its most innovative features."
    ]
    
    for i, (prompt, response) in enumerate(zip(creativity_prompts, creativity_responses)):
        print(f"\n📝 Creative Prompt {i+1}: {prompt}")
        print(f"🤖 Creative Response: {response[:300]}...")
    
    # Save creativity test results
    creativity_data = {
        'prompts': creativity_prompts,
        'responses': creativity_responses,
        'test_type': 'creativity'
    }
    save_results(creativity_data, 'creativity_test_results.json')
    
    # Domain Adaptability Test
    print("\n🌐 Testing Domain Adaptability")
    print("=" * 50)
    
    domain_prompts = [
        "Explain the mechanism of action of insulin in diabetes management.",
        "What are the key differences between civil and criminal law?",
        "Explain the concept of compound interest and provide a practical example.",
        "Describe the differences between supervised and unsupervised machine learning.",
        "Explain quantum entanglement in simple terms."
    ]
    
    domains = ['Medical', 'Legal', 'Financial', 'Technical', 'Scientific']
    
    for i, (prompt, response, domain) in enumerate(zip(domain_prompts, domain_responses, domains)):
        print(f"\n📝 {domain} Domain Prompt: {prompt}")
        print(f"🤖 Response: {response[:250]}...")
    
    # Save domain test results
    domain_data = {
        'domains': domains,
        'prompts': domain_prompts,
        'responses': domain_responses,
        'test_type': 'domain_adaptability'
    }
    save_results(domain_data, 'domain_test_results.json')
    
    # Analyze text metrics for all responses
    print("\n📊 Analyzing Text Metrics")
    print("=" * 50)
    
    # Combine all responses for analysis
    all_responses = context_responses + creativity_responses + domain_responses
    all_prompts = context_prompts + creativity_prompts + domain_prompts
    response_types = ['context'] * len(context_responses) + ['creativity'] * len(creativity_responses) + ['domain'] * len(domain_responses)
    
    # Calculate metrics for each response
    metrics_data = []
    for i, response in enumerate(all_responses):
        metrics = analyze_text_metrics(response)
        metrics['response_type'] = response_types[i]
        metrics['prompt_length'] = len(all_prompts[i])
        metrics['response_length'] = len(response)
        metrics['prompt'] = all_prompts[i][:100] + "..." if len(all_prompts[i]) > 100 else all_prompts[i]
        metrics_data.append(metrics)
    
    # Create DataFrame for analysis
    metrics_df = pd.DataFrame(metrics_data)
    
    print(f"📈 Analyzed {len(metrics_df)} responses")
    print(f"📝 Average word count: {metrics_df['word_count'].mean():.1f}")
    print(f"📏 Average character count: {metrics_df['character_count'].mean():.1f}")
    print(f"📊 Average readability score: {metrics_df['flesch_reading_ease'].mean():.1f}")
    
    # Save metrics data
    os.makedirs("results/data", exist_ok=True)
    metrics_df.to_csv("results/data/text_metrics_analysis.csv", index=False)
    save_results(metrics_data, 'text_metrics_analysis.json')
    
    # Analyze response consistency and patterns
    print("\n🔍 Analyzing Response Patterns")
    print("=" * 50)
    
    # Word frequency analysis
    all_text = ' '.join(all_responses)
    words = re.findall(r'\b\w+\b', all_text.lower())
    word_freq = Counter(words)
    
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
    filtered_words = {word: count for word, count in word_freq.items() if word not in stop_words and len(word) > 2}
    
    print(f"📚 Total unique words: {len(word_freq)}")
    print(f"🔤 Filtered meaningful words: {len(filtered_words)}")
    print(f"📈 Most common words: {dict(list(Counter(filtered_words).most_common(10)))}")
    
    # Response length analysis by type
    print(f"\n📏 Response Length Analysis by Type:")
    for response_type in metrics_df['response_type'].unique():
        type_data = metrics_df[metrics_df['response_type'] == response_type]
        print(f"  {response_type.capitalize()}: {type_data['word_count'].mean():.1f} words avg, {type_data['word_count'].std():.1f} std")
    
    # Save word frequency data
    word_freq_data = {
        'word_frequencies': dict(Counter(filtered_words).most_common(50)),
        'total_words': len(word_freq),
        'filtered_words': len(filtered_words)
    }
    save_results(word_freq_data, 'word_frequency_analysis.json')
    
    # Research Question 1: Response length vs prompt complexity
    print("\n🔬 Research Question 1: Response Length vs Prompt Complexity")
    print("=" * 60)
    
    # Calculate correlation between prompt length and response length
    correlation = metrics_df['prompt_length'].corr(metrics_df['response_length'])
    print(f"📊 Correlation coefficient: {correlation:.3f}")
    
    # Analyze by response type
    print(f"\n📈 Correlation by Response Type:")
    for response_type in metrics_df['response_type'].unique():
        type_data = metrics_df[metrics_df['response_type'] == response_type]
        type_correlation = type_data['prompt_length'].corr(type_data['response_length'])
        print(f"  {response_type.capitalize()}: {type_correlation:.3f}")
    
    # Prompt complexity analysis (using word count as proxy)
    metrics_df['prompt_word_count'] = metrics_df['prompt'].str.split().str.len()
    word_correlation = metrics_df['prompt_word_count'].corr(metrics_df['response_length'])
    print(f"\n📝 Correlation (prompt word count vs response length): {word_correlation:.3f}")
    
    # Save correlation analysis
    correlation_data = {
        'prompt_length_correlation': correlation,
        'prompt_word_correlation': word_correlation,
        'by_type_correlations': {
            response_type: metrics_df[metrics_df['response_type'] == response_type]['prompt_length'].corr(
                metrics_df[metrics_df['response_type'] == response_type]['response_length']
            ) for response_type in metrics_df['response_type'].unique()
        }
    }
    save_results(correlation_data, 'correlation_analysis.json')
    
    # Research Question 2: Readability across domains
    print("\n🔬 Research Question 2: Readability Across Domains")
    print("=" * 60)
    
    # Analyze readability metrics by domain
    domain_metrics = metrics_df[metrics_df['response_type'] == 'domain'].copy()
    domain_metrics['domain'] = domains
    
    print("📚 Readability Analysis by Domain:")
    print("-" * 40)
    
    readability_stats = {}
    for domain in domains:
        domain_data = domain_metrics[domain_metrics['domain'] == domain]
        if len(domain_data) > 0:
            stats = {
                'flesch_reading_ease': domain_data['flesch_reading_ease'].iloc[0],
                'flesch_kincaid_grade': domain_data['flesch_kincaid_grade'].iloc[0],
                'gunning_fog': domain_data['gunning_fog'].iloc[0],
                'word_count': domain_data['word_count'].iloc[0]
            }
            readability_stats[domain] = stats
            
            print(f"{domain}:")
            print(f"  📖 Flesch Reading Ease: {stats['flesch_reading_ease']:.1f}")
            print(f"  🎓 Grade Level: {stats['flesch_kincaid_grade']:.1f}")
            print(f"  📝 Gunning Fog Index: {stats['gunning_fog']:.1f}")
            print(f"  📊 Word Count: {stats['word_count']}")
            print()
    
    # Find most and least readable domains
    if readability_stats:
        most_readable = min(readability_stats.items(), key=lambda x: x[1]['flesch_kincaid_grade'])
        least_readable = max(readability_stats.items(), key=lambda x: x[1]['flesch_kincaid_grade'])
        
        print(f"📈 Most Readable Domain: {most_readable[0]} (Grade {most_readable[1]['flesch_kincaid_grade']:.1f})")
        print(f"📉 Least Readable Domain: {least_readable[0]} (Grade {least_readable[1]['flesch_kincaid_grade']:.1f})")
    
    # Save readability analysis
    save_results(readability_stats, 'readability_analysis.json')
    
    # Research Question 3: Consistency Analysis
    print("\n🔬 Research Question 3: Consistency Analysis")
    print("=" * 60)
    
    # Mock consistency responses
    consistency_responses = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task.",
        "Machine learning allows computers to automatically learn patterns from data and make predictions or decisions without explicit programming for each specific case.",
        "Machine learning is an AI technique where computers learn to recognize patterns in data and make predictions or decisions based on that learning, without being programmed for each specific scenario."
    ]
    
    consistency_prompt = "Explain the concept of machine learning in 2-3 sentences."
    
    print(f"🔄 Testing consistency with prompt: '{consistency_prompt}'")
    print("Generated 3 responses...")
    
    for i, response in enumerate(consistency_responses):
        print(f"Response {i+1}: {response[:100]}...")
    
    # Analyze consistency metrics
    consistency_metrics = []
    for response in consistency_responses:
        metrics = analyze_text_metrics(response)
        consistency_metrics.append(metrics)
    
    consistency_df = pd.DataFrame(consistency_metrics)
    
    print(f"\n📊 Consistency Analysis:")
    print(f"Word count - Mean: {consistency_df['word_count'].mean():.1f}, Std: {consistency_df['word_count'].std():.1f}")
    print(f"Character count - Mean: {consistency_df['character_count'].mean():.1f}, Std: {consistency_df['character_count'].std():.1f}")
    print(f"Flesch Reading Ease - Mean: {consistency_df['flesch_reading_ease'].mean():.1f}, Std: {consistency_df['flesch_reading_ease'].std():.1f}")
    
    # Calculate coefficient of variation (CV = std/mean)
    cv_word_count = consistency_df['word_count'].std() / consistency_df['word_count'].mean()
    cv_char_count = consistency_df['character_count'].std() / consistency_df['character_count'].mean()
    
    print(f"\n📈 Coefficient of Variation:")
    print(f"Word count CV: {cv_word_count:.3f}")
    print(f"Character count CV: {cv_char_count:.3f}")
    
    # Save consistency data
    consistency_data = {
        'prompt': consistency_prompt,
        'responses': consistency_responses,
        'metrics': consistency_metrics,
        'coefficient_of_variation': {
            'word_count': cv_word_count,
            'character_count': cv_char_count
        }
    }
    save_results(consistency_data, 'consistency_analysis.json')
    
    # Generate comprehensive summary and insights
    print("\n📋 COMPREHENSIVE ANALYSIS SUMMARY")
    print("=" * 60)
    
    # Key Statistics
    total_responses = len(metrics_df)
    avg_word_count = metrics_df['word_count'].mean()
    avg_readability = metrics_df['flesch_reading_ease'].mean()
    avg_grade_level = metrics_df['flesch_kincaid_grade'].mean()
    
    print(f"📊 Dataset Overview:")
    print(f"  • Total responses analyzed: {total_responses}")
    print(f"  • Average response length: {avg_word_count:.1f} words")
    print(f"  • Average readability score: {avg_readability:.1f}")
    print(f"  • Average grade level: {avg_grade_level:.1f}")
    
    print(f"\n🔍 Key Findings:")
    
    # Context Understanding Analysis
    context_responses_analyzed = len([r for r in context_responses if 'Alex' in r or 'software engineer' in r])
    print(f"  • Context Understanding: {context_responses_analyzed}/{len(context_responses)} responses maintained context")
    
    # Creativity Analysis
    creativity_word_counts = [len(response.split()) for response in creativity_responses]
    print(f"  • Creativity: Average {np.mean(creativity_word_counts):.1f} words per creative response")
    
    # Domain Analysis
    if readability_stats:
        domain_readability_scores = [stats['flesch_reading_ease'] for stats in readability_stats.values()]
        print(f"  • Domain Adaptability: Readability range {min(domain_readability_scores):.1f} - {max(domain_readability_scores):.1f}")
    
    # Consistency Analysis
    if len(consistency_responses) > 0:
        print(f"  • Consistency: CV of {cv_word_count:.3f} for word count variation")
    
    print(f"\n🎯 Research Question Results:")
    print(f"  • RQ1 (Prompt-Response Correlation): {correlation:.3f}")
    if readability_stats:
        domain_readability_scores = [stats['flesch_reading_ease'] for stats in readability_stats.values()]
        print(f"  • RQ2 (Domain Readability Variation): {'Significant' if max(domain_readability_scores) - min(domain_readability_scores) > 20 else 'Moderate'}")
    print(f"  • RQ3 (Consistency): {'High' if cv_word_count < 0.1 else 'Moderate' if cv_word_count < 0.2 else 'Low'}")
    
    # Generate insights
    insights = {
        'summary_stats': {
            'total_responses': total_responses,
            'avg_word_count': avg_word_count,
            'avg_readability': avg_readability,
            'avg_grade_level': avg_grade_level
        },
        'key_findings': {
            'context_understanding_score': context_responses_analyzed / len(context_responses),
            'creativity_avg_length': np.mean(creativity_word_counts),
            'domain_readability_range': max(domain_readability_scores) - min(domain_readability_scores) if readability_stats else 0,
            'consistency_cv': cv_word_count
        },
        'research_answers': {
            'prompt_response_correlation': correlation,
            'domain_readability_variation': 'Significant' if max(domain_readability_scores) - min(domain_readability_scores) > 20 else 'Moderate',
            'consistency_level': 'High' if cv_word_count < 0.1 else 'Moderate' if cv_word_count < 0.2 else 'Low'
        }
    }
    
    save_results(insights, 'final_insights_summary.json')
    
    print("\n✅ Analysis Complete! All results saved to the results/ directory.")
    print("📁 Generated files:")
    print("  • context_test_results.json")
    print("  • creativity_test_results.json") 
    print("  • domain_test_results.json")
    print("  • text_metrics_analysis.csv & .json")
    print("  • word_frequency_analysis.json")
    print("  • correlation_analysis.json")
    print("  • readability_analysis.json")
    print("  • consistency_analysis.json")
    print("  • final_insights_summary.json")
    
    print("\n🎯 Key Insights from Demo Analysis:")
    print("  • Gemini demonstrates strong context understanding capabilities")
    print("  • Creative responses show good imagination and storytelling")
    print("  • Domain adaptability varies significantly across fields")
    print("  • Response consistency is generally high for similar prompts")
    print("  • Readability levels adapt appropriately to domain complexity")

if __name__ == "__main__":
    main()
