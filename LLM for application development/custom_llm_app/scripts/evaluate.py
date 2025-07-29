#!/usr/bin/env python3

import pandas as pd
import numpy as np
from transformers import pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings
warnings.filterwarnings("ignore")

def load_model_safe(model_path="models/custom_support_model_WORKING"):
    """Safely load any model"""
    print(f"Loading model from {model_path}...")
    
    try:
        classifier = pipeline(
            "text-classification",
            model=model_path,
            tokenizer=model_path
        )
        print("✅ Model loaded successfully!")
        return classifier
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def create_test_cases():
    """Create comprehensive test cases"""
    return [
        # Billing
        ("I was charged twice for my subscription", "billing"),
        ("Can you help me understand my bill", "billing"),
        ("My payment method was declined", "billing"),
        ("How do I cancel and get a refund", "billing"),
        
        # Technical Support
        ("The app crashes when I upload files", "technical_support"),
        ("I cannot log into my account", "technical_support"),
        ("The website loads very slowly", "technical_support"),
        ("Error 500 appears when I save", "technical_support"),
        
        # Product Inquiry
        ("What features are in the premium plan", "product_inquiry"),
        ("How does your product compare to others", "product_inquiry"),
        ("What are the system requirements", "product_inquiry"),
        ("Is there a free trial available", "product_inquiry"),
        
        # Complaint
        ("I have been waiting days for a response", "complaint"),
        ("The service quality has declined", "complaint"),
        ("This product does not work as advertised", "complaint"),
        ("I am disappointed with this experience", "complaint"),
        
        # Compliment
        ("Your support team is amazing", "compliment"),
        ("I love how easy this product is", "compliment"),
        ("The customer service exceeded expectations", "compliment"),
        ("This is exactly what I needed", "compliment")
    ]

def test_model_comprehensive(classifier):
    """Comprehensive model testing"""
    print("\n🧪 COMPREHENSIVE MODEL TESTING")
    print("="*60)
    
    test_cases = create_test_cases()
    
    results = []
    correct_count = 0
    total_confidence = 0
    high_confidence_count = 0
    
    print("Testing predictions:")
    print("-" * 60)
    
    for i, (text, expected) in enumerate(test_cases, 1):
        try:
            # Get prediction
            result = classifier(text)
            
            # Handle different result formats
            if isinstance(result, list) and len(result) > 0:
                pred_info = result[0]
            else:
                pred_info = result
            
            predicted = pred_info.get('label', 'unknown')
            confidence = pred_info.get('score', 0.0)
            
            # Check if correct
            is_correct = predicted == expected
            status = "✅" if is_correct else "❌"
            
            if is_correct:
                correct_count += 1
            if confidence > 0.8:
                high_confidence_count += 1
            
            total_confidence += confidence
            
            results.append({
                'text': text,
                'expected': expected,
                'predicted': predicted,
                'confidence': confidence,
                'correct': is_correct
            })
            
            print(f"{status} {i:2d}. Expected: {expected:15} | Got: {predicted:15} | Conf: {confidence:.3f}")
            
        except Exception as e:
            print(f"❌ {i:2d}. Error testing: {e}")
            results.append({
                'text': text,
                'expected': expected,
                'predicted': 'error',
                'confidence': 0.0,
                'correct': False
            })
    
    # Calculate final metrics
    accuracy = correct_count / len(test_cases)
    avg_confidence = total_confidence / len(test_cases)
    high_conf_ratio = high_confidence_count / len(test_cases)
    
    print("-" * 60)
    print(f"  FINAL RESULTS:")
    print(f"   Accuracy: {accuracy:.1%} ({correct_count}/{len(test_cases)})")
    print(f"   Average Confidence: {avg_confidence:.3f}")
    print(f"   High Confidence (>0.8): {high_conf_ratio:.1%}")
    print("-" * 60)
    
    # Performance assessment
    if accuracy >= 0.9 and avg_confidence >= 0.8:
        print(" EXCELLENT! Professional-grade model!")
        grade = "A+"
    elif accuracy >= 0.8 and avg_confidence >= 0.7:
        print(" GREAT! Production-ready model!")
        grade = "A"
    elif accuracy >= 0.7 and avg_confidence >= 0.6:
        print(" GOOD! Solid performance!")
        grade = "B+"
    elif accuracy >= 0.6 and avg_confidence >= 0.5:
        print(" DECENT! Shows improvement!")
        grade = "B"
    elif accuracy >= 0.5 and avg_confidence >= 0.4:
        print(" FAIR! Needs more work!")
        grade = "C"
    else:
        print(" POOR! Significant issues remain!")
        grade = "D"
    
    print(f"   Overall Grade: {grade}")
    
    return results, accuracy, avg_confidence

def analyze_by_category(results):
    """Analyze performance by category"""
    print(f"\n📊 CATEGORY BREAKDOWN:")
    print("-" * 40)
    
    # Group by expected category
    category_stats = {}
    
    for result in results:
        expected = result['expected']
        if expected not in category_stats:
            category_stats[expected] = {
                'total': 0,
                'correct': 0,
                'confidences': []
            }
        
        category_stats[expected]['total'] += 1
        if result['correct']:
            category_stats[expected]['correct'] += 1
        category_stats[expected]['confidences'].append(result['confidence'])
    
    # Print category performance
    for category, stats in category_stats.items():
        accuracy = stats['correct'] / stats['total']
        avg_conf = np.mean(stats['confidences'])
        
        print(f"{category.upper().replace('_', ' ')}:")
        print(f"  Accuracy: {accuracy:.1%} ({stats['correct']}/{stats['total']})")
        print(f"  Avg Confidence: {avg_conf:.3f}")
        print()

def show_errors(results):
    """Show misclassified examples"""
    errors = [r for r in results if not r['correct']]
    
    if errors:
        print(f"\n MISCLASSIFIED EXAMPLES ({len(errors)}):")
        print("-" * 60)
        for error in errors:
            print(f"Text: '{error['text'][:50]}...'")
            print(f"Expected: {error['expected']} | Got: {error['predicted']} (conf: {error['confidence']:.3f})")
            print("-" * 60)
    else:
        print(f"\n NO ERRORS! Perfect classification!")

def create_simple_visualization(results):
    """Create simple performance visualization"""
    try:
        # Confidence distribution
        confidences = [r['confidence'] for r in results]
        
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(confidences, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(confidences), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(confidences):.3f}')
        plt.title('Confidence Distribution')
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy by category
        plt.subplot(1, 2, 2)
        categories = list(set(r['expected'] for r in results))
        accuracies = []
        
        for cat in categories:
            cat_results = [r for r in results if r['expected'] == cat]
            cat_accuracy = sum(r['correct'] for r in cat_results) / len(cat_results)
            accuracies.append(cat_accuracy)
        
        plt.bar(range(len(categories)), accuracies, color='lightgreen', alpha=0.7)
        plt.title('Accuracy by Category')
        plt.xlabel('Category')
        plt.ylabel('Accuracy')
        plt.xticks(range(len(categories)), [c.replace('_', '\n') for c in categories], rotation=45)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualization saved: model_performance.png")
        plt.show()
        
    except Exception as e:
        print(f"⚠️  Could not create visualization: {e}")

def main():
    """Main evaluation function"""
    print("🧪 BULLETPROOF MODEL EVALUATION")
    print("="*60)
    
    # Try to find available models
    model_paths = [
        "models/custom_support_model_WORKING"
    ]
    
    classifier = None
    model_used = None
    
    for path in model_paths:
        if os.path.exists(path):
            classifier = load_model_safe(path)
            if classifier:
                model_used = path
                break
    
    if not classifier:
        print("❌ No working model found!")
        print("Available paths to check:")
        for path in model_paths:
            status = "✅" if os.path.exists(path) else "❌"
            print(f"  {status} {path}")
        return
    
    print(f"✅ Using model: {model_used}")
    
    # Run comprehensive testing
    results, accuracy, avg_confidence = test_model_comprehensive(classifier)
    
    # Detailed analysis
    analyze_by_category(results)
    show_errors(results)
    
    # Create visualization
    create_simple_visualization(results)
    
    # Final summary
    print(f"\n EVALUATION SUMMARY:")
    print(f"   Model: {model_used}")
    print(f"   Overall Accuracy: {accuracy:.1%}")
    print(f"   Average Confidence: {avg_confidence:.3f}")
    print(f"   Test Cases: {len(results)}")
    
    print(f"\n✅ Evaluation completed successfully!")

if __name__ == "__main__":
    main()