#!/usr/bin/env python3

import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from sklearn.metrics import accuracy_score
import torch
import os
import json
import warnings
warnings.filterwarnings("ignore")

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def create_clean_dataset():
    """Create clean, well-formatted dataset"""
    
    # Clean, diverse samples - no special characters that might cause issues
    billing_samples = [
        "I was charged twice for my subscription this month",
        "Can you help me understand my bill with these unexpected charges",
        "My payment method was declined but my bank says it is valid",
        "How do I cancel my subscription and get a refund",
        "The billing cycle does not match what I signed up for",
        "I need to update my payment information on file",
        "Why am I being charged when my subscription was cancelled",
        "The invoice amount does not match the price I was quoted",
        "I want to downgrade my plan to save money",
        "Can I get a receipt for my recent payment",
        "There is an error in my billing address information",
        "I would like to switch to annual billing for the discount",
        "My credit card expired how do I update it",
        "I was overcharged for last month usage",
        "Can you explain these fees on my statement",
        "I need to dispute a charge from last week",
        "My payment failed but I was still billed",
        "How do I change my billing address",
        "I want to add tax exemption to my account",
        "The auto renewal charged me unexpectedly"
    ]
    
    technical_samples = [
        "The app crashes every time I try to upload a file",
        "I cannot log into my account password reset is not working",
        "The website loads very slowly on my browser",
        "My data is not syncing across devices properly",
        "Error 500 appears when I try to save my work",
        "The mobile app will not connect to the server",
        "I am getting a file not found error message",
        "The search function returns no results",
        "My account dashboard is completely blank",
        "The export feature is not working correctly",
        "I cannot access my files from yesterday",
        "The desktop application freezes during startup",
        "My backup process keeps failing with errors",
        "The integration with our CRM is not working",
        "I lost all my data after the last update",
        "The API keeps returning authentication errors",
        "My notifications stopped working completely",
        "The file upload progress bar is stuck at 99 percent",
        "I cannot delete files from my storage",
        "The two factor authentication is not working"
    ]
    
    inquiry_samples = [
        "What features are included in the premium plan",
        "How does your product compare to competitors",
        "Can I integrate this with my existing workflow",
        "What is the difference between the basic and pro versions",
        "Do you offer enterprise level solutions",
        "What are the system requirements for installation",
        "Is there a free trial available",
        "How many users can I add to my account",
        "What kind of customer support do you provide",
        "Do you have any training materials or tutorials",
        "What is included in the API access tier",
        "Can I customize the interface for my team",
        "What are your data retention policies",
        "Do you offer white label solutions",
        "What integrations do you support",
        "Can I export my data if I cancel",
        "What security measures do you have in place",
        "How often do you release new features",
        "Do you offer volume discounts",
        "What is your uptime guarantee"
    ]
    
    complaint_samples = [
        "I have been waiting 5 days for a response to my ticket",
        "The service quality has declined significantly lately",
        "Your customer support is completely unhelpful",
        "This product does not work as advertised",
        "I am extremely disappointed with this experience",
        "The recent update broke several important features",
        "Your website is always down when I need it most",
        "I regret purchasing this product it is useless",
        "The promised features were never delivered",
        "This is the worst customer service I have ever experienced",
        "The software is full of bugs and crashes constantly",
        "I am switching to a competitor due to poor service",
        "Your support team never follows up on issues",
        "The product is nothing like what was demonstrated",
        "I feel like I was misled during the sales process",
        "This platform is unreliable and causes productivity loss",
        "The interface is confusing and poorly designed",
        "Your pricing model is deceptive and unclear",
        "I have experienced multiple outages this month",
        "The performance has gotten worse since I started"
    ]
    
    compliment_samples = [
        "Your support team resolved my issue quickly and professionally",
        "I love how easy this product is to use",
        "The customer service exceeded my expectations",
        "This is exactly what I was looking for perfect",
        "Thank you for the outstanding help and support",
        "The user interface is intuitive and well designed",
        "Your team went above and beyond to help me",
        "I am impressed with the quality of your service",
        "The product works flawlessly great job",
        "Best customer support experience I have ever had",
        "The new features you added are fantastic",
        "I recommend this product to all my colleagues",
        "Your documentation is clear and comprehensive",
        "The onboarding process was smooth and helpful",
        "I appreciate the regular updates and improvements",
        "The training materials helped me get started quickly",
        "Your team is responsive and knowledgeable",
        "This tool has improved our productivity significantly",
        "The integration process was seamless",
        "I am happy with my decision to choose your service"
    ]
    
    # Combine all samples
    texts = billing_samples + technical_samples + inquiry_samples + complaint_samples + compliment_samples
    categories = (['billing'] * len(billing_samples) + 
                 ['technical_support'] * len(technical_samples) +
                 ['product_inquiry'] * len(inquiry_samples) +
                 ['complaint'] * len(complaint_samples) +
                 ['compliment'] * len(compliment_samples))
    
    return pd.DataFrame({
        'text': texts,
        'category': categories
    })

def prepare_data_clean(df, model_name):
    
    # Create label mapping
    categories = sorted(df['category'].unique())
    label_mapping = {cat: idx for idx, cat in enumerate(categories)}
    id2label = {idx: cat for cat, idx in label_mapping.items()}
    label2id = label_mapping
    
    print(f"📋 Categories: {categories}")
    print(f"📊 Total samples: {len(df)}")
    print(f"📊 Distribution: {df['category'].value_counts().to_dict()}")
    
    # Add numerical labels
    df['labels'] = df['category'].map(label_mapping)
    
    # Clean split
    train_data = []
    test_data = []
    
    for category in categories:
        category_data = df[df['category'] == category].copy().reset_index(drop=True)
        
        # Take 3 samples for test, rest for train
        n_test = 3
        test_indices = list(range(n_test))
        train_indices = list(range(n_test, len(category_data)))
        
        test_data.append(category_data.iloc[test_indices])
        train_data.append(category_data.iloc[train_indices])
    
    train_df = pd.concat(train_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)
    
    # Shuffle
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Clean tokenization
    def tokenize_function(examples):
        # Ensure we have proper text input
        texts = examples['text']
        if isinstance(texts, str):
            texts = [texts]
        
        return tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors=None  # Let datasets handle tensor conversion
        )
    
    # Create clean datasets
    train_dataset = Dataset.from_pandas(train_df[['text', 'labels']])
    test_dataset = Dataset.from_pandas(test_df[['text', 'labels']])
    
    # Apply tokenization
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    
    # Set format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    return train_dataset, test_dataset, tokenizer, label_mapping, id2label, label2id

def compute_metrics_simple(eval_pred):
    """Simple accuracy computation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': accuracy}

def main():
    print("TRAINING")
    print("="*60)
    
    # Configuration
    model_name = "distilbert-base-uncased"
    output_dir = "models/custom_support_model_WORKING"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create clean dataset
    print("\nCreating CLEAN dataset...")
    df = create_clean_dataset()
    print(f"Dataset shape: {df.shape}")
    
    # Prepare data cleanly
    train_dataset, eval_dataset, tokenizer, label_mapping, id2label, label2id = prepare_data_clean(df, model_name)
    
    print(f"\n📊 Final dataset info:")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Test samples: {len(eval_dataset)}")
    print(f"   Categories: {list(label_mapping.keys())}")
    
    # Load model
    print("\nLoading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_mapping),
        id2label=id2label,
        label2id=label2id
    )
    print("Model loaded successfully!")
    
    # Ultra aggressive training arguments
    print("\n Setting up training...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=12,  # More epochs
        per_device_train_batch_size=2,  # Small batch size
        per_device_eval_batch_size=4,
        warmup_steps=200,
        weight_decay=0.01,
        learning_rate=8e-5,  # High learning rate
        logging_dir=f'{output_dir}/logs',
        logging_steps=5,
        save_steps=100,
        eval_steps=100,
        seed=42,
        gradient_accumulation_steps=2,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,  # Avoid multiprocessing issues
    )
    
    print(f"📊 Training configuration:")
    print(f"   Learning Rate: {training_args.learning_rate}")
    print(f"   Epochs: {training_args.num_train_epochs}")
    print(f"   Batch Size: {training_args.per_device_train_batch_size}")
    
    # Initialize trainer
    print("\n Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_simple,
    )
    
    # Start training
    print("\nSTARTING TRAINING...")
    print("="*60)
    
    try:
        trainer.train()
        print("TRAINING COMPLETED!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print(f"Error details: {type(e).__name__}: {str(e)}")
        return
    
    # Save model
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save mappings
    mappings = {
        'label_mapping': label_mapping,
        'id2label': id2label,
        'label2id': label2id,
        'categories': list(label_mapping.keys()),
        'num_labels': len(label_mapping)
    }
    
    with open(f"{output_dir}/label_mapping.json", "w") as f:
        json.dump(mappings, f, indent=2)
    
    print("✅ Model saved successfully!")
    
    # Immediate testing
    print("\n IMMEDIATE COMPREHENSIVE TESTING")
    print("="*50)
    
    try:
        from transformers import pipeline
        
        classifier = pipeline(
            "text-classification",
            model=output_dir,
            tokenizer=output_dir
        )
        
        # Test cases
        test_cases = [
            ("I was charged twice for my subscription", "billing"),
            ("The app crashes when I upload files", "technical_support"),
            ("Your customer service is amazing", "compliment"),
            ("What features are in the premium plan", "product_inquiry"),
            ("I am disappointed with this service", "complaint"),
            ("Can you help with my billing statement", "billing"),
            ("I cannot log into my account", "technical_support"),
            ("This product works perfectly", "compliment"),
            ("Do you offer enterprise solutions", "product_inquiry"),
            ("The quality has declined lately", "complaint")
        ]
        
        print("Testing predictions:")
        print("-" * 60)
        
        correct = 0
        total_conf = 0
        high_conf_count = 0
        
        for i, (text, expected) in enumerate(test_cases, 1):
            result = classifier(text)
            predicted = result[0]['label']
            confidence = result[0]['score']
            
            is_correct = predicted == expected
            status = "✅" if is_correct else "❌"
            
            if is_correct:
                correct += 1
            if confidence > 0.8:
                high_conf_count += 1
            
            total_conf += confidence
            
            print(f"{status} {i:2d}. Expected: {expected:15} | Got: {predicted:15} | Conf: {confidence:.3f}")
        
        accuracy = correct / len(test_cases)
        avg_conf = total_conf / len(test_cases)
        high_conf_ratio = high_conf_count / len(test_cases)
        
        print("-" * 60)
        print(f" RESULTS:")
        print(f"   Accuracy: {accuracy:.1%} ({correct}/{len(test_cases)})")
        print(f"   Avg Confidence: {avg_conf:.3f}")
        print(f"   High Confidence (>0.8): {high_conf_ratio:.1%}")
    except Exception as e:
        print(f"❌ Testing failed: {e}")
    
    print(f"\nWORKING TRAINING COMPLETE!")
    print(f"📁 Model saved to: {output_dir}")

if __name__ == "__main__":
    main()
