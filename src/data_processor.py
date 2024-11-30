"""Data processing utilities for medical datasets"""

import json
from typing import Dict, Any, List
from datasets import Dataset, concatenate_datasets
import logging
from config import DATASET_CONFIGS

logger = logging.getLogger(__name__)

class DatasetPreprocessor:
    """Handles preprocessing for different medical datasets"""
    
    @staticmethod
    def preprocess_medical_qa(example: Dict[str, Any]) -> Dict[str, str]:
        """Preprocess Medical QA format"""
        return {
            "text": (
                "### Instruction: Based on the following symptoms, provide a detailed medical diagnosis.\n\n"
                f"### Input: {example['question']}\n\n"
                f"### Response: {example['answer']}"
            )
        }
    
    @staticmethod
    def preprocess_mimic(example: Dict[str, Any]) -> Dict[str, str]:
        """Preprocess MIMIC-III format"""
        return {
            "text": (
                "### Instruction: Analyze the following clinical case and provide a diagnosis.\n\n"
                f"### Input: Patient History: {example['clinical_history']}\n"
                f"Symptoms: {example['symptoms']}\n\n"
                f"### Response: {example['diagnosis']}\n"
                f"Reasoning: {example['clinical_reasoning']}"
            )
        }
    
    @staticmethod
    def preprocess_healthcaremagic(example: Dict[str, Any]) -> Dict[str, str]:
        """Preprocess HealthCareMagic format"""
        return {
            "text": (
                "### Instruction: Provide a medical diagnosis based on the patient's symptoms.\n\n"
                f"### Input: {example['patient_complaint']}\n\n"
                f"### Response: {example['doctor_response']}"
            )
        }

class DatasetManager:
    """Manages dataset loading, validation, and combination"""
    
    def __init__(self):
        self.preprocessor = DatasetPreprocessor()
    
    def validate_dataset(self, dataset_path: str) -> bool:
        """Validate the format of a dataset"""
        try:
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            if not isinstance(data, (list, dict)):
                raise ValueError(f"Dataset at {dataset_path} must be a JSON array or object")
            return True
        except Exception as e:
            logger.error(f"Error validating dataset {dataset_path}: {str(e)}")
            return False
    
    def load_dataset(self, dataset_name: str, config: Dict[str, Any]) -> Dataset:
        """Load and preprocess a single dataset"""
        if not self.validate_dataset(config['path']):
            raise ValueError(f"Invalid dataset: {dataset_name}")
        
        # Load dataset
        dataset = Dataset.from_json(config['path'])
        
        # Get preprocessing function
        preprocess_fn = getattr(DatasetPreprocessor, config['preprocessing_fn'])
        
        # Preprocess dataset
        processed_dataset = dataset.map(
            preprocess_fn,
            remove_columns=dataset.column_names
        )
        
        return processed_dataset
    
    def combine_datasets(self) -> Dataset:
        """Load and combine all configured datasets"""
        datasets = []
        
        for dataset_name, config in DATASET_CONFIGS.items():
            try:
                processed_dataset = self.load_dataset(dataset_name, config)
                datasets.append(processed_dataset)
                logger.info(f"Successfully processed {dataset_name}")
            except Exception as e:
                logger.error(f"Error processing {dataset_name}: {str(e)}")
        
        if not datasets:
            raise ValueError("No datasets were successfully loaded")
        
        # Combine all datasets
        combined_dataset = concatenate_datasets(datasets)
        logger.info(f"Combined dataset size: {len(combined_dataset)} examples")
        
        return combined_dataset
    
    def get_dataset_statistics(self, dataset: Dataset) -> Dict[str, Any]:
        """Calculate statistics for the dataset"""
        return {
            "total_examples": len(dataset),
            "avg_text_length": sum(len(ex["text"]) for ex in dataset) / len(dataset),
            "max_text_length": max(len(ex["text"]) for ex in dataset),
            "min_text_length": min(len(ex["text"]) for ex in dataset)
        }

def create_data_splits(dataset: Dataset, split_ratio: float = 0.1, seed: int = 42) -> Dict[str, Dataset]:
    """Create train/validation splits"""
    splits = dataset.train_test_split(test_size=split_ratio, seed=seed)
    return {
        "train": splits["train"],
        "validation": splits["test"]
    }