"""
Data loading and preprocessing utilities for legal documents.
"""

import os
import json
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LegalDocument:
    """Data class for legal documents."""
    id: str
    judgment: str
    summary: str
    jurisdiction: str  # IN, UK
    case_type: str     # Abs, Ext
    split: str         # train, test

class LegalDataLoader:
    """
    Data loader for legal documents with judgment-summary pairs.
    """
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.documents: List[LegalDocument] = []
        
    def load_all_documents(self) -> List[LegalDocument]:
        """Load all legal documents from the dataset."""
        logger.info(f"Loading documents from {self.dataset_path}")
        
        # Load Indian Abstract documents
        self._load_jurisdiction_documents("IN-Abs", "IN", "Abs")
        
        # Load Indian Extended documents  
        self._load_jurisdiction_documents("IN-Ext", "IN", "Ext")
        
        # Load UK Abstract documents
        self._load_jurisdiction_documents("UK-Abs", "UK", "Abs")
        
        logger.info(f"Loaded {len(self.documents)} total documents")
        return self.documents
    
    def _load_jurisdiction_documents(self, folder: str, jurisdiction: str, case_type: str):
        """Load documents from a specific jurisdiction folder."""
        folder_path = self.dataset_path / folder
        
        if not folder_path.exists():
            logger.warning(f"Folder {folder_path} does not exist")
            return
            
        # Load training documents
        train_path = folder_path / "train-data"
        if train_path.exists():
            self._load_split_documents(train_path, jurisdiction, case_type, "train")
            
        # Load test documents
        test_path = folder_path / "test-data"
        if test_path.exists():
            self._load_split_documents(test_path, jurisdiction, case_type, "test")
    
    def _load_split_documents(self, split_path: Path, jurisdiction: str, case_type: str, split: str):
        """Load documents from train/test split."""
        judgment_path = split_path / "judgement"
        summary_path = split_path / "summary"
        
        if not (judgment_path.exists() and summary_path.exists()):
            logger.warning(f"Missing judgment or summary folder in {split_path}")
            return
            
        # Get all judgment files
        judgment_files = list(judgment_path.glob("*.txt"))
        
        for judgment_file in judgment_files:
            doc_id = judgment_file.stem
            summary_file = summary_path / f"{doc_id}.txt"
            
            if summary_file.exists():
                try:
                    # Read judgment text
                    with open(judgment_file, 'r', encoding='utf-8') as f:
                        judgment_text = f.read().strip()
                    
                    # Read summary text
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        summary_text = f.read().strip()
                    
                    # Create document object
                    doc = LegalDocument(
                        id=f"{jurisdiction}_{case_type}_{split}_{doc_id}",
                        judgment=judgment_text,
                        summary=summary_text,
                        jurisdiction=jurisdiction,
                        case_type=case_type,
                        split=split
                    )
                    
                    self.documents.append(doc)
                    
                except Exception as e:
                    logger.error(f"Error loading document {doc_id}: {e}")
    
    def get_train_documents(self) -> List[LegalDocument]:
        """Get training documents."""
        return [doc for doc in self.documents if doc.split == "train"]
    
    def get_test_documents(self) -> List[LegalDocument]:
        """Get test documents."""
        return [doc for doc in self.documents if doc.split == "test"]
    
    def get_documents_by_jurisdiction(self, jurisdiction: str) -> List[LegalDocument]:
        """Get documents by jurisdiction (IN or UK)."""
        return [doc for doc in self.documents if doc.jurisdiction == jurisdiction]
    
    def save_processed_data(self, output_path: str):
        """Save processed data in JSON format for training."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare training data
        train_data = []
        test_data = []
        
        for doc in self.documents:
            data_point = {
                "id": doc.id,
                "input": doc.judgment,
                "target": doc.summary,
                "jurisdiction": doc.jurisdiction,
                "case_type": doc.case_type
            }
            
            if doc.split == "train":
                train_data.append(data_point)
            else:
                test_data.append(data_point)
        
        # Save training data
        with open(output_path / "train.json", 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        
        # Save test data
        with open(output_path / "test.json", 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        
        # Create validation split from training data (80-20 split)
        split_idx = int(0.8 * len(train_data))
        train_split = train_data[:split_idx]
        val_split = train_data[split_idx:]
        
        # Save splits
        with open(output_path / "train_split.json", 'w', encoding='utf-8') as f:
            json.dump(train_split, f, indent=2, ensure_ascii=False)
            
        with open(output_path / "validation.json", 'w', encoding='utf-8') as f:
            json.dump(val_split, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved processed data to {output_path}")
        logger.info(f"Train: {len(train_split)}, Validation: {len(val_split)}, Test: {len(test_data)}")
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            "total_documents": len(self.documents),
            "train_documents": len(self.get_train_documents()),
            "test_documents": len(self.get_test_documents()),
            "jurisdictions": {},
            "case_types": {},
            "avg_judgment_length": 0,
            "avg_summary_length": 0
        }
        
        # Calculate jurisdiction and case type distributions
        for doc in self.documents:
            stats["jurisdictions"][doc.jurisdiction] = stats["jurisdictions"].get(doc.jurisdiction, 0) + 1
            stats["case_types"][doc.case_type] = stats["case_types"].get(doc.case_type, 0) + 1
        
        # Calculate average lengths
        if self.documents:
            total_judgment_len = sum(len(doc.judgment.split()) for doc in self.documents)
            total_summary_len = sum(len(doc.summary.split()) for doc in self.documents)
            
            stats["avg_judgment_length"] = total_judgment_len / len(self.documents)
            stats["avg_summary_length"] = total_summary_len / len(self.documents)
        
        return stats

def main():
    """Example usage of the data loader."""
    loader = LegalDataLoader("dataset")
    documents = loader.load_all_documents()
    
    # Print statistics
    stats = loader.get_statistics()
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Save processed data
    loader.save_processed_data("data/processed")

if __name__ == "__main__":
    main()
