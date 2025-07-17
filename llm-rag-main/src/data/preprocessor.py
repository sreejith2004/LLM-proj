"""
Text preprocessing utilities for legal documents.
"""

import re
import string
import nltk
from typing import List, Dict, Optional
import logging
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logger = logging.getLogger(__name__)

class LegalTextPreprocessor:
    """
    Text preprocessing utilities specifically designed for legal documents.
    """
    
    def __init__(self, remove_stopwords: bool = False, lowercase: bool = False):
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        
        # Legal-specific patterns
        self.legal_patterns = {
            'case_citations': r'\b\d{4}\s+\w+\s+\d+\b',  # e.g., "2023 SC 123"
            'section_references': r'[Ss]ection\s+\d+(?:\(\d+\))?(?:\([a-z]\))?',  # e.g., "Section 9(1)(iv)"
            'act_references': r'\b\w+\s+Act,?\s+\d{4}\b',  # e.g., "Income Tax Act, 1961"
            'court_names': r'\b(?:Supreme Court|High Court|District Court|Tribunal)\b',
            'judge_names': r'\b[A-Z][a-z]+\s+J\.?\b',  # e.g., "Mahajan J."
            'legal_terms': r'\b(?:appellant|respondent|petitioner|defendant|plaintiff)\b'
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean legal text while preserving important legal formatting.
        """
        if not text:
            return ""
        
        # Remove excessive whitespace but preserve paragraph breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove page numbers and line numbers at the beginning of lines
        text = re.sub(r'^\s*\d+\s+', '', text, flags=re.MULTILINE)
        
        # Clean up common OCR errors in legal documents
        text = re.sub(r'\b([a-z])([A-Z])', r'\1 \2', text)  # Fix missing spaces
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)  # Ensure space after punctuation
        
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r'['']', "'", text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_legal_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract legal entities like case citations, section references, etc.
        """
        entities = {}
        
        for entity_type, pattern in self.legal_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities[entity_type] = list(set(matches))  # Remove duplicates
        
        return entities
    
    def segment_judgment(self, text: str) -> Dict[str, str]:
        """
        Segment legal judgment into different sections.
        """
        segments = {
            'header': '',
            'facts': '',
            'arguments': '',
            'reasoning': '',
            'conclusion': '',
            'full_text': text
        }
        
        # Simple heuristic-based segmentation
        lines = text.split('\n')
        current_section = 'header'
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect section transitions based on common legal document patterns
            line_lower = line.lower()
            
            if any(keyword in line_lower for keyword in ['facts', 'background', 'case history']):
                current_section = 'facts'
            elif any(keyword in line_lower for keyword in ['arguments', 'submissions', 'contentions']):
                current_section = 'arguments'
            elif any(keyword in line_lower for keyword in ['reasoning', 'analysis', 'discussion', 'held']):
                current_section = 'reasoning'
            elif any(keyword in line_lower for keyword in ['conclusion', 'order', 'judgment', 'disposed']):
                current_section = 'conclusion'
            
            segments[current_section] += line + ' '
        
        # Clean up segments
        for key in segments:
            if key != 'full_text':
                segments[key] = segments[key].strip()
        
        return segments
    
    def tokenize_for_training(self, text: str, max_length: int = 1024) -> List[str]:
        """
        Tokenize text into chunks suitable for training.
        """
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Rough word count estimation
            if len((current_chunk + " " + sentence).split()) <= max_length:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def preprocess_for_summarization(self, judgment: str, summary: str) -> Dict[str, str]:
        """
        Preprocess judgment-summary pairs for fine-tuning.
        """
        # Clean both texts
        clean_judgment = self.clean_text(judgment)
        clean_summary = self.clean_text(summary)
        
        # Apply case conversion if specified
        if self.lowercase:
            clean_judgment = clean_judgment.lower()
            clean_summary = clean_summary.lower()
        
        return {
            'input': clean_judgment,
            'target': clean_summary,
            'input_length': len(clean_judgment.split()),
            'target_length': len(clean_summary.split())
        }
    
    def create_training_prompt(self, judgment: str, summary: str = None) -> str:
        """
        Create training prompt in a consistent format.
        """
        if summary:
            # Training format
            prompt = f"""### Legal Document Summarization

**Instruction:** Summarize the following legal judgment concisely, highlighting the key facts, legal issues, reasoning, and conclusion.

**Judgment:**
{judgment}

**Summary:**
{summary}"""
        else:
            # Inference format
            prompt = f"""### Legal Document Summarization

**Instruction:** Summarize the following legal judgment concisely, highlighting the key facts, legal issues, reasoning, and conclusion.

**Judgment:**
{judgment}

**Summary:**"""
        
        return prompt
    
    def validate_text_quality(self, text: str) -> Dict[str, bool]:
        """
        Validate text quality for training.
        """
        checks = {
            'not_empty': len(text.strip()) > 0,
            'reasonable_length': 10 <= len(text.split()) <= 5000,
            'has_legal_content': any(pattern in text.lower() for pattern in 
                                   ['court', 'judge', 'case', 'law', 'section', 'act']),
            'proper_encoding': text.isascii() or len(text.encode('utf-8')) == len(text.encode('utf-8', errors='ignore'))
        }
        
        return checks

def main():
    """Example usage of the preprocessor."""
    preprocessor = LegalTextPreprocessor()
    
    # Example text
    sample_text = """
    Appeal No. LXVI of 1949.
    Appeal from the High Court of judicature, Bombay, in a reference under section 66 of the Indian Income tax Act, 1022.
    The judgment of the Court was delivered by MEHR CHAND MAHAJAN J.
    This is an appeal against a judgment of the High Court of Judicature at Bombay in an income tax matter.
    """
    
    # Clean text
    cleaned = preprocessor.clean_text(sample_text)
    print("Cleaned text:", cleaned[:200] + "...")
    
    # Extract entities
    entities = preprocessor.extract_legal_entities(sample_text)
    print("Legal entities:", entities)
    
    # Segment judgment
    segments = preprocessor.segment_judgment(sample_text)
    print("Segments:", list(segments.keys()))

if __name__ == "__main__":
    main()
