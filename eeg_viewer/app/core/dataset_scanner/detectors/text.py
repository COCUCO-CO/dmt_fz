"""
Text/NLP dataset detector.

Handles plain text, JSON, JSONL, and other text formats.
"""

from pathlib import Path
from typing import Set, List, Optional, Dict, Any
import json

from .base import BaseDetector, DetectionResult
from ..models import TextSpecificInfo


class TextDetector(BaseDetector):
    """Detector for text/NLP datasets."""
    
    EXTENSIONS: Set[str] = {
        '.txt', '.text',
        '.json', '.jsonl', '.ndjson',
        '.md', '.markdown',
        '.xml',
        '.html', '.htm',
    }
    
    # Sentiment-related class names
    SENTIMENT_CLASSES = {'positive', 'negative', 'neutral', 'pos', 'neg'}
    
    def detect(self, path: Path, analyze_samples: bool = False,
               sample_size: int = 10) -> DetectionResult:
        """Detect text dataset."""
        result = DetectionResult()
        
        if not path.exists():
            result.warnings.append(f"Path does not exist: {path}")
            return result
        
        # Find all text files
        files = self._find_files(path, self.EXTENSIONS)
        
        if not files:
            return result
        
        # Filter out non-text files (binary)
        text_files = []
        for f in files:
            if self._is_text_file(f):
                text_files.append(f)
        
        if not text_files:
            return result
        
        result.is_detected = True
        result.file_count = len(text_files)
        result.sample_files = text_files[:sample_size]
        result.total_size_bytes = self._calculate_total_size(text_files)
        
        extensions = set(f.suffix.lower() for f in text_files)
        result.extensions_found = sorted(extensions)
        
        # Calculate confidence
        if result.file_count >= 100:
            result.confidence = 0.90
        elif result.file_count >= 10:
            result.confidence = 0.80
        else:
            result.confidence = 0.65
        
        # Detect structure
        has_splits, splits, split_counts = self._detect_splits(path)
        result.has_splits = has_splits
        result.splits_found = splits
        result.split_counts = split_counts
        
        has_classes, classes, class_counts = self._detect_classes(path)
        result.has_classes = has_classes
        result.classes_found = classes
        result.class_counts = class_counts
        
        # Check if sentiment dataset
        if classes:
            result.is_sentiment_dataset = bool(
                set(c.lower() for c in classes) & self.SENTIMENT_CLASSES
            )
        
        # Analyze samples
        if analyze_samples:
            result.text_info = self._analyze_text_files(text_files[:sample_size])
            result.text_info.is_sentiment_dataset = result.is_sentiment_dataset
        
        result.suggestions = self._generate_suggestions(result)
        result.suggested_loader = self._suggest_loader(result)
        
        return result
    
    def _is_text_file(self, path: Path) -> bool:
        """Check if file is likely text (not binary)."""
        try:
            with open(path, 'rb') as f:
                chunk = f.read(512)
                # Check for null bytes (binary indicator)
                if b'\x00' in chunk:
                    return False
                # Try to decode as UTF-8
                try:
                    chunk.decode('utf-8')
                    return True
                except UnicodeDecodeError:
                    # Try latin-1 (accepts all bytes)
                    try:
                        chunk.decode('latin-1')
                        # Check if mostly printable
                        printable = sum(1 for b in chunk if 32 <= b < 127 or b in (9, 10, 13))
                        return printable > len(chunk) * 0.7
                    except Exception:
                        return False
        except Exception:
            return False
    
    def _analyze_text_files(self, files: List[Path]) -> TextSpecificInfo:
        """Analyze text files."""
        info = TextSpecificInfo()
        
        total_lines = 0
        total_words = 0
        total_chars = 0
        doc_lengths = []
        all_words = set()
        json_keys = set()
        record_count = 0
        is_json = False
        is_jsonl = False
        is_conversational = False
        
        json_structure = ''
        
        for f in files:
            try:
                suffix = f.suffix.lower()
                
                if suffix in {'.json', '.jsonl', '.ndjson'}:
                    keys, records, convo, struct = self._analyze_json_file(f)
                    json_keys.update(keys)
                    record_count += records
                    is_json = suffix == '.json'
                    is_jsonl = suffix in {'.jsonl', '.ndjson'}
                    is_conversational = is_conversational or convo
                    if struct:
                        json_structure = struct
                else:
                    # Plain text
                    with open(f, 'r', encoding='utf-8', errors='ignore') as file:
                        content = file.read()
                        lines = content.split('\n')
                        words = content.split()
                        
                        total_lines += len(lines)
                        total_words += len(words)
                        total_chars += len(content)
                        doc_lengths.append(len(content))
                        all_words.update(words[:1000])  # Sample words
                        
            except Exception:
                continue
        
        info.document_count = len(files)
        info.total_documents = len(files)
        info.total_lines = total_lines
        info.total_words = total_words
        info.total_characters = total_chars
        
        if doc_lengths:
            info.avg_document_length = sum(doc_lengths) / len(doc_lengths)
        if files:
            info.avg_words_per_document = total_words / len(files)
        
        info.estimated_vocabulary_size = len(all_words)
        info.vocabulary_size = len(all_words)
        
        info.is_json = is_json
        info.is_jsonl = is_jsonl
        info.json_keys = sorted(json_keys)
        info.json_structure = json_structure
        info.record_count = record_count
        info.is_conversational = is_conversational
        
        if is_json:
            info.format = 'json'
        elif is_jsonl:
            info.format = 'jsonl'
        else:
            info.format = 'txt'
        
        return info
    
    def _analyze_json_file(self, path: Path) -> tuple:
        """
        Analyze JSON/JSONL file.
        
        Returns:
            (keys, record_count, is_conversational, json_structure)
        """
        keys = set()
        record_count = 0
        is_conversational = False
        json_structure = ''
        
        try:
            suffix = path.suffix.lower()
            
            if suffix in {'.jsonl', '.ndjson'}:
                # JSONL format
                json_structure = 'records'
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                obj = json.loads(line)
                                if isinstance(obj, dict):
                                    keys.update(obj.keys())
                                    record_count += 1
                                    if 'messages' in obj or 'conversation' in obj:
                                        is_conversational = True
                            except json.JSONDecodeError:
                                continue
            else:
                # Regular JSON
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    json_structure = 'array'
                    record_count = len(data)
                    if data and isinstance(data[0], dict):
                        keys.update(data[0].keys())
                        if 'messages' in data[0] or 'conversation' in data[0]:
                            is_conversational = True
                elif isinstance(data, dict):
                    json_structure = 'object'
                    keys.update(data.keys())
                    record_count = 1
                    if 'messages' in data or 'conversation' in data:
                        is_conversational = True
                        
        except Exception:
            pass
        
        return keys, record_count, is_conversational, json_structure
    
    def _generate_suggestions(self, result: DetectionResult) -> List[str]:
        """Generate suggestions."""
        suggestions = []
        
        if result.text_info:
            if result.text_info.is_json or result.text_info.is_jsonl:
                suggestions.append("Use HuggingFace datasets for structured JSON data")
            
            if result.text_info.is_conversational:
                suggestions.append("Dataset contains conversational data - suitable for chat/dialogue models")
        
        if result.has_classes:
            suggestions.append("Use datasets for class-based text classification")
        
        if result.is_sentiment_dataset:
            suggestions.append("Suitable for sentiment analysis tasks")
        
        suggestions.append("Consider using HuggingFace tokenizers for preprocessing")
        
        return suggestions
    
    def _suggest_loader(self, result: DetectionResult) -> str:
        """Suggest data loader."""
        if result.text_info:
            if result.text_info.is_json or result.text_info.is_jsonl:
                return "datasets.load_dataset('json', data_files=...)"
        
        if result.has_classes:
            return "datasets.load_dataset with data_dir"
        
        return "Custom text loader or HuggingFace datasets"

