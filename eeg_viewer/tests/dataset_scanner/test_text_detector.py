"""
Tests for TextDetector.

Verifies detection of text/NLP datasets including plain text, JSON, and JSONL.
"""

from pathlib import Path
import json


class TestTextDetectorBasic:
    """Basic text detection tests."""
    
    def test_detects_txt_files(self, text_flat_dataset: Path):
        """Verify .txt files are detected."""
        from app.core.dataset_scanner.detectors import TextDetector
        
        detector = TextDetector()
        result = detector.detect(text_flat_dataset)
        
        assert result.is_detected is True
        assert result.confidence > 0.7
        assert '.txt' in result.extensions_found
    
    def test_counts_files_correctly(self, text_flat_dataset: Path):
        """Verify file count is accurate."""
        from app.core.dataset_scanner.detectors import TextDetector
        
        detector = TextDetector()
        result = detector.detect(text_flat_dataset)
        
        # Created 10 text files
        assert result.file_count == 10
    
    def test_detects_json_files(self, json_dataset: Path):
        """Verify JSON files are detected."""
        from app.core.dataset_scanner.detectors import TextDetector
        
        detector = TextDetector()
        result = detector.detect(json_dataset)
        
        assert result.is_detected is True
        assert '.json' in result.extensions_found
    
    def test_detects_jsonl_files(self, jsonl_dataset: Path):
        """Verify JSONL files are detected."""
        from app.core.dataset_scanner.detectors import TextDetector
        
        detector = TextDetector()
        result = detector.detect(jsonl_dataset)
        
        assert result.is_detected is True
        assert '.jsonl' in result.extensions_found
    
    def test_empty_directory_not_detected(self, empty_dataset: Path):
        """Verify empty directory is not detected as text dataset."""
        from app.core.dataset_scanner.detectors import TextDetector
        
        detector = TextDetector()
        result = detector.detect(empty_dataset)
        
        assert result.is_detected is False


class TestTextDetectorStructures:
    """Test text detection with various structures."""
    
    def test_detects_class_folders(self, text_by_class_dataset: Path):
        """Verify class-based structure is detected."""
        from app.core.dataset_scanner.detectors import TextDetector
        
        detector = TextDetector()
        result = detector.detect(text_by_class_dataset)
        
        assert result.is_detected is True
        assert result.has_classes is True
        assert set(result.classes_found) == {'positive', 'negative', 'neutral'}
    
    def test_counts_per_class(self, text_by_class_dataset: Path):
        """Verify correct count per class."""
        from app.core.dataset_scanner.detectors import TextDetector
        
        detector = TextDetector()
        result = detector.detect(text_by_class_dataset)
        
        # 10 files per class
        assert result.class_counts['positive'] == 10
        assert result.class_counts['negative'] == 10
        assert result.class_counts['neutral'] == 10


class TestTextDetectorMetadata:
    """Test text metadata extraction."""
    
    def test_extracts_document_count(self, text_flat_dataset: Path):
        """Verify document count is extracted."""
        from app.core.dataset_scanner.detectors import TextDetector
        
        detector = TextDetector()
        result = detector.detect(text_flat_dataset, analyze_samples=True)
        
        assert result.text_info is not None
        assert result.text_info.document_count == 10
    
    def test_extracts_average_length(self, text_flat_dataset: Path):
        """Verify average document length is calculated."""
        from app.core.dataset_scanner.detectors import TextDetector
        
        detector = TextDetector()
        result = detector.detect(text_flat_dataset, analyze_samples=True)
        
        assert result.text_info.avg_document_length > 0
    
    def test_extracts_total_lines(self, text_flat_dataset: Path):
        """Verify total lines are counted."""
        from app.core.dataset_scanner.detectors import TextDetector
        
        detector = TextDetector()
        result = detector.detect(text_flat_dataset, analyze_samples=True)
        
        # 10 files × 50 lines = 500 total
        assert result.text_info.total_lines == 500
    
    def test_estimates_vocabulary(self, text_flat_dataset: Path):
        """Verify vocabulary size is estimated."""
        from app.core.dataset_scanner.detectors import TextDetector
        
        detector = TextDetector()
        result = detector.detect(text_flat_dataset, analyze_samples=True)
        
        assert result.text_info.estimated_vocabulary_size > 0


class TestTextDetectorJSONFormat:
    """Test JSON-specific detection."""
    
    def test_analyzes_json_structure(self, json_dataset: Path):
        """Verify JSON structure is analyzed."""
        from app.core.dataset_scanner.detectors import TextDetector
        
        detector = TextDetector()
        result = detector.detect(json_dataset, analyze_samples=True)
        
        assert result.text_info is not None
        assert result.text_info.is_json is True
    
    def test_detects_json_keys(self, json_dataset: Path):
        """Verify JSON keys are detected."""
        from app.core.dataset_scanner.detectors import TextDetector
        
        detector = TextDetector()
        result = detector.detect(json_dataset, analyze_samples=True)
        
        expected_keys = {'id', 'text', 'label', 'score'}
        assert set(result.text_info.json_keys) == expected_keys
    
    def test_detects_json_array_vs_object(self, temp_dir: Path):
        """Verify JSON array structure is detected."""
        from app.core.dataset_scanner.detectors import TextDetector
        
        dataset_path = temp_dir / "json_array"
        dataset_path.mkdir()
        
        data = [{'text': f'item {i}', 'label': i % 3} for i in range(50)]
        with open(dataset_path / "data.json", 'w') as f:
            json.dump(data, f)
        
        detector = TextDetector()
        result = detector.detect(dataset_path, analyze_samples=True)
        
        assert result.text_info.json_structure == 'array'
        assert result.text_info.record_count == 50
    
    def test_analyzes_jsonl_structure(self, jsonl_dataset: Path):
        """Verify JSONL structure is analyzed."""
        from app.core.dataset_scanner.detectors import TextDetector
        
        detector = TextDetector()
        result = detector.detect(jsonl_dataset, analyze_samples=True)
        
        assert result.text_info.is_jsonl is True
        assert result.text_info.record_count == 100  # Created 100 records


class TestTextDetectorFormats:
    """Test different text formats."""
    
    def test_detects_markdown(self, temp_dir: Path):
        """Verify markdown files are detected."""
        from app.core.dataset_scanner.detectors import TextDetector
        
        dataset_path = temp_dir / "markdown"
        dataset_path.mkdir()
        
        for i in range(5):
            with open(dataset_path / f"doc_{i}.md", 'w') as f:
                f.write(f"# Heading {i}\n\nSome **bold** text.\n")
        
        detector = TextDetector()
        result = detector.detect(dataset_path)
        
        assert result.is_detected is True
        assert '.md' in result.extensions_found
    
    def test_detects_xml(self, temp_dir: Path):
        """Verify XML files are detected."""
        from app.core.dataset_scanner.detectors import TextDetector
        
        dataset_path = temp_dir / "xml_data"
        dataset_path.mkdir()
        
        for i in range(3):
            with open(dataset_path / f"data_{i}.xml", 'w') as f:
                f.write(f'<?xml version="1.0"?>\n<root><item id="{i}">Content</item></root>')
        
        detector = TextDetector()
        result = detector.detect(dataset_path)
        
        assert result.is_detected is True
        assert '.xml' in result.extensions_found


class TestTextDetectorEdgeCases:
    """Test edge cases for text detection."""
    
    def test_handles_empty_files(self, temp_dir: Path):
        """Verify empty text files are handled."""
        from app.core.dataset_scanner.detectors import TextDetector
        
        dataset_path = temp_dir / "empty_texts"
        dataset_path.mkdir()
        
        (dataset_path / "empty.txt").touch()
        
        detector = TextDetector()
        result = detector.detect(dataset_path, analyze_samples=True)
        
        # Should handle without crashing
        assert result.is_detected is True or result.file_count >= 0
    
    def test_handles_binary_files(self, temp_dir: Path):
        """Verify binary files are not mistaken for text."""
        from app.core.dataset_scanner.detectors import TextDetector
        
        dataset_path = temp_dir / "binary"
        dataset_path.mkdir()
        
        # Create binary file with .txt extension
        with open(dataset_path / "binary.txt", 'wb') as f:
            f.write(bytes(range(256)))
        
        detector = TextDetector()
        result = detector.detect(dataset_path, analyze_samples=True)
        
        # Should either not detect or report issues
        assert result.confidence < 0.8 or len(result.warnings) > 0 or not result.is_detected
    
    def test_handles_different_encodings(self, temp_dir: Path):
        """Verify different text encodings are handled."""
        from app.core.dataset_scanner.detectors import TextDetector
        
        dataset_path = temp_dir / "encodings"
        dataset_path.mkdir()
        
        # UTF-8
        with open(dataset_path / "utf8.txt", 'w', encoding='utf-8') as f:
            f.write("Hello, 世界! 🌍\n")
        
        # Latin-1
        with open(dataset_path / "latin1.txt", 'w', encoding='latin-1') as f:
            f.write("Café résumé\n")
        
        detector = TextDetector()
        result = detector.detect(dataset_path, analyze_samples=True)
        
        assert result.is_detected is True
    
    def test_handles_very_long_lines(self, temp_dir: Path):
        """Verify files with very long lines are handled."""
        from app.core.dataset_scanner.detectors import TextDetector
        
        dataset_path = temp_dir / "long_lines"
        dataset_path.mkdir()
        
        with open(dataset_path / "long.txt", 'w') as f:
            f.write("x" * 100000 + "\n")
        
        detector = TextDetector()
        result = detector.detect(dataset_path, analyze_samples=True)
        
        assert result.is_detected is True
    
    def test_handles_malformed_json(self, temp_dir: Path):
        """Verify malformed JSON is handled gracefully."""
        from app.core.dataset_scanner.detectors import TextDetector
        
        dataset_path = temp_dir / "bad_json"
        dataset_path.mkdir()
        
        with open(dataset_path / "bad.json", 'w') as f:
            f.write('{"incomplete": "json",')
        
        detector = TextDetector()
        result = detector.detect(dataset_path, analyze_samples=True)
        
        # Should not crash
        assert True


class TestTextDetectorNLP:
    """Test NLP-specific features."""
    
    def test_detects_sentiment_labels(self, text_by_class_dataset: Path):
        """Verify sentiment-like labels are detected."""
        from app.core.dataset_scanner.detectors import TextDetector
        
        detector = TextDetector()
        result = detector.detect(text_by_class_dataset)
        
        # Classes are 'positive', 'negative', 'neutral' - sentiment indicators
        assert result.is_sentiment_dataset is True or \
               set(result.classes_found) == {'positive', 'negative', 'neutral'}
    
    def test_detects_conversational_format(self, temp_dir: Path):
        """Verify conversational format is detected."""
        from app.core.dataset_scanner.detectors import TextDetector
        
        dataset_path = temp_dir / "conversational"
        dataset_path.mkdir()
        
        conversations = [
            {
                'messages': [
                    {'role': 'user', 'content': 'Hello'},
                    {'role': 'assistant', 'content': 'Hi there!'}
                ]
            }
            for _ in range(10)
        ]
        with open(dataset_path / "convos.jsonl", 'w') as f:
            for conv in conversations:
                f.write(json.dumps(conv) + '\n')
        
        detector = TextDetector()
        result = detector.detect(dataset_path, analyze_samples=True)
        
        assert result.text_info.is_conversational is True or \
               'messages' in str(result.text_info.json_keys)
    
    def test_estimates_tokenization_needs(self, text_flat_dataset: Path):
        """Verify tokenization recommendations are provided."""
        from app.core.dataset_scanner.detectors import TextDetector
        
        detector = TextDetector()
        result = detector.detect(text_flat_dataset, analyze_samples=True)
        
        # Should provide some tokenization info
        assert result.text_info is not None
        assert result.text_info.avg_words_per_document > 0 or \
               result.text_info.avg_document_length > 0


class TestTextDetectorStatistics:
    """Test statistical analysis of text datasets."""
    
    def test_calculates_word_statistics(self, text_flat_dataset: Path):
        """Verify word statistics are calculated."""
        from app.core.dataset_scanner.detectors import TextDetector
        
        detector = TextDetector()
        result = detector.detect(text_flat_dataset, analyze_samples=True)
        
        assert result.text_info.total_words > 0
        assert result.text_info.avg_words_per_document > 0
    
    def test_calculates_character_statistics(self, text_flat_dataset: Path):
        """Verify character statistics are calculated."""
        from app.core.dataset_scanner.detectors import TextDetector
        
        detector = TextDetector()
        result = detector.detect(text_flat_dataset, analyze_samples=True)
        
        assert result.text_info.total_characters > 0
    
    def test_detects_language(self, temp_dir: Path):
        """Verify language detection works (if enabled)."""
        from app.core.dataset_scanner.detectors import TextDetector
        
        dataset_path = temp_dir / "english_text"
        dataset_path.mkdir()
        
        with open(dataset_path / "english.txt", 'w') as f:
            f.write("This is a sample English text document.\n" * 10)
        
        detector = TextDetector()
        result = detector.detect(dataset_path, analyze_samples=True)
        
        # Language detection may not be implemented, but should not crash
        assert result.is_detected is True


class TestTextDetectorOutput:
    """Test output format and completeness."""
    
    def test_returns_detection_result(self, text_flat_dataset: Path):
        """Verify detection result has all required fields."""
        from app.core.dataset_scanner.detectors import TextDetector
        
        detector = TextDetector()
        result = detector.detect(text_flat_dataset)
        
        assert hasattr(result, 'is_detected')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'file_count')
        assert hasattr(result, 'extensions_found')
    
    def test_to_dict_conversion(self, text_flat_dataset: Path):
        """Verify result can be converted to dict."""
        from app.core.dataset_scanner.detectors import TextDetector
        
        detector = TextDetector()
        result = detector.detect(text_flat_dataset)
        
        d = result.to_dict()
        
        assert isinstance(d, dict)
        assert 'is_detected' in d
    
    def test_generates_suggestions(self, text_flat_dataset: Path):
        """Verify detector generates useful suggestions."""
        from app.core.dataset_scanner.detectors import TextDetector
        
        detector = TextDetector()
        result = detector.detect(text_flat_dataset)
        
        assert len(result.suggestions) > 0 or result.suggested_loader is not None








