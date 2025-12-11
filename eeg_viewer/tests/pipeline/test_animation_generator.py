"""
Tests for Animation Generator component.

Tests the folder/subject detection functionality.
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestAnimationGeneratorFolderDetection:
    """Test folder and subject detection in animation generator."""
    
    @pytest.fixture
    def sample_data_dir(self, tmp_path):
        """Create a sample data directory structure."""
        # Create subfolders like DMT, EC, EO
        for condition in ['DMT', 'EC', 'EO']:
            cond_dir = tmp_path / condition
            cond_dir.mkdir()
            
            # Create sample pkl files
            for subj_num in range(1, 6):
                subj_id = f'S{subj_num:02d}'
                (cond_dir / f'phases-{subj_id}-{condition}.pkl').touch()
                (cond_dir / f'syncro-{subj_id}-{condition}.pkl').touch()
                (cond_dir / f'order-{subj_id}-{condition}.pkl').touch()
        
        return tmp_path
    
    @pytest.fixture
    def real_data_dir(self):
        """Use real data directory if available."""
        real_path = Path('/media/storage_hdd/dmt_fz/fwd-inv-stc')
        if real_path.exists():
            return real_path
        pytest.skip("Real data directory not available")
    
    def test_list_subfolders_sample(self, sample_data_dir):
        """Test listing subfolders from sample data."""
        subfolders = []
        for item in sorted(sample_data_dir.iterdir()):
            if item.is_dir() and not item.name.startswith('.'):
                subfolders.append(item.name)
        
        assert len(subfolders) == 3
        assert 'DMT' in subfolders
        assert 'EC' in subfolders
        assert 'EO' in subfolders
    
    def test_list_subfolders_real(self, real_data_dir):
        """Test listing subfolders from real data."""
        subfolders = []
        for item in sorted(real_data_dir.iterdir()):
            if item.is_dir() and not item.name.startswith('.'):
                subfolders.append(item.name)
        
        assert len(subfolders) >= 3
        assert 'DMT' in subfolders
        assert 'EC' in subfolders
        assert 'EO' in subfolders
    
    def test_extract_subjects_from_pkl_files(self, sample_data_dir):
        """Test extracting subject IDs from pkl filenames."""
        folder = sample_data_dir / 'DMT'
        subjects = set()
        
        for f in folder.glob('*.pkl'):
            name = f.stem
            for prefix in ['phases-', 'syncro-', 'order_all-', 'order-']:
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    break
            if name:
                subjects.add(name)
        
        subjects = sorted(subjects)
        
        assert len(subjects) == 5
        assert 'S01-DMT' in subjects
        assert 'S05-DMT' in subjects
    
    def test_extract_subjects_real_data(self, real_data_dir):
        """Test extracting subjects from real DMT folder."""
        folder = real_data_dir / 'DMT'
        subjects = set()
        
        pkl_files = list(folder.glob('*.pkl'))
        assert len(pkl_files) > 0, "Should have pkl files"
        
        for f in pkl_files:
            name = f.stem
            for prefix in ['phases-', 'syncro-', 'order_all-', 'order-']:
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    break
            
            # Exclude non-subject files
            # Exclude non-subject files (including subject_phases_*)
            if name and not name.startswith('subject_phases') and name not in [
                'con_mat', 'eigen_all', 'eigen_correct', 'kuramoto_all', 
                'extra', 'extras', 'clusters_splits0'
            ]:
                subjects.add(name)
        
        subjects = sorted(subjects)
        
        assert len(subjects) > 20, f"Expected >20 subjects, got {len(subjects)}"
        # Check format S##-DMT
        for s in subjects:
            assert '-DMT' in s, f"Subject {s} should have -DMT suffix"
    
    def test_all_conditions_have_subjects(self, real_data_dir):
        """Test that all conditions (DMT, EC, EO) have subjects."""
        for condition in ['DMT', 'EC', 'EO']:
            folder = real_data_dir / condition
            assert folder.exists(), f"Folder {condition} should exist"
            
            pkl_files = list(folder.glob('*.pkl'))
            assert len(pkl_files) > 0, f"{condition} should have pkl files"
            
            # Count subjects
            subjects = set()
            for f in pkl_files:
                name = f.stem
                for prefix in ['phases-', 'syncro-', 'order_all-', 'order-']:
                    if name.startswith(prefix):
                        name = name[len(prefix):]
                        break
                if name and not name.startswith(('con_', 'eigen', 'kura', 'extra', 'cluster', 'subject')):
                    subjects.add(name)
            
            assert len(subjects) > 0, f"{condition} should have subjects"


class TestAnimationGeneratorPathHandling:
    """Test path handling in animation generator."""
    
    def test_path_conversion_string_to_path(self):
        """Test that string paths are converted to Path objects."""
        path_str = '/media/storage_hdd/dmt_fz/fwd-inv-stc'
        path_obj = Path(path_str)
        
        assert isinstance(path_obj, Path)
        assert str(path_obj) == path_str
    
    def test_path_exists_check(self):
        """Test path existence checking."""
        valid_path = Path('/media/storage_hdd/dmt_fz/fwd-inv-stc')
        invalid_path = Path('/nonexistent/path/12345')
        
        # Only check if the valid path actually exists on this system
        if valid_path.exists():
            assert valid_path.is_dir()
        
        assert not invalid_path.exists()
    
    def test_glob_pkl_files(self, tmp_path):
        """Test globbing pkl files."""
        # Create test files
        (tmp_path / 'file1.pkl').touch()
        (tmp_path / 'file2.pkl').touch()
        (tmp_path / 'file3.txt').touch()
        
        pkl_files = list(tmp_path.glob('*.pkl'))
        
        assert len(pkl_files) == 2
        assert all(f.suffix == '.pkl' for f in pkl_files)


class TestSubjectParsing:
    """Test subject ID parsing from filenames."""
    
    @pytest.mark.parametrize("filename,expected", [
        ("phases-S01-DMT.pkl", "S01-DMT"),
        ("syncro-S01-DMT.pkl", "S01-DMT"),
        ("order-S01-DMT.pkl", "S01-DMT"),
        ("order_all-S01-DMT.pkl", "S01-DMT"),
        ("phases-S35-EC.pkl", "S35-EC"),
        ("syncro-S10-EO.pkl", "S10-EO"),
    ])
    def test_subject_extraction(self, filename, expected):
        """Test extracting subject ID from various filename formats."""
        name = Path(filename).stem
        
        for prefix in ['phases-', 'syncro-', 'order_all-', 'order-']:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        
        assert name == expected
    
    @pytest.mark.parametrize("filename", [
        "con_mat.pkl",
        "eigen_all.pkl",
        "eigen_correct.pkl",
        "kuramoto_all.pkl",
        "extra.pkl",
        "extras.pickle",
        "clusters_splits0.pkl",
    ])
    def test_non_subject_files_excluded(self, filename):
        """Test that non-subject files are properly identified."""
        name = Path(filename).stem
        
        excluded = ['con_mat', 'eigen_all', 'eigen_correct', 'kuramoto_all', 
                    'extra', 'extras', 'clusters_splits0', 'subject_phases']
        
        assert name in excluded or name.startswith('eigen') or name.startswith('kura')

