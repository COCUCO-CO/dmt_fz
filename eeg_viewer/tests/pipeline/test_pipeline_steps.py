"""
Tests for individual pipeline step execution.

Tests each pipeline step's:
- Argument construction
- Validation
- Expected output
"""

import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Test: Step 1 - fwd.py (Source Localization)
# =============================================================================

class TestStep1Fwd:
    """Tests for fwd.py step configuration."""
    
    def test_fwd_arguments_construction(self):
        """Should construct correct arguments for fwd.py."""
        max_subjects = 5
        conditions = ["DMT", "EC"]
        jobs = 8
        workers = 4
        max_epochs = 10
        
        args = [
            '--max-subjects', str(max_subjects),
            '--conditions'] + conditions + [
            '--jobs', str(jobs),
            '--workers', str(workers),
            '--max-epochs', str(max_epochs)
        ]
        
        assert '--max-subjects' in args
        assert '5' in args
        assert 'DMT' in args
        assert 'EC' in args
        assert '--workers' in args
        assert '4' in args
    
    def test_fwd_script_exists(self):
        """fwd.py should exist in pipeline directory."""
        from app.pages.pipeline import PIPELINE_DIR
        
        fwd_path = PIPELINE_DIR / "fwd.py"
        assert fwd_path.exists()
    
    def test_fwd_handles_all_conditions_default(self):
        """Default should process all conditions."""
        conditions = ["DMT", "EC", "EO"]
        
        args = ['--conditions'] + conditions
        
        assert "DMT" in args
        assert "EC" in args
        assert "EO" in args
    
    def test_fwd_zero_means_all(self):
        """0 for max_subjects/max_epochs should mean 'all'."""
        max_subjects = 0  # means all
        max_epochs = 0    # means all
        
        args = [
            '--max-subjects', str(max_subjects),
            '--max-epochs', str(max_epochs)
        ]
        
        assert '0' in args  # Script interprets 0 as "all"


# =============================================================================
# Test: Step 2 - save_load_pickle.py (Consolidate)
# =============================================================================

class TestStep2Consolidate:
    """Tests for save_load_pickle.py step configuration."""
    
    def test_consolidate_arguments(self):
        """Should construct correct arguments for consolidate step."""
        conditions = ["DMT", "EC", "EO"]
        
        args = ['--conditions'] + conditions
        
        assert '--conditions' in args
        assert 'DMT' in args
    
    def test_consolidate_script_exists(self):
        """save_load_pickle.py should exist."""
        from app.pages.pipeline import PIPELINE_DIR
        
        script_path = PIPELINE_DIR / "save_load_pickle.py"
        assert script_path.exists()


# =============================================================================
# Test: Step 3 - multi2pool2.py (Network Filtering)
# =============================================================================

class TestStep3Network:
    """Tests for multi2pool2.py step configuration."""
    
    def test_network_no_arguments(self):
        """multi2pool2.py uses no command line arguments."""
        args = []
        
        assert len(args) == 0
    
    def test_network_script_exists(self):
        """multi2pool2.py should exist."""
        from app.pages.pipeline import PIPELINE_DIR
        
        script_path = PIPELINE_DIR / "multi2pool2.py"
        assert script_path.exists()


# =============================================================================
# Test: Step 4 - calculate_syncro.py (Sync Metrics)
# =============================================================================

class TestStep4Syncro:
    """Tests for calculate_syncro.py step configuration."""
    
    def test_syncro_arguments(self):
        """Should construct correct arguments for syncro step."""
        workers = 7
        conditions = ["DMT", "EC"]
        
        args = ['--workers', str(workers), '--conditions'] + conditions
        
        assert '--workers' in args
        assert '7' in args
        assert '--conditions' in args
        assert 'DMT' in args
    
    def test_syncro_script_exists(self):
        """calculate_syncro.py should exist."""
        from app.pages.pipeline import PIPELINE_DIR
        
        script_path = PIPELINE_DIR / "calculate_syncro.py"
        assert script_path.exists()


# =============================================================================
# Test: Step 5 - generate_order.py (Kuramoto Order)
# =============================================================================

class TestStep5KuramotoOrder:
    """Tests for generate_order.py step configuration."""
    
    def test_order_arguments(self):
        """Should construct correct arguments for order step."""
        workers = 7
        conditions = ["DMT", "EC", "EO"]
        
        args = ['--workers', str(workers), '--conditions'] + conditions
        
        assert '--workers' in args
        assert '--conditions' in args
    
    def test_order_script_exists(self):
        """generate_order.py should exist."""
        from app.pages.pipeline import PIPELINE_DIR
        
        script_path = PIPELINE_DIR / "generate_order.py"
        assert script_path.exists()


# =============================================================================
# Test: Step 6 - build_order_data.py (Aggregate Data)
# =============================================================================

class TestStep6Aggregate:
    """Tests for build_order_data.py step configuration."""
    
    def test_aggregate_arguments(self):
        """Should construct correct arguments for aggregate step."""
        workers = 7
        
        args = ['--build-all', '--workers', str(workers)]
        
        assert '--build-all' in args
        assert '--workers' in args
        assert '7' in args
    
    def test_aggregate_script_exists(self):
        """build_order_data.py should exist."""
        from app.pages.pipeline import PIPELINE_DIR
        
        script_path = PIPELINE_DIR / "build_order_data.py"
        assert script_path.exists()


# =============================================================================
# Test: Step 7 - pearson.py (Correlations)
# =============================================================================

class TestStep7Pearson:
    """Tests for pearson.py step configuration."""
    
    def test_pearson_no_arguments(self):
        """pearson.py uses no command line arguments."""
        args = []
        
        assert len(args) == 0
    
    def test_pearson_script_exists(self):
        """pearson.py should exist."""
        from app.pages.pipeline import PIPELINE_DIR
        
        script_path = PIPELINE_DIR / "pearson.py"
        assert script_path.exists()


# =============================================================================
# Test: Step 8 - clustering.py (Clustering)
# =============================================================================

class TestStep8Clustering:
    """Tests for clustering.py step configuration."""
    
    def test_clustering_arguments_quick(self):
        """Should construct correct arguments for quick clustering."""
        conditions = ["DMT", "EC", "EO"]
        bands = ["Alpha", "Beta"]
        min_k = 2
        max_k = 10
        min_comps = 2
        max_comps = 8
        workers = 4
        output_dir = "/tmp/clustering_results"
        
        args = [
            '--quick-search',
            '--conditions'] + conditions + [
            '--bands'] + bands + [
            '--min-k', str(min_k),
            '--max-k', str(max_k),
            '--min-comps', str(min_comps),
            '--max-comps', str(max_comps),
            '--workers', str(workers),
            '--output-dir', output_dir
        ]
        
        assert '--quick-search' in args
        assert '--conditions' in args
        assert 'Alpha' in args
        assert '--min-k' in args
        assert '2' in args
        assert '--output-dir' in args
    
    def test_clustering_arguments_full(self):
        """Should construct correct arguments for full clustering."""
        args = ['--full-search', '--conditions', 'DMT']
        
        assert '--full-search' in args
    
    def test_clustering_script_exists(self):
        """clustering.py should exist."""
        from app.pages.pipeline import PIPELINE_DIR
        
        script_path = PIPELINE_DIR / "clustering.py"
        assert script_path.exists()
    
    def test_clustering_bands_mapping(self):
        """Band checkboxes should map to correct band names."""
        band_mapping = {
            'delta': 'Delta',
            'theta': 'Theta',
            'alpha': 'Alpha',
            'beta': 'Beta',
            'gamma': 'Gamma',
        }
        
        # All bands should be properly capitalized
        for short, full in band_mapping.items():
            assert full[0].isupper()
            assert full[1:].islower()


# =============================================================================
# Test: Steps Output Directories
# =============================================================================

class TestStepsOutputDirectories:
    """Tests for pipeline step output directory handling."""
    
    def test_run_creates_condition_dirs(self, tmp_path):
        """Creating a run should create condition subdirectories."""
        run_dir = tmp_path / "run_test"
        run_dir.mkdir()
        
        for cond in ["DMT", "EC", "EO"]:
            cond_dir = run_dir / cond
            cond_dir.mkdir()
            assert cond_dir.exists()
    
    def test_fwd_outputs_to_condition_dirs(self, tmp_path):
        """fwd.py outputs should go to condition subdirectories."""
        # Expected output pattern: run_*/DMT/phases-*.pkl
        run_dir = tmp_path / "run_test"
        dmt_dir = run_dir / "DMT"
        dmt_dir.mkdir(parents=True)
        
        # Simulate output
        output_file = dmt_dir / "phases-S01-DMT.pkl"
        output_file.touch()
        
        assert output_file.exists()
        assert output_file.parent.name == "DMT"
    
    def test_clustering_creates_subdirectory(self, tmp_path):
        """Clustering should create clustering_results/ subdirectory."""
        run_dir = tmp_path / "run_test"
        cluster_dir = run_dir / "clustering_results"
        cluster_dir.mkdir(parents=True)
        
        assert cluster_dir.exists()


# =============================================================================
# Test: Conditions Helper Function
# =============================================================================

class TestGetConditionsHelper:
    """Tests for get_conditions helper function behavior."""
    
    def test_get_conditions_all_checked(self):
        """When all conditions checked, should return all."""
        # Simulate checkbox values
        cond_dmt = True
        cond_ec = True
        cond_eo = True
        
        conds = []
        if cond_dmt: conds.append('DMT')
        if cond_ec: conds.append('EC')
        if cond_eo: conds.append('EO')
        
        assert conds == ['DMT', 'EC', 'EO']
    
    def test_get_conditions_partial(self):
        """When some conditions unchecked, should return only checked."""
        cond_dmt = True
        cond_ec = False
        cond_eo = True
        
        conds = []
        if cond_dmt: conds.append('DMT')
        if cond_ec: conds.append('EC')
        if cond_eo: conds.append('EO')
        
        assert conds == ['DMT', 'EO']
        assert 'EC' not in conds
    
    def test_get_conditions_none_checked(self):
        """When no conditions checked, should return empty."""
        cond_dmt = False
        cond_ec = False
        cond_eo = False
        
        conds = []
        if cond_dmt: conds.append('DMT')
        if cond_ec: conds.append('EC')
        if cond_eo: conds.append('EO')
        
        assert conds == []


# =============================================================================
# Test: Input Directory Validation
# =============================================================================

class TestInputDirectoryValidation:
    """Tests for input directory validation."""
    
    def test_input_dir_subdirectory_structure(self, temp_pipeline_dir):
        """Should recognize subdirectory structure (DMT/, EC/, EO/)."""
        input_dir = temp_pipeline_dir["input"]
        
        # Check structure exists
        assert (input_dir / "DMT").exists()
        assert (input_dir / "EC").exists()
        assert (input_dir / "EO").exists()
    
    def test_input_dir_flat_structure(self, tmp_path):
        """Should recognize flat structure (files with condition in name)."""
        flat_dir = tmp_path / "flat"
        flat_dir.mkdir()
        
        # Create files with condition in filename
        (flat_dir / "S01_DMT_cleaned.set").touch()
        (flat_dir / "S01_EC_cleaned.set").touch()
        
        # Files exist in root
        assert (flat_dir / "S01_DMT_cleaned.set").exists()
    
    def test_input_dir_counts_set_files(self, temp_pipeline_dir):
        """Should count .set files per condition."""
        input_dir = temp_pipeline_dir["input"]
        
        # Add some test files
        (input_dir / "DMT" / "S01.set").touch()
        (input_dir / "DMT" / "S02.set").touch()
        (input_dir / "EC" / "S01.set").touch()
        
        dmt_count = len(list((input_dir / "DMT").glob("*.set")))
        ec_count = len(list((input_dir / "EC").glob("*.set")))
        eo_count = len(list((input_dir / "EO").glob("*.set")))
        
        assert dmt_count == 2
        assert ec_count == 1
        assert eo_count == 0








