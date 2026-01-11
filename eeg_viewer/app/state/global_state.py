"""
Global state classes for EEG Viewer application.

These are the actual state classes used by main.py, moved here for organization.
Each page has its own state class to avoid coupling.
"""


class State:
    """Global state for the main EEG viewer page."""
    def __init__(self):
        # EEG Data
        self.eeg_data = None
        self.eeg_data2 = None  # Second EEG for comparison
        self.compare_mode = False  # Whether comparison mode is active
        
        # View settings
        self.selected_channels = []
        self.view_start = 0.0
        self.view_duration = 5.0
        self.is_playing = False
        self.scale_factor = 1.0
        self.playback_speed = 1.0
        self.playback_reverse = False
        
        # Filter settings
        self.notch_freq = 50.0
        self.notch_enabled = False
        self.bandpass_low = 1.0
        self.bandpass_high = 45.0
        self.bandpass_enabled = False
        
        # Hilbert
        self.hilbert_channel = ""
        self.hilbert_channel2 = ""  # Channel for Hilbert 2
        
        # Epochs
        self.epoch_duration = 2.0
        self.epochs = []
        
        # Current amplitudes
        self.current_amplitudes = {}
        
        # Fixed axis ranges (calculated on load)
        self.fft_y_max = None  # Max Y for FFT plot
        self.fft_y_max2 = None  # Max Y for FFT2 plot
        self.fft_diff_y_max = None  # Max Y for FFT difference plot
        self.hilbert_amp_max = None  # Max amplitude for Hilbert envelope
        self.hilbert_amp_max2 = None  # Max amplitude for Hilbert2 envelope
        self.hilbert_diff_y_max = None  # Max Y for Hilbert difference plot
        
        # UI references
        self.eeg_plot = None
        self.eeg_plot2 = None
        self.fft_plot = None
        self.fft_plot2 = None
        self.fft_diff_plot = None  # FFT difference plot (EEG1 - EEG2)
        self.hilbert_plot = None
        self.hilbert_plot2 = None
        self.hilbert_diff_plot = None  # Hilbert difference plot (EEG1 - EEG2)
        self.hilbert_select_container2 = None  # Channel selector for Hilbert 2
        self.brain_plot = None
        self.brain_plot2 = None
        self.time_label = None
        self.info_container = None
        self.info_container2 = None
        self.channel_container = None
        self.hilbert_select_container = None


class PipelineState:
    """State for the pipeline execution page."""
    def __init__(self):
        self.running = False
        self.current_step = ""
        self.start_time = None
        self.current_process = None
        self.log_container = None
        self.log_scroll = None
        self.status_label = None
        self.progress = 0
        self.refresh_files = None  # Function to refresh file browser
        
        # Persistence for tab switching
        self.log_history = []  # Store log messages for restoration
        self.running_task_name = ""  # Name of currently running task (for global indicator)
        self.selected_run = None  # Persists selected run directory across tab switches
        
        # Pipeline parameters (persist across page navigations)
        self.max_subjects = 0  # 0 = all
        self.max_epochs = 0  # 0 = all
        self.conditions = ["DMT", "EC", "EO"]
        self.jobs = 0  # 0 = auto
        self.workers = 7
        self.bands = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
        self.min_k = 2
        self.max_k = 15
        self.min_comps = 2
        self.max_comps = 10
        self.clustering_search_mode = "Quick"  # 'Quick' or 'Full'
        self.input_dir = None  # Persists input directory path
        self.dataset_metadata = None  # Metadata extracted from input files (sfreq, n_channels, etc.)
        
        # Visualization panel state
        self.viz_panel_height = "300px"  # Persists panel height across step changes
        self.viz_active_step = 1  # Persists active step


class ModelState:
    """State for model training page."""
    def __init__(self):
        self.dataset_path = ""
        self.dataset_info = {}
        self.dataset_type = None  # 'graph', 'image', etc.
        self.model_type = "vae"
        self.training = False
        self.current_process = None
        self.history = {'train_loss': [], 'val_loss': [], 'recon_loss': [], 'kl_loss': [], 'epoch': []}
        self.log_container = None
        self.log_scroll = None
        self.loss_plot = None
        self.config = {}
        
        # Persistence for tab switching
        self.status = 'idle'  # 'idle', 'training', 'completed', 'error'
        self.log_history = []  # Store log messages
        self.running_task_name = ""  # Name of currently running task (for global indicator)
        self.status_indicator = None
        self.update_plots = None
        self.tabs = None
        self.tab_console = None
        
        # UI parameters persistence (survives page navigation)
        self.ui_params = {
            'data_source': 'EEG (channels)',
            'bands': {'Delta': False, 'Theta': False, 'Alpha': True, 'Beta': False, 'Gamma': False},
            'subsample': 0.3,
            # Model type (SimCLR is now default)
            'model_type': 'SimCLR (Graph)',
            # Model architecture
            'latent_dim': 128,  # Embedding dim for SimCLR
            'hidden_dim': 64,
            'gat_layers': 3,
            'attention_heads': 4,
            'dropout': 0.2,
            'attn_dropout': 0.1,
            'decoder_gat_layers': 0,
            'decoder_hidden_dims': '256,128',
            # Training
            'num_epochs': 100,
            'batch_size': 256,
            'learning_rate': '1e-3',
            'weight_decay': '1e-5',
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'patience': 25,
            'grad_clip': 1.0,
            # SimCLR Loss params
            'temperature': 0.5,
            'edge_drop_prob': 0.2,
            'feature_noise_std': 0.1,
            # VAE Loss params
            'kl_weight': 0.01,
            'beta_annealing': True,
            'node_weight': 0.3,
            'edge_weight': 1.0,
            # Workers (low defaults for HDD; increase for SSD)
            'num_workers': 2,
            'dataset_workers': 2,
            # Quick testing
            'max_files': 0,  # 0 = all files
        }
        
        # Polling counters
        self._last_log_count = 0
        self._last_epoch_count = 0


class AnalysisState:
    """State for model analysis page."""
    def __init__(self):
        # Model
        self.model = None
        self.model_path = ""
        self.model_config = {}
        self.model_params = {}
        self.device = 'cpu'
        self.model_type = 'graph'  # 'graph' or 'image'
        
        # Dataset
        self.dataset = None
        self.dataset_path = ""
        self.current_split = 'test'  # train, val, test
        self.current_idx = 0
        self.total_samples = 0
        self.dataset_type = 'graph'  # 'graph' or 'image'
        self.class_names = []  # For image datasets with classes
        
        # Playback
        self.playing = False
        self.speed = 1.0  # samples per second
        self.play_timer = None
        
        # EEG Synchronization (for graph datasets with EEG metadata)
        self.eeg_sync_enabled = True  # Toggle EEG sync
        self.eeg_data = None  # Loaded EEG data (EEGData object)
        self.current_eeg_file = None  # Path to currently loaded EEG
        self.eeg_view_start = 0.0  # Current view position in EEG
        self.epoch_duration = 2.0  # Duration of each epoch in seconds
        self.eeg_channels = []  # Selected channels for visualization
        self.eeg_plot = None  # UI reference for EEG plot
        self.eeg_info_container = None  # UI reference for EEG info
        self.suggested_eeg_path = None  # Auto-suggested EEG path from phases file
        
        # Activations (stored during forward pass)
        self.activations = {}  # layer_name -> tensor
        self.attention_weights = {}  # layer_name -> attention matrix
        self.latent_codes = []  # history of z vectors for PCA
        self.latent_labels = []  # labels for each z
        
        # Current sample info
        self.current_sample = None
        self.current_recon = None
        self.current_z = None
        self.current_label = None  # For image classification
        self.current_kuramoto_comparison = None  # Kuramoto original vs proxy
        
        # SimCLR specific
        self.is_encoder_only = False  # True for SimCLR
        self.current_attention_sync = 0.0  # Mean attention as sync proxy
        self.current_attention_matrix = None  # NxN attention matrix from last GAT layer
        self.current_original_sync = None  # NxN original sync matrix
        self.current_kuramoto_original = 0.0  # Original Kuramoto order parameter
        
        # Fixed axis ranges (computed from dataset)
        self.axis_ranges = {
            'node_features': {'min': -5, 'max': 5},
            'latent': {'x_min': -5, 'x_max': 5, 'y_min': -5, 'y_max': 5},
            'attention': {'min': 0, 'max': 1},
            'activations': {'min': -500, 'max': 500},
            'diff': {'min': 0, 'max': 1},
            'kuramoto': {'min': 0, 'max': 1}
        }
        self.ranges_computed = False
        
        # Kuramoto tracking (for graph models)
        self.kuramoto_history = []  # (idx, kuramoto_mean) for each processed sample
        self.kuramoto_proxy_history = []  # (idx, kuramoto_proxy) from reconstruction
        self.kuramoto_avg = 0.5  # Average across test set
        self.kuramoto_std = 0.1  # Std (metastability proxy)
        self.kuramoto_proxy_avg = 0.5  # Average proxy across test set
        
        # UI references
        self.log_container = None
        self.arch_diagram = None
        self.activation_plots = {}
        self.attention_plots = {}
        self.latent_plot = None
        self.recon_plot = None
        self.progress_slider = None
        self.sample_info_container = None
        # Layer info for image models
        self.layer_info = {'encoder': [], 'decoder': []}


# Global state instances
S = State()
PS = PipelineState()
MS = ModelState()
AS = AnalysisState()

