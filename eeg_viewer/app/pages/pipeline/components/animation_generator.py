"""
Animation Generator Component.

Generates Kuramoto visualization frames and videos.
"""
import asyncio
import os
from pathlib import Path
from typing import Callable, Optional
from nicegui import ui

from config import THEME_PRIMARY, THEME_SECONDARY, THEME_TEXT_DIM, THEME_ERROR, THEME_WARN


class AnimationGenerator:
    """
    Generator for Kuramoto visualization animations.
    
    Features:
    - Generate PNG frames for different modes (stc, eeg, all)
    - Create MP4 videos from frames
    - Progress logging
    """
    
    def __init__(self, get_run_dir: Callable[[], Optional[Path]]):
        """
        Initialize generator.
        
        Args:
            get_run_dir: Function that returns current run directory
        """
        self._get_run_dir = get_run_dir
        
        # Settings
        self._mode = 'stc'
        self._quality = 'medium'
        self._format = 'png'
        self._start_epoch = 0
        self._end_epoch = 10
        self._fps = 5
        self._subject = None
        self._band = 'Alpha'
        self._subfolder = None  # For nested folder structure
        self._custom_path = None  # Custom path override
        
        # UI refs
        self._log_container = None
        self._subject_select = None
        self._subfolder_select = None
        self._path_input = None
    
    def render(self) -> None:
        """Render the animation generator UI."""
        with ui.expansion('🎬 ANIMACIONES', icon='movie').classes('w-full').style(
            'background: #1a1a1a;'
        ):
            with ui.column().classes('w-full p-3 gap-3'):
                ui.label('Genera frames y videos de Kuramoto').style(
                    f'color: {THEME_TEXT_DIM}; font-size: 0.75rem;'
                )
                
                # Path input row
                with ui.row().classes('w-full gap-2 items-center'):
                    ui.label('Directorio:').style(f'color: {THEME_TEXT_DIM}; font-size: 0.75rem;')
                    self._path_input = ui.input(
                        value='/media/storage_hdd/dmt_fz/fwd-inv-stc',
                        placeholder='Path al directorio de datos'
                    ).props('dense dark').classes('flex-1')
                    ui.button('Cargar', on_click=self._load_subjects, icon='folder_open').props('dense')
                
                # Settings row 1 - Folder and subject selection
                with ui.row().classes('gap-3 items-center flex-wrap'):
                    # Subfolder selector (for nested structures like DMT/, EC/, EO/)
                    self._subfolder_select = ui.select(
                        [], label='Carpeta'
                    ).props('dense dark').classes('w-24').on(
                        'update:model-value',
                        lambda e: self._on_subfolder_change(e.args)
                    )
                    
                    self._subject_select = ui.select(
                        [], label='Subject'
                    ).props('dense dark').classes('w-28')
                    
                    ui.select(
                        ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'],
                        value=self._band, label='Banda'
                    ).props('dense dark').classes('w-24').on(
                        'update:model-value',
                        lambda e: setattr(self, '_band', e.args)
                    )
                    
                    ui.select(
                        ['stc', 'eeg', 'all', 'advanced'],
                        value=self._mode, label='Mode'
                    ).props('dense dark').classes('w-24').on(
                        'update:model-value',
                        lambda e: setattr(self, '_mode', e.args)
                    )
                    
                    ui.select(
                        ['high', 'medium', 'low'],
                        value=self._quality, label='Calidad'
                    ).props('dense dark').classes('w-24').on(
                        'update:model-value',
                        lambda e: setattr(self, '_quality', e.args)
                    )
                
                # Settings row 2
                with ui.row().classes('gap-3 items-center flex-wrap'):
                    ui.number(
                        value=self._start_epoch, min=0, max=200, label='Start'
                    ).props('dense').classes('w-20').on(
                        'update:model-value',
                        lambda e: setattr(self, '_start_epoch', int(e.args or 0))
                    )
                    
                    ui.number(
                        value=self._end_epoch, min=1, max=200, label='End'
                    ).props('dense').classes('w-20').on(
                        'update:model-value',
                        lambda e: setattr(self, '_end_epoch', int(e.args or 10))
                    )
                    
                    ui.number(
                        value=self._fps, min=1, max=30, label='FPS'
                    ).props('dense').classes('w-16').on(
                        'update:model-value',
                        lambda e: setattr(self, '_fps', int(e.args or 5))
                    )
                
                # Log area
                self._log_container = ui.column().classes('w-full').style(
                    'max-height: 200px; overflow-y: auto; background: #050505; '
                    'border-radius: 4px; padding: 8px;'
                )
                
                # Action buttons
                with ui.row().classes('gap-2'):
                    ui.button(
                        'Generar Frames', 
                        on_click=self._generate_frames,
                        icon='photo_library'
                    ).props('dense').style(f'background: {THEME_PRIMARY}; color: black;')
                    
                    ui.button(
                        'Crear Video',
                        on_click=self._generate_video,
                        icon='movie'
                    ).props('dense').style(f'background: {THEME_SECONDARY}; color: black;')
                    
                    ui.button(
                        'Clear Log',
                        on_click=lambda: self._log_container.clear() if self._log_container else None,
                        icon='delete'
                    ).props('dense flat')
    
    def _log(self, message: str, color: str = None) -> None:
        """Add message to log."""
        if not self._log_container:
            return
        with self._log_container:
            ui.label(message).style(
                f'color: {color or THEME_TEXT_DIM}; font-family: JetBrains Mono; '
                f'font-size: 0.7rem;'
            )
    
    def _load_subjects(self) -> None:
        """Load available subfolders and subjects - simple directory listing."""
        # First try custom path from input, then fallback to get_run_dir
        if self._path_input and self._path_input.value and self._path_input.value.strip():
            run_dir = Path(self._path_input.value.strip())
        else:
            run_dir = self._get_run_dir()
            if run_dir:
                run_dir = Path(run_dir)
        
        if not run_dir:
            ui.notify('Especificá un directorio', type='warning')
            self._log('ERROR: No hay directorio especificado', THEME_ERROR)
            return
        
        self._log(f'Directorio: {run_dir}', THEME_PRIMARY)
        
        if not run_dir.exists():
            self._log(f'ERROR: No existe {run_dir}', THEME_ERROR)
            return
        
        # Simply list all subdirectories
        subfolders = []
        for item in sorted(run_dir.iterdir()):
            if item.is_dir() and not item.name.startswith('.'):
                subfolders.append(item.name)
                self._log(f'  📁 {item.name}', THEME_TEXT_DIM)
        
        if subfolders:
            if self._subfolder_select:
                self._subfolder_select.options = subfolders
                self._subfolder_select.value = subfolders[0]
                self._subfolder = subfolders[0]
            self._load_subjects_from_folder(run_dir / subfolders[0])
            self._log(f'✓ {len(subfolders)} carpetas encontradas', THEME_PRIMARY)
        else:
            # No subfolders, use root
            self._log('No hay subcarpetas, buscando en raíz...', THEME_TEXT_DIM)
            if self._subfolder_select:
                self._subfolder_select.options = ['(raíz)']
                self._subfolder_select.value = '(raíz)'
                self._subfolder = None
            self._load_subjects_from_folder(run_dir)
    
    def _load_subjects_from_folder(self, folder: Path) -> None:
        """Load subjects from a specific folder - simple pkl file listing."""
        folder = Path(folder)
        self._log(f'Buscando en: {folder.name}/', THEME_TEXT_DIM)
        
        # List all .pkl files and extract subject identifiers
        subjects = set()
        pkl_files = list(folder.glob('*.pkl'))
        
        self._log(f'  Archivos .pkl: {len(pkl_files)}', THEME_TEXT_DIM)
        
        for f in pkl_files:
            name = f.stem
            # Extract subject ID from filename patterns like:
            # phases-S01-DMT, syncro-S01-DMT, order-S01-DMT, order_all-S01-DMT
            for prefix in ['phases-', 'syncro-', 'order_all-', 'order-']:
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    break
            
            # Exclude non-subject files
            if name and not name.startswith('subject_phases') and name not in [
                'con_mat', 'eigen_all', 'eigen_correct', 'kuramoto_all', 
                'extra', 'extras', 'clusters_splits0'
            ]:
                subjects.add(name)
        
        subjects = sorted(subjects)
        
        if subjects:
            for s in subjects[:10]:
                self._log(f'    👤 {s}', THEME_TEXT_DIM)
            if len(subjects) > 10:
                self._log(f'    ... y {len(subjects) - 10} más', THEME_TEXT_DIM)
            
            if self._subject_select:
                self._subject_select.options = subjects
                self._subject_select.value = subjects[0]
                self._subject = subjects[0]
            self._log(f'✓ {len(subjects)} sujetos encontrados', THEME_PRIMARY)
        else:
            self._log(f'No se encontraron sujetos en {folder.name}', THEME_WARN)
            if self._subject_select:
                self._subject_select.options = []
                self._subject_select.value = None
    
    def _on_subfolder_change(self, value) -> None:
        """Handle subfolder selection change."""
        from ..visualizers.base import extract_event_value
        self._subfolder = extract_event_value(value)
        
        if self._subfolder and self._subfolder != '(raíz)':
            run_dir = self._get_run_dir()
            if run_dir:
                self._load_subjects_from_folder(run_dir / self._subfolder)
    
    async def _generate_frames(self) -> None:
        """Generate visualization frames."""
        self._log_container.clear()
        self._log('🎬 Starting frame generation...', THEME_PRIMARY)
        
        run_dir = self._get_run_dir()
        if not run_dir:
            self._log('Error: No run selected', THEME_ERROR)
            return
        
        subject = self._subject_select.value if self._subject_select else None
        if not subject:
            self._log('Error: No subject selected', THEME_ERROR)
            return
        
        # Parse subject/condition
        subj = subject
        cond = 'DMT'
        if '-' in str(subject):
            parts = str(subject).split('-')
            subj = parts[0]
            cond = parts[1] if len(parts) > 1 else 'DMT'
        
        # Build command
        script_path = Path('/media/storage_hdd/dmt_fz/viz_scripts/plot.py')
        conda_prefix = os.environ.get('CONDA_PREFIX', os.path.expanduser('~/anaconda3/envs/dmt_fz'))
        python_path = Path(conda_prefix) / 'bin' / 'python'
        
        if not python_path.exists():
            python_path = 'python'
        
        cmd = [
            str(python_path), str(script_path),
            '--subject', subj,
            '--condition', cond,
            '--band', self._band,
            '--mode', self._mode,
            '--epochs', f'{self._start_epoch}:{self._end_epoch}',
        ]
        
        self._log(f'Command: {" ".join(cmd)}')
        
        expected_output = Path('/media/storage_hdd/dmt_fz/visualizations/plot') / self._mode / f'{subj}_{cond}_{self._band}'
        self._log(f'Output dir: {expected_output}')
        
        try:
            env = os.environ.copy()
            env['PIPELINE_OUTPUT_DIR'] = str(run_dir)
            base_path = Path('/media/storage_hdd/dmt_fz')
            pythonpath = [str(base_path / 'viz_scripts'), str(base_path / 'pipeline'), str(base_path)]
            env['PYTHONPATH'] = ':'.join(pythonpath)
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            # Read output
            async def read_stream(stream):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    text = line.decode().strip()
                    if text:
                        self._log(text)
            
            await asyncio.gather(
                read_stream(process.stdout),
                read_stream(process.stderr)
            )
            await process.wait()
            
            if process.returncode == 0:
                self._log('✓ Frames generated!', THEME_PRIMARY)
                
                # Show preview if available
                if expected_output.exists():
                    frames = sorted(expected_output.glob(f'*.{self._format}'))
                    if frames:
                        self._log(f'Generated {len(frames)} frames')
                        with self._log_container:
                            ui.image(str(frames[0])).style('max-height: 200px;')
            else:
                self._log(f'Process exited with code {process.returncode}', THEME_WARN)
                
        except Exception as e:
            self._log(f'Error: {e}', THEME_ERROR)
    
    async def _generate_video(self) -> None:
        """Create video from frames."""
        self._log_container.clear()
        self._log('🎬 Creating video...', THEME_PRIMARY)
        
        subject = self._subject_select.value if self._subject_select else None
        if not subject:
            self._log('Error: No subject selected', THEME_ERROR)
            return
        
        subj = subject
        cond = 'DMT'
        if '-' in str(subject):
            parts = str(subject).split('-')
            subj = parts[0]
            cond = parts[1] if len(parts) > 1 else 'DMT'
        
        frames_dir = Path('/media/storage_hdd/dmt_fz/visualizations/plot') / self._mode / f'{subj}_{cond}_{self._band}'
        
        if not frames_dir.exists():
            self._log(f'Frames not found: {frames_dir}', THEME_ERROR)
            self._log('Generate frames first', THEME_TEXT_DIM)
            return
        
        output_video = frames_dir.parent / f'{subj}_{cond}_{self._band}_animation.mp4'
        
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(self._fps),
            '-pattern_type', 'glob',
            '-i', str(frames_dir / f'*.{self._format}'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            str(output_video)
        ]
        
        self._log(f'Output: {output_video.name}')
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await process.communicate()
            
            if process.returncode == 0:
                self._log(f'✓ Video saved: {output_video}', THEME_PRIMARY)
                if output_video.exists():
                    with self._log_container:
                        ui.video(str(output_video)).style('max-height: 300px;')
            else:
                self._log(f'ffmpeg error', THEME_ERROR)
                
        except Exception as e:
            self._log(f'Error: {e}', THEME_ERROR)

