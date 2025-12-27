"""Pipeline page for EEG processing."""
from pathlib import Path
import asyncio
from datetime import datetime
from nicegui import ui

from config import (
    THEME_BG, THEME_CARD, THEME_BORDER, THEME_PRIMARY, THEME_SECONDARY,
    THEME_WARN, THEME_ERROR, THEME_TEXT, THEME_TEXT_DIM
)
from app.state import PS
from app.visualization.styles.css import STYLE
from app.visualization.components.running_indicator import render_running_indicator

PIPELINE_DIR = Path(__file__).parent.parent.parent.parent / "pipeline"
PIPELINE_OUTPUTS = Path(__file__).parent.parent.parent.parent / "eeg_viewer" / "pipeline_outputs"
RESULTS_BASE = Path(__file__).parent.parent.parent / "fwd-inv-stc"
DEFAULT_INPUT_DIR = Path(__file__).parent.parent.parent.parent / "EEG_CLEAN"

def get_run_dirs():
    """List existing pipeline runs"""
    if not PIPELINE_OUTPUTS.exists():
        return []
    return sorted([d.name for d in PIPELINE_OUTPUTS.iterdir() if d.is_dir() and d.name.startswith('run_')], reverse=True)

def create_new_run():
    """Create a new run directory with timestamp"""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = PIPELINE_OUTPUTS / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for cond in ["DMT", "EC", "EO"]:
        (run_dir / cond).mkdir(exist_ok=True)
    return run_dir

def pipeline_log(msg):
    """Add message to pipeline log - persists even when tab switches"""
    # Always store in history for persistence
    PS.log_history.append(msg)
    
    # Try to update UI if container exists and client is connected
    if PS.log_container:
        try:
            with PS.log_container:
                # Add line break before new steps/sections
                if any(x in msg for x in ['[RUN]', 'Starting:', '[INFO]', '[SETUP]', '===', 'COMPLETED', 'FAILED']):
                    ui.label('').style('height: 12px;')
                ui.label(msg).style(f'color:{THEME_TEXT}; font-family: JetBrains Mono; font-size: 0.75rem;')
            # Force UI update and scroll to bottom
            PS.log_container.update()
            if hasattr(PS, 'log_scroll') and PS.log_scroll:
                PS.log_scroll.scroll_to(percent=1.0)
        except RuntimeError:
            # Client disconnected (tab switched), log is still stored in history
            pass

async def run_pipeline_step(script_name, args_list, step_name, output_dir=None, input_dir=None):
    """Run a pipeline script with arguments"""
    import os as _os
    
    if PS.running:
        try:
            ui.notify('Pipeline already running', type='warning')
        except RuntimeError:
            pass
        return
    
    PS.running = True
    PS.current_step = step_name
    PS.running_task_name = script_name  # For global indicator
    PS.start_time = datetime.now()
    
    script_path = PIPELINE_DIR / script_name
    if not script_path.exists():
        try:
            ui.notify(f'Script not found: {script_path}', type='negative')
        except RuntimeError:
            pass
        PS.running = False
        PS.running_task_name = ""
        return
    
    # Use -u for unbuffered output so we see logs in real-time
    cmd = ["python", "-u", str(script_path)] + args_list
    pipeline_log(f"[{step_name}] Starting: {' '.join(cmd)}")
    
    # Set environment variables for input/output directories
    env = _os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'  # Force unbuffered output
    if input_dir:
        env['PIPELINE_INPUT_DIR'] = str(input_dir)
        pipeline_log(f"[{step_name}] Input dir: {input_dir}")
    if output_dir:
        env['PIPELINE_OUTPUT_DIR'] = str(output_dir)
        pipeline_log(f"[{step_name}] Output dir: {output_dir}")
    
    try:
        # Use larger buffer limit for tqdm progress bars
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(PIPELINE_DIR.parent.parent),
            env=env,
            limit=1024*1024  # 1MB buffer limit
        )
        
        PS.current_process = process
        
        # Read both stdout and stderr with chunked reading to handle long lines
        async def read_stream(stream, is_stderr=False):
            buffer = b''
            while True:
                try:
                    chunk = await stream.read(8192)  # Read in 8KB chunks
                    if not chunk:
                        # Process remaining buffer
                        if buffer:
                            try:
                                text = buffer.decode('utf-8', errors='replace').strip()
                                if text and not any(x in text for x in ['%|', 'it/s]', '█', '▌', '\r']):
                                    if is_stderr:
                                        pipeline_log(f"[WARN] {text[:500]}")
                                    else:
                                        pipeline_log(text[:500])
                            except:
                                pass
                        break
                    
                    buffer += chunk
                    
                    # Process complete lines
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        try:
                            text = line.decode('utf-8', errors='replace').strip()
                            if text:
                                # Skip tqdm progress bars (they clutter the log)
                                is_progress_bar = any(x in text for x in ['%|', 'it/s]', '0%|', '100%|', '█', '▌', '\r'])
                                if is_progress_bar:
                                    continue
                                
                                # Truncate very long lines
                                if len(text) > 500:
                                    text = text[:500] + '...'
                                
                                if is_stderr:
                                    pipeline_log(f"[WARN] {text}")
                                else:
                                    pipeline_log(text)
                                await asyncio.sleep(0.01)
                        except:
                            pass
                    
                    # Prevent buffer from growing too large (discard if > 100KB)
                    if len(buffer) > 100*1024:
                        buffer = b''
                        
                except Exception:
                    # Continue on read errors
                    await asyncio.sleep(0.1)
                    continue
        
        # Read both streams concurrently
        await asyncio.gather(
            read_stream(process.stdout, is_stderr=False),
            read_stream(process.stderr, is_stderr=True)
        )
        
        await process.wait()
        
        if process.returncode == 0:
            pipeline_log(f"[{step_name}] Completed successfully")
            try:
                ui.notify(f'{step_name} completed!', type='positive')
            except RuntimeError:
                pass
        else:
            pipeline_log(f"[{step_name}] Failed with code {process.returncode}")
            try:
                ui.notify(f'{step_name} failed', type='negative')
            except RuntimeError:
                pass
            
    except Exception as e:
        import traceback
        pipeline_log(f"[{step_name}] ERROR: {str(e)}")
        pipeline_log(traceback.format_exc())
        try:
            ui.notify(f'Error: {e}', type='negative')
        except RuntimeError:
            pass
    finally:
        PS.running = False
        PS.current_step = ""
        PS.running_task_name = ""
        PS.start_time = None
        PS.current_process = None


@ui.page('/cleaner')
def cleaner_page_route():
    """EEG Cleaning Pipeline page."""
    ui.add_head_html(f'<style>{STYLE}</style>')
    cleaner_page()


@ui.page('/pipeline')
def pipeline_page():
    ui.add_head_html(f'<style>{STYLE}</style>')
    
    # Header with navigation
    with ui.header().classes('items-center px-4 py-1').style(f'background: {THEME_BG}; border-bottom: 1px solid {THEME_BORDER};'):
        ui.label('▶').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.75rem;')
        ui.label('EEG_PIPELINE').classes('text-base font-medium ml-2').style(f'color: {THEME_PRIMARY}; font-family: JetBrains Mono;')
        ui.label('v1.0').classes('text-xs ml-2').style(f'color: {THEME_TEXT_DIM}; font-family: JetBrains Mono;')
        
        # Global running indicator
        render_running_indicator()
        
        with ui.row().classes('ml-auto gap-2'):
            ui.button('VIEWER', on_click=lambda: ui.navigate.to('/')).props('flat dense').style(f'color:{THEME_TEXT_DIM};')
            ui.button('CLEANER', on_click=lambda: ui.navigate.to('/cleaner')).props('flat dense').style(f'color:{THEME_TEXT_DIM};')
            ui.button('PIPELINE', on_click=lambda: ui.navigate.to('/pipeline')).props('flat dense').style(f'color:{THEME_PRIMARY};')
            ui.button('MODEL', on_click=lambda: ui.navigate.to('/model')).props('flat dense').style(f'color:{THEME_TEXT_DIM};')
            ui.button('ANALYSIS', on_click=lambda: ui.navigate.to('/analysis')).props('flat dense').style(f'color:{THEME_TEXT_DIM};')
    
    with ui.row().classes('w-full p-4 gap-4').style('height: calc(100vh - 50px); align-items: stretch; overflow: hidden;'):
        
        # LEFT: Pipeline Controls (scrollable independently)
        with ui.scroll_area().style('width: 450px; height: 100%;'):
          with ui.column().classes('gap-4 pr-2'):
            
            # INPUT/OUTPUT CONFIGURATION
            with ui.card().classes('dark-card p-4 w-full').style(f'border: 1px solid {THEME_PRIMARY};'):
                ui.label('// INPUT_OUTPUT_DIRS').classes('terminal-header')
                
                # INPUT DIRECTORY
                with ui.row().classes('items-center gap-2 mt-2 w-full'):
                    ui.label('INPUT:').style(f'color:{THEME_SECONDARY}; font-family: JetBrains Mono; font-size: 0.75rem; min-width: 60px;')
                    input_dir_field = ui.input(value=str(DEFAULT_INPUT_DIR)).props('dense').classes('flex-1')
                    
                    def scan_input_dir():
                        p = Path(input_dir_field.value)
                        if p.exists():
                            # Check for subdirectory structure (DMT/, EC/, EO/)
                            conds = [d.name for d in p.iterdir() if d.is_dir() and d.name in ['DMT', 'EC', 'EO']]
                            
                            if conds:
                                # Subdirectory structure
                                files = sum(len(list((p / c).glob('*.set'))) for c in conds)
                                ui.notify(f'Found: {conds}, {files} .set files', type='info')
                                pipeline_log(f"[INPUT] Scanned {p}: {conds}, {files} .set files")
                            else:
                                # Flat structure (files in root with condition in filename)
                                all_set = list(p.glob('*.set'))
                                dmt = len([f for f in all_set if 'DMT' in f.name.upper()])
                                ec = len([f for f in all_set if 'EC' in f.name.upper()])
                                eo = len([f for f in all_set if 'EO' in f.name.upper()])
                                total = dmt + ec + eo
                                ui.notify(f'Flat: DMT={dmt}, EC={ec}, EO={eo} .set files', type='info')
                                pipeline_log(f"[INPUT] Scanned {p}: Flat - DMT={dmt}, EC={ec}, EO={eo} .set files")
                        else:
                            ui.notify('Directory not found', type='warning')
                    
                    ui.button(icon='search', on_click=scan_input_dir).props('flat dense size=sm')
                
                ui.label('Directorio con DMT/, EC/, EO/ o archivos .set con condición en nombre').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; margin-left: 68px;')
                
                # Mutable container for input dir
                current_input_dir = [DEFAULT_INPUT_DIR]
                def get_input_dir():
                    return Path(input_dir_field.value) if input_dir_field.value else DEFAULT_INPUT_DIR
                
                ui.separator().classes('my-2')
                
                # OUTPUT DIRECTORY
                with ui.row().classes('items-center gap-2 w-full'):
                    ui.label('OUTPUT:').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.75rem; min-width: 60px;')
                    
                    run_label = ui.label('(crear NEW RUN)').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.8rem;')
                    # Use PS.selected_run for persistence across tab switches
                    current_run_dir = [PS.selected_run]
                    
                    def refresh_run_label():
                        if current_run_dir[0]:
                            run_label.text = str(current_run_dir[0].name)
                            run_label.style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.8rem;')
                        else:
                            run_label.text = '(crear NEW RUN)'
                            run_label.style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.8rem;')
                    
                    # Restore label if we have a persisted run
                    if PS.selected_run:
                        refresh_run_label()
                    
                    async def new_run():
                        run_dir = await asyncio.get_event_loop().run_in_executor(None, create_new_run)
                        current_run_dir[0] = run_dir
                        PS.selected_run = run_dir  # Persist selection
                        refresh_run_label()
                        if PS.refresh_files: PS.refresh_files()
                        ui.notify(f'Nuevo run: {run_dir.name}', type='positive')
                        pipeline_log(f"[RUN] Created: {run_dir}")
                    
                    ui.button('NEW RUN', on_click=new_run, icon='add').props('dense').style(f'background:{THEME_PRIMARY}; color:black;')
                    
                    existing_runs = get_run_dirs()
                    if existing_runs:
                        # Set initial value if we have a persisted run
                        initial_run = PS.selected_run.name if PS.selected_run and PS.selected_run.name in existing_runs else None
                        run_select = ui.select(existing_runs, value=initial_run, label='continuar:').props('dense').classes('w-36')
                        def use_existing():
                            if run_select.value:
                                current_run_dir[0] = PIPELINE_OUTPUTS / run_select.value
                                PS.selected_run = current_run_dir[0]  # Persist selection
                                refresh_run_label()
                                if PS.refresh_files: PS.refresh_files()
                                pipeline_log(f"[RUN] Using existing: {current_run_dir[0]}")
                        run_select.on('update:model-value', lambda e: use_existing())
                
                ui.label(f'Output: {PIPELINE_OUTPUTS}/run_* (no pisa fwd-inv-stc/)').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; margin-left: 68px;')
            
            # GLOBAL PARAMETERS - Clean grid layout
            with ui.card().classes('dark-card p-4 w-full'):
                ui.label('// GLOBAL_PARAMS').classes('terminal-header')
                
                # Use CSS grid for aligned parameters
                with ui.element('div').classes('w-full').style('display: grid; grid-template-columns: 100px 1fr; gap: 8px 12px; align-items: center;'):
                    # Conditions row
                    ui.label('Conditions').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.75rem;')
                    with ui.row().classes('gap-3 items-center'):
                        cond_dmt = ui.checkbox('DMT', value=True).props('dense')
                        cond_ec = ui.checkbox('EC', value=True).props('dense')
                        cond_eo = ui.checkbox('EO', value=True).props('dense')
                    
                    # Max subjects row
                    ui.label('Max subjects').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.75rem;')
                    with ui.row().classes('gap-2 items-center'):
                        max_subj = ui.number(value=0, min=0, max=100).props('dense').classes('w-20')
                        ui.label('0 = all').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                    
                    # Max epochs row
                    ui.label('Max epochs').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.75rem;')
                    with ui.row().classes('gap-2 items-center'):
                        max_epochs = ui.number(value=0, min=0, max=500).props('dense').classes('w-20')
                        ui.label('0 = all').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                    
                    # Workers row
                    ui.label('Workers').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.75rem;')
                    with ui.row().classes('gap-2 items-center'):
                        workers_num = ui.number(value=7, min=1, max=32).props('dense').classes('w-20')
                    
                    # Jobs row
                    ui.label('Jobs').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.75rem;')
                    with ui.row().classes('gap-2 items-center'):
                        jobs_num = ui.number(value=0, min=0, max=64).props('dense').classes('w-20')
                        ui.label('0 = auto').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                
                def get_conditions():
                    conds = []
                    if cond_dmt.value: conds.append('DMT')
                    if cond_ec.value: conds.append('EC')
                    if cond_eo.value: conds.append('EO')
                    return conds
            
            # STEP 1: FWD.PY
            with ui.card().classes('dark-card p-4 w-full'):
                ui.label('// STEP_1: SOURCE_LOCALIZATION').classes('terminal-header')
                ui.label('fwd.py - Forward/Inverse Solution + Metrics').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                ui.label('→ run_*/phases-{subj}.pkl').style(f'color:{THEME_PRIMARY}; font-size: 0.65rem;')
                
                with ui.row().classes('gap-2 mt-3'):
                    async def run_fwd():
                        if not current_run_dir[0]:
                            ui.notify('Primero creá un NEW RUN', type='warning')
                            return
                        args = [
                            '--max-subjects', str(int(max_subj.value or 0)),
                            '--conditions'] + get_conditions() + [
                            '--jobs', str(int(jobs_num.value or 0)),
                            '--workers', str(int(workers_num.value or 7)),
                            '--max-epochs', str(int(max_epochs.value or 0))
                        ]
                        await run_pipeline_step('fwd.py', args, 'Source Localization', current_run_dir[0], get_input_dir())
                    
                    ui.button('RUN fwd.py', on_click=run_fwd, icon='play_arrow').props('dense').style(f'background:{THEME_PRIMARY}; color:black;')
                    ui.label('~3-4h (o menos con max_epochs)').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
            
            # STEP 2: SAVE_LOAD_PICKLE.PY (Consolidate phases)
            with ui.card().classes('dark-card p-4 w-full'):
                ui.label('// STEP_2: CONSOLIDATE_PHASES').classes('terminal-header')
                ui.label('save_load_pickle.py - Merge phases-*.pkl into subject_phases_{cond}.pkl').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                ui.label('→ run_*/{cond}/subject_phases_{cond}.pkl').style(f'color:{THEME_PRIMARY}; font-size: 0.65rem;')
                
                with ui.row().classes('gap-2 mt-3'):
                    async def run_consolidate():
                        if not current_run_dir[0]:
                            ui.notify('Primero creá un NEW RUN', type='warning')
                            return
                        args = ['--conditions'] + get_conditions()
                        await run_pipeline_step('save_load_pickle.py', args, 'Consolidate Phases', current_run_dir[0])
                    
                    ui.button('RUN save_load_pickle.py', on_click=run_consolidate, icon='play_arrow').props('dense').style('background:#06b6d4; color:black;')
                    ui.label('~1-2 min').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
            
            # STEP 3: MULTI2POOL2.PY
            with ui.card().classes('dark-card p-4 w-full'):
                ui.label('// STEP_3: NETWORK_FILTERING').classes('terminal-header')
                ui.label('multi2pool2.py - Filter by brain networks (DMN, FPN, etc)').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                ui.label('→ run_*/order_all-{subj}.pkl').style(f'color:{THEME_PRIMARY}; font-size: 0.65rem;')
                
                with ui.row().classes('gap-2 mt-3'):
                    async def run_multi():
                        if not current_run_dir[0]:
                            ui.notify('Primero creá un NEW RUN', type='warning')
                            return
                        await run_pipeline_step('multi2pool2.py', [], 'Network Filtering', current_run_dir[0])
                    
                    ui.button('RUN multi2pool2.py', on_click=run_multi, icon='play_arrow').props('dense').style(f'background:{THEME_SECONDARY}; color:black;')
                    ui.label('~2-5 min').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
            
            # STEP 4: CALCULATE_SYNCRO.PY
            with ui.card().classes('dark-card p-4 w-full'):
                ui.label('// STEP_4: SYNC_METRICS').classes('terminal-header')
                ui.label('calculate_syncro.py - Calculate sync matrices & Kuramoto').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                ui.label('→ run_*/{cond}/syncro-{subj}.pkl').style(f'color:{THEME_PRIMARY}; font-size: 0.65rem;')
                
                with ui.row().classes('gap-2 mt-3'):
                    async def run_syncro():
                        if not current_run_dir[0]:
                            ui.notify('Primero creá un NEW RUN', type='warning')
                            return
                        args = ['--workers', str(int(workers_num.value or 7)), '--conditions'] + get_conditions()
                        await run_pipeline_step('calculate_syncro.py', args, 'Sync Metrics', current_run_dir[0])
                    
                    ui.button('RUN calculate_syncro.py', on_click=run_syncro, icon='play_arrow').props('dense').style('background:#10b981; color:black;')
                    ui.label('~10-30 min').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
            
            # STEP 5: GENERATE_ORDER.PY
            with ui.card().classes('dark-card p-4 w-full'):
                ui.label('// STEP_5: KURAMOTO_ORDER').classes('terminal-header')
                ui.label('generate_order.py - Calculate Kuramoto order parameter').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                ui.label('→ run_*/order-{subj}.pkl').style(f'color:{THEME_PRIMARY}; font-size: 0.65rem;')
                
                with ui.row().classes('gap-2 mt-3'):
                    async def run_order():
                        if not current_run_dir[0]:
                            ui.notify('Primero creá un NEW RUN', type='warning')
                            return
                        args = ['--workers', str(int(workers_num.value or 7)), '--conditions'] + get_conditions()
                        await run_pipeline_step('generate_order.py', args, 'Kuramoto Order', current_run_dir[0])
                    
                    ui.button('RUN generate_order.py', on_click=run_order, icon='play_arrow').props('dense').style(f'background:{THEME_WARN}; color:black;')
                    ui.label('~1 min').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
            
            # STEP 6: BUILD_ORDER_DATA.PY
            with ui.card().classes('dark-card p-4 w-full'):
                ui.label('// STEP_6: AGGREGATE_DATA').classes('terminal-header')
                ui.label('build_order_data.py - Aggregate Kuramoto metrics').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                ui.label('-> r_kuramoto_nets_*.pkl').style(f'color:{THEME_PRIMARY}; font-size: 0.65rem;')
                
                with ui.row().classes('gap-2 mt-3'):
                    async def run_build_order():
                        if not current_run_dir[0]:
                            ui.notify('Primero creá un NEW RUN', type='warning')
                            return
                        args = ['--build-all', '--workers', str(int(workers_num.value or 7))]
                        await run_pipeline_step('build_order_data.py', args, 'Aggregate Data', current_run_dir[0])
                    
                    ui.button('RUN build_order_data.py', on_click=run_build_order, icon='play_arrow').props('dense').style('background:#60a5fa; color:black;')
                    ui.label('~2-5 min').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
            
            # STEP 7: PEARSON.PY
            with ui.card().classes('dark-card p-4 w-full'):
                ui.label('// STEP_7: CORRELATIONS').classes('terminal-header')
                ui.label('pearson.py - Correlate with questionnaires').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                ui.label('-> pearson_results/').style(f'color:{THEME_PRIMARY}; font-size: 0.65rem;')
                
                with ui.row().classes('gap-2 mt-3'):
                    async def run_pearson():
                        if not current_run_dir[0]:
                            ui.notify('Primero creá un NEW RUN', type='warning')
                            return
                        await run_pipeline_step('pearson.py', [], 'Correlations', current_run_dir[0])
                    
                    ui.button('RUN pearson.py', on_click=run_pearson, icon='play_arrow').props('dense').style('background:#a78bfa; color:black;')
                    ui.label('~3-5 min').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
            
            # STEP 8: CLUSTERING.PY (Optional)
            with ui.card().classes('dark-card p-4 w-full'):
                ui.label('// STEP_8: CLUSTERING (OPTIONAL)').classes('terminal-header')
                ui.label('clustering.py - Brain state identification').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                ui.label('→ run_*/clustering_results/').style(f'color:{THEME_PRIMARY}; font-size: 0.65rem;')
                
                with ui.row().classes('gap-4 items-center mt-2'):
                    ui.label('Bands:').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.7rem;')
                    band_delta = ui.checkbox('δ', value=True).props('dense')
                    band_theta = ui.checkbox('θ', value=True).props('dense')
                    band_alpha = ui.checkbox('α', value=True).props('dense')
                    band_beta = ui.checkbox('β', value=True).props('dense')
                    band_gamma = ui.checkbox('γ', value=True).props('dense')
                
                with ui.row().classes('gap-3 items-center mt-2'):
                    ui.label('Clusters:').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.7rem;')
                    min_k = ui.number(value=2, min=2, max=20).props('dense').classes('w-16')
                    ui.label('-').style(f'color:{THEME_TEXT_DIM};')
                    max_k = ui.number(value=15, min=2, max=30).props('dense').classes('w-16')
                    
                    ui.label('PCA:').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.7rem;')
                    min_pca = ui.number(value=2, min=2, max=20).props('dense').classes('w-16')
                    ui.label('-').style(f'color:{THEME_TEXT_DIM};')
                    max_pca = ui.number(value=10, min=2, max=30).props('dense').classes('w-16')
                
                search_mode = ui.toggle(['Quick', 'Full'], value='Quick').props('dense')
                
                with ui.row().classes('gap-2 mt-3'):
                    async def run_clustering():
                        if not current_run_dir[0]:
                            ui.notify('Primero creá un NEW RUN', type='warning')
                            return
                        bands = []
                        if band_delta.value: bands.append('Delta')
                        if band_theta.value: bands.append('Theta')
                        if band_alpha.value: bands.append('Alpha')
                        if band_beta.value: bands.append('Beta')
                        if band_gamma.value: bands.append('Gamma')
                        
                        mode_arg = '--quick-search' if search_mode.value == 'Quick' else '--full-search'
                        cluster_out = str(current_run_dir[0] / "clustering_results")
                        args = [
                            mode_arg,
                            '--conditions'] + get_conditions() + [
                            '--bands'] + bands + [
                            '--min-k', str(int(min_k.value)),
                            '--max-k', str(int(max_k.value)),
                            '--min-comps', str(int(min_pca.value)),
                            '--max-comps', str(int(max_pca.value)),
                            '--workers', str(int(workers_num.value or 4)),
                            '--output-dir', cluster_out
                        ]
                        await run_pipeline_step('clustering.py', args, 'Clustering', current_run_dir[0])
                    
                    ui.button('RUN clustering.py', on_click=run_clustering, icon='play_arrow').props('dense').style('background:#ff6b9d; color:black;')
                    ui.label('Quick: ~30min, Full: ~4h').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
        
        # RIGHT: Tabbed Panel (Console, Files, System, Visualize)
        with ui.column().classes('flex-1').style('height: 100%; min-height: 0; display: flex; flex-direction: column; overflow: hidden;'):
            with ui.card().classes('dark-card p-2 w-full flex-1').style('display: flex; flex-direction: column; min-height: 0;'):
                with ui.tabs().classes('w-full').style(f'background: {THEME_BG};') as tabs:
                    tab_console = ui.tab('CONSOLE', icon='terminal').style(f'color:{THEME_PRIMARY};')
                    tab_files = ui.tab('FILES', icon='folder').style(f'color:{THEME_SECONDARY};')
                    tab_system = ui.tab('SYSTEM', icon='memory').style(f'color:{THEME_WARN};')
                    tab_viz = ui.tab('VISUALIZE', icon='analytics').style('color:#a78bfa;')
                
                with ui.tab_panels(tabs, value=tab_console).classes('w-full').style('flex: 1; min-height: 0; overflow: hidden;'):
                    # CONSOLE TAB
                    with ui.tab_panel(tab_console).classes('p-2').style('height: 100%; display: flex; flex-direction: column; overflow: hidden;'):
                        with ui.row().classes('items-center gap-3 mb-2 shrink-0'):
                            ui.label('// OUTPUT_LOG').classes('terminal-header')
                            
                            # Status indicator
                            # Reset pipeline state on page load
                            PS.running = False
                            PS.current_step = ""
                            PS.start_time = None
                            PS.current_process = None
                            
                            status_label = ui.label('Idle').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.7rem; margin-left: auto;')
                            PS.status_label = status_label
                            
                            def update_status():
                                if PS.running and PS.start_time:
                                    elapsed = (datetime.now() - PS.start_time).seconds
                                    mins, secs = divmod(elapsed, 60)
                                    status_label.text = f'Running: {PS.current_step} ({mins}m {secs}s)'
                                    status_label.style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.7rem;')
                                else:
                                    status_label.text = 'Idle'
                                    status_label.style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.7rem;')
                            
                            ui.timer(1.0, update_status)
                            
                            async def stop_pipeline():
                                if PS.current_process:
                                    try:
                                        PS.current_process.terminate()
                                        pipeline_log(f"[{PS.current_step}] STOPPED by user")
                                        ui.notify('Pipeline stopped', type='warning')
                                    except:
                                        pass
                            
                            ui.button('STOP', on_click=stop_pipeline, icon='stop').props('flat dense size=sm color=negative')
                            
                            def clear_log():
                                if PS.log_container:
                                    PS.log_container.clear()
                                PS.log_history.clear()  # Also clear history
                            ui.button('CLEAR', on_click=clear_log, icon='delete').props('flat dense size=sm')
                        
                        PS.log_scroll = ui.scroll_area().classes('w-full').style('background: #050505; border-radius: 4px; flex: 1; min-height: 0;')
                        with PS.log_scroll:
                            PS.log_container = ui.column().classes('w-full p-3 gap-0')
                            with PS.log_container:
                                # Restore logs from history if available
                                if PS.log_history:
                                    for msg in PS.log_history:
                                        if any(x in msg for x in ['[RUN]', 'Starting:', '[INFO]', '[SETUP]', '===', 'COMPLETED', 'FAILED']):
                                            ui.label('').style('height: 12px;')
                                        ui.label(msg).style(f'color:{THEME_TEXT}; font-family: JetBrains Mono; font-size: 0.75rem;')
                                else:
                                    ui.label('Pipeline ready. Select a step and click RUN.').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.75rem;')
                                    ui.label(f'Pipeline directory: {PIPELINE_DIR}').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.7rem;')
                    
                    # FILES TAB
                    with ui.tab_panel(tab_files).classes('p-2').style('height: 100%; display: flex; flex-direction: column;'):
                        file_browser_container = ui.column().classes('w-full flex-1').style('min-height: 0; overflow: hidden;')
                        
                        def refresh_files():
                            file_browser_container.clear()
                            if not current_run_dir[0] or not current_run_dir[0].exists():
                                with file_browser_container:
                                    ui.label('No run selected').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem;')
                                return
                            
                            run_path = current_run_dir[0]
                            with file_browser_container:
                                def count_files(pattern):
                                    return len(list(run_path.rglob(pattern)))
                                
                                stats = {
                                    'syncro': count_files('syncro-*.pkl') + count_files('phases-*.pkl'),
                                    'order_all': count_files('order_all-*.pkl'),
                                    'order': count_files('order-*.pkl'),
                                    'clustering': count_files('clustering_results/**/*.pkl') + count_files('clustering_results/**/*.csv'),
                                    'pearson': count_files('pearson_results/**/*'),
                                }
                                
                                ui.label(f'{run_path.name}').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.8rem;')
                                
                                with ui.row().classes('gap-4 mt-2 flex-wrap'):
                                    for name, count in stats.items():
                                        color = THEME_PRIMARY if count > 0 else THEME_TEXT_DIM
                                        ui.label(f'{name}: {count}').style(f'color:{color}; font-family: JetBrains Mono; font-size: 0.7rem;')
                                
                                ui.separator().classes('my-2')
                                
                                with ui.scroll_area().classes('w-full flex-1').style('min-height: 150px;'):
                                    for item in sorted(run_path.iterdir()):
                                        if item.is_dir():
                                            file_count = len(list(item.rglob('*')))
                                            ui.label(f'[dir] {item.name}/ ({file_count} files)').style(f'color:{THEME_SECONDARY}; font-family: JetBrains Mono; font-size: 0.7rem;')
                                        else:
                                            size_kb = item.stat().st_size / 1024
                                            size_str = f'{size_kb:.1f}KB' if size_kb < 1024 else f'{size_kb/1024:.1f}MB'
                                            ui.label(f'[file] {item.name} ({size_str})').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.7rem;')
                        
                        with ui.row().classes('mt-2'):
                            ui.button('REFRESH', on_click=refresh_files, icon='refresh').props('flat dense size=sm')
                        
                        PS.refresh_files = refresh_files
                        refresh_files()
                    
                    # SYSTEM TAB
                    with ui.tab_panel(tab_system).classes('p-2'):
                        ui.label('// SYSTEM_MONITOR').classes('terminal-header mb-2')
                        system_container = ui.column().classes('w-full gap-4')
                        
                        def update_system_stats():
                            import psutil
                            system_container.clear()
                            with system_container:
                                cpu_percent = psutil.cpu_percent(interval=0.1)
                                cpu_count = psutil.cpu_count()
                                cpu_color = THEME_PRIMARY if cpu_percent < 50 else (THEME_WARN if cpu_percent < 80 else '#ff4444')
                                
                                with ui.row().classes('gap-4 items-center'):
                                    with ui.column().classes('gap-0'):
                                        ui.label(f'CPU {cpu_percent:.0f}%').style(f'color:{cpu_color}; font-family: JetBrains Mono; font-size: 1rem; font-weight: bold;')
                                        ui.label(f'{cpu_count} cores').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                                    
                                    per_cpu = psutil.cpu_percent(percpu=True)
                                    with ui.row().classes('gap-1 flex-wrap'):
                                        for i, pct in enumerate(per_cpu[:16]):
                                            color = THEME_PRIMARY if pct < 50 else (THEME_WARN if pct < 80 else '#ff4444')
                                            ui.label(f'{pct:.0f}').style(f'color:{color}; font-family: JetBrains Mono; font-size: 0.65rem; min-width: 22px; text-align: center;')
                                
                                mem = psutil.virtual_memory()
                                mem_used_gb = mem.used / (1024**3)
                                mem_total_gb = mem.total / (1024**3)
                                mem_color = THEME_PRIMARY if mem.percent < 60 else (THEME_WARN if mem.percent < 85 else '#ff4444')
                                
                                with ui.row().classes('gap-4 items-center'):
                                    ui.label(f'RAM {mem.percent:.0f}%').style(f'color:{mem_color}; font-family: JetBrains Mono; font-size: 1rem; font-weight: bold;')
                                    ui.label(f'{mem_used_gb:.1f} / {mem_total_gb:.0f} GB').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                                
                                try:
                                    import GPUtil
                                    gpus = GPUtil.getGPUs()
                                    if gpus:
                                        gpu = gpus[0]
                                        gpu_color = THEME_PRIMARY if gpu.load*100 < 50 else (THEME_WARN if gpu.load*100 < 80 else '#ff4444')
                                        with ui.row().classes('gap-4 items-center'):
                                            ui.label(f'GPU {gpu.load*100:.0f}%').style(f'color:{gpu_color}; font-family: JetBrains Mono; font-size: 1rem; font-weight: bold;')
                                            ui.label(f'{gpu.memoryUsed:.0f} / {gpu.memoryTotal:.0f} MB').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                                except:
                                    ui.label('GPU: N/A').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                        
                        ui.timer(2.0, update_system_stats)
                        update_system_stats()
                    
                    # VISUALIZE TAB - Unified Visualization Dashboard
                    with ui.tab_panel(tab_viz).classes('p-0').style('height: 100%; overflow: hidden;'):
                        # State for visualization - use nonlocal dict to persist across tab switches
                        if not hasattr(PS, 'viz_state'):
                            PS.viz_state = {'data': None, 'file': None, 'loaded': False}
                        viz_state = PS.viz_state
                        
                        with ui.column().classes('w-full h-full').style('display: flex; flex-direction: column; overflow: hidden;'):
                            # FIXED HEADER - Data Selection with custom path support (outside scroll area)
                            with ui.card().classes('dark-card p-3 w-full').style('flex-shrink: 0;'):
                                # Custom data path
                                with ui.row().classes('items-center gap-2 w-full mb-2'):
                                    ui.label('Data Path:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                                    custom_data_path = ui.input(placeholder='/path/to/data or leave empty for run dir').props('dense dark').classes('flex-1')
                                    
                                    def browse_path():
                                        """Load data from custom path"""
                                        path = custom_data_path.value.strip() if custom_data_path.value else None
                                        if path:
                                            from pathlib import Path
                                            p = Path(path)
                                            if p.exists():
                                                viz_state['custom_path'] = p
                                                load_subjects_from_path(p)
                                            else:
                                                ui.notify(f'Path not found: {path}', type='warning')
                                        elif current_run_dir[0]:
                                            viz_state['custom_path'] = None
                                            load_subjects_from_path(current_run_dir[0])
                                    
                                    ui.button('Load', on_click=browse_path, icon='folder_open').props('dense flat')
                                
                                with ui.row().classes('items-center gap-4 flex-wrap'):
                                    ui.label('▌VISUALIZATION').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.9rem; letter-spacing: 1px;')
                                    
                                    viz_band = ui.select(['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'], value='Alpha', label='Band').props('dense dark').classes('w-24')
                                    viz_subject = ui.select([], label='Subject').props('dense dark').classes('w-32')
                                    viz_epoch = ui.number(value=0, min=0, max=100, label='Epoch').props('dense').classes('w-20')
                                    
                                    def load_subjects_from_path(path):
                                        """Load subjects from given path and auto-refresh plots"""
                                        from pathlib import Path
                                        p = Path(path)
                                        viz_subject.options = []
                                        all_files = list(p.rglob('syncro-*.pkl')) + list(p.rglob('phases-*.pkl'))
                                        subjects = sorted(list(set([f.stem.split('-')[1] if '-' in f.stem else f.stem for f in all_files])))[:30]
                                        viz_subject.options = subjects
                                        if subjects:
                                            viz_subject.value = subjects[0]
                                            ui.notify(f'Found {len(subjects)} subjects in {p.name}', type='info')
                                            # Auto-refresh plots with first subject
                                            if 'refresh_all_plots' in viz_state:
                                                viz_state['refresh_all_plots']()
                                        else:
                                            ui.notify(f'No subjects found in {p.name}', type='warning')
                                    
                                    def load_subjects():
                                        path = viz_state.get('custom_path') or current_run_dir[0]
                                        if path:
                                            load_subjects_from_path(path)
                                    
                                    ui.button('Load Subjects', on_click=load_subjects, icon='refresh').props('dense flat')
                                    
                                    viz_status = ui.label('No data loaded').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem; margin-left: auto;')
                                    
                                    # Function to refresh all plots when parameters change
                                    def refresh_all_plots():
                                        """Refresh all visualizations with current parameters"""
                                        # Load data for current subject
                                        data_path = viz_state.get('custom_path') or current_run_dir[0]
                                        if data_path and viz_subject.value:
                                            import pickle
                                            from pathlib import Path
                                            try:
                                                p = Path(data_path)
                                                subj = viz_subject.value
                                                
                                                # Prefer phases-*.pkl (has phases_stc for Hilbert viz)
                                                # Fall back to syncro-*.pkl (has kuramoto data)
                                                phases_files = list(p.rglob(f'phases-*{subj}*.pkl'))
                                                syncro_files = list(p.rglob(f'syncro-*{subj}*.pkl'))
                                                
                                                # Load phases file for phase visualizations
                                                if phases_files:
                                                    with open(phases_files[0], 'rb') as f:
                                                        viz_state['data'] = pickle.load(f)
                                                    viz_state['file'] = phases_files[0]
                                                    viz_status.text = f'Loaded: {phases_files[0].name}'
                                                    viz_status.style(f'color:{THEME_PRIMARY}; font-size: 0.7rem;')
                                                elif syncro_files:
                                                    # syncro files have kuramoto but no phases
                                                    with open(syncro_files[0], 'rb') as f:
                                                        viz_state['data'] = pickle.load(f)
                                                    viz_state['file'] = syncro_files[0]
                                                    viz_status.text = f'Loaded: {syncro_files[0].name} (no phases)'
                                                    viz_status.style(f'color:{THEME_WARN}; font-size: 0.7rem;')
                                                else:
                                                    viz_status.text = f'No files for {subj}'
                                                    viz_status.style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                                            except Exception as e:
                                                viz_status.text = f'Error: {e}'
                                        
                                        # Update all plots
                                        try:
                                            # Update brain plot using stored reference
                                            if 'update_brain_plot' in viz_state:
                                                viz_state['update_brain_plot']()
                                            update_network_plot()
                                            update_kuramoto_timeline()
                                            update_band_comparison()
                                            update_phase_plot()
                                            update_sync_matrix()
                                            update_connectivity()
                                            # Use stored references for Hilbert functions (defined later)
                                            if 'update_hilbert_2d' in viz_state:
                                                viz_state['update_hilbert_2d']()
                                            if 'update_hilbert_3d' in viz_state:
                                                viz_state['update_hilbert_3d']()
                                        except:
                                            pass  # Some functions might not be defined yet
                                    
                                    # Store reference for load_subjects_from_path to use
                                    viz_state['refresh_all_plots'] = refresh_all_plots
                                    
                                    # Connect selectors to auto-refresh
                                    viz_band.on('update:model-value', lambda e: refresh_all_plots())
                                    viz_subject.on('update:model-value', lambda e: refresh_all_plots())
                                    viz_epoch.on('update:model-value', lambda e: refresh_all_plots())
                            
                            # VISUALIZATION CONTENT - scroll area fills remaining space
                            with ui.scroll_area().classes('w-full flex-1').style('min-height: 0;'):
                                with ui.column().classes('w-full p-3 gap-3'):
                                    # MAIN VISUALIZATION AREA - 3D Brain + Stats
                                    with ui.row().classes('w-full gap-3'):
                                        # LEFT: 3D Brain Visualization
                                        with ui.card().classes('dark-card p-3').style('flex: 2; min-width: 400px;'):
                                            ui.label('▌3D BRAIN NETWORK').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                                            
                                            brain_plot_container = ui.column().classes('w-full')
                                            
                                            # Track current plot type for auto-refresh
                                            viz_state['current_brain_plot'] = 'network'
                                            
                                            def update_brain_plot(plot_type=None):
                                                if plot_type is None:
                                                    plot_type = viz_state.get('current_brain_plot', 'network')
                                                else:
                                                    viz_state['current_brain_plot'] = plot_type
                                                
                                                brain_plot_container.clear()
                                                try:
                                                    from viz_scripts import brain_3d
                                                    
                                                    # Use already loaded data from refresh_all_plots
                                                    data = viz_state.get('data')
                                                    
                                                    with brain_plot_container:
                                                        if plot_type == 'network':
                                                            fig = brain_3d.create_brain_network_figure()
                                                        elif plot_type == 'colored':
                                                            fig = brain_3d.create_colored_brain_figure()
                                                        elif plot_type == 'sync':
                                                            fig = brain_3d.create_sync_brain_figure(data, viz_band.value, int(viz_epoch.value or 0))
                                                        elif plot_type == 'all_bands':
                                                            fig = brain_3d.create_all_bands_brain_figure(data)
                                                        else:
                                                            fig = brain_3d.create_brain_network_figure()
                                                        
                                                        ui.plotly(fig).classes('w-full').style('height: 450px;')
                                                    
                                                    # Update button states
                                                    update_brain_buttons(plot_type)
                                                except Exception as e:
                                                    with brain_plot_container:
                                                        ui.label(f'Error: {e}').style(f'color:{THEME_ERROR}; font-size: 0.75rem;')
                                                        import traceback
                                                        ui.label(traceback.format_exc()[:500]).style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; white-space: pre-wrap;')
                                            
                                            # Store reference for refresh_all_plots
                                            viz_state['update_brain_plot'] = update_brain_plot
                                        
                                        # Brain plot type buttons with active state
                                        brain_buttons = {}
                                        with ui.row().classes('gap-2 mb-2'):
                                            brain_buttons['network'] = ui.button('Networks', on_click=lambda: update_brain_plot('network')).props('dense')
                                            brain_buttons['colored'] = ui.button('Parcellation', on_click=lambda: update_brain_plot('colored')).props('dense')
                                            brain_buttons['sync'] = ui.button('Sync Map', on_click=lambda: update_brain_plot('sync')).props('dense')
                                            brain_buttons['all_bands'] = ui.button('All Bands', on_click=lambda: update_brain_plot('all_bands')).props('dense')
                                        
                                        def update_brain_buttons(active_type):
                                            for btn_type, btn in brain_buttons.items():
                                                if btn_type == active_type:
                                                    btn.style(f'background:{THEME_PRIMARY}; color:black;')
                                                else:
                                                    btn.style(f'background:transparent; color:{THEME_TEXT_DIM}; border: 1px solid {THEME_BORDER};')
                                        
                                        # Initial button state
                                        update_brain_buttons('network')
                                        
                                        # Load initial plot
                                        update_brain_plot('network')
                                    
                                    # RIGHT: Network Stats
                                    with ui.card().classes('dark-card p-3').style('flex: 1; min-width: 300px;'):
                                        ui.label('▌NETWORK SYNC').style(f'color:{THEME_SECONDARY}; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                                        
                                        network_plot_container = ui.column().classes('w-full')
                                        
                                        def update_network_plot():
                                            network_plot_container.clear()
                                            data = viz_state.get('data')
                                            if not data:
                                                with network_plot_container:
                                                    ui.label('Load data first').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem;')
                                                return
                                            try:
                                                from viz_scripts import brain_3d
                                                with network_plot_container:
                                                    fig = brain_3d.create_network_comparison_figure(data, viz_band.value)
                                                    ui.plotly(fig).classes('w-full').style('height: 350px;')
                                            except Exception as e:
                                                with network_plot_container:
                                                    ui.label(f'Error: {e}').style(f'color:{THEME_TEXT_DIM};')
                                        
                                        # Don't auto-load without data
                                        if viz_state.get('data'):
                                            update_network_plot()
                                
                                # SECOND ROW - Kuramoto Analysis
                                with ui.row().classes('w-full gap-3'):
                                    # Timeline
                                    with ui.card().classes('dark-card p-3 flex-1'):
                                        ui.label('▌KURAMOTO TIMELINE').style(f'color:{THEME_WARN}; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                                        
                                        kura_timeline_container = ui.column().classes('w-full')
                                        
                                        def update_kuramoto_timeline():
                                            kura_timeline_container.clear()
                                            data = viz_state.get('data')
                                            if not data:
                                                with kura_timeline_container:
                                                    ui.label('Load data first').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem;')
                                                return
                                            try:
                                                from viz_scripts import kuramoto_viz
                                                with kura_timeline_container:
                                                    fig = kuramoto_viz.create_timeline_figure(data, viz_band.value)
                                                    ui.plotly(fig).classes('w-full').style('height: 280px;')
                                            except Exception as e:
                                                with kura_timeline_container:
                                                    ui.label(f'Error: {e}').style(f'color:{THEME_TEXT_DIM};')
                                        
                                        def show_all_bands():
                                            kura_timeline_container.clear()
                                            try:
                                                from viz_scripts import kuramoto_viz
                                                data = viz_state.get('data')
                                                with kura_timeline_container:
                                                    fig = kuramoto_viz.create_all_bands_timeline(data)
                                                    ui.plotly(fig).classes('w-full').style('height: 280px;')
                                            except Exception as e:
                                                with kura_timeline_container:
                                                    ui.label(f'Error: {e}').style(f'color:{THEME_TEXT_DIM};')
                                        
                                        ui.button('All Bands', on_click=show_all_bands).props('dense flat size=sm').classes('mb-1')
                                        if viz_state.get('data'):
                                            update_kuramoto_timeline()
                                    
                                    # Band Comparison
                                    with ui.card().classes('dark-card p-3 flex-1'):
                                        ui.label('▌BAND COMPARISON').style(f'color:{THEME_SECONDARY}; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                                        
                                        band_comp_container = ui.column().classes('w-full')
                                        
                                        def update_band_comparison():
                                            band_comp_container.clear()
                                            data = viz_state.get('data')
                                            if not data:
                                                with band_comp_container:
                                                    ui.label('Load data first').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem;')
                                                return
                                            try:
                                                from viz_scripts import kuramoto_viz
                                                with band_comp_container:
                                                    fig = kuramoto_viz.create_band_comparison_figure(data)
                                                    ui.plotly(fig).classes('w-full').style('height: 280px;')
                                            except Exception as e:
                                                with band_comp_container:
                                                    ui.label(f'Error: {e}').style(f'color:{THEME_TEXT_DIM};')
                                        
                                        if viz_state.get('data'):
                                            update_band_comparison()
                                
                                # THIRD ROW - More Analysis
                                with ui.row().classes('w-full gap-3'):
                                    # Phase Distribution
                                    with ui.card().classes('dark-card p-3 flex-1'):
                                        ui.label('▌PHASE DISTRIBUTION').style('color:#a78bfa; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                                        
                                        phase_container = ui.column().classes('w-full')
                                        
                                        def update_phase_plot():
                                            phase_container.clear()
                                            data = viz_state.get('data')
                                            if not data:
                                                with phase_container:
                                                    ui.label('Load data first').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem;')
                                                return
                                            try:
                                                from viz_scripts import kuramoto_viz
                                                with phase_container:
                                                    fig = kuramoto_viz.create_phase_distribution_figure(data, viz_band.value, int(viz_epoch.value or 0))
                                                    ui.plotly(fig).classes('w-full').style('height: 280px;')
                                            except Exception as e:
                                                with phase_container:
                                                    ui.label(f'Error: {e}').style(f'color:{THEME_TEXT_DIM};')
                                        
                                        if viz_state.get('data'):
                                            update_phase_plot()
                                    
                                    # Sync Matrix
                                    with ui.card().classes('dark-card p-3 flex-1'):
                                        ui.label('▌SYNC MATRIX').style('color:#60a5fa; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                                        
                                        sync_matrix_container = ui.column().classes('w-full')
                                        
                                        def update_sync_matrix():
                                            sync_matrix_container.clear()
                                            data = viz_state.get('data')
                                            if not data:
                                                with sync_matrix_container:
                                                    ui.label('Load data first').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem;')
                                                return
                                            try:
                                                from viz_scripts import kuramoto_viz
                                                with sync_matrix_container:
                                                    fig = kuramoto_viz.create_heatmap_figure(data, viz_band.value, int(viz_epoch.value or 0))
                                                    ui.plotly(fig).classes('w-full').style('height: 280px;')
                                            except Exception as e:
                                                with sync_matrix_container:
                                                    ui.label(f'Error: {e}').style(f'color:{THEME_TEXT_DIM};')
                                        
                                        if viz_state.get('data'):
                                            update_sync_matrix()
                                    
                                    # Connectivity Graph
                                    with ui.card().classes('dark-card p-3 flex-1'):
                                        ui.label('▌ROI CONNECTIVITY').style('color:#22c55e; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                                        
                                        connectivity_container = ui.column().classes('w-full')
                                        conn_threshold = ui.slider(min=0.3, max=0.9, step=0.1, value=0.5).props('label-always').classes('w-full')
                                        
                                        def update_connectivity():
                                            connectivity_container.clear()
                                            data = viz_state.get('data')
                                            if not data:
                                                with connectivity_container:
                                                    ui.label('Load data first').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem;')
                                                return
                                            try:
                                                from viz_scripts import kuramoto_viz
                                                with connectivity_container:
                                                    fig = kuramoto_viz.create_roi_connectivity_figure(data, viz_band.value, conn_threshold.value)
                                                    ui.plotly(fig).classes('w-full').style('height: 250px;')
                                            except Exception as e:
                                                with connectivity_container:
                                                    ui.label(f'Error: {e}').style(f'color:{THEME_TEXT_DIM};')
                                        
                                        conn_threshold.on('update:model-value', lambda e: update_connectivity())
                                        if viz_state.get('data'):
                                            update_connectivity()
                                
                                # FOURTH ROW - Hilbert Transform Visualizations
                                with ui.row().classes('w-full gap-3'):
                                    # Hilbert 2D
                                    with ui.card().classes('dark-card p-3 flex-1'):
                                        ui.label('▌HILBERT 2D').style('color:#f472b6; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                                        
                                        hilbert_2d_container = ui.column().classes('w-full')
                                        
                                        def update_hilbert_2d():
                                            hilbert_2d_container.clear()
                                            data = viz_state.get('data')
                                            if not data:
                                                with hilbert_2d_container:
                                                    ui.label('Load data first').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem;')
                                                return
                                            try:
                                                from viz_scripts import kuramoto_viz
                                                with hilbert_2d_container:
                                                    fig = kuramoto_viz.create_hilbert_2d_figure(data, viz_band.value, int(viz_epoch.value or 0))
                                                    ui.plotly(fig).classes('w-full').style('height: 500px;')
                                            except Exception as e:
                                                with hilbert_2d_container:
                                                    ui.label(f'Error: {e}').style(f'color:{THEME_TEXT_DIM};')
                                        
                                        # Store reference for later updates
                                        viz_state['update_hilbert_2d'] = update_hilbert_2d
                                        if viz_state.get('data'):
                                            update_hilbert_2d()
                                    
                                    # Hilbert 3D
                                    with ui.card().classes('dark-card p-3 flex-1'):
                                        ui.label('▌HILBERT 3D PHASE SPACE').style('color:#c084fc; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                                        
                                        hilbert_3d_container = ui.column().classes('w-full')
                                        
                                        def update_hilbert_3d():
                                            hilbert_3d_container.clear()
                                            data = viz_state.get('data')
                                            if not data:
                                                with hilbert_3d_container:
                                                    ui.label('Load data first').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem;')
                                                return
                                            try:
                                                from viz_scripts import kuramoto_viz
                                                subj = viz_subject.value or 'S01'
                                                # Extract condition from subject (S01-DMT -> DMT)
                                                cond = 'DMT'
                                                if '-' in str(subj):
                                                    cond = str(subj).split('-')[-1]
                                                
                                                with hilbert_3d_container:
                                                    fig = kuramoto_viz.create_hilbert_3d_figure(
                                                        data, 
                                                        viz_band.value, 
                                                        int(viz_epoch.value or 0),
                                                        roi_idx=0,
                                                        subject=subj,
                                                        condition=cond
                                                    )
                                                    ui.plotly(fig).classes('w-full').style('height: 500px;')
                                            except Exception as e:
                                                with hilbert_3d_container:
                                                    ui.label(f'Error: {e}').style(f'color:{THEME_TEXT_DIM};')
                                        
                                        # Store reference for later updates
                                        viz_state['update_hilbert_3d'] = update_hilbert_3d
                                        if viz_state.get('data'):
                                            update_hilbert_3d()
                                
                                # FIFTH ROW - Clustering (if available)
                                with ui.expansion('CLUSTERING ANALYSIS', icon='analytics').classes('w-full').style(f'background:{THEME_CARD};'):
                                    with ui.row().classes('w-full gap-3 p-2'):
                                        # Clustering Scores
                                        with ui.card().classes('dark-card p-3 flex-1'):
                                            ui.label('▌CLUSTER SCORES').style('color:#ec4899; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                                            
                                            cluster_scores_container = ui.column().classes('w-full')
                                            
                                            def update_cluster_scores():
                                                cluster_scores_container.clear()
                                                try:
                                                    from viz_scripts import clustering_viz
                                                    if current_run_dir[0]:
                                                        with cluster_scores_container:
                                                            fig = clustering_viz.create_band_comparison_figure(current_run_dir[0])
                                                            ui.plotly(fig).classes('w-full').style('height: 280px;')
                                                    else:
                                                        with cluster_scores_container:
                                                            ui.label('Select a run first').style(f'color:{THEME_TEXT_DIM};')
                                                except Exception as e:
                                                    with cluster_scores_container:
                                                        ui.label(f'Error: {e}').style(f'color:{THEME_TEXT_DIM};')
                                            
                                            ui.button('Load', on_click=update_cluster_scores).props('dense flat size=sm').classes('mb-2')
                                        
                                        # PCA Scatter
                                        with ui.card().classes('dark-card p-3 flex-1'):
                                            ui.label('▌PCA CLUSTERS').style('color:#f97316; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                                            
                                            pca_container = ui.column().classes('w-full')
                                            
                                            def update_pca_scatter():
                                                pca_container.clear()
                                                try:
                                                    from viz_scripts import clustering_viz
                                                    if current_run_dir[0]:
                                                        with pca_container:
                                                            fig = clustering_viz.create_pca_scatter_figure(current_run_dir[0], viz_band.value)
                                                            ui.plotly(fig).classes('w-full').style('height: 280px;')
                                                    else:
                                                        with pca_container:
                                                            ui.label('Select a run first').style(f'color:{THEME_TEXT_DIM};')
                                                except Exception as e:
                                                    with pca_container:
                                                        ui.label(f'Error: {e}').style(f'color:{THEME_TEXT_DIM};')
                                            
                                            ui.button('Load', on_click=update_pca_scatter).props('dense flat size=sm').classes('mb-2')
                                
                                # FIFTH ROW - Pearson Results Gallery
                                with ui.expansion('PEARSON CORRELATIONS', icon='insights').classes('w-full').style(f'background:{THEME_CARD};'):
                                    with ui.column().classes('w-full p-2'):
                                        with ui.row().classes('gap-2 items-center mb-2'):
                                            prs_metric = ui.select(['All', 'Coherence', 'Metastability'], value='All', label='Metric').props('dense').classes('w-28')
                                            prs_cond = ui.select(['All', 'DMT', 'EC', 'EO'], value='All', label='Condition').props('dense').classes('w-20')
                                            prs_band = ui.select(['All', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'], value='All', label='Band').props('dense').classes('w-24')
                                        
                                        pearson_gallery = ui.column().classes('w-full')
                                        pearson_image_dialog = ui.dialog().classes('w-full max-w-4xl')
                                        
                                        def refresh_pearson_gallery():
                                            pearson_gallery.clear()
                                            if not current_run_dir[0]:
                                                with pearson_gallery:
                                                    ui.label('No run selected').style(f'color:{THEME_TEXT_DIM};')
                                                return
                                            
                                            pearson_dir = current_run_dir[0] / 'pearson_results'
                                            if not pearson_dir.exists():
                                                with pearson_gallery:
                                                    ui.label('No Pearson results. Run pearson.py first.').style(f'color:{THEME_TEXT_DIM};')
                                                return
                                            
                                            all_files = list(pearson_dir.glob('*.png')) + list(pearson_dir.glob('*.svg'))
                                            filtered = [f for f in all_files if 
                                                (prs_metric.value == 'All' or prs_metric.value.lower() in f.name.lower()) and
                                                (prs_cond.value == 'All' or prs_cond.value.lower() in f.name.lower()) and
                                                (prs_band.value == 'All' or prs_band.value.lower() in f.name.lower())]
                                            
                                            with pearson_gallery:
                                                ui.label(f'{len(filtered)} images').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                                                with ui.element('div').classes('grid gap-3 mt-2').style('grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));'):
                                                    for img_file in sorted(filtered)[:30]:
                                                        def show_full_image(p=img_file):
                                                            pearson_image_dialog.clear()
                                                            with pearson_image_dialog:
                                                                with ui.column().classes('w-full items-center'):
                                                                    ui.label(p.name).style(f'color:{THEME_PRIMARY}; font-size: 0.9rem; margin-bottom: 10px;')
                                                                    if p.suffix == '.png':
                                                                        ui.image(str(p)).classes('w-full').style('max-height: 70vh;')
                                                                    else:
                                                                        ui.html(f'<object data="{p}" type="image/svg+xml" style="width:100%; max-height: 70vh;"></object>', sanitize=False)
                                                                    ui.button('Close', on_click=pearson_image_dialog.close).props('flat').classes('mt-3')
                                                            pearson_image_dialog.open()
                                                        
                                                        with ui.card().classes('cursor-pointer p-2').style(f'background:{THEME_CARD};').on('click', show_full_image):
                                                            ui.label(img_file.stem[:30] + ('...' if len(img_file.stem) > 30 else '')).style(f'color:{THEME_TEXT}; font-size: 0.65rem;')
                                        
                                        for sel in [prs_metric, prs_cond, prs_band]:
                                            sel.on('update:model-value', lambda e: refresh_pearson_gallery())
                                        
                                        ui.button('Load Images', on_click=refresh_pearson_gallery, icon='refresh').props('dense flat').classes('mt-2')
                                
                                # SEVENTH ROW - Animation/Frame Generator
                                with ui.expansion('ANIMATION GENERATOR', icon='movie').classes('w-full').style(f'background:{THEME_CARD};'):
                                    with ui.column().classes('w-full p-3 gap-3'):
                                        ui.label('Generate Kuramoto visualization frames and animations').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem;')
                                        
                                        with ui.row().classes('gap-4 items-center flex-wrap'):
                                            anim_mode = ui.select(['stc', 'eeg', 'all', 'advanced'], value='stc', label='Mode').props('dense').classes('w-28')
                                            anim_quality = ui.select(['high', 'medium', 'low'], value='medium', label='Quality').props('dense').classes('w-24')
                                            anim_format = ui.select(['png', 'jpg'], value='png', label='Format').props('dense').classes('w-20')
                                            anim_start_epoch = ui.number(value=0, min=0, max=100, label='Start').props('dense').classes('w-20')
                                            anim_end_epoch = ui.number(value=10, min=1, max=200, label='End').props('dense').classes('w-20')
                                            anim_fps = ui.number(value=5, min=1, max=30, label='FPS').props('dense').classes('w-16')
                                        
                                        with ui.row().classes('gap-4 items-center'):
                                            anim_output_dir = ui.input(value='', placeholder='/path/to/output or auto').props('dense').classes('flex-1')
                                            ui.label('Output dir (leave empty for auto)').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                                        
                                        anim_log = ui.column().classes('w-full').style('max-height: 800px; overflow-y: auto; background: #050505; border-radius: 4px; padding: 8px;')
                                        
                                        async def generate_frames():
                                            """Generate visualization frames"""
                                            anim_log.clear()
                                            with anim_log:
                                                ui.label('Starting frame generation...').style(f'color:{THEME_PRIMARY}; font-size: 0.7rem;')
                                            
                                            data_path = viz_state.get('custom_path') or current_run_dir[0]
                                            if not data_path or not viz_subject.value:
                                                with anim_log:
                                                    ui.label('Error: No data loaded. Load a subject first.').style(f'color:{THEME_ERROR}; font-size: 0.7rem;')
                                                return
                                            
                                            # Build command
                                            from pathlib import Path
                                            
                                            # Use plot.py directly (generate_frames.py is deprecated)
                                            script_path = Path('/media/storage_hdd/dmt_fz/viz_scripts/plot.py')
                                            
                                            # Use conda environment python
                                            import os
                                            conda_prefix = os.environ.get('CONDA_PREFIX', os.path.expanduser('~/anaconda3/envs/dmt_fz'))
                                            python_path = Path(conda_prefix) / 'bin' / 'python'
                                            if not python_path.exists():
                                                python_path = 'python'  # Fallback
                                            
                                            # Extract condition from subject if present (e.g., S01-DMT -> DMT)
                                            subj = viz_subject.value or 'S01'
                                            cond = 'DMT'
                                            if '-' in str(subj):
                                                parts = str(subj).split('-')
                                                subj = parts[0]
                                                cond = parts[1] if len(parts) > 1 else 'DMT'
                                            
                                            # Map mode names
                                            mode_map = {'stc': 'stc', 'eeg': 'eeg', 'all': 'all', 'advanced': 'advanced'}
                                            mode = mode_map.get(anim_mode.value, 'stc')
                                            
                                            # Build epochs range
                                            start_ep = int(anim_start_epoch.value or 0)
                                            end_ep = int(anim_end_epoch.value or 10)
                                            
                                            cmd = [
                                                str(python_path), str(script_path),
                                                '--subject', subj,
                                                '--condition', cond,
                                                '--band', viz_band.value,
                                                '--mode', mode,
                                                '--epochs', f'{start_ep}:{end_ep}',
                                            ]
                                            
                                            # plot.py saves frames to: visualizations/plot/{mode}/{subj}_{cond}_{band}
                                            band = viz_band.value or 'Alpha'
                                            expected_output = Path('/media/storage_hdd/dmt_fz/visualizations/plot') / mode / f'{subj}_{cond}_{band}'
                                            
                                            with anim_log:
                                                ui.label(f'Command: {" ".join(cmd)}').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                                                ui.label(f'Frames will be saved to: {expected_output}').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                                            
                                            try:
                                                import os
                                                env = os.environ.copy()
                                                env['PIPELINE_OUTPUT_DIR'] = str(data_path)
                                                # Add required paths for viz_scripts modules
                                                base_path = Path('/media/storage_hdd/dmt_fz')
                                                pythonpath = [
                                                    str(base_path / 'viz_scripts'),
                                                    str(base_path / 'pipeline'),
                                                    str(base_path),
                                                ]
                                                existing_pythonpath = env.get('PYTHONPATH', '')
                                                env['PYTHONPATH'] = ':'.join(pythonpath) + (':' + existing_pythonpath if existing_pythonpath else '')
                                                
                                                process = await asyncio.create_subprocess_exec(
                                                    *cmd,
                                                    stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE,
                                                    env=env
                                                )
                                                
                                                async def read_output(stream):
                                                    while True:
                                                        line = await stream.readline()
                                                        if not line:
                                                            break
                                                        text = line.decode().strip()
                                                        if text:
                                                            with anim_log:
                                                                ui.label(text).style(f'color:{THEME_TEXT}; font-size: 0.65rem;')
                                                
                                                await asyncio.gather(
                                                    read_output(process.stdout),
                                                    read_output(process.stderr)
                                                )
                                                
                                                await process.wait()
                                                
                                                with anim_log:
                                                    if process.returncode == 0:
                                                        ui.label('✓ Frames generated successfully!').style(f'color:{THEME_PRIMARY}; font-size: 0.75rem;')
                                                        # Show preview of first frame
                                                        frames = sorted(expected_output.glob(f'*.{anim_format.value}'))
                                                        if frames:
                                                            ui.label(f'Generated {len(frames)} frames').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                                                            # Show first frame as preview
                                                            with ui.card().classes('mt-2 p-2').style('background: #1a1a1a;'):
                                                                ui.label('Preview (first frame):').style(f'color:{THEME_TEXT_DIM}; font-size: 0.6rem;')
                                                                ui.image(str(frames[0])).classes('w-full').style('max-height: 1024px; max-width: 1024px; object-fit: contain;')
                                                    else:
                                                        ui.label(f'Process exited with code {process.returncode}').style(f'color:{THEME_WARN}; font-size: 0.7rem;')
                                            except Exception as e:
                                                with anim_log:
                                                    ui.label(f'Error: {e}').style(f'color:{THEME_ERROR}; font-size: 0.7rem;')
                                        
                                        async def generate_video():
                                            """Generate video from frames"""
                                            anim_log.clear()
                                            with anim_log:
                                                ui.label('🎬 Generating video from frames...').style(f'color:{THEME_PRIMARY}; font-size: 0.7rem;')
                                            
                                            from pathlib import Path
                                            
                                            # Build the correct frames path based on plot.py output structure
                                            # plot.py saves to: visualizations/plot/{mode}/{subject}_{condition}_{band}
                                            subj = viz_subject.value or 'S01'
                                            cond = 'DMT'
                                            if '-' in str(subj):
                                                parts = str(subj).split('-')
                                                subj = parts[0]
                                                cond = parts[1] if len(parts) > 1 else 'DMT'
                                            
                                            mode = anim_mode.value or 'stc'
                                            band = viz_band.value or 'Alpha'
                                            
                                            # Check custom path first, then default visualizations path
                                            if anim_output_dir.value and anim_output_dir.value.strip():
                                                frames_dir = Path(anim_output_dir.value.strip())
                                            else:
                                                frames_dir = Path('/media/storage_hdd/dmt_fz/visualizations/plot') / mode / f'{subj}_{cond}_{band}'
                                            
                                            with anim_log:
                                                ui.label(f'Looking for frames in: {frames_dir}').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                                            
                                            if not frames_dir.exists():
                                                with anim_log:
                                                    ui.label(f'Frames directory not found: {frames_dir}').style(f'color:{THEME_ERROR}; font-size: 0.7rem;')
                                                return
                                            
                                            output_video = frames_dir.parent / f'{viz_subject.value}_{viz_band.value}_animation.mp4'
                                            
                                            # Use ffmpeg to create video
                                            cmd = [
                                                'ffmpeg', '-y',
                                                '-framerate', str(int(anim_fps.value or 5)),
                                                '-pattern_type', 'glob',
                                                '-i', str(frames_dir / f'*.{anim_format.value}'),
                                                '-c:v', 'libx264',
                                                '-pix_fmt', 'yuv420p',
                                                str(output_video)
                                            ]
                                            
                                            with anim_log:
                                                ui.label(f'Running: ffmpeg -> {output_video.name}').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                                            
                                            try:
                                                process = await asyncio.create_subprocess_exec(
                                                    *cmd,
                                                    stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE
                                                )
                                                _, stderr = await process.communicate()
                                                
                                                with anim_log:
                                                    if process.returncode == 0:
                                                        ui.label(f'✓ Video saved: {output_video}').style(f'color:{THEME_PRIMARY}; font-size: 0.75rem;')
                                                        # Show video player
                                                        if output_video.exists():
                                                            with ui.card().classes('mt-2 p-2 w-full').style('background: #1a1a1a;'):
                                                                ui.label('Generated video:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.6rem;')
                                                                ui.video(str(output_video)).classes('w-full').style('max-height: 1024px; max-width: 1024px;')
                                                    else:
                                                        ui.label(f'ffmpeg error: {stderr.decode()[:200]}').style(f'color:{THEME_ERROR}; font-size: 0.65rem;')
                                            except Exception as e:
                                                with anim_log:
                                                    ui.label(f'Error: {e}').style(f'color:{THEME_ERROR}; font-size: 0.7rem;')
                                        
                                        with ui.row().classes('gap-2'):
                                            ui.button('Generate Frames', on_click=generate_frames, icon='photo_library').props('dense').style(f'background:{THEME_PRIMARY}; color:black;')
                                            ui.button('Create Video', on_click=generate_video, icon='movie').props('dense').style(f'background:{THEME_SECONDARY}; color:black;')
                                            
                                            def clear_log():
                                                anim_log.clear()
                                            ui.button('Clear Log', on_click=clear_log, icon='delete').props('dense flat')


# =============================================================================
# MODEL TRAINING PAGE
# =============================================================================

# ModelState (MS) and AnalysisState (AS) imported from app.state

# Autoencoder paths
AUTOENCODER_DIR = Path(__file__).parent.parent / "machine_learning" / "autoencoder"
AUTOENCODER_CACHE_DIR = Path(__file__).parent / "cache" / "autoencoder"


