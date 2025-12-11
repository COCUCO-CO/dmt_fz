"""
Step 2: Consolidate Phases (save_load_pickle.py)

Merges individual phases-*.pkl files into condition-specific files.
"""
from .base import BasePipelineStep


class Step2Consolidate(BasePipelineStep):
    """Phase consolidation step using save_load_pickle.py."""
    
    script_name = "save_load_pickle.py"
    display_name = "Consolidate Phases"
    title = "// STEP_2: CONSOLIDATE_PHASES"
    description = "save_load_pickle.py - Merge phases-*.pkl into subject_phases_{cond}.pkl"
    output_pattern = "→ run_*/{cond}/subject_phases_{cond}.pkl"
    color = "#06b6d4"  # Cyan
    estimated_time = "~1-2 min"
    
    def build_args(self) -> list[str]:
        """Build arguments for save_load_pickle.py."""
        ctx = self.context
        return ['--conditions', *ctx.conditions]


