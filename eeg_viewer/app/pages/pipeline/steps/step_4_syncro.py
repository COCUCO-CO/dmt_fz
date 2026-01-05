"""
Step 4: Sync Metrics (calculate_syncro.py)

Calculates synchronization matrices and Kuramoto metrics.
"""
from .base import BasePipelineStep


class Step4Syncro(BasePipelineStep):
    """Synchronization metrics step using calculate_syncro.py."""
    
    script_name = "calculate_syncro.py"
    display_name = "Sync Metrics"
    title = "// STEP_4: SYNC_METRICS"
    description = "calculate_syncro.py - Calculate sync matrices & Kuramoto"
    output_pattern = "→ run_*/{cond}/syncro-{subj}.pkl"
    color = "#10b981"  # Emerald
    estimated_time = "~10-30 min"
    
    def build_args(self) -> list[str]:
        """Build arguments for calculate_syncro.py."""
        ctx = self.context
        return [
            '--workers', str(ctx.workers),
            '--conditions', *ctx.conditions,
        ]








