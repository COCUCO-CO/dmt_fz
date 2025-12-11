"""
Step 1: Source Localization (fwd.py)

Computes forward/inverse solutions and extracts phase information.
"""
from .base import BasePipelineStep


class Step1Fwd(BasePipelineStep):
    """Source localization step using fwd.py."""
    
    script_name = "fwd.py"
    display_name = "Source Localization"
    title = "// STEP_1: SOURCE_LOCALIZATION"
    description = "fwd.py - Forward/Inverse Solution + Metrics"
    output_pattern = "→ run_*/phases-{subj}.pkl"
    color = "#00ff88"  # THEME_PRIMARY
    estimated_time = "~3-4h (o menos con max_epochs)"
    
    def build_args(self) -> list[str]:
        """Build arguments for fwd.py."""
        ctx = self.context
        return [
            '--max-subjects', str(ctx.max_subjects),
            '--conditions', *ctx.conditions,
            '--jobs', str(ctx.jobs),
            '--workers', str(ctx.workers),
            '--max-epochs', str(ctx.max_epochs),
        ]


