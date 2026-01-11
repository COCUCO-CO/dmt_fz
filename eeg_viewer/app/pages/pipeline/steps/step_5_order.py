"""
Step 5: Kuramoto Order (generate_order.py)

Calculates Kuramoto order parameter from synchronization data.
"""
from .base import BasePipelineStep


class Step5Order(BasePipelineStep):
    """Kuramoto order step using generate_order.py."""
    
    script_name = "generate_order.py"
    display_name = "Kuramoto Order"
    title = "// STEP_5: KURAMOTO_ORDER"
    description = "generate_order.py - Calculate Kuramoto order parameter"
    output_pattern = "→ run_*/order-{subj}.pkl"
    color = "#ffcc00"  # THEME_WARN
    estimated_time = "~1 min"
    
    def build_args(self) -> list[str]:
        """Build arguments for generate_order.py."""
        ctx = self.context
        return [
            '--workers', str(ctx.workers),
            '--conditions', *ctx.conditions,
        ]









