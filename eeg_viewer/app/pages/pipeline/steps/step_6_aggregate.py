"""
Step 6: Aggregate Data (build_order_data.py)

Aggregates Kuramoto metrics across subjects and conditions.
"""
from .base import BasePipelineStep


class Step6Aggregate(BasePipelineStep):
    """Data aggregation step using build_order_data.py."""
    
    script_name = "build_order_data.py"
    display_name = "Aggregate Data"
    title = "// STEP_6: AGGREGATE_DATA"
    description = "build_order_data.py - Aggregate Kuramoto metrics"
    output_pattern = "-> r_kuramoto_nets_*.pkl"
    color = "#60a5fa"  # Blue
    estimated_time = "~2-5 min"
    
    def build_args(self) -> list[str]:
        """Build arguments for build_order_data.py."""
        ctx = self.context
        return [
            '--build-all',
            '--workers', str(ctx.workers),
        ]

