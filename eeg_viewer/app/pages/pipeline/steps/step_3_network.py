"""
Step 3: Network Filtering (multi2pool2.py)

Filters phase data by brain networks (DMN, FPN, etc).
"""
from .base import BasePipelineStep


class Step3Network(BasePipelineStep):
    """Network filtering step using multi2pool2.py."""
    
    script_name = "multi2pool2.py"
    display_name = "Network Filtering"
    title = "// STEP_3: NETWORK_FILTERING"
    description = "multi2pool2.py - Filter by brain networks (DMN, FPN, etc)"
    output_pattern = "→ run_*/order_all-{subj}.pkl"
    color = "#00d4ff"  # THEME_SECONDARY
    estimated_time = "~2-5 min"
    
    def build_args(self) -> list[str]:
        """Build arguments for multi2pool2.py (no extra args needed)."""
        return []





