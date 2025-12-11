"""
Step 7: Correlations (pearson.py)

Computes Pearson correlations with questionnaire data.
"""
from .base import BasePipelineStep


class Step7Pearson(BasePipelineStep):
    """Correlation analysis step using pearson.py."""
    
    script_name = "pearson.py"
    display_name = "Correlations"
    title = "// STEP_7: CORRELATIONS"
    description = "pearson.py - Correlate with questionnaires"
    output_pattern = "-> pearson_results/"
    color = "#a78bfa"  # Purple
    estimated_time = "~3-5 min"
    
    def build_args(self) -> list[str]:
        """Build arguments for pearson.py (no extra args needed)."""
        return []


