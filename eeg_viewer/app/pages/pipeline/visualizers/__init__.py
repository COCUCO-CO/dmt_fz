"""
Pipeline visualizers module.

Each step has its own visualizer for contextual data display.
"""

from .base import BaseVisualizer, extract_event_value
from .step_1_viz import Step1Visualizer
from .step_2_viz import Step2Visualizer
from .step_3_viz import Step3Visualizer
from .step_4_viz import Step4Visualizer
from .step_5_viz import Step5Visualizer
from .step_6_viz import Step6Visualizer
from .step_7_viz import Step7Visualizer
from .step_8_viz import Step8Visualizer

# Registry mapping step number to visualizer class
VISUALIZER_REGISTRY = {
    1: Step1Visualizer,
    2: Step2Visualizer,
    3: Step3Visualizer,
    4: Step4Visualizer,
    5: Step5Visualizer,
    6: Step6Visualizer,
    7: Step7Visualizer,
    8: Step8Visualizer,
}


def get_visualizer(step_number: int, get_run_dir):
    """
    Get visualizer instance for a step.
    
    Args:
        step_number: Pipeline step number (1-8)
        get_run_dir: Function that returns current run directory
        
    Returns:
        Visualizer instance or None if step not found
    """
    viz_class = VISUALIZER_REGISTRY.get(step_number)
    if viz_class:
        return viz_class(get_run_dir)
    return None


__all__ = [
    'BaseVisualizer',
    'VISUALIZER_REGISTRY',
    'get_visualizer',
    'Step1Visualizer',
    'Step2Visualizer', 
    'Step3Visualizer',
    'Step4Visualizer',
    'Step5Visualizer',
    'Step6Visualizer',
    'Step7Visualizer',
    'Step8Visualizer',
]

