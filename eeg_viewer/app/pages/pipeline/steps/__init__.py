"""
Pipeline steps module.

Each step follows the Strategy pattern:
- Common interface via BasePipelineStep
- Each step encapsulates its own argument building logic
- Steps can be added/modified without changing the orchestrator

Exports all step classes and the registry for dynamic access.
"""

from .base import BasePipelineStep, StepContext
from .step_1_fwd import Step1Fwd
from .step_2_consolidate import Step2Consolidate
from .step_3_network import Step3Network
from .step_4_syncro import Step4Syncro
from .step_5_order import Step5Order
from .step_6_aggregate import Step6Aggregate
from .step_7_pearson import Step7Pearson
from .step_8_clustering import Step8Clustering


# Registry of all available steps (ordered)
STEPS = [
    Step1Fwd,
    Step2Consolidate,
    Step3Network,
    Step4Syncro,
    Step5Order,
    Step6Aggregate,
    Step7Pearson,
    Step8Clustering,
]


def get_step_by_script(script_name: str) -> type[BasePipelineStep]:
    """Get step class by script name."""
    for step_cls in STEPS:
        if step_cls.script_name == script_name:
            return step_cls
    return None


__all__ = [
    'BasePipelineStep',
    'StepContext',
    'Step1Fwd',
    'Step2Consolidate',
    'Step3Network',
    'Step4Syncro',
    'Step5Order',
    'Step6Aggregate',
    'Step7Pearson',
    'Step8Clustering',
    'STEPS',
    'get_step_by_script',
]

