"""
evaluation/judges/__init__.py
所有 Judge 的统一入口

用法：
    from evaluation.judges import ALL_JUDGES
    for judge_fn in ALL_JUDGES:
        result = judge_fn(question=..., answer=..., diary=..., contexts=...)
"""

from .empathy_judge      import judge as empathy_judge
from .diary_judge        import judge as diary_judge
from .theory_bridge_judge import judge as theory_bridge_judge
from .actionability_judge import judge as actionability_judge
from .structure_judge    import judge as structure_judge
from .safety_judge       import judge as safety_judge, infer_scene_type

ALL_JUDGES = [
    empathy_judge,
    diary_judge,
    theory_bridge_judge,
    actionability_judge,
    structure_judge,
    safety_judge,
]

JUDGE_NAMES = [
    "empathy",
    "diary_integration",
    "theory_bridge",
    "actionability",
    "structure",
    "safety",
]

__all__ = [
    "empathy_judge",
    "diary_judge",
    "theory_bridge_judge",
    "actionability_judge",
    "structure_judge",
    "safety_judge",
    "infer_scene_type",
    "ALL_JUDGES",
    "JUDGE_NAMES",
]
