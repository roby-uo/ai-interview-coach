from .base import _execute_skill_with_tools
from .socratic import socratic_node
from .scaffold import scaffold_node
from .judge import judge_node
from .redirection import redirection_node
from .teaching import teaching_node

SKILL_NODE_MAP = {
    "socratic_probe": socratic_node,
    "scaffold_rescue": scaffold_node,
    "judge_strike": judge_node,
    "smart_redirection": redirection_node,
    "teaching_supplement": teaching_node
}

__all__ = [
    "_execute_skill_with_tools",
    "socratic_node",
    "scaffold_node",
    "judge_node",
    "redirection_node",
    "teaching_node",
    "SKILL_NODE_MAP"
]
