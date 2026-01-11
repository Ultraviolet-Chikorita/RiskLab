"""
Resource management for evaluator agents.

Evaluators must allocate limited resources to decide which checks to perform.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from enum import Enum


class ResourceType(str, Enum):
    """Types of resources evaluators can consume."""
    TOKENS = "tokens"
    TIME_MS = "time_ms"
    TOOL_CALLS = "tool_calls"
    LLM_CALLS = "llm_calls"


class ResourceBudget(BaseModel):
    """Budget allocation for an evaluator."""
    max_tokens: int = Field(default=4096, ge=100)
    max_time_ms: int = Field(default=30000, ge=1000)
    max_tool_calls: int = Field(default=5, ge=1)
    max_llm_calls: int = Field(default=3, ge=1)
    
    def to_dict(self) -> Dict[str, int]:
        return {
            ResourceType.TOKENS.value: self.max_tokens,
            ResourceType.TIME_MS.value: self.max_time_ms,
            ResourceType.TOOL_CALLS.value: self.max_tool_calls,
            ResourceType.LLM_CALLS.value: self.max_llm_calls,
        }


class ResourceUsage(BaseModel):
    """Tracked resource usage."""
    tokens_used: int = 0
    time_ms_used: int = 0
    tool_calls_used: int = 0
    llm_calls_used: int = 0
    
    def to_dict(self) -> Dict[str, int]:
        return {
            ResourceType.TOKENS.value: self.tokens_used,
            ResourceType.TIME_MS.value: self.time_ms_used,
            ResourceType.TOOL_CALLS.value: self.tool_calls_used,
            ResourceType.LLM_CALLS.value: self.llm_calls_used,
        }


class ResourceTracker:
    """
    Tracks resource consumption for evaluator agents.
    
    Induces trade-offs that expose evaluator bias and systematic blind spots.
    """
    
    def __init__(self, budget: ResourceBudget):
        self.budget = budget
        self.usage = ResourceUsage()
        self.allocation_log: List[Dict[str, Any]] = []
    
    def can_spend(self, resource_type: ResourceType, amount: int) -> bool:
        """Check if spending is within budget."""
        current = self._get_current(resource_type)
        limit = self._get_limit(resource_type)
        return current + amount <= limit
    
    def spend(
        self, 
        resource_type: ResourceType, 
        amount: int,
        action: Optional[str] = None,
    ) -> bool:
        """
        Attempt to spend resources. Returns True if successful.
        
        Logs the allocation for later analysis.
        """
        if not self.can_spend(resource_type, amount):
            self.allocation_log.append({
                "action": action or "unknown",
                "resource": resource_type.value,
                "requested": amount,
                "granted": False,
                "reason": "budget_exceeded",
            })
            return False
        
        self._add_usage(resource_type, amount)
        self.allocation_log.append({
            "action": action or "unknown",
            "resource": resource_type.value,
            "requested": amount,
            "granted": True,
            "remaining": self.remaining(resource_type),
        })
        return True
    
    def remaining(self, resource_type: ResourceType) -> int:
        """Get remaining budget for a resource type."""
        return self._get_limit(resource_type) - self._get_current(resource_type)
    
    def utilization(self, resource_type: ResourceType) -> float:
        """Get utilization ratio (0-1) for a resource type."""
        limit = self._get_limit(resource_type)
        if limit == 0:
            return 0.0
        return self._get_current(resource_type) / limit
    
    def overall_utilization(self) -> float:
        """Get average utilization across all resource types."""
        utils = [
            self.utilization(rt) 
            for rt in ResourceType
        ]
        return sum(utils) / len(utils)
    
    def is_exhausted(self) -> bool:
        """Check if any critical resource is exhausted."""
        return (
            self.remaining(ResourceType.TOKENS) < 100 or
            self.remaining(ResourceType.LLM_CALLS) == 0 or
            self.remaining(ResourceType.TOOL_CALLS) == 0
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of resource usage."""
        return {
            "budget": self.budget.to_dict(),
            "usage": self.usage.to_dict(),
            "utilization": {
                rt.value: self.utilization(rt)
                for rt in ResourceType
            },
            "overall_utilization": self.overall_utilization(),
            "is_exhausted": self.is_exhausted(),
            "allocation_count": len(self.allocation_log),
        }
    
    def get_allocation_analysis(self) -> Dict[str, Any]:
        """Analyze allocation patterns for bias detection."""
        if not self.allocation_log:
            return {"allocations": 0}
        
        # Group by action type
        action_usage: Dict[str, Dict[str, int]] = {}
        for entry in self.allocation_log:
            action = entry["action"]
            if action not in action_usage:
                action_usage[action] = {"count": 0, "granted": 0, "denied": 0}
            action_usage[action]["count"] += 1
            if entry["granted"]:
                action_usage[action]["granted"] += 1
            else:
                action_usage[action]["denied"] += 1
        
        # Identify potential biases
        biases = []
        total_allocations = len(self.allocation_log)
        for action, stats in action_usage.items():
            if stats["denied"] > stats["granted"]:
                biases.append({
                    "action": action,
                    "issue": "frequently_denied",
                    "denial_rate": stats["denied"] / stats["count"],
                })
        
        return {
            "allocations": total_allocations,
            "by_action": action_usage,
            "potential_biases": biases,
        }
    
    def _get_current(self, resource_type: ResourceType) -> int:
        mapping = {
            ResourceType.TOKENS: self.usage.tokens_used,
            ResourceType.TIME_MS: self.usage.time_ms_used,
            ResourceType.TOOL_CALLS: self.usage.tool_calls_used,
            ResourceType.LLM_CALLS: self.usage.llm_calls_used,
        }
        return mapping[resource_type]
    
    def _get_limit(self, resource_type: ResourceType) -> int:
        mapping = {
            ResourceType.TOKENS: self.budget.max_tokens,
            ResourceType.TIME_MS: self.budget.max_time_ms,
            ResourceType.TOOL_CALLS: self.budget.max_tool_calls,
            ResourceType.LLM_CALLS: self.budget.max_llm_calls,
        }
        return mapping[resource_type]
    
    def _add_usage(self, resource_type: ResourceType, amount: int) -> None:
        if resource_type == ResourceType.TOKENS:
            self.usage.tokens_used += amount
        elif resource_type == ResourceType.TIME_MS:
            self.usage.time_ms_used += amount
        elif resource_type == ResourceType.TOOL_CALLS:
            self.usage.tool_calls_used += amount
        elif resource_type == ResourceType.LLM_CALLS:
            self.usage.llm_calls_used += amount
