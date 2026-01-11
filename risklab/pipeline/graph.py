"""
Component Graph Abstraction.

Provides a formal pipeline representation where:
- Nodes = components (classifier, retriever, generator, validator)
- Edges = data flow between components

Each node logs inputs, outputs, confidence, cost, and decision impact.
"""

from typing import Optional, List, Dict, Any, Set, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid
import json

if TYPE_CHECKING:
    from risklab.pipeline.components import Component, ComponentOutput


class EdgeType(str, Enum):
    """Types of edges in the component graph."""
    DATA_FLOW = "data_flow"           # Normal data passing
    GATE = "gate"                      # Conditional flow (can block)
    MODIFICATION = "modification"      # Transform data
    APPROVAL = "approval"              # Final approval/rejection
    FEEDBACK = "feedback"              # Feedback loop


@dataclass
class Edge:
    """
    An edge in the component graph representing data flow.
    """
    source_id: str
    target_id: str
    edge_type: EdgeType = EdgeType.DATA_FLOW
    condition: Optional[str] = None  # Optional condition for flow
    weight: float = 1.0  # Edge weight for importance
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.edge_type.value,
            "condition": self.condition,
            "weight": self.weight,
        }


@dataclass
class NodeExecution:
    """Record of a single node execution."""
    node_id: str
    component_type: str
    timestamp: datetime
    
    # Inputs and outputs
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    
    # Metrics
    confidence: float = 0.0
    cost: float = 0.0  # Computational/API cost
    latency_ms: float = 0.0
    
    # Decision impact
    decision_impact: str = "none"  # none, modified, gated, approved, rejected
    gate_passed: Optional[bool] = None
    modification_applied: Optional[str] = None
    
    # Error tracking
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "component_type": self.component_type,
            "timestamp": self.timestamp.isoformat(),
            "inputs": self._safe_serialize(self.inputs),
            "outputs": self._safe_serialize(self.outputs),
            "confidence": self.confidence,
            "cost": self.cost,
            "latency_ms": self.latency_ms,
            "decision_impact": self.decision_impact,
            "gate_passed": self.gate_passed,
            "error": self.error,
        }
    
    def _safe_serialize(self, obj: Any) -> Any:
        """Safely serialize objects for JSON."""
        if isinstance(obj, dict):
            return {k: self._safe_serialize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._safe_serialize(v) for v in obj]
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)


@dataclass
class ExecutionTrace:
    """
    Complete trace of a pipeline execution.
    
    Records all node executions, data flow, and decisions.
    """
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Execution records
    node_executions: List[NodeExecution] = field(default_factory=list)
    
    # Summary metrics
    total_cost: float = 0.0
    total_latency_ms: float = 0.0
    
    # Flow tracking
    gates_passed: List[str] = field(default_factory=list)
    gates_blocked: List[str] = field(default_factory=list)
    modifications: List[Dict[str, str]] = field(default_factory=list)
    
    # Final outcome
    final_output: Optional[Any] = None
    blocked: bool = False
    blocked_by: Optional[str] = None
    
    def add_execution(self, execution: NodeExecution) -> None:
        """Add a node execution to the trace."""
        self.node_executions.append(execution)
        self.total_cost += execution.cost
        self.total_latency_ms += execution.latency_ms
        
        if execution.gate_passed is not None:
            if execution.gate_passed:
                self.gates_passed.append(execution.node_id)
            else:
                self.gates_blocked.append(execution.node_id)
                self.blocked = True
                self.blocked_by = execution.node_id
        
        if execution.modification_applied:
            self.modifications.append({
                "node_id": execution.node_id,
                "modification": execution.modification_applied,
            })
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "node_executions": [e.to_dict() for e in self.node_executions],
            "total_cost": self.total_cost,
            "total_latency_ms": self.total_latency_ms,
            "gates_passed": self.gates_passed,
            "gates_blocked": self.gates_blocked,
            "modifications": self.modifications,
            "blocked": self.blocked,
            "blocked_by": self.blocked_by,
        }
    
    def get_node_execution(self, node_id: str) -> Optional[NodeExecution]:
        """Get execution record for a specific node."""
        for exec in self.node_executions:
            if exec.node_id == node_id:
                return exec
        return None


@dataclass
class ComponentNode:
    """
    A node in the component graph.
    
    Wraps a component with graph-specific metadata.
    """
    node_id: str
    component: "Component"
    
    # Graph position
    position: int = 0  # Execution order hint
    
    # Configuration
    enabled: bool = True
    budget_limit: Optional[float] = None  # Max cost allowed
    timeout_ms: Optional[float] = None    # Max latency allowed
    
    # Edges
    incoming_edges: List[str] = field(default_factory=list)
    outgoing_edges: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "component_type": self.component.component_type.value,
            "component_role": self.component.role.value,
            "position": self.position,
            "enabled": self.enabled,
            "budget_limit": self.budget_limit,
            "incoming_edges": self.incoming_edges,
            "outgoing_edges": self.outgoing_edges,
        }


@dataclass
class PipelineResult:
    """Result of running a pipeline on an input."""
    
    trace: ExecutionTrace
    
    # Final output
    output: Optional[Any] = None
    blocked: bool = False
    
    # Component-level scores
    component_scores: Dict[str, float] = field(default_factory=dict)
    
    # Aggregate metrics
    aggregate_risk: float = 0.0
    aggregate_confidence: float = 0.0
    
    # Issues detected
    issues: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace": self.trace.to_dict(),
            "output": str(self.output) if self.output else None,
            "blocked": self.blocked,
            "component_scores": self.component_scores,
            "aggregate_risk": self.aggregate_risk,
            "aggregate_confidence": self.aggregate_confidence,
            "issues": self.issues,
        }


class ComponentGraph:
    """
    A directed graph of components representing an AI pipeline.
    
    This is a first-class artifact in the Lab that can be:
    - Visualized
    - Audited
    - Run with budget constraints
    - Compared across framings
    """
    
    def __init__(self, name: str = "default_pipeline"):
        self.name = name
        self.graph_id = str(uuid.uuid4())[:8]
        self._nodes: Dict[str, ComponentNode] = {}
        self._edges: Dict[str, Edge] = {}
        self._execution_order: List[str] = []
        self._entry_nodes: List[str] = []
        self._exit_nodes: List[str] = []
    
    def add_node(
        self,
        node_id: str,
        component: "Component",
        position: Optional[int] = None,
        budget_limit: Optional[float] = None,
    ) -> "ComponentGraph":
        """Add a component node to the graph."""
        if position is None:
            position = len(self._nodes)
        
        node = ComponentNode(
            node_id=node_id,
            component=component,
            position=position,
            budget_limit=budget_limit,
        )
        self._nodes[node_id] = node
        self._update_execution_order()
        return self
    
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType = EdgeType.DATA_FLOW,
        condition: Optional[str] = None,
    ) -> "ComponentGraph":
        """Add an edge between nodes."""
        if source_id not in self._nodes:
            raise ValueError(f"Source node {source_id} not found")
        if target_id not in self._nodes:
            raise ValueError(f"Target node {target_id} not found")
        
        edge_id = f"{source_id}->{target_id}"
        edge = Edge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            condition=condition,
        )
        self._edges[edge_id] = edge
        
        # Update node references
        self._nodes[source_id].outgoing_edges.append(edge_id)
        self._nodes[target_id].incoming_edges.append(edge_id)
        
        self._update_execution_order()
        return self
    
    def get_node(self, node_id: str) -> Optional[ComponentNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)
    
    def get_nodes(self) -> List[ComponentNode]:
        """Get all nodes in execution order."""
        return [self._nodes[nid] for nid in self._execution_order]
    
    def get_edges(self) -> List[Edge]:
        """Get all edges."""
        return list(self._edges.values())
    
    def get_predecessors(self, node_id: str) -> List[str]:
        """Get predecessor node IDs."""
        node = self._nodes.get(node_id)
        if not node:
            return []
        return [
            self._edges[eid].source_id 
            for eid in node.incoming_edges
        ]
    
    def get_successors(self, node_id: str) -> List[str]:
        """Get successor node IDs."""
        node = self._nodes.get(node_id)
        if not node:
            return []
        return [
            self._edges[eid].target_id 
            for eid in node.outgoing_edges
        ]
    
    def _update_execution_order(self) -> None:
        """Update execution order using topological sort."""
        # Find entry nodes (no incoming edges)
        self._entry_nodes = [
            nid for nid, node in self._nodes.items()
            if not node.incoming_edges
        ]
        
        # Find exit nodes (no outgoing edges)
        self._exit_nodes = [
            nid for nid, node in self._nodes.items()
            if not node.outgoing_edges
        ]
        
        # Topological sort
        visited: Set[str] = set()
        order: List[str] = []
        
        def visit(node_id: str) -> None:
            if node_id in visited:
                return
            visited.add(node_id)
            for succ in self.get_successors(node_id):
                visit(succ)
            order.append(node_id)
        
        for entry in self._entry_nodes:
            visit(entry)
        
        # Add any remaining nodes
        for nid in self._nodes:
            if nid not in visited:
                visit(nid)
        
        self._execution_order = list(reversed(order))
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary."""
        return {
            "name": self.name,
            "graph_id": self.graph_id,
            "nodes": [self._nodes[nid].to_dict() for nid in self._execution_order],
            "edges": [e.to_dict() for e in self._edges.values()],
            "entry_nodes": self._entry_nodes,
            "exit_nodes": self._exit_nodes,
            "execution_order": self._execution_order,
        }
    
    def to_mermaid(self) -> str:
        """Generate Mermaid diagram of the graph."""
        lines = ["graph TD"]
        
        # Add nodes with labels
        for nid, node in self._nodes.items():
            label = f"{node.component.name}"
            lines.append(f"    {nid}[{label}]")
        
        # Add edges
        for edge in self._edges.values():
            arrow = "-->" if edge.edge_type == EdgeType.DATA_FLOW else "-.->|{edge.edge_type.value}|"
            lines.append(f"    {edge.source_id} {arrow} {edge.target_id}")
        
        return "\n".join(lines)
    
    def __len__(self) -> int:
        return len(self._nodes)
    
    def __repr__(self) -> str:
        return f"ComponentGraph(name={self.name}, nodes={len(self._nodes)}, edges={len(self._edges)})"
