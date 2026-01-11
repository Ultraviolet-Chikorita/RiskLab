"""
Core visualization plots for the Risk-Conditioned AI Evaluation Lab.

All visuals are generated deterministically from logged artifacts.
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


class FramingDeltaHeatmap:
    """
    Framing Delta Heatmaps showing behavioral changes across framings per scenario.
    """
    
    def __init__(self, figsize: tuple = (12, 8)):
        self.figsize = figsize
    
    def plot(
        self,
        delta_matrix: Dict[str, Dict[str, float]],
        title: str = "Framing Delta Heatmap",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot framing delta heatmap.
        
        Args:
            delta_matrix: Dict mapping framing pairs to metric deltas
                         e.g., {"neutral_vs_oversight": {"refusal": 0.3, ...}}
        """
        # Convert to 2D array
        framing_pairs = list(delta_matrix.keys())
        if not framing_pairs:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return fig
        
        metrics = list(delta_matrix[framing_pairs[0]].keys())
        
        data = np.array([
            [delta_matrix[pair].get(metric, 0.0) for metric in metrics]
            for pair in framing_pairs
        ])
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="light", as_cmap=True)
        im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
        
        # Labels
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_yticks(range(len(framing_pairs)))
        ax.set_yticklabels(framing_pairs)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Behavioral Delta')
        
        # Annotations
        for i in range(len(framing_pairs)):
            for j in range(len(metrics)):
                value = data[i, j]
                color = 'white' if abs(value) > 0.5 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', color=color, fontsize=8)
        
        ax.set_title(title)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_multi_episode(
        self,
        episode_deltas: Dict[str, Dict[str, Dict[str, float]]],
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Plot heatmaps for multiple episodes."""
        n_episodes = len(episode_deltas)
        if n_episodes == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            return fig
        
        cols = min(3, n_episodes)
        rows = (n_episodes + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_episodes == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (episode_name, delta_matrix) in enumerate(episode_deltas.items()):
            ax = axes[idx]
            
            if not delta_matrix:
                ax.text(0.5, 0.5, "No data", ha='center', va='center')
                ax.set_title(episode_name[:30])
                continue
            
            framing_pairs = list(delta_matrix.keys())
            metrics = list(delta_matrix[framing_pairs[0]].keys())
            
            data = np.array([
                [delta_matrix[pair].get(metric, 0.0) for metric in metrics]
                for pair in framing_pairs
            ])
            
            im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto', vmin=-1, vmax=1)
            ax.set_xticks(range(len(metrics)))
            ax.set_xticklabels([m[:8] for m in metrics], rotation=45, ha='right', fontsize=7)
            ax.set_yticks(range(len(framing_pairs)))
            ax.set_yticklabels([p[:15] for p in framing_pairs], fontsize=7)
            ax.set_title(episode_name[:30], fontsize=9)
        
        # Hide unused axes
        for idx in range(len(episode_deltas), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


class RiskScoreLadder:
    """
    Risk-Adjusted Score Ladders plotting scenarios against decision thresholds.
    """
    
    def __init__(self, figsize: tuple = (10, 8)):
        self.figsize = figsize
    
    def plot(
        self,
        scenarios: List[Dict[str, Any]],
        thresholds: Dict[str, float] = None,
        title: str = "Risk Score Ladder",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot risk score ladder.
        
        Args:
            scenarios: List of dicts with 'name' and 'risk_score' keys
            thresholds: Dict of threshold names to values
        """
        if thresholds is None:
            thresholds = {
                "acceptable": 0.25,
                "monitor": 0.40,
                "mitigated": 0.55,
                "escalate": 0.75,
                "block": 0.90,
            }
        
        # Sort by risk score
        scenarios = sorted(scenarios, key=lambda x: x.get('risk_score', 0), reverse=True)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot threshold zones
        colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad']
        zone_labels = ['Acceptable', 'Monitor', 'Mitigated', 'Escalate', 'Block']
        threshold_values = [0] + list(thresholds.values()) + [1.0]
        
        for i in range(len(threshold_values) - 1):
            ax.axhspan(threshold_values[i], threshold_values[i+1], 
                      alpha=0.2, color=colors[min(i, len(colors)-1)])
        
        # Plot horizontal threshold lines
        for name, value in thresholds.items():
            ax.axhline(y=value, color='gray', linestyle='--', alpha=0.5)
            ax.text(len(scenarios) + 0.5, value, name, va='center', fontsize=8)
        
        # Plot scenario scores
        names = [s.get('name', f'Scenario {i}')[:25] for i, s in enumerate(scenarios)]
        scores = [s.get('risk_score', 0) for s in scenarios]
        
        bars = ax.barh(range(len(scenarios)), scores, color=[
            colors[min(int(s * 5), 4)] for s in scores
        ])
        
        ax.set_yticks(range(len(scenarios)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Risk-Adjusted Score')
        ax.set_title(title)
        
        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + 0.02, i, f'{score:.2f}', va='center', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


class OversightGapPlot:
    """
    Oversight Gap Plots visualizing mismatch between risk and detection confidence.
    """
    
    def __init__(self, figsize: tuple = (10, 8)):
        self.figsize = figsize
    
    def plot(
        self,
        data_points: List[Dict[str, Any]],
        title: str = "Oversight Gap Analysis",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot oversight gap.
        
        Args:
            data_points: List of dicts with 'name', 'risk_score', 'evaluator_confidence'
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if not data_points:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return fig
        
        risks = [d.get('risk_score', 0) for d in data_points]
        confidences = [d.get('evaluator_confidence', 0.5) for d in data_points]
        names = [d.get('name', '')[:20] for d in data_points]
        
        # Calculate gap scores
        gaps = [r * c for r, c in zip(risks, confidences)]
        
        # Scatter plot
        scatter = ax.scatter(confidences, risks, c=gaps, cmap='RdYlGn_r', 
                            s=100, alpha=0.7, edgecolors='black')
        
        # Add labels
        for i, name in enumerate(names):
            ax.annotate(name, (confidences[i], risks[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=7)
        
        # Diagonal line (equal risk and confidence)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Equal line')
        
        # Danger zone (high risk, high confidence = might be missed)
        ax.fill_between([0.6, 1], [0.6, 0.6], [1, 1], alpha=0.1, color='red',
                        label='Oversight gap zone')
        
        ax.set_xlabel('Evaluator Confidence')
        ax.set_ylabel('Risk Score')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.legend(loc='lower right')
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Gap Score (Risk × Confidence)')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


class EvaluatorDisagreementMatrix:
    """
    Evaluator Disagreement Matrices highlighting uncertainty and institutional fragility.
    """
    
    def __init__(self, figsize: tuple = (10, 8)):
        self.figsize = figsize
    
    def plot(
        self,
        judge_assessments: List[Dict[str, Any]],
        title: str = "Evaluator Disagreement Matrix",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot disagreement matrix.
        
        Args:
            judge_assessments: List of dicts with 'judge_name', 'episode_name', 'risk_score'
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if not judge_assessments:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return fig
        
        # Organize data
        judges = list(set(a.get('judge_name', 'Unknown') for a in judge_assessments))
        episodes = list(set(a.get('episode_name', 'Unknown') for a in judge_assessments))
        
        # Create matrix
        matrix = np.zeros((len(judges), len(episodes)))
        for a in judge_assessments:
            j_idx = judges.index(a.get('judge_name', 'Unknown'))
            e_idx = episodes.index(a.get('episode_name', 'Unknown'))
            matrix[j_idx, e_idx] = a.get('risk_score', 0)
        
        # Compute disagreement (std across judges per episode)
        disagreement = np.std(matrix, axis=0)
        
        # Plot heatmap
        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(range(len(episodes)))
        ax.set_xticklabels([e[:15] for e in episodes], rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(judges)))
        ax.set_yticklabels(judges, fontsize=9)
        
        # Annotations
        for i in range(len(judges)):
            for j in range(len(episodes)):
                ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center', 
                       fontsize=8, color='white' if matrix[i,j] > 0.5 else 'black')
        
        ax.set_title(title)
        
        # Add disagreement bar at bottom
        ax2 = fig.add_axes([0.125, 0.02, 0.775, 0.03])
        ax2.bar(range(len(episodes)), disagreement, color='gray', alpha=0.7)
        ax2.set_xlim(-0.5, len(episodes) - 0.5)
        ax2.set_ylim(0, 0.5)
        ax2.set_ylabel('Std', fontsize=8)
        ax2.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        
        plt.colorbar(im, ax=ax, label='Risk Score')
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


class BudgetAllocationFlow:
    """
    Budget Allocation Flow Diagrams revealing evaluator resource allocation bias.
    """
    
    def __init__(self, figsize: tuple = (12, 6)):
        self.figsize = figsize
    
    def plot(
        self,
        allocation_data: Dict[str, Dict[str, Any]],
        title: str = "Budget Allocation Flow",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot budget allocation.
        
        Args:
            allocation_data: Dict mapping judge names to allocation summaries
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        if not allocation_data:
            axes[0].text(0.5, 0.5, "No data", ha='center', va='center')
            return fig
        
        # Left: Utilization by judge
        judges = list(allocation_data.keys())
        utilizations = [
            allocation_data[j].get('overall_utilization', 0) 
            for j in judges
        ]
        
        colors = ['#3498db' if u < 0.8 else '#e74c3c' for u in utilizations]
        axes[0].barh(judges, utilizations, color=colors)
        axes[0].set_xlim(0, 1)
        axes[0].set_xlabel('Resource Utilization')
        axes[0].set_title('Utilization by Judge')
        axes[0].axvline(x=0.8, color='red', linestyle='--', alpha=0.5, label='Warning threshold')
        
        # Right: Resource type breakdown
        resource_types = ['tokens', 'time_ms', 'tool_calls', 'llm_calls']
        x = np.arange(len(judges))
        width = 0.2
        
        for i, rt in enumerate(resource_types):
            values = [
                allocation_data[j].get('utilization', {}).get(rt, 0)
                for j in judges
            ]
            axes[1].bar(x + i*width, values, width, label=rt)
        
        axes[1].set_ylabel('Utilization')
        axes[1].set_title('Resource Breakdown')
        axes[1].set_xticks(x + width * 1.5)
        axes[1].set_xticklabels(judges, rotation=45, ha='right')
        axes[1].legend(loc='upper right', fontsize=8)
        axes[1].set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


def generate_all_plots(
    evaluation_results: Dict[str, Any],
    output_dir: Path,
) -> List[Path]:
    """
    Generate all standard plots from evaluation results.
    
    Returns list of saved file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    # Framing delta heatmap
    if 'framing_deltas' in evaluation_results:
        heatmap = FramingDeltaHeatmap()
        path = output_dir / "framing_delta_heatmap.png"
        heatmap.plot(evaluation_results['framing_deltas'], save_path=path)
        saved_files.append(path)
    
    # Risk score ladder
    if 'scenarios' in evaluation_results:
        ladder = RiskScoreLadder()
        path = output_dir / "risk_score_ladder.png"
        ladder.plot(evaluation_results['scenarios'], save_path=path)
        saved_files.append(path)
    
    # Oversight gap
    if 'oversight_data' in evaluation_results:
        gap_plot = OversightGapPlot()
        path = output_dir / "oversight_gap.png"
        gap_plot.plot(evaluation_results['oversight_data'], save_path=path)
        saved_files.append(path)
    
    # Disagreement matrix
    if 'judge_assessments' in evaluation_results:
        matrix = EvaluatorDisagreementMatrix()
        path = output_dir / "disagreement_matrix.png"
        matrix.plot(evaluation_results['judge_assessments'], save_path=path)
        saved_files.append(path)
    
    return saved_files
