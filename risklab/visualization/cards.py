"""
Scenario Narrative Cards for rapid human review.

Cards combine context, responses, risk scores, and evaluator assessments.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from pathlib import Path


class ScenarioNarrativeCard(BaseModel):
    """
    Concise scenario summary for rapid human review.
    
    Combines context, model responses, risk scores, evaluator assessments, and final decisions.
    """
    card_id: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Scenario info
    episode_name: str
    episode_description: str = ""
    category: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    
    # Context
    domain: str = "general"
    stakes_level: str = "medium"
    vulnerability_level: str = "low"
    context_summary: str = ""
    
    # Prompt and response
    original_prompt: str
    framing_applied: str = "neutral"
    model_response: str
    response_truncated: bool = False
    
    # Risk assessment
    risk_score: float = 0.0
    risk_label: str = "unknown"
    primary_concerns: List[str] = Field(default_factory=list)
    
    # Evaluator assessments
    num_evaluators: int = 0
    consensus_reached: bool = False
    evaluator_agreement: float = 0.0
    evaluator_notes: List[str] = Field(default_factory=list)
    
    # Decision
    recommended_action: str = "acceptable"
    confidence: float = 0.0
    
    # Evidence
    evidence_highlights: List[str] = Field(default_factory=list)
    
    def to_text(self) -> str:
        """Generate plain text representation."""
        lines = [
            f"{'='*60}",
            f"SCENARIO: {self.episode_name}",
            f"{'='*60}",
            f"",
            f"Context: {self.domain} | Stakes: {self.stakes_level} | Vulnerability: {self.vulnerability_level}",
            f"Framing: {self.framing_applied}",
            f"",
            f"--- PROMPT ---",
            self.original_prompt[:500] + ("..." if len(self.original_prompt) > 500 else ""),
            f"",
            f"--- RESPONSE ---",
            self.model_response[:800] + ("..." if len(self.model_response) > 800 else ""),
            f"",
            f"--- ASSESSMENT ---",
            f"Risk Score: {self.risk_score:.2f} ({self.risk_label})",
            f"Recommended Action: {self.recommended_action.upper()}",
            f"Confidence: {self.confidence:.0%}",
            f"",
        ]
        
        if self.primary_concerns:
            lines.append("Primary Concerns:")
            for concern in self.primary_concerns[:3]:
                lines.append(f"  • {concern}")
            lines.append("")
        
        if self.evidence_highlights:
            lines.append("Evidence:")
            for evidence in self.evidence_highlights[:3]:
                lines.append(f"  • {evidence}")
            lines.append("")
        
        lines.append(f"Evaluators: {self.num_evaluators} | Agreement: {self.evaluator_agreement:.0%}")
        lines.append(f"{'='*60}")
        
        return "\n".join(lines)
    
    def to_html(self) -> str:
        """Generate HTML representation."""
        risk_color = self._get_risk_color()
        action_color = self._get_action_color()
        
        html = f"""
        <div class="scenario-card" style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px 0; font-family: sans-serif;">
            <div class="card-header" style="border-bottom: 1px solid #eee; padding-bottom: 8px; margin-bottom: 12px;">
                <h3 style="margin: 0; color: #333;">{self.episode_name}</h3>
                <div class="tags" style="margin-top: 4px;">
                    <span style="background: #e0e0e0; padding: 2px 8px; border-radius: 4px; font-size: 12px;">{self.domain}</span>
                    <span style="background: #e0e0e0; padding: 2px 8px; border-radius: 4px; font-size: 12px;">Stakes: {self.stakes_level}</span>
                    <span style="background: #e0e0e0; padding: 2px 8px; border-radius: 4px; font-size: 12px;">{self.framing_applied}</span>
                </div>
            </div>
            
            <div class="prompt-section" style="background: #f9f9f9; padding: 12px; border-radius: 4px; margin-bottom: 12px;">
                <strong>Prompt:</strong>
                <p style="margin: 8px 0 0 0; font-size: 14px;">{self.original_prompt[:300]}{'...' if len(self.original_prompt) > 300 else ''}</p>
            </div>
            
            <div class="response-section" style="background: #f0f7ff; padding: 12px; border-radius: 4px; margin-bottom: 12px;">
                <strong>Response:</strong>
                <p style="margin: 8px 0 0 0; font-size: 14px;">{self.model_response[:500]}{'...' if len(self.model_response) > 500 else ''}</p>
            </div>
            
            <div class="assessment" style="display: flex; gap: 16px; margin-bottom: 12px;">
                <div style="flex: 1; text-align: center; padding: 12px; background: {risk_color}20; border-radius: 4px;">
                    <div style="font-size: 24px; font-weight: bold; color: {risk_color};">{self.risk_score:.2f}</div>
                    <div style="font-size: 12px; color: #666;">Risk Score</div>
                </div>
                <div style="flex: 1; text-align: center; padding: 12px; background: {action_color}20; border-radius: 4px;">
                    <div style="font-size: 16px; font-weight: bold; color: {action_color};">{self.recommended_action.upper()}</div>
                    <div style="font-size: 12px; color: #666;">Recommendation</div>
                </div>
                <div style="flex: 1; text-align: center; padding: 12px; background: #f0f0f0; border-radius: 4px;">
                    <div style="font-size: 24px; font-weight: bold;">{self.confidence:.0%}</div>
                    <div style="font-size: 12px; color: #666;">Confidence</div>
                </div>
            </div>
            
            {'<div class="concerns" style="margin-bottom: 12px;"><strong>Concerns:</strong><ul style="margin: 4px 0; padding-left: 20px;">' + ''.join(f'<li style="font-size: 13px;">{c}</li>' for c in self.primary_concerns[:3]) + '</ul></div>' if self.primary_concerns else ''}
            
            <div class="footer" style="font-size: 12px; color: #888; border-top: 1px solid #eee; padding-top: 8px;">
                Evaluators: {self.num_evaluators} | Agreement: {self.evaluator_agreement:.0%} | Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M')}
            </div>
        </div>
        """
        return html
    
    def _get_risk_color(self) -> str:
        if self.risk_score < 0.3:
            return "#27ae60"
        elif self.risk_score < 0.5:
            return "#f39c12"
        elif self.risk_score < 0.75:
            return "#e67e22"
        else:
            return "#e74c3c"
    
    def _get_action_color(self) -> str:
        colors = {
            "acceptable": "#27ae60",
            "monitor": "#3498db",
            "mitigated": "#f39c12",
            "escalate": "#e67e22",
            "block": "#e74c3c",
        }
        return colors.get(self.recommended_action.lower(), "#666")


class CardGenerator:
    """
    Generates Scenario Narrative Cards from evaluation results.
    """
    
    def generate_card(
        self,
        episode_name: str,
        prompt: str,
        response: str,
        assessment: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        evaluator_data: Optional[Dict[str, Any]] = None,
    ) -> ScenarioNarrativeCard:
        """Generate a single narrative card."""
        card = ScenarioNarrativeCard(
            card_id=f"card_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            episode_name=episode_name,
            original_prompt=prompt,
            model_response=response,
        )
        
        # Context
        if context:
            card.domain = context.get('domain', 'general')
            card.stakes_level = context.get('stakes_level', 'medium')
            card.vulnerability_level = context.get('vulnerability_level', 'low')
            card.framing_applied = context.get('framing', 'neutral')
            card.context_summary = context.get('summary', '')
        
        # Assessment
        card.risk_score = assessment.get('risk_score', 0.0)
        card.risk_label = assessment.get('risk_label', 'unknown')
        card.recommended_action = assessment.get('recommended_action', 'acceptable')
        card.confidence = assessment.get('confidence', 0.0)
        card.primary_concerns = assessment.get('concerns', [])[:5]
        card.evidence_highlights = assessment.get('evidence', [])[:5]
        
        # Evaluator data
        if evaluator_data:
            card.num_evaluators = evaluator_data.get('num_evaluators', 0)
            card.consensus_reached = evaluator_data.get('consensus', False)
            card.evaluator_agreement = evaluator_data.get('agreement', 0.0)
            card.evaluator_notes = evaluator_data.get('notes', [])
        
        return card
    
    def generate_batch(
        self,
        evaluations: List[Dict[str, Any]],
    ) -> List[ScenarioNarrativeCard]:
        """Generate cards for multiple evaluations."""
        cards = []
        for eval_data in evaluations:
            card = self.generate_card(
                episode_name=eval_data.get('episode_name', 'Unknown'),
                prompt=eval_data.get('prompt', ''),
                response=eval_data.get('response', ''),
                assessment=eval_data.get('assessment', {}),
                context=eval_data.get('context'),
                evaluator_data=eval_data.get('evaluator_data'),
            )
            cards.append(card)
        return cards
    
    def export_cards_html(
        self,
        cards: List[ScenarioNarrativeCard],
        output_path: Path,
        title: str = "Scenario Evaluation Cards",
    ) -> None:
        """Export cards to an HTML file."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .summary {{ background: white; padding: 16px; border-radius: 8px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="summary">
        <strong>Total Scenarios:</strong> {len(cards)} |
        <strong>Generated:</strong> {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
    </div>
    {''.join(card.to_html() for card in cards)}
</body>
</html>
        """
        
        Path(output_path).write_text(html_content, encoding='utf-8')
    
    def export_cards_text(
        self,
        cards: List[ScenarioNarrativeCard],
        output_path: Path,
    ) -> None:
        """Export cards to a plain text file."""
        content = "\n\n".join(card.to_text() for card in cards)
        Path(output_path).write_text(content, encoding='utf-8')
