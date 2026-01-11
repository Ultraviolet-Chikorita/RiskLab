"""
RiskLab Integration Module.

Provides integrations for:
- CI/CD pipelines (GitHub Actions, GitLab CI)
- Export formats (SARIF, JUnit)
- External tools and services
"""

from risklab.integration.ci_cd import (
    CIResult,
    CIIssue,
    CIReport,
    ThresholdConfig,
    SARIFExporter,
    JUnitExporter,
    GitHubActionsIntegration,
    GitLabCIIntegration,
    CIRunner,
    create_github_workflow,
    create_gitlab_ci,
)

__all__ = [
    "CIResult",
    "CIIssue",
    "CIReport",
    "ThresholdConfig",
    "SARIFExporter",
    "JUnitExporter",
    "GitHubActionsIntegration",
    "GitLabCIIntegration",
    "CIRunner",
    "create_github_workflow",
    "create_gitlab_ci",
]
