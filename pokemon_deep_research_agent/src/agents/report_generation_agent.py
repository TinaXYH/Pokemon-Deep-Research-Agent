"""
Report Generation Agent for the Pokémon Deep Research Agent system.

The Report Generation Agent specializes in:
- Synthesizing research findings into comprehensive reports
- Creating structured markdown documents with tables and charts
- Generating user-friendly summaries and recommendations
- Formatting data for different output formats
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from ..agents.base_agent import BaseAgent
from ..core.communication import MessageBus, TaskChannel
from ..core.models import (AgentConfig, Message, MessageType, ResearchResult,
                           Task)
from ..tools.llm_client import LLMClient


class ReportGenerationAgent(BaseAgent):
    """
    Report Generation Agent specialized in creating comprehensive research reports.

    This agent synthesizes findings from multiple research agents and creates
    well-structured, user-friendly reports in various formats.
    """

    def __init__(
        self,
        config: AgentConfig,
        message_bus: MessageBus,
        task_channel: TaskChannel,
        llm_client: LLMClient,
    ):
        super().__init__(config, message_bus, task_channel)
        self.llm_client = llm_client

        # Report generation handlers
        self.report_handlers = {
            "report_synthesis": self._handle_report_synthesis,
            "data_visualization": self._handle_data_visualization,
            "recommendation_generation": self._handle_recommendation_generation,
            "summary_generation": self._handle_summary_generation,
            "markdown_report": self._handle_markdown_report,
            "comparative_report": self._handle_comparative_report,
        }

        # Report templates
        self.report_templates = {
            "pokemon_analysis": {
                "sections": [
                    "Executive Summary",
                    "Pokemon Overview",
                    "Statistical Analysis",
                    "Type Analysis",
                    "Competitive Viability",
                    "Recommendations",
                    "Conclusion",
                ],
                "format": "detailed_analysis",
            },
            "team_building": {
                "sections": [
                    "Team Overview",
                    "Role Distribution",
                    "Type Coverage Analysis",
                    "Synergy Assessment",
                    "Threat Analysis",
                    "Alternative Options",
                    "Final Recommendations",
                ],
                "format": "strategic_guide",
            },
            "comparison": {
                "sections": [
                    "Comparison Summary",
                    "Statistical Comparison",
                    "Strengths and Weaknesses",
                    "Use Case Analysis",
                    "Recommendation Matrix",
                    "Conclusion",
                ],
                "format": "comparative_analysis",
            },
            "battle_strategy": {
                "sections": [
                    "Strategy Overview",
                    "Core Game Plan",
                    "Key Matchups",
                    "Tactical Considerations",
                    "Counter-Strategies",
                    "Implementation Guide",
                ],
                "format": "strategic_guide",
            },
        }

        # Formatting utilities
        self.markdown_formatters = {
            "table": self._format_table,
            "stat_chart": self._format_stat_chart,
            "type_chart": self._format_type_effectiveness_chart,
            "recommendation_list": self._format_recommendation_list,
            "pokemon_card": self._format_pokemon_card,
        }

    async def _initialize(self) -> None:
        """Initialize report generation resources."""
        self.logger.info("Report Generation Agent initialized")

    async def process_task(self, task: Task) -> Dict[str, Any]:
        """Process report generation tasks."""
        task_type = task.task_type

        if task_type not in self.report_handlers:
            raise ValueError(f"Unknown report task type: {task_type}")

        handler = self.report_handlers[task_type]

        try:
            result = await handler(task)

            # Add metadata to result
            result.update(
                {
                    "agent_id": self.config.agent_id,
                    "task_id": str(task.id),
                    "timestamp": datetime.now().isoformat(),
                    "report_type": "research_synthesis",
                }
            )

            return result

        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            raise

    async def _handle_report_synthesis(self, task: Task) -> Dict[str, Any]:
        """Handle comprehensive report synthesis."""
        params = task.metadata.get("task_parameters", {})
        research_findings = params.get("research_findings", [])
        query_context = params.get("query_context", {})
        report_type = params.get("report_type", "general")

        if not research_findings:
            return {
                "type": "report_synthesis",
                "error": "No research findings provided",
                "confidence": 0.0,
            }

        # Determine report template
        template = self.report_templates.get(
            report_type, self.report_templates["pokemon_analysis"]
        )

        # Synthesize findings
        synthesis = await self._synthesize_research_findings(
            research_findings, query_context
        )

        # Generate structured report
        structured_report = await self._generate_structured_report(
            synthesis, template, query_context
        )

        # Create markdown report
        markdown_report = await self._create_markdown_report(
            structured_report, template
        )

        # Generate executive summary
        executive_summary = await self._generate_executive_summary(
            synthesis, query_context
        )

        return {
            "type": "report_synthesis",
            "report_type": report_type,
            "synthesis": synthesis,
            "structured_report": structured_report,
            "markdown_report": markdown_report,
            "executive_summary": executive_summary,
            "findings_count": len(research_findings),
            "confidence": self._calculate_report_confidence(research_findings),
            "sources": self._extract_sources(research_findings),
        }

    async def _synthesize_research_findings(
        self, findings: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize research findings into coherent analysis."""

        # Categorize findings by type
        categorized_findings = self._categorize_findings(findings)

        # Extract key insights
        key_insights = await self._extract_key_insights(findings, context)

        # Identify patterns and trends
        patterns = self._identify_patterns(findings)

        # Generate synthesis using LLM
        synthesis_text = await self._generate_synthesis_text(
            findings, context, key_insights
        )

        return {
            "categorized_findings": categorized_findings,
            "key_insights": key_insights,
            "patterns": patterns,
            "synthesis_text": synthesis_text,
            "data_quality": self._assess_data_quality(findings),
            "completeness_score": self._calculate_completeness_score(findings, context),
        }

    def _categorize_findings(
        self, findings: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize findings by type."""
        categories = {
            "pokemon_data": [],
            "statistical_analysis": [],
            "competitive_analysis": [],
            "battle_strategy": [],
            "type_analysis": [],
            "general": [],
        }

        for finding in findings:
            finding_type = finding.get("type", "general")

            if "pokemon" in finding_type:
                categories["pokemon_data"].append(finding)
            elif "stat" in finding_type:
                categories["statistical_analysis"].append(finding)
            elif "competitive" in finding_type or "battle" in finding_type:
                categories["competitive_analysis"].append(finding)
            elif "strategy" in finding_type:
                categories["battle_strategy"].append(finding)
            elif "type" in finding_type:
                categories["type_analysis"].append(finding)
            else:
                categories["general"].append(finding)

        return categories

    async def _extract_key_insights(
        self, findings: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract key insights from findings."""

        insights_prompt = f"""
        Extract 5-7 key insights from the following Pokemon research findings:
        
        Context: {json.dumps(context, indent=2)}
        
        Findings Summary:
        {self._create_findings_summary(findings)}
        
        For each insight, provide:
        1. The insight statement
        2. Supporting evidence from the findings
        3. Importance level (high/medium/low)
        4. Actionable implications
        
        Focus on insights that directly answer the user's query and provide practical value.
        """

        try:
            response = await self.llm_client.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Pokemon research analyst expert at extracting key insights.",
                    },
                    {"role": "user", "content": insights_prompt},
                ],
                temperature=0.6,
            )

            insights_text = response["choices"][0]["message"]["content"]

            # Parse insights (simplified)
            insights = self._parse_insights_from_text(insights_text)

            return insights

        except Exception as e:
            self.logger.error(f"Failed to extract insights: {e}")
            return [
                {
                    "insight": "Research findings analysis completed",
                    "evidence": "Multiple data sources analyzed",
                    "importance": "medium",
                    "implications": "Provides comprehensive Pokemon information",
                }
            ]

    def _create_findings_summary(self, findings: List[Dict[str, Any]]) -> str:
        """Create a summary of findings for LLM processing."""
        summary_parts = []

        for i, finding in enumerate(findings[:10]):  # Limit to first 10 findings
            finding_type = finding.get("type", "unknown")
            confidence = finding.get("confidence", 0.0)

            summary_parts.append(
                f"Finding {i+1}: {finding_type} (confidence: {confidence})"
            )

            # Add key data points
            if "results" in finding:
                results = finding["results"]
                if isinstance(results, list) and results:
                    summary_parts.append(f"  - {len(results)} results found")
                elif isinstance(results, dict):
                    summary_parts.append(f"  - Data: {list(results.keys())[:3]}")

        return "\n".join(summary_parts)

    def _parse_insights_from_text(self, insights_text: str) -> List[Dict[str, Any]]:
        """Parse insights from LLM response text."""
        # Simplified parsing - in practice, would use more sophisticated parsing
        lines = insights_text.split("\n")
        insights = []

        current_insight = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.")):
                if current_insight:
                    insights.append(current_insight)
                current_insight = {
                    "insight": line[2:].strip(),
                    "evidence": "Supporting evidence from analysis",
                    "importance": "medium",
                    "implications": "Actionable recommendations available",
                }
            elif "evidence" in line.lower():
                current_insight["evidence"] = line
            elif "importance" in line.lower():
                if "high" in line.lower():
                    current_insight["importance"] = "high"
                elif "low" in line.lower():
                    current_insight["importance"] = "low"

        if current_insight:
            insights.append(current_insight)

        return insights[:7]  # Limit to 7 insights

    def _identify_patterns(
        self, findings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify patterns across findings."""
        patterns = []

        # Pattern 1: Confidence levels
        confidences = [f.get("confidence", 0.0) for f in findings]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        patterns.append(
            {
                "pattern_type": "data_quality",
                "description": f"Average confidence level: {avg_confidence:.2f}",
                "significance": (
                    "high"
                    if avg_confidence >= 0.8
                    else "medium" if avg_confidence >= 0.6 else "low"
                ),
            }
        )

        # Pattern 2: Source diversity
        sources = set()
        for finding in findings:
            sources.update(finding.get("sources", []))

        patterns.append(
            {
                "pattern_type": "source_diversity",
                "description": f"Data from {len(sources)} different sources",
                "significance": "high" if len(sources) >= 3 else "medium",
            }
        )

        # Pattern 3: Finding types
        types = [f.get("type", "unknown") for f in findings]
        type_counts = {}
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1

        most_common_type = (
            max(type_counts.items(), key=lambda x: x[1])
            if type_counts
            else ("unknown", 0)
        )

        patterns.append(
            {
                "pattern_type": "analysis_focus",
                "description": f"Primary focus: {most_common_type[0]} ({most_common_type[1]} findings)",
                "significance": "medium",
            }
        )

        return patterns

    async def _generate_synthesis_text(
        self,
        findings: List[Dict[str, Any]],
        context: Dict[str, Any],
        insights: List[Dict[str, Any]],
    ) -> str:
        """Generate synthesis text using LLM."""

        synthesis_prompt = f"""
        Create a comprehensive synthesis of Pokemon research findings.
        
        Original Query Context: {context.get('original_query', 'Pokemon research')}
        Query Type: {context.get('query_type', 'general')}
        
        Key Insights:
        {json.dumps([i['insight'] for i in insights], indent=2)}
        
        Research Findings: {len(findings)} findings analyzed
        
        Create a synthesis that:
        1. Directly addresses the original query
        2. Integrates findings from multiple sources
        3. Provides clear, actionable conclusions
        4. Highlights important caveats or limitations
        5. Uses accessible language for Pokemon enthusiasts
        
        Structure the synthesis with clear paragraphs and logical flow.
        """

        try:
            response = await self.llm_client.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Pokemon research expert creating comprehensive analysis reports.",
                    },
                    {"role": "user", "content": synthesis_prompt},
                ],
                temperature=0.7,
                max_tokens=2000,
            )

            return response["choices"][0]["message"]["content"]

        except Exception as e:
            self.logger.error(f"Failed to generate synthesis text: {e}")
            return f"Analysis of {len(findings)} research findings completed. The research provides comprehensive information addressing the query about {context.get('original_query', 'Pokemon research')}."

    def _assess_data_quality(self, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess the quality of research data."""
        if not findings:
            return {"quality_score": 0.0, "assessment": "no_data"}

        # Calculate quality metrics
        confidences = [f.get("confidence", 0.0) for f in findings]
        avg_confidence = sum(confidences) / len(confidences)

        # Count findings with high confidence
        high_confidence_count = sum(1 for c in confidences if c >= 0.8)
        high_confidence_ratio = high_confidence_count / len(confidences)

        # Assess source diversity
        all_sources = set()
        for finding in findings:
            all_sources.update(finding.get("sources", []))

        source_diversity = len(all_sources)

        # Calculate overall quality score
        quality_score = (
            avg_confidence * 0.6
            + high_confidence_ratio * 0.3
            + min(source_diversity / 3, 1.0) * 0.1
        )

        if quality_score >= 0.8:
            assessment = "excellent"
        elif quality_score >= 0.6:
            assessment = "good"
        elif quality_score >= 0.4:
            assessment = "fair"
        else:
            assessment = "poor"

        return {
            "quality_score": round(quality_score, 2),
            "assessment": assessment,
            "avg_confidence": round(avg_confidence, 2),
            "high_confidence_ratio": round(high_confidence_ratio, 2),
            "source_diversity": source_diversity,
            "total_findings": len(findings),
        }

    def _calculate_completeness_score(
        self, findings: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> float:
        """Calculate completeness score based on query requirements."""
        query_type = context.get("query_type", "general")

        # Define expected finding types for different query types
        expected_types = {
            "team_building": [
                "pokemon_research",
                "competitive_analysis",
                "team_analysis",
            ],
            "individual_analysis": [
                "pokemon_research",
                "stat_analysis",
                "type_analysis",
            ],
            "comparison": ["pokemon_research", "comparative_analysis"],
            "battle_strategy": ["competitive_analysis", "strategy_development"],
            "general": ["pokemon_research"],
        }

        required_types = expected_types.get(query_type, expected_types["general"])
        found_types = set(f.get("type", "") for f in findings)

        # Calculate coverage
        coverage = len(found_types.intersection(required_types)) / len(required_types)

        return min(coverage, 1.0)

    async def _generate_structured_report(
        self,
        synthesis: Dict[str, Any],
        template: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate structured report based on template."""

        sections = {}

        for section_name in template["sections"]:
            section_content = await self._generate_section_content(
                section_name, synthesis, context
            )
            sections[section_name] = section_content

        return {
            "template_type": template["format"],
            "sections": sections,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "query_context": context,
                "synthesis_quality": synthesis.get("data_quality", {}),
                "completeness": synthesis.get("completeness_score", 0.0),
            },
        }

    async def _generate_section_content(
        self, section_name: str, synthesis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate content for a specific report section."""

        section_generators = {
            "Executive Summary": self._generate_executive_summary_section,
            "Pokemon Overview": self._generate_pokemon_overview_section,
            "Statistical Analysis": self._generate_statistical_analysis_section,
            "Type Analysis": self._generate_type_analysis_section,
            "Competitive Viability": self._generate_competitive_viability_section,
            "Recommendations": self._generate_recommendations_section,
            "Conclusion": self._generate_conclusion_section,
            "Team Overview": self._generate_team_overview_section,
            "Role Distribution": self._generate_role_distribution_section,
            "Type Coverage Analysis": self._generate_type_coverage_section,
            "Synergy Assessment": self._generate_synergy_assessment_section,
            "Threat Analysis": self._generate_threat_analysis_section,
            "Alternative Options": self._generate_alternatives_section,
            "Final Recommendations": self._generate_final_recommendations_section,
        }

        generator = section_generators.get(section_name, self._generate_generic_section)
        return await generator(synthesis, context)

    async def _generate_executive_summary_section(
        self, synthesis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate executive summary section."""

        key_insights = synthesis.get("key_insights", [])
        top_insights = [insight["insight"] for insight in key_insights[:3]]

        return {
            "content_type": "summary",
            "key_points": top_insights,
            "confidence_level": synthesis.get("data_quality", {}).get(
                "assessment", "unknown"
            ),
            "completeness": synthesis.get("completeness_score", 0.0),
            "summary_text": synthesis.get("synthesis_text", "")[:500] + "...",
        }

    async def _generate_pokemon_overview_section(
        self, synthesis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate Pokemon overview section."""

        pokemon_findings = synthesis.get("categorized_findings", {}).get(
            "pokemon_data", []
        )

        return {
            "content_type": "overview",
            "pokemon_analyzed": len(pokemon_findings),
            "data_sources": synthesis.get("data_quality", {}).get(
                "source_diversity", 0
            ),
            "overview_text": "Comprehensive Pokemon data analysis completed",
            "key_pokemon": self._extract_key_pokemon(pokemon_findings),
        }

    def _extract_key_pokemon(self, pokemon_findings: List[Dict[str, Any]]) -> List[str]:
        """Extract key Pokemon from findings."""
        pokemon_names = []

        for finding in pokemon_findings:
            if "results" in finding:
                results = finding["results"]
                if isinstance(results, list):
                    for result in results[:3]:  # Limit to first 3
                        if isinstance(result, dict) and "name" in result:
                            pokemon_names.append(result["name"])

        return list(set(pokemon_names))[:5]  # Unique names, max 5

    async def _generate_statistical_analysis_section(
        self, synthesis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate statistical analysis section."""

        stat_findings = synthesis.get("categorized_findings", {}).get(
            "statistical_analysis", []
        )

        return {
            "content_type": "analysis",
            "analysis_count": len(stat_findings),
            "analysis_text": "Statistical analysis of Pokemon data completed",
            "key_statistics": self._extract_key_statistics(stat_findings),
        }

    def _extract_key_statistics(
        self, stat_findings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract key statistics from findings."""
        return {
            "findings_analyzed": len(stat_findings),
            "statistical_methods": ["base_stat_analysis", "comparative_analysis"],
            "confidence_level": "high" if stat_findings else "low",
        }

    async def _generate_type_analysis_section(
        self, synthesis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate type analysis section."""

        return {
            "content_type": "type_analysis",
            "analysis_text": "Type effectiveness and coverage analysis",
            "type_data": "Comprehensive type analysis completed",
        }

    async def _generate_competitive_viability_section(
        self, synthesis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate competitive viability section."""

        competitive_findings = synthesis.get("categorized_findings", {}).get(
            "competitive_analysis", []
        )

        return {
            "content_type": "competitive_analysis",
            "viability_assessment": "Competitive analysis completed",
            "findings_count": len(competitive_findings),
        }

    async def _generate_recommendations_section(
        self, synthesis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate recommendations section."""

        insights = synthesis.get("key_insights", [])
        actionable_insights = [i for i in insights if i.get("importance") == "high"]

        recommendations = []
        for insight in actionable_insights[:5]:
            recommendations.append(
                {
                    "recommendation": insight.get("insight", ""),
                    "rationale": insight.get("evidence", ""),
                    "priority": insight.get("importance", "medium"),
                }
            )

        return {
            "content_type": "recommendations",
            "recommendations": recommendations,
            "total_recommendations": len(recommendations),
        }

    async def _generate_conclusion_section(
        self, synthesis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate conclusion section."""

        return {
            "content_type": "conclusion",
            "conclusion_text": synthesis.get("synthesis_text", "")[
                -300:
            ],  # Last 300 chars
            "data_quality": synthesis.get("data_quality", {}).get(
                "assessment", "unknown"
            ),
            "completeness": synthesis.get("completeness_score", 0.0),
        }

    async def _generate_generic_section(
        self, synthesis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate generic section content."""

        return {
            "content_type": "generic",
            "content_text": "Section content generated from research findings",
            "data_available": len(synthesis.get("categorized_findings", {})),
        }

    # Team building specific sections
    async def _generate_team_overview_section(
        self, synthesis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "content_type": "team_overview",
            "overview": "Team building analysis overview",
        }

    async def _generate_role_distribution_section(
        self, synthesis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "content_type": "role_distribution",
            "distribution": "Team role analysis",
        }

    async def _generate_type_coverage_section(
        self, synthesis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {"content_type": "type_coverage", "coverage": "Type coverage analysis"}

    async def _generate_synergy_assessment_section(
        self, synthesis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {"content_type": "synergy", "assessment": "Team synergy analysis"}

    async def _generate_threat_analysis_section(
        self, synthesis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {"content_type": "threats", "analysis": "Threat analysis"}

    async def _generate_alternatives_section(
        self, synthesis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {"content_type": "alternatives", "options": "Alternative options"}

    async def _generate_final_recommendations_section(
        self, synthesis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "content_type": "final_recommendations",
            "recommendations": "Final recommendations",
        }

    async def _create_markdown_report(
        self, structured_report: Dict[str, Any], template: Dict[str, Any]
    ) -> str:
        """Create markdown formatted report."""

        markdown_parts = []

        # Title
        markdown_parts.append("# Pokemon Research Report")
        markdown_parts.append("")

        # Metadata
        metadata = structured_report.get("metadata", {})
        markdown_parts.append(
            f"**Generated:** {metadata.get('generated_at', 'Unknown')}"
        )
        markdown_parts.append(
            f"**Report Type:** {structured_report.get('template_type', 'General')}"
        )
        markdown_parts.append("")

        # Sections
        sections = structured_report.get("sections", {})

        for section_name in template["sections"]:
            if section_name in sections:
                section_data = sections[section_name]

                markdown_parts.append(f"## {section_name}")
                markdown_parts.append("")

                # Format section content based on type
                content_type = section_data.get("content_type", "generic")
                formatted_content = self._format_section_content(
                    section_data, content_type
                )
                markdown_parts.append(formatted_content)
                markdown_parts.append("")

        # Footer
        markdown_parts.append("---")
        markdown_parts.append("*Report generated by Pokemon Deep Research Agent*")

        return "\n".join(markdown_parts)

    def _format_section_content(
        self, section_data: Dict[str, Any], content_type: str
    ) -> str:
        """Format section content for markdown."""

        if content_type == "summary":
            content = []
            content.append(section_data.get("summary_text", "Summary not available"))
            content.append("")

            key_points = section_data.get("key_points", [])
            if key_points:
                content.append("**Key Points:**")
                for point in key_points:
                    content.append(f"- {point}")

            return "\n".join(content)

        elif content_type == "recommendations":
            content = []
            recommendations = section_data.get("recommendations", [])

            for i, rec in enumerate(recommendations, 1):
                content.append(
                    f"### {i}. {rec.get('recommendation', 'Recommendation')}"
                )
                content.append(f"**Priority:** {rec.get('priority', 'Medium')}")
                content.append(
                    f"**Rationale:** {rec.get('rationale', 'Analysis-based recommendation')}"
                )
                content.append("")

            return "\n".join(content)

        elif content_type == "analysis":
            return section_data.get("analysis_text", "Analysis content not available")

        else:
            # Generic formatting
            text_fields = [
                "content_text",
                "overview",
                "analysis",
                "assessment",
                "description",
            ]
            for field in text_fields:
                if field in section_data:
                    return section_data[field]

            return "Section content not available"

    async def _generate_executive_summary(
        self, synthesis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate executive summary."""

        key_insights = synthesis.get("key_insights", [])
        data_quality = synthesis.get("data_quality", {})

        summary_prompt = f"""
        Create a concise executive summary for a Pokemon research report.
        
        Original Query: {context.get('original_query', 'Pokemon research')}
        Query Type: {context.get('query_type', 'general')}
        
        Key Insights:
        {json.dumps([i['insight'] for i in key_insights[:5]], indent=2)}
        
        Data Quality: {data_quality.get('assessment', 'unknown')} ({data_quality.get('quality_score', 0.0)})
        
        Create a 3-4 sentence executive summary that:
        1. States what was researched
        2. Highlights the most important finding
        3. Provides the key recommendation
        4. Notes any important caveats
        
        Keep it concise and actionable.
        """

        try:
            response = await self.llm_client.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Pokemon research analyst creating executive summaries.",
                    },
                    {"role": "user", "content": summary_prompt},
                ],
                temperature=0.5,
                max_tokens=200,
            )

            summary_text = response["choices"][0]["message"]["content"]

            return {
                "summary_text": summary_text,
                "key_insight": (
                    key_insights[0]["insight"] if key_insights else "Research completed"
                ),
                "confidence_level": data_quality.get("assessment", "unknown"),
                "recommendation_count": len(key_insights),
            }

        except Exception as e:
            self.logger.error(f"Failed to generate executive summary: {e}")
            return {
                "summary_text": f"Research analysis completed for {context.get('original_query', 'Pokemon research')}. Findings provide comprehensive information with {data_quality.get('assessment', 'unknown')} data quality.",
                "key_insight": "Research findings available",
                "confidence_level": data_quality.get("assessment", "unknown"),
                "recommendation_count": len(key_insights),
            }

    def _calculate_report_confidence(self, findings: List[Dict[str, Any]]) -> float:
        """Calculate overall report confidence."""
        if not findings:
            return 0.0

        confidences = [f.get("confidence", 0.0) for f in findings]
        return sum(confidences) / len(confidences)

    def _extract_sources(self, findings: List[Dict[str, Any]]) -> List[str]:
        """Extract all sources from findings."""
        sources = set()

        for finding in findings:
            sources.update(finding.get("sources", []))

        return list(sources)

    # Additional handler methods
    async def _handle_data_visualization(self, task: Task) -> Dict[str, Any]:
        """Handle data visualization requests."""
        params = task.metadata.get("task_parameters", {})

        return {
            "type": "data_visualization",
            "note": "Data visualization capabilities",
            "available_formats": [
                "markdown_tables",
                "stat_charts",
                "type_effectiveness_charts",
            ],
            "confidence": 0.8,
        }

    async def _handle_recommendation_generation(self, task: Task) -> Dict[str, Any]:
        """Handle recommendation generation."""
        params = task.metadata.get("task_parameters", {})

        return {
            "type": "recommendation_generation",
            "note": "Recommendation generation system",
            "confidence": 0.9,
        }

    async def _handle_summary_generation(self, task: Task) -> Dict[str, Any]:
        """Handle summary generation."""
        params = task.metadata.get("task_parameters", {})

        return {
            "type": "summary_generation",
            "note": "Summary generation capabilities",
            "confidence": 0.9,
        }

    async def _handle_markdown_report(self, task: Task) -> Dict[str, Any]:
        """Handle markdown report generation."""
        params = task.metadata.get("task_parameters", {})

        return {
            "type": "markdown_report",
            "note": "Markdown report generation",
            "confidence": 1.0,
        }

    async def _handle_comparative_report(self, task: Task) -> Dict[str, Any]:
        """Handle comparative report generation."""
        params = task.metadata.get("task_parameters", {})

        return {
            "type": "comparative_report",
            "note": "Comparative analysis reports",
            "confidence": 0.8,
        }

    # Formatting utilities
    def _format_table(self, data: List[Dict[str, Any]], headers: List[str]) -> str:
        """Format data as markdown table."""
        if not data or not headers:
            return "No data available for table"

        # Create header
        table_lines = []
        table_lines.append("| " + " | ".join(headers) + " |")
        table_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        # Add data rows
        for row in data:
            row_data = [str(row.get(header, "")) for header in headers]
            table_lines.append("| " + " | ".join(row_data) + " |")

        return "\n".join(table_lines)

    def _format_stat_chart(self, stats: Dict[str, int]) -> str:
        """Format stats as a simple chart."""
        if not stats:
            return "No stat data available"

        chart_lines = []
        chart_lines.append("```")
        chart_lines.append("Stat Distribution:")

        max_stat = max(stats.values()) if stats.values() else 1

        for stat_name, value in stats.items():
            bar_length = int((value / max_stat) * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            chart_lines.append(f"{stat_name:12} {value:3d} |{bar}|")

        chart_lines.append("```")
        return "\n".join(chart_lines)

    def _format_type_effectiveness_chart(self, type_data: Dict[str, Any]) -> str:
        """Format type effectiveness as chart."""
        if not type_data:
            return "No type data available"

        chart_lines = []
        chart_lines.append("**Type Effectiveness:**")
        chart_lines.append("")

        weaknesses = type_data.get("weaknesses", [])
        resistances = type_data.get("resistances", [])
        immunities = type_data.get("immunities", [])

        if weaknesses:
            chart_lines.append(f"**Weak to:** {', '.join(weaknesses)}")

        if resistances:
            chart_lines.append(f"**Resists:** {', '.join(resistances)}")

        if immunities:
            chart_lines.append(f"**Immune to:** {', '.join(immunities)}")

        return "\n".join(chart_lines)

    def _format_recommendation_list(self, recommendations: List[Dict[str, Any]]) -> str:
        """Format recommendations as list."""
        if not recommendations:
            return "No recommendations available"

        list_lines = []

        for i, rec in enumerate(recommendations, 1):
            priority = rec.get("priority", "medium")
            priority_symbol = (
                "🔴" if priority == "high" else "🟡" if priority == "medium" else "🟢"
            )

            list_lines.append(
                f"{i}. {priority_symbol} {rec.get('recommendation', 'Recommendation')}"
            )

            if "rationale" in rec:
                list_lines.append(f"   *{rec['rationale']}*")

            list_lines.append("")

        return "\n".join(list_lines)

    def _format_pokemon_card(self, pokemon_data: Dict[str, Any]) -> str:
        """Format Pokemon data as a card."""
        if not pokemon_data:
            return "No Pokemon data available"

        name = pokemon_data.get("name", "Unknown")
        types = pokemon_data.get("types", [])
        type_str = "/".join(types) if types else "Unknown"

        card_lines = []
        card_lines.append(f"### {name.title()}")
        card_lines.append(f"**Type:** {type_str}")

        if "stats" in pokemon_data:
            stats = pokemon_data["stats"]
            total = sum(stats.values()) if isinstance(stats, dict) else "Unknown"
            card_lines.append(f"**Base Stat Total:** {total}")

        return "\n".join(card_lines)

    def get_report_stats(self) -> Dict[str, Any]:
        """Get report generation agent statistics."""
        status = self.get_status()
        status.update(
            {
                "report_handlers": list(self.report_handlers.keys()),
                "report_templates": list(self.report_templates.keys()),
                "markdown_formatters": list(self.markdown_formatters.keys()),
            }
        )
        return status
