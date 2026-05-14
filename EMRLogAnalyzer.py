#!/usr/bin/env python3
"""
EMRLogAnalyzer.py - LLM-powered analysis of EMR error clusters

Takes JSON output from EMRLogParser.py and enriches it with AI analysis using Claude.

Usage:
  pip install anthropic
  export ANTHROPIC_API_KEY="sk-ant-..."

  python EMRLogAnalyzer.py <INPUT_JSON> [OUTPUT_JSON] [--max-clusters 10]

  Example:
  python EMRLogAnalyzer.py emr_analysis_output/emr_clusters_j-NKUTYZ21RLTY_20260228_093607.json

  Environment variables:
    ANTHROPIC_API_KEY=...  # Required
    ANTHROPIC_MODEL=...    # Default: claude-opus-4-7
"""

from __future__ import annotations

import os
import re
import sys
import json
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

try:
    from anthropic import Anthropic
    from pydantic import BaseModel, Field
except Exception:
    Anthropic = None
    BaseModel = None
    Field = None


# ----------------------------
# Error Cluster Data Structure
# ----------------------------

@dataclass
class ErrorCluster:
    """Represents a clustered error pattern."""
    cluster_id: str
    exception_class: str
    normalized_signature: str
    count: int
    examples: List[str]
    sources: List[str]


# ----------------------------
# LLM Provider Abstraction
# ----------------------------

class SparkConfig(BaseModel):
    key: str = Field(description="Spark config key")
    value: str = Field(description="Recommended value")
    why: str = Field(description="Reason for this setting")


class ClusterAnalysis(BaseModel):
    severity: str = Field(description="CRITICAL, HIGH, MEDIUM, or LOW")
    category: str = Field(description="DATA, CODE, CONFIG, INFRA, PERMISSIONS, or UNKNOWN")
    root_cause: str = Field(description="Root cause explanation")
    quick_fix: str = Field(description="Immediate remediation steps")
    recommended_spark_configs: List[SparkConfig] = Field(description="Spark config recommendations")
    prevention: str = Field(description="How to prevent this error in the future")
    verification_steps: str = Field(description="Steps to verify the fix worked")


class ClaudeProvider:
    """Anthropic Claude API provider for LLM analysis."""

    SYSTEM_PROMPT = (
        "You are an expert in Apache Spark running on AWS EMR (EC2, YARN). "
        "Analyze the given error cluster and provide structured recommendations. "
        "If the error seems like application logic (e.g., IllegalArgumentException requirement failed), "
        "focus on code/data validation fixes rather than Spark tuning."
    )

    def __init__(self):
        if Anthropic is None:
            raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

        import httpx
        base_url = os.environ.get("ANTHROPIC_BASE_URL")
        auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
        ssl_cert_file = os.environ.get("SSL_CERT_FILE")

        kwargs = {}
        if base_url:
            logging.info(f"Using custom API base URL: {base_url}")
            kwargs["base_url"] = base_url
        if auth_token:
            kwargs["api_key"] = auth_token
        elif os.environ.get("ANTHROPIC_API_KEY"):
            kwargs["api_key"] = os.environ.get("ANTHROPIC_API_KEY")

        if ssl_cert_file and os.path.exists(ssl_cert_file):
            logging.info(f"Using SSL certificate: {ssl_cert_file}")
            http_client = httpx.Client(verify=ssl_cert_file, timeout=60.0, follow_redirects=True)
            kwargs["http_client"] = http_client

        self.client = Anthropic(**kwargs)
        self.model = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-7")
        logging.info(f"Initialized Claude provider with model: {self.model}")

    def analyze_cluster(self, cluster: ErrorCluster, spark_version: Optional[str]) -> Dict:
        """Analyze cluster using Claude API with structured output."""
        logging.info(f"Analyzing cluster {cluster.cluster_id} ({cluster.exception_class}) with Claude...")

        payload = {
            "environment": "AWS EMR on EC2 (YARN)",
            "spark_version": spark_version or "unknown",
            "exception_class": cluster.exception_class,
            "normalized_signature": cluster.normalized_signature,
            "count": cluster.count,
            "examples": cluster.examples,
        }
        prompt = f"Analyze this error cluster:\n{json.dumps(payload, indent=2)}"

        try:
            response = self.client.messages.parse(
                model=self.model,
                max_tokens=4096,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                output_format=ClusterAnalysis,
            )
            result = response.parsed_output
            logging.info(f"Claude analysis completed for cluster {cluster.cluster_id}")
            return {
                "severity": result.severity,
                "category": result.category,
                "root_cause": result.root_cause,
                "quick_fix": result.quick_fix,
                "recommended_spark_configs": [
                    {"key": c.key, "value": c.value, "why": c.why}
                    for c in result.recommended_spark_configs
                ],
                "prevention": result.prevention,
                "verification_steps": result.verification_steps,
            }
        except Exception as e:
            logging.error(f"Claude API call failed: {type(e).__name__}: {e}")
            raise


def get_llm_provider(provider_name: Optional[str] = None) -> ClaudeProvider:
    """Create the Claude LLM provider."""
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        raise RuntimeError("ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN not set")
    return ClaudeProvider()


# ----------------------------
# Analysis Pipeline
# ----------------------------

def analyze_clusters(
    clusters_json: Dict,
    provider: LLMProvider,
    max_clusters: int = 10,
    spark_version: Optional[str] = None
) -> Dict:
    """
    Enrich error clusters with LLM analysis.

    Args:
        clusters_json: JSON output from EMRLogParser.py
        provider: LLM provider instance
        max_clusters: Maximum number of clusters to analyze
        spark_version: Spark version (optional)

    Returns:
        Enriched JSON with LLM analysis
    """
    logging.info("=" * 60)
    logging.info("Starting LLM Analysis")
    logging.info(f"Provider: {type(provider).__name__}")
    logging.info(f"Max clusters to analyze: {max_clusters}")
    logging.info("=" * 60)

    top_clusters = clusters_json.get("top_clusters", [])[:max_clusters]
    llm_enriched = []
    llm_failures = 0

    for idx, cluster_dict in enumerate(top_clusters, 1):
        try:
            # Convert dict to ErrorCluster object
            cluster = ErrorCluster(**cluster_dict)

            logging.info(f"Analyzing cluster {idx}/{len(top_clusters)}: {cluster.exception_class}")
            analysis = provider.analyze_cluster(cluster, spark_version)
            llm_enriched.append({
                "cluster": cluster_dict,
                "analysis": analysis
            })
        except Exception as e:
            llm_failures += 1
            logging.error(f"LLM analysis failed for cluster {idx}: {type(e).__name__}: {e}")
            llm_enriched.append({
                "cluster": cluster_dict,
                "analysis": {"error": type(e).__name__, "message": str(e)}
            })

    logging.info(f"LLM analysis completed. Failures: {llm_failures}")
    logging.info("=" * 60)

    # Create enriched output
    result = clusters_json.copy()
    result["llm_enabled"] = True
    result["llm_provider"] = type(provider).__name__.replace("Provider", "").lower()
    result["llm_failures"] = llm_failures
    result["llm_enriched"] = llm_enriched

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze EMR error clusters with Claude LLM"
    )
    parser.add_argument(
        "input_json",
        help="Input JSON file from EMRLogParser.py"
    )
    parser.add_argument(
        "output_json",
        nargs="?",
        help="Output JSON file (default: input_json with _analyzed suffix)"
    )
    parser.add_argument(
        "--provider",
        choices=["claude"],
        default="claude",
        help="LLM provider (only claude is supported)"
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=10,
        help="Maximum number of clusters to analyze (default: 10)"
    )
    parser.add_argument(
        "--spark-version",
        help="Spark version for analysis context (optional)"
    )

    args = parser.parse_args()

    # Read input JSON
    logging.info(f"Reading clusters from: {args.input_json}")
    with open(args.input_json, 'r') as f:
        clusters_json = json.load(f)

    # Determine output path
    if args.output_json:
        out_path = args.output_json
    else:
        # Auto-generate output path
        base = args.input_json.replace(".json", "")
        out_path = f"{base}_analyzed.json"

    logging.info(f"Output will be written to: {out_path}")

    # Get LLM provider
    provider = get_llm_provider(args.provider)

    # Run analysis
    result = analyze_clusters(
        clusters_json=clusters_json,
        provider=provider,
        max_clusters=args.max_clusters,
        spark_version=args.spark_version
    )

    # Write output
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    logging.info(f"Analysis written to: {out_path}")
    print(f"\nAnalysis complete! Output written to: {out_path}")

    # Print summary
    print(f"\nSummary:")
    print(f"  - Clusters analyzed: {len(result.get('llm_enriched', []))}")
    print(f"  - Provider used: {result.get('llm_provider', 'unknown')}")
    print(f"  - Failures: {result.get('llm_failures', 0)}")


if __name__ == "__main__":
    main()