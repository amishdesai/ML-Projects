#!/usr/bin/env python3
"""
EMRLogAnalyzer.py - LLM-powered analysis of EMR error clusters

Takes JSON output from EMRLogParser.py and enriches it with AI analysis.
Supports both Claude (Anthropic) and OpenAI APIs.

Usage:
  pip install anthropic  # or: pip install openai
  export ANTHROPIC_API_KEY="sk-ant-..."  # or OPENAI_API_KEY="sk-..."

  python EMRLogAnalyzer.py <INPUT_JSON> [OUTPUT_JSON] [--provider claude|openai] [--max-clusters 10]

  Example:
  python EMRLogAnalyzer.py emr_analysis_output/emr_clusters_j-NKUTYZ21RLTY_20260228_093607.json

  Environment variables:
    EMR_LLM_PROVIDER=claude|openai  # Specify LLM provider (auto-detected if not set)
    ANTHROPIC_API_KEY=...           # For Claude API
    ANTHROPIC_MODEL=...             # Default: claude-opus-4-6
    OPENAI_API_KEY=...              # For OpenAI API
    OPENAI_MODEL=...                # Default: gpt-5.2
"""

from __future__ import annotations

import os
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
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from anthropic import Anthropic
except Exception:
    Anthropic = None


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

class LLMProvider:
    """Abstract base class for LLM providers."""

    def analyze_cluster(self, cluster: ErrorCluster, spark_version: Optional[str]) -> Dict:
        """Analyze an error cluster and return structured analysis."""
        raise NotImplementedError("Subclasses must implement analyze_cluster")

    def _build_prompt(self, cluster: ErrorCluster, spark_version: Optional[str]) -> str:
        """Build the analysis prompt (shared across providers)."""
        payload = {
            "environment": "AWS EMR on EC2 (YARN)",
            "spark_version": spark_version or "unknown",
            "exception_class": cluster.exception_class,
            "normalized_signature": cluster.normalized_signature,
            "count": cluster.count,
            "examples": cluster.examples,
        }

        return (
            "You are an expert in Apache Spark running on AWS EMR (EC2, YARN).\n"
            "Given an error cluster from logs, output ONLY valid JSON with keys:\n"
            "severity (CRITICAL|HIGH|MEDIUM|LOW), category (DATA|CODE|CONFIG|INFRA|PERMISSIONS|UNKNOWN),\n"
            "root_cause, quick_fix, recommended_spark_configs (array of {key,value,why}),\n"
            "prevention, verification_steps.\n"
            "If the error seems like application logic (e.g., IllegalArgumentException requirement failed),\n"
            "focus on code/data validation fixes rather than Spark tuning.\n\n"
            f"Cluster:\n{json.dumps(payload, indent=2)}"
        )


class ClaudeProvider(LLMProvider):
    """Anthropic Claude API provider for LLM analysis."""

    def __init__(self):
        if Anthropic is None:
            raise RuntimeError("anthropic package not installed. Run: pip install anthropic")
        self.client = Anthropic()
        self.model = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-6")
        logging.info(f"Initialized Claude provider with model: {self.model}")

    def analyze_cluster(self, cluster: ErrorCluster, spark_version: Optional[str]) -> Dict:
        """Analyze cluster using Claude API."""
        logging.info(f"Analyzing cluster {cluster.cluster_id} ({cluster.exception_class}) with Claude...")
        prompt = self._build_prompt(cluster, spark_version)

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Extract text from response
            text = message.content[0].text.strip()

            try:
                result = json.loads(text)
                logging.info(f"Claude analysis completed for cluster {cluster.cluster_id}")
                return result
            except Exception:
                logging.warning(f"Failed to parse Claude output as JSON for cluster {cluster.cluster_id}")
                return {"unparsed_output": text}
        except Exception as e:
            logging.error(f"Claude API call failed: {type(e).__name__}: {e}")
            raise


class OpenAIProvider(LLMProvider):
    """OpenAI API provider for LLM analysis."""

    def __init__(self):
        if OpenAI is None:
            raise RuntimeError("openai package not installed. Run: pip install openai")
        self.client = OpenAI()
        self.model = os.environ.get("OPENAI_MODEL", "gpt-5.2")
        logging.info(f"Initialized OpenAI provider with model: {self.model}")

    def analyze_cluster(self, cluster: ErrorCluster, spark_version: Optional[str]) -> Dict:
        """Analyze cluster using OpenAI Responses API."""
        logging.info(f"Analyzing cluster {cluster.cluster_id} ({cluster.exception_class}) with OpenAI...")
        prompt = self._build_prompt(cluster, spark_version)

        resp = self.client.responses.create(
            model=self.model,
            input=prompt,
        )
        text = (resp.output_text or "").strip()

        try:
            result = json.loads(text)
            logging.info(f"OpenAI analysis completed for cluster {cluster.cluster_id}")
            return result
        except Exception:
            logging.warning(f"Failed to parse OpenAI output as JSON for cluster {cluster.cluster_id}")
            return {"unparsed_output": text}


def get_llm_provider(provider_name: Optional[str] = None) -> LLMProvider:
    """
    Factory function to create the appropriate LLM provider.

    Args:
        provider_name: 'openai', 'claude', or None (auto-detect from env vars)

    Returns:
        An instance of LLMProvider (OpenAIProvider or ClaudeProvider)
    """
    if provider_name is None:
        # Auto-detect based on environment variables
        provider_name = os.environ.get("EMR_LLM_PROVIDER", "").strip().lower()

    if not provider_name:
        # Default to Claude if ANTHROPIC_API_KEY exists, otherwise OpenAI
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider_name = "claude"
        elif os.environ.get("OPENAI_API_KEY"):
            provider_name = "openai"
        else:
            raise RuntimeError(
                "No LLM provider configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY, "
                "or specify EMR_LLM_PROVIDER=openai|claude"
            )

    if provider_name in ("claude", "anthropic"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("Claude provider selected but ANTHROPIC_API_KEY not set")
        return ClaudeProvider()
    elif provider_name == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OpenAI provider selected but OPENAI_API_KEY not set")
        return OpenAIProvider()
    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}. Use 'openai' or 'claude'")


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
        description="Analyze EMR error clusters with LLM (Claude or OpenAI)"
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
        choices=["claude", "openai"],
        help="Force specific LLM provider (auto-detects if not specified)"
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