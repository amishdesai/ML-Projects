#!/usr/bin/env python3
"""
EMR Log Parser - LLM Analysis Only
Loads existing cluster data and runs LLM analysis on it using Claude.

Usage:
  pip install anthropic
  export ANTHROPIC_API_KEY="sk-ant-..."

  python EMRLogParser_LLM_Only.py <INPUT_JSON> [MAX_CLUSTERS]

  Environment variables:
    ANTHROPIC_API_KEY=...  # Required
    ANTHROPIC_MODEL=...    # Default: claude-opus-4-7
"""

import json
import os
import sys
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

# Fix SSL certificate issue - remove invalid path if it exists
if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    print(f"Warning: Removing invalid SSL_CERT_FILE: {os.environ['SSL_CERT_FILE']}")
    del os.environ["SSL_CERT_FILE"]

try:
    from anthropic import Anthropic
    from pydantic import BaseModel, Field
except ImportError:
    Anthropic = None
    BaseModel = None
    Field = None


@dataclass
class ErrorCluster:
    cluster_id: str
    exception_class: str
    normalized_signature: str
    count: int
    examples: List[str]
    sources: List[str]


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


SYSTEM_PROMPT = (
    "You are an expert in Apache Spark running on AWS EMR (EC2, YARN). "
    "Analyze the given error cluster and provide structured recommendations. "
    "If the error seems like application logic (e.g., IllegalArgumentException requirement failed), "
    "focus on code/data validation fixes rather than Spark tuning."
)


def llm_analyze_cluster(
    client: "Anthropic",
    model: str,
    cluster: ErrorCluster,
    spark_version: Optional[str],
) -> Dict:
    """Calls Claude to summarize root cause + fixes for one cluster."""
    print(f"Analyzing cluster {cluster.cluster_id} ({cluster.exception_class})...")

    payload = {
        "environment": "AWS EMR on EC2 (YARN)",
        "spark_version": spark_version or "unknown",
        "exception_class": cluster.exception_class,
        "normalized_signature": cluster.normalized_signature,
        "count": cluster.count,
        "examples": cluster.examples[:3],
    }
    prompt = f"Analyze this error cluster:\n{json.dumps(payload, indent=2)}"

    response = client.messages.parse(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        output_format=ClusterAnalysis,
    )
    result = response.parsed_output
    print(f"  ✓ Analysis completed for {cluster.exception_class}")
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


def main():
    if len(sys.argv) < 2:
        print("Usage: python EMRLogParser_LLM_Only.py <INPUT_JSON> [MAX_CLUSTERS]")
        print("  INPUT_JSON: Path to existing EMR analysis JSON file")
        print("  MAX_CLUSTERS: Number of top clusters to analyze (default: 10)")
        sys.exit(1)

    input_path = sys.argv[1]
    max_llm = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    if Anthropic is None:
        print("Error: anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set in environment")
        sys.exit(1)

    # Load existing report
    print(f"Loading existing report from: {input_path}")
    with open(input_path, 'r') as f:
        report = json.load(f)

    if "top_clusters" not in report:
        print("Error: Input file does not contain 'top_clusters' data")
        sys.exit(1)

    clusters_data = report["top_clusters"]
    print(f"Found {len(clusters_data)} clusters in report")

    # Convert to ErrorCluster objects
    clusters = []
    for c_data in clusters_data[:max_llm]:
        cluster = ErrorCluster(
            cluster_id=c_data["cluster_id"],
            exception_class=c_data["exception_class"],
            normalized_signature=c_data["normalized_signature"],
            count=c_data["count"],
            examples=c_data["examples"],
            sources=c_data["sources"],
        )
        clusters.append(cluster)

    client = Anthropic()
    model = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-7")
    print(f"\nStarting LLM analysis for top {len(clusters)} clusters (model: {model})...\n")

    llm_enriched = []
    llm_failures = 0

    for idx, cluster in enumerate(clusters, 1):
        try:
            print(f"[{idx}/{len(clusters)}] ", end="")
            analysis = llm_analyze_cluster(client, model, cluster, report.get("spark_version"))
            llm_enriched.append({
                "cluster": {
                    "cluster_id": cluster.cluster_id,
                    "exception_class": cluster.exception_class,
                    "normalized_signature": cluster.normalized_signature,
                    "count": cluster.count,
                    "examples": cluster.examples[:3],
                    "sources": cluster.sources[:10],
                },
                "analysis": analysis
            })
        except Exception as e:
            llm_failures += 1
            print(f"  ✗ LLM analysis failed: {type(e).__name__}: {e}")
            llm_enriched.append({
                "cluster": {
                    "cluster_id": cluster.cluster_id,
                    "exception_class": cluster.exception_class,
                    "count": cluster.count,
                },
                "analysis": {"error": type(e).__name__, "message": str(e)}
            })

    # Update report with LLM analysis
    report["llm_enabled"] = True
    report["llm_provider"] = "claude"
    report["llm_failures"] = llm_failures
    report["llm_enriched"] = llm_enriched
    report["llm_analyzed_at"] = datetime.now().isoformat()

    # Save enriched report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = input_path.replace(".json", f"_llm_enriched_{timestamp}.json")

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✓ LLM Analysis Complete!")
    print(f"  Analyzed: {len(clusters)} clusters")
    print(f"  Failures: {llm_failures}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
