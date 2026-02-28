#!/usr/bin/env python3
"""
EMRLogParser.py - Core log parsing without LLM dependencies

Minimal working Spark log analysis agent for AWS EMR on EC2 (YARN):
- Scans EMR logs in S3 (steps/, containers/, applications/)
- Extracts:
    * Exception/Error stack blocks
    * YARN "diagnostics:" blocks (often contain the true root cause)
- Normalizes noisy tokens (IDs, numbers, S3 paths, domains, arrow chains)
- Clusters by (exception_class, normalized_signature)
- Outputs a JSON report

Usage:
  pip install boto3 python-dotenv
  python EMRLogParser.py <S3_BUCKET> <CLUSTER_PREFIX> [OUTPUT_PATH]

  Example:
  python EMRLogParser.py aws-logs-310532615780-us-east-1 j-NKUTYZ21RLTY

  Environment variables:
    EMR_DAYS=2                      # Days to scan
    EMR_READ_BYTES=2000000          # Bytes to read per file
    EMR_MAX_BLOCKS_PER_FILE=200     # Max error blocks per file
    EMR_MAX_CONTAINERS=10000        # Max container files to scan
"""

from __future__ import annotations

import os
import re
import json
import time
import hashlib
import gzip
import sys
import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

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
    import boto3
except Exception:
    boto3 = None  # allow printing usage even if deps are missing


# ----------------------------
# S3 helpers
# ----------------------------

def get_aws_session(profile_name: Optional[str] = None):
    """
    Create AWS session with profile or environment credentials.
    """
    if boto3 is None:
        raise RuntimeError("boto3 package not installed. Run: pip install boto3")

    # Try profile first, then fall back to environment variables
    if profile_name:
        logging.info(f"Using AWS profile: {profile_name}")
        session = boto3.Session(profile_name=profile_name)
    else:
        # Use environment variables or default credentials
        session = boto3.Session()

    # Test credentials
    try:
        sts = session.client('sts')
        identity = sts.get_caller_identity()
        logging.info(f"Authenticated as: {identity['Arn']}")
        return session
    except Exception as e:
        logging.error(f"AWS authentication failed: {e}")
        raise RuntimeError(f"AWS authentication failed. Run: aws sso login --profile {profile_name or 'default'}")


def s3_list_keys(bucket: str, prefix: str, days: int = 2, max_keys: int = 10000, session=None) -> List[str]:
    """
    List S3 keys that were modified within the last N days.
    """
    if boto3 is None:
        raise RuntimeError("boto3 package not installed. Run: pip install boto3")
    logging.info(f"Listing S3 keys in bucket '{bucket}' with prefix '{prefix}' (last {days} days, max: {max_keys})")

    if session:
        s3 = session.client("s3")
    else:
        s3 = boto3.client("s3")

    keys: List[str] = []
    token: Optional[str] = None

    # Calculate cutoff time
    cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
    logging.info(f"Filtering files modified after: {cutoff_time.isoformat()}")

    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token

        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            last_modified = obj.get("LastModified")
            if last_modified and last_modified >= cutoff_time:
                keys.append(obj["Key"])
                if len(keys) >= max_keys:
                    logging.info(f"Found {len(keys)} keys (reached max_keys limit)")
                    return keys

        if not resp.get("IsTruncated"):
            logging.info(f"Found {len(keys)} keys in the last {days} days")
            return keys
        token = resp.get("NextContinuationToken")


def s3_read_text(bucket: str, key: str, byte_limit: int = 2_000_000, session=None) -> str:
    """
    Reads up to byte_limit bytes (MVP).
    For huge logs, improve by streaming and scanning line-by-line.
    """
    if boto3 is None:
        raise RuntimeError("boto3 package not installed. Run: pip install boto3")
    logging.debug(f"Reading S3 object: s3://{bucket}/{key}")

    if session:
        s3 = session.client("s3")
    else:
        s3 = boto3.client("s3")

    obj = s3.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read(byte_limit)
    if key.lower().endswith(".gz"):
        try:
            data = gzip.decompress(data)
            logging.debug(f"Decompressed gzip file: {key}")
        except Exception:
            # If the read is truncated mid-stream, decompression can fail. Fall back to raw bytes.
            logging.warning(f"Failed to decompress {key}, using raw bytes")
            pass
    return data.decode("utf-8", errors="ignore")


# ----------------------------
# Parsing: exceptions + diagnostics
# ----------------------------

# Typical Spark log timestamp line on EMR
TS_LINE = re.compile(r"^\d{2}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\s+(INFO|WARN|ERROR|DEBUG|TRACE)\b")

# Exception start line (captures fully-qualified exception class)
EXC_START = re.compile(r".*\b([A-Za-z0-9_.]+(?:Exception|Error))\b:?\s*(.*)")
STACK_LINE = re.compile(r"^\s+at\s+.+|^\s*Caused by:\s+.+|^\s*\.\.\.\s+\d+\s+more")

# YARN diagnostics line
DIAG_START = re.compile(r"^\s*diagnostics:\s*(.*)$", re.IGNORECASE)
DIAG_EXCEPTION = re.compile(
    r"(?:User class threw exception:\s*)?([A-Za-z0-9_.]+(?:Exception|Error))\s*:\s*(.*)",
    re.IGNORECASE
)

def extract_exception_blocks(lines: List[str], max_blocks: int = 500) -> List[str]:
    """
    Extract multi-line exception + stack blocks starting at a line containing *Exception/*Error.
    """
    blocks: List[str] = []
    i, n = 0, len(lines)

    while i < n and len(blocks) < max_blocks:
        m = EXC_START.match(lines[i])
        if not m:
            i += 1
            continue

        block = [lines[i].rstrip()]
        i += 1

        # Pull stack trace lines and common continuations
        while i < n:
            line = lines[i]
            if STACK_LINE.match(line) or line.strip() == "" or line.startswith("\t") or line.startswith(" "):
                # Allow indented lines (common in stack traces)
                block.append(line.rstrip())
                i += 1
                continue
            break

        blocks.append("\n".join(block).strip())

    logging.debug(f"Extracted {len(blocks)} exception blocks from {n} lines")
    return blocks


def extract_diagnostics_blocks(lines: List[str], max_blocks: int = 500) -> List[str]:
    """
    Extract YARN diagnostics blocks, which often contain the real application exception message.
    We capture the diagnostics line and subsequent continuation lines until the next timestamped log line.
    """
    blocks: List[str] = []
    i, n = 0, len(lines)

    while i < n and len(blocks) < max_blocks:
        m = DIAG_START.match(lines[i])
        if not m:
            i += 1
            continue

        block = [("diagnostics: " + (m.group(1) or "")).rstrip()]
        i += 1

        while i < n:
            line = lines[i]
            if line.strip() == "":
                block.append("")
                i += 1
                continue
            if TS_LINE.match(line):
                break

            # diagnostics continuations are usually indented; but we allow non-timestamp lines too
            block.append(line.rstrip())
            i += 1

        blocks.append("\n".join(block).strip())

    logging.debug(f"Extracted {len(blocks)} diagnostics blocks from {n} lines")
    return blocks


# ----------------------------
# Normalization + clustering
# ----------------------------

DOMAIN = re.compile(r"\b([a-z0-9-]+\.)+[a-z]{2,}\b", re.IGNORECASE)

def normalize_for_cluster(text: str) -> str:
    """
    Aggressive normalization to prevent 1 error = 1 cluster because of IDs / paths / domains.
    Tune to your log reality.
    """
    # Paths & URIs
    text = re.sub(r"s3://[^\s]+", "s3://<PATH>", text)
    text = re.sub(r"hdfs://[^\s]+", "hdfs://<PATH>", text)
    text = re.sub(r"file:/[^\s]+", "file:/<PATH>", text)
    text = re.sub(r"/var/[^\s]+", "/var/<PATH>", text)

    # IPs
    text = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "<IP>", text)

    # Container / application / attempt-ish IDs (common in YARN)
    text = re.sub(r"\b(application|container|attempt)_[0-9_]+\b", r"\1_<ID>", text, flags=re.IGNORECASE)

    # Long hex / UUID-ish tokens
    text = re.sub(r"\b[0-9a-f]{8,}\b", "<HEX>", text, flags=re.IGNORECASE)

    # Numbers
    text = re.sub(r"\b\d+\b", "<NUM>", text)

    # Domains (your example uses domains + arrow chains)
    text = DOMAIN.sub("<DOMAIN>", text)
    # Replace arrow chains like <DOMAIN>-><DOMAIN>->...
    text = re.sub(r"(?:<DOMAIN>\s*->\s*)+<DOMAIN>", "<DOMAIN_CHAIN>", text)

    # Whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def stable_cluster_id(exc_class: str, norm_sig: str) -> str:
    h = hashlib.sha1(f"{exc_class}||{norm_sig}".encode("utf-8")).hexdigest()
    return h[:12]


def signature_from_exception_block(block: str) -> Tuple[str, str]:
    first = block.splitlines()[0]
    m = EXC_START.match(first)
    if not m:
        return ("UnknownException", normalize_for_cluster(first))
    exc_class = m.group(1)
    msg = m.group(2) or ""
    sig = normalize_for_cluster(f"{exc_class}: {msg}".strip())
    return (exc_class, sig)


def signature_from_diagnostics_block(block: str) -> Tuple[str, str]:
    first = block.splitlines()[0]
    s = re.sub(r"^\s*diagnostics:\s*", "", first, flags=re.IGNORECASE).strip()
    m = DIAG_EXCEPTION.search(s)
    if m:
        exc_class = m.group(1)
        msg = m.group(2) or ""
        sig = normalize_for_cluster(f"{exc_class}: {msg}".strip())
        return (exc_class, sig)
    return ("Diagnostics", normalize_for_cluster(s))


def is_diagnostics_block(block: str) -> bool:
    return block.lower().startswith("diagnostics:")


@dataclass
class ErrorCluster:
    cluster_id: str
    exception_class: str
    normalized_signature: str
    count: int
    examples: List[str]
    sources: List[str]  # S3 keys


def cluster_blocks(found: List[Tuple[str, str, str]]) -> List[ErrorCluster]:
    """
    found entries: (s3_key, block_text, label)
    """
    logging.info(f"Clustering {len(found)} error blocks...")
    groups: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)
    for s3_key, block, _label in found:
        if is_diagnostics_block(block):
            exc_class, norm_sig = signature_from_diagnostics_block(block)
        else:
            exc_class, norm_sig = signature_from_exception_block(block)

        # Skip "Diagnostics" exception class
        if exc_class == "Diagnostics":
            continue

        groups[(exc_class, norm_sig)].append((s3_key, block))

    clusters: List[ErrorCluster] = []
    for (exc_class, norm_sig), items in groups.items():
        cid = stable_cluster_id(exc_class, norm_sig)
        examples = [b for _k, b in items[:3]]
        sources = sorted({k for k, _b in items})[:50]
        clusters.append(
            ErrorCluster(
                cluster_id=cid,
                exception_class=exc_class,
                normalized_signature=norm_sig,
                count=len(items),
                examples=examples,
                sources=sources,
            )
        )

    clusters.sort(key=lambda c: c.count, reverse=True)
    logging.info(f"Created {len(clusters)} unique error clusters")
    if clusters:
        logging.info(f"Top cluster: {clusters[0].exception_class} with {clusters[0].count} occurrences")
    return clusters


# ----------------------------
# Main pipeline
# ----------------------------

def run(
    bucket: str,
    cluster_prefix: str,
    days: int,
    read_byte_limit: int,
    max_blocks_per_file: int,
    max_containers: int,
    aws_session=None,
) -> Dict:
    logging.info("=" * 60)
    logging.info("Starting EMR Log Parser")
    logging.info(f"Bucket: {bucket}")
    logging.info(f"Cluster Prefix: {cluster_prefix}")
    logging.info(f"Days to scan: {days}")
    logging.info(f"Read Byte Limit: {read_byte_limit}")
    logging.info(f"Max Blocks Per File: {max_blocks_per_file}")
    logging.info("=" * 60)

    # Simplified: just use elasticmapreduce/ as root
    cluster_prefix = cluster_prefix.strip()
    if cluster_prefix and not cluster_prefix.endswith("/"):
        cluster_prefix += "/"

    # If cluster_prefix doesn't start with elasticmapreduce/, add it
    if not cluster_prefix.startswith("elasticmapreduce/"):
        root = f"elasticmapreduce/{cluster_prefix}"
    else:
        root = cluster_prefix

    prefixes: List[Tuple[str, str]] = [
        ("steps", f"{root}steps/"),
        ("containers", f"{root}containers/"),
        ("applications", f"{root}applications/"),
    ]

    logging.info(f"Scanning {len(prefixes)} S3 prefixes for logs...")
    found: List[Tuple[str, str, str]] = []
    scanned_files = 0
    files_per_prefix: Dict[str, int] = {}

    for label, pfx in prefixes:
        logging.info(f"Checking prefix: {pfx} ({label})")
        probe = s3_list_keys(bucket, pfx, days=days, max_keys=1, session=aws_session)
        if not probe:
            logging.info(f"No files found in {pfx}")
            files_per_prefix[pfx] = 0
            continue

        # Use max_containers limit for containers, 10000 for others
        max_keys_for_prefix = max_containers if label == "containers" else 10000
        keys = s3_list_keys(bucket, pfx, days=days, max_keys=max_keys_for_prefix, session=aws_session)
        files_per_prefix[pfx] = len(keys)
        logging.info(f"Processing {len(keys)} files from {pfx}")

        for idx, key in enumerate(keys, 1):
            scanned_files += 1
            if idx % 10 == 0:
                logging.info(f"  Progress: {idx}/{len(keys)} files processed in {label}")

            txt = s3_read_text(bucket, key, byte_limit=read_byte_limit, session=aws_session)
            lines = txt.splitlines()

            exc_blocks = extract_exception_blocks(lines, max_blocks=max_blocks_per_file)
            diag_blocks = extract_diagnostics_blocks(lines, max_blocks=max_blocks_per_file)

            # Keep both; diagnostics blocks frequently provide better clustering for user-code failures
            for b in exc_blocks:
                found.append((key, b, label))
            for b in diag_blocks:
                found.append((key, b, label))

    logging.info(f"Scanned {scanned_files} files total, found {len(found)} error blocks")
    clusters = cluster_blocks(found)

    logging.info("=" * 60)
    logging.info("EMR Log Parser completed successfully")
    logging.info("=" * 60)

    return {
        "bucket": bucket,
        "cluster_prefix": cluster_prefix,
        "s3_prefixes_scanned": [pfx for _label, pfx in prefixes],
        "files_per_prefix": files_per_prefix,
        "scanned_files": scanned_files,
        "total_blocks_extracted": len(found),
        "top_clusters": [asdict(c) for c in clusters[:50]],
        "generated_at_epoch": int(time.time()),
    }


def main():
    def env_int(name: str, default: int) -> int:
        v = (os.environ.get(name) or "").strip()
        if not v:
            return default
        try:
            return int(v)
        except ValueError:
            return default

    argv = sys.argv
    if len(argv) < 3:
        print(
            "Usage: python EMRLogParser.py <S3_BUCKET> <CLUSTER_PREFIX e.g. j-XXXX/> [OUTPUT_PATH]",
            file=sys.stderr,
        )
        print(
            "  OUTPUT_PATH: Optional JSON output file path (default: auto-generated with timestamp in emr_analysis_output/)",
            file=sys.stderr,
        )
        print(
            "Optional env: EMR_DAYS=2 EMR_READ_BYTES=2000000 EMR_MAX_BLOCKS_PER_FILE=200 EMR_MAX_CONTAINERS=10000",
            file=sys.stderr,
        )
        raise SystemExit(2)

    bucket = argv[1]
    cluster_prefix = argv[2]

    # Determine output path
    if len(argv) > 3:
        out_path = argv[3]
    else:
        # Generate default timestamped output path
        cluster_id = cluster_prefix.rstrip('/').split('/')[-1] if cluster_prefix else 'unknown'
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "/Users/amish.desai/ML-Projects/emr_analysis_output"
        os.makedirs(output_dir, exist_ok=True)
        out_path = f"{output_dir}/emr_clusters_{cluster_id}_{timestamp}.json"

    days = env_int("EMR_DAYS", 2)
    read_bytes = env_int("EMR_READ_BYTES", 2_000_000)
    max_blocks_per_file = env_int("EMR_MAX_BLOCKS_PER_FILE", 200)
    max_containers = env_int("EMR_MAX_CONTAINERS", 10000)

    logging.info(f"Configuration: days={days}, read_bytes={read_bytes}, max_blocks_per_file={max_blocks_per_file}, max_containers={max_containers}")
    logging.info(f"Output path: {out_path}")

    # Setup AWS authentication
    aws_profile = os.environ.get("AWS_PROFILE")
    if aws_profile:
        logging.info(f"Using AWS_PROFILE from environment: {aws_profile}")
    aws_session = get_aws_session(aws_profile)

    report = run(
        bucket=bucket,
        cluster_prefix=cluster_prefix,
        days=days,
        read_byte_limit=read_bytes,
        max_blocks_per_file=max_blocks_per_file,
        max_containers=max_containers,
        aws_session=aws_session,
    )

    out_json = json.dumps(report, indent=2)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out_json)
    logging.info(f"Report written to {out_path}")
    print(f"Report written to: {out_path}")


if __name__ == "__main__":
    main()
