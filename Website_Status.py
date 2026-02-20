import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from time import sleep
from typing import Dict, Optional, Tuple

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

DEFAULT_INPUT_CSV = os.getenv("WEBSITE_STATUS_INPUT", "Unmatched_Domains_10.csv")
DEFAULT_OUTPUT_CSV = os.getenv("WEBSITE_STATUS_OUTPUT", "domain_description_industry_results.csv")


@dataclass(frozen=True)
class DomainStatus:
    status: str  # Reachable|Unreachable
    probe_url: Optional[str]
    http_code: Optional[int]
    is_redirect: bool
    redirect_location: Optional[str]
    error: Optional[str]


def normalize_domain(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    s = re.sub(r"^https?://", "", s, flags=re.IGNORECASE)
    s = s.split("/", 1)[0]
    return s.strip().lower()


def check_domain_status(domain: str, timeout_s: float, headers: Dict[str, str]) -> DomainStatus:
    try:
        import requests
    except ModuleNotFoundError as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: requests. Install: pip install requests") from e

    domain = normalize_domain(domain)
    if not domain:
        return DomainStatus(
            status="Unreachable",
            probe_url=None,
            http_code=None,
            is_redirect=False,
            redirect_location=None,
            error="empty domain",
        )

    last_error: Optional[str] = None
    for scheme in ("https", "http"):
        url = f"{scheme}://{domain}"
        try:
            resp = requests.get(url, timeout=timeout_s, allow_redirects=False, headers=headers)
            code = int(resp.status_code)
            is_redirect = 300 <= code < 400
            return DomainStatus(
                status="Reachable",
                probe_url=url,
                http_code=code,
                is_redirect=is_redirect,
                redirect_location=resp.headers.get("location"),
                error=None,
            )
        except requests.exceptions.RequestException as e:
            last_error = f"{type(e).__name__}: {e}"
            continue

    return DomainStatus(
        status="Unreachable",
        probe_url=None,
        http_code=None,
        is_redirect=False,
        redirect_location=None,
        error=last_error,
    )


def extract_visible_text(html: str, max_chars: int) -> str:
    try:
        from bs4 import BeautifulSoup
    except ModuleNotFoundError as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: beautifulsoup4. Install: pip install beautifulsoup4") from e

    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def fetch_site_text(
    probe_url: str,
    timeout_s: float,
    headers: Dict[str, str],
    max_chars: int,
) -> Tuple[str, str]:
    """
    Returns (final_url, extracted_text).
    """
    try:
        import requests
    except ModuleNotFoundError as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: requests. Install: pip install requests") from e

    resp = requests.get(probe_url, timeout=timeout_s, allow_redirects=True, headers=headers)
    resp.raise_for_status()

    content_type = (resp.headers.get("content-type") or "").lower()
    if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
        raise ValueError(f"non-HTML content-type: {content_type or 'unknown'}")

    final_url = resp.url
    text = extract_visible_text(resp.text, max_chars=max_chars)
    if not text:
        raise ValueError("no visible text extracted")
    return final_url, text


def llm_describe_site(client: "OpenAI", model: str, domain: str, text: str) -> Dict[str, str]:
    prompt = (
        "You are given a snippet of visible text from a website.\n"
        "Return ONLY valid JSON with keys:\n"
        '  "description": a short (1-2 sentence) description of the website\n'
        '  "industry": a short industry label (e.g., healthcare, finance, retail, technology, education). '
        'Use "Unknown" if you cannot tell.\n\n'
        f"Domain: {domain}\n"
        f"Text:\n{text}"
    )
    resp = client.responses.create(model=model, input=prompt)
    out = (resp.output_text or "").strip()
    if not out:
        return {"description": "", "industry": "Unknown"}

    try:
        data = json.loads(out)
        desc = str(data.get("description", "")).strip()
        industry = str(data.get("industry", "Unknown")).strip() or "Unknown"
        return {"description": desc, "industry": industry}
    except Exception:
        return {"description": out, "industry": "Unknown"}


def main() -> int:
    p = argparse.ArgumentParser(
        description="Check website status for domains in a CSV and optionally describe each site via OpenAI."
    )
    p.add_argument(
        "input_csv",
        nargs="?",
        default=DEFAULT_INPUT_CSV,
        help=f"Input CSV containing a domain column (default: {DEFAULT_INPUT_CSV})",
    )
    p.add_argument(
        "output_csv",
        nargs="?",
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV to write results (default: {DEFAULT_OUTPUT_CSV})",
    )
    p.add_argument("--domain-col", default="DOMAIN", help="Column name containing domains (default: DOMAIN)")
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.2"), help="OpenAI model name")
    p.add_argument("--no-llm", action="store_true", help="Skip OpenAI description/industry")
    p.add_argument("--timeout", type=float, default=10.0, help="Request timeout in seconds (default: 10)")
    p.add_argument("--max-text-chars", type=int, default=3000, help="Max visible text chars for LLM (default: 3000)")
    p.add_argument("--sleep", type=float, default=0.1, help="Delay between domains in seconds (default: 0.1)")
    p.add_argument("--max-rows", type=int, default=0, help="Process only first N rows (0=all)")
    args = p.parse_args()

    try:
        import requests  # noqa: F401
    except ModuleNotFoundError:
        print("Missing dependency: requests. Install: pip install requests", file=sys.stderr)
        return 2

    if not args.no_llm:
        try:
            import bs4  # noqa: F401
        except ModuleNotFoundError:
            print("Missing dependency: beautifulsoup4. Install: pip install beautifulsoup4", file=sys.stderr)
            return 2

    headers = {"User-Agent": DEFAULT_USER_AGENT}

    client = None
    if not args.no_llm:
        if OpenAI is None:
            print("openai package not installed. Run: pip install openai", file=sys.stderr)
            return 2
        if not (os.environ.get("OPENAI_API_KEY") or "").strip():
            print("Missing OPENAI_API_KEY env var (required unless --no-llm).", file=sys.stderr)
            return 2
        client = OpenAI()

    out_fields = [
        args.domain_col,
        "STATUS",
        "PROBE_URL",
        "HTTP_CODE",
        "IS_REDIRECT",
        "REDIRECT_LOCATION",
        "FINAL_URL",
        "DESCRIPTION",
        "INDUSTRY",
        "ERROR",
    ]

    def csv_val(v) -> str:
        if v is None:
            return ""
        return str(v)

    wrote = 0
    with open(args.input_csv, "r", newline="", encoding="utf-8", errors="replace") as f_in:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames or args.domain_col not in set(reader.fieldnames):
            print(f"Missing column {args.domain_col!r} in {args.input_csv}", file=sys.stderr)
            return 2

        with open(args.output_csv, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=out_fields, extrasaction="ignore")
            writer.writeheader()

            for idx, row in enumerate(reader, 1):
                if args.max_rows and args.max_rows > 0 and idx > args.max_rows:
                    break

                raw_domain = row.get(args.domain_col, "")
                domain = normalize_domain(str(raw_domain))
                if not domain:
                    continue

                print(f"Processing: {domain}")
                st = check_domain_status(domain, timeout_s=min(args.timeout, 10.0), headers=headers)

                description = "Not applicable"
                industry = "Not applicable"
                final_url = None
                err = st.error

                if st.status == "Reachable" and st.probe_url and not args.no_llm:
                    try:
                        final_url, text = fetch_site_text(
                            st.probe_url,
                            timeout_s=args.timeout,
                            headers=headers,
                            max_chars=args.max_text_chars,
                        )
                        info = llm_describe_site(client, model=args.model, domain=domain, text=text)
                        description = info.get("description", "") or ""
                        industry = info.get("industry", "Unknown") or "Unknown"
                        err = None
                    except Exception as e:
                        description = f"Description failed: {type(e).__name__}: {e}"
                        industry = "Unknown"
                        err = f"{type(e).__name__}: {e}"

                writer.writerow(
                    {
                        args.domain_col: domain,
                        "STATUS": st.status,
                        "PROBE_URL": csv_val(st.probe_url),
                        "HTTP_CODE": csv_val(st.http_code),
                        "IS_REDIRECT": "1" if st.is_redirect else "0",
                        "REDIRECT_LOCATION": csv_val(st.redirect_location),
                        "FINAL_URL": csv_val(final_url),
                        "DESCRIPTION": description,
                        "INDUSTRY": industry,
                        "ERROR": csv_val(err),
                    }
                )
                wrote += 1

                if args.sleep and args.sleep > 0:
                    sleep(args.sleep)

    print(f"Done. Wrote {wrote} rows to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
