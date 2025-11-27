# Test Runner for Industry Classifier with Ground Truth Evaluation
from langchain.agents import create_agent
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import json
from datetime import datetime

load_dotenv()

# configure the LLM
model = "claude-sonnet-4-5-20250929"


class IndustryInfo(BaseModel):
    """Industry classification information."""
    industryCodes: str = Field(description="Primary Industry of the Domain")
    naicsCodes: str = Field(description="Primary 2022 NAICS code of the Domain")
    sicCodes: str = Field(description="Primary SIC code of the Domain")
    crosscheckValid: str = Field(description="Crosscheck NAICS code with North American Industry Classification System and classify Valid or Non-Valid")
    accuracyScore: int = Field(description="Primary accuracy score of the Domain")


# Create classifier agent
agent = create_agent(
    model=model,
    tools=[],
    response_format=IndustryInfo,
    system_prompt=(
        "You are Industry Classifier for Company Profiles "
        "Understand the top level domain and generate 'Primary Industry', 2022 'North American Industry Classification System' code and 2022 'SIC' code for a given domain. "
        "Give me confidence score you feel this is right values."
        "Can you crosscheck the NAICS with North American Industry Classification System for 2022 "
    ),
)


def classify_domain(domain):
    """Classify a single domain using the agent."""
    messages = [{"role": "user", "content": domain}]
    result = agent.invoke({"messages": messages})

    try:
        return result["structured_response"]
    except Exception as e:
        return str(e)


def load_test_dataset(file_path):
    """Load test dataset from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def calculate_code_match(predicted, ground_truth):
    """
    Calculate if codes match exactly or partially.
    Returns: 'exact', 'partial', or 'no_match'
    """
    if predicted == ground_truth:
        return 'exact'

    # Check for partial match (same first digits for NAICS/SIC)
    # NAICS: first 2-4 digits, SIC: first 2 digits
    if len(predicted) >= 2 and len(ground_truth) >= 2:
        if predicted[:2] == ground_truth[:2]:
            return 'partial'

    return 'no_match'


def normalize_industry_name(industry):
    """Normalize industry names for comparison."""
    return industry.lower().strip().replace('&', 'and').replace(',', '')


def calculate_industry_similarity(predicted, ground_truth):
    """Calculate similarity between industry names."""
    pred_norm = normalize_industry_name(predicted)
    truth_norm = normalize_industry_name(ground_truth)

    if pred_norm == truth_norm:
        return 'exact'

    # Check if one contains the other (partial match)
    if pred_norm in truth_norm or truth_norm in pred_norm:
        return 'partial'

    # Check for common words
    pred_words = set(pred_norm.split())
    truth_words = set(truth_norm.split())
    common = pred_words & truth_words

    if len(common) > 0:
        overlap = len(common) / max(len(pred_words), len(truth_words))
        if overlap >= 0.5:
            return 'partial'

    return 'no_match'


def evaluate_classification(domain, prediction, ground_truth):
    """Evaluate a single classification against ground truth."""
    results = {
        'domain': domain,
        'prediction': {
            'industry': prediction.industryCodes,
            'naics': prediction.naicsCodes,
            'sic': prediction.sicCodes,
            'crosscheck': prediction.crosscheckValid,
            'confidence': prediction.accuracyScore
        },
        'ground_truth': ground_truth,
        'evaluation': {}
    }

    # Evaluate industry
    results['evaluation']['industry_match'] = calculate_industry_similarity(
        prediction.industryCodes,
        ground_truth['industryCodes']
    )

    # Evaluate NAICS code
    results['evaluation']['naics_match'] = calculate_code_match(
        prediction.naicsCodes,
        ground_truth['naicsCodes']
    )

    # Evaluate SIC code
    results['evaluation']['sic_match'] = calculate_code_match(
        prediction.sicCodes,
        ground_truth['sicCodes']
    )

    # Overall match (all three must be exact)
    results['evaluation']['overall_match'] = (
        results['evaluation']['industry_match'] == 'exact' and
        results['evaluation']['naics_match'] == 'exact' and
        results['evaluation']['sic_match'] == 'exact'
    )

    return results


def calculate_metrics(evaluation_results):
    """Calculate overall accuracy metrics."""
    total = len(evaluation_results)

    metrics = {
        'total_tests': total,
        'industry': {'exact': 0, 'partial': 0, 'no_match': 0},
        'naics': {'exact': 0, 'partial': 0, 'no_match': 0},
        'sic': {'exact': 0, 'partial': 0, 'no_match': 0},
        'overall_exact_match': 0
    }

    for result in evaluation_results:
        eval_data = result['evaluation']

        # Count industry matches
        metrics['industry'][eval_data['industry_match']] += 1

        # Count NAICS matches
        metrics['naics'][eval_data['naics_match']] += 1

        # Count SIC matches
        metrics['sic'][eval_data['sic_match']] += 1

        # Count overall matches
        if eval_data['overall_match']:
            metrics['overall_exact_match'] += 1

    # Calculate percentages
    metrics['accuracy'] = {
        'industry_exact': (metrics['industry']['exact'] / total) * 100,
        'industry_partial': (metrics['industry']['partial'] / total) * 100,
        'naics_exact': (metrics['naics']['exact'] / total) * 100,
        'naics_partial': (metrics['naics']['partial'] / total) * 100,
        'sic_exact': (metrics['sic']['exact'] / total) * 100,
        'sic_partial': (metrics['sic']['partial'] / total) * 100,
        'overall_exact': (metrics['overall_exact_match'] / total) * 100
    }

    return metrics


def print_detailed_results(evaluation_results):
    """Print detailed results for each test case."""
    print("\n" + "="*80)
    print("DETAILED TEST RESULTS")
    print("="*80 + "\n")

    for idx, result in enumerate(evaluation_results, 1):
        print(f"[{idx}] {result['domain']}")
        print(f"  Industry:")
        print(f"    Predicted:    {result['prediction']['industry']}")
        print(f"    Ground Truth: {result['ground_truth']['industryCodes']}")
        print(f"    Match:        {result['evaluation']['industry_match'].upper()}")

        print(f"  NAICS:")
        print(f"    Predicted:    {result['prediction']['naics']}")
        print(f"    Ground Truth: {result['ground_truth']['naicsCodes']}")
        print(f"    Match:        {result['evaluation']['naics_match'].upper()}")

        print(f"  SIC:")
        print(f"    Predicted:    {result['prediction']['sic']}")
        print(f"    Ground Truth: {result['ground_truth']['sicCodes']}")
        print(f"    Match:        {result['evaluation']['sic_match'].upper()}")

        overall = "✓ PASS" if result['evaluation']['overall_match'] else "✗ FAIL"
        print(f"  Overall:      {overall}")
        print("-"*80)


def print_metrics_summary(metrics):
    """Print summary of all metrics."""
    print("\n" + "="*80)
    print("METRICS SUMMARY")
    print("="*80 + "\n")

    print(f"Total Test Cases: {metrics['total_tests']}")
    print(f"\nOverall Exact Match: {metrics['overall_exact_match']}/{metrics['total_tests']} ({metrics['accuracy']['overall_exact']:.1f}%)")

    print(f"\n{'='*80}")
    print("INDUSTRY CLASSIFICATION ACCURACY")
    print(f"{'='*80}")
    print(f"  Exact Match:   {metrics['industry']['exact']}/{metrics['total_tests']} ({metrics['accuracy']['industry_exact']:.1f}%)")
    print(f"  Partial Match: {metrics['industry']['partial']}/{metrics['total_tests']} ({metrics['accuracy']['industry_partial']:.1f}%)")
    print(f"  No Match:      {metrics['industry']['no_match']}/{metrics['total_tests']}")

    print(f"\n{'='*80}")
    print("NAICS CODE ACCURACY")
    print(f"{'='*80}")
    print(f"  Exact Match:   {metrics['naics']['exact']}/{metrics['total_tests']} ({metrics['accuracy']['naics_exact']:.1f}%)")
    print(f"  Partial Match: {metrics['naics']['partial']}/{metrics['total_tests']} ({metrics['accuracy']['naics_partial']:.1f}%)")
    print(f"  No Match:      {metrics['naics']['no_match']}/{metrics['total_tests']}")

    print(f"\n{'='*80}")
    print("SIC CODE ACCURACY")
    print(f"{'='*80}")
    print(f"  Exact Match:   {metrics['sic']['exact']}/{metrics['total_tests']} ({metrics['accuracy']['sic_exact']:.1f}%)")
    print(f"  Partial Match: {metrics['sic']['partial']}/{metrics['total_tests']} ({metrics['accuracy']['sic_partial']:.1f}%)")
    print(f"  No Match:      {metrics['sic']['no_match']}/{metrics['total_tests']}")
    print("="*80 + "\n")


def save_results_to_file(evaluation_results, metrics, output_file):
    """Save results to JSON file."""
    output = {
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics,
        'detailed_results': evaluation_results
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {output_file}")


def run_tests(test_dataset_path='test_dataset.json', save_output=True):
    """Run all tests and generate report."""
    print("="*80)
    print("INDUSTRY CLASSIFIER TEST SUITE")
    print("="*80)
    print(f"\nLoading test dataset from: {test_dataset_path}\n")

    # Load test data
    test_data = load_test_dataset(test_dataset_path)
    print(f"Loaded {len(test_data)} test cases\n")

    # Run classifications
    print("Running classifications...\n")
    evaluation_results = []

    for idx, test_case in enumerate(test_data, 1):
        domain = test_case['domain']
        ground_truth = test_case['ground_truth']

        print(f"[{idx}/{len(test_data)}] Classifying: {domain}")

        try:
            prediction = classify_domain(domain)
            result = evaluate_classification(domain, prediction, ground_truth)
            evaluation_results.append(result)

            match_status = "✓" if result['evaluation']['overall_match'] else "✗"
            print(f"  {match_status} Industry: {result['evaluation']['industry_match']}, "
                  f"NAICS: {result['evaluation']['naics_match']}, "
                  f"SIC: {result['evaluation']['sic_match']}\n")
        except Exception as e:
            print(f"  ✗ Error: {e}\n")

    # Calculate metrics
    metrics = calculate_metrics(evaluation_results)

    # Print results
    print_detailed_results(evaluation_results)
    print_metrics_summary(metrics)

    # Save to file
    if save_output:
        output_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_results_to_file(evaluation_results, metrics, output_file)

    return evaluation_results, metrics


if __name__ == "__main__":
    print("\n🧪 Starting Industry Classifier Tests...\n")
    run_tests()