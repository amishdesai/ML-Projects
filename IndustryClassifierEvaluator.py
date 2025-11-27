# LLM-based Evaluator for Industry Classification Results
from langchain.agents import create_agent
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import json

load_dotenv()

# configure the LLM
model = "claude-sonnet-4-5-20250929"


class EvaluationResult(BaseModel):
    """Evaluation results for industry classification."""
    industryAccuracy: int = Field(description="Score 1-10: How accurate is the industry classification for this domain")
    naicsValidity: int = Field(description="Score 1-10: Is the NAICS code valid and appropriate for the classified industry")
    sicValidity: int = Field(description="Score 1-10: Is the SIC code valid and appropriate for the classified industry")
    consistency: int = Field(description="Score 1-10: Are industry, NAICS, and SIC codes consistent with each other")
    overallScore: int = Field(description="Score 1-10: Overall quality of the classification")
    reasoning: str = Field(description="Brief explanation of the evaluation scores")


evaluator_agent = create_agent(
    model=model,
    tools=[],
    response_format=EvaluationResult,
    system_prompt=(
        "You are an expert evaluator for industry classification results. "
        "Your job is to critically assess the quality of industry classifications. "
        "Evaluate based on: "
        "1. Industry Accuracy: Does the industry match what you know about this domain? "
        "2. NAICS Validity: Is this a real 2022 NAICS code and does it match the industry? "
        "3. SIC Validity: Is this a valid SIC code and does it match the industry? "
        "4. Consistency: Do all the codes and industry name align logically? "
        "5. Overall Score: Holistic assessment of classification quality. "
        "Be strict but fair. A score of 10 means perfect, 7-9 is good, 5-6 is acceptable, below 5 needs improvement."
    ),
)


def evaluate_classification(domain, classification):
    """
    Evaluate the quality of a classification using the evaluator agent.

    Args:
        domain: The domain name that was classified
        classification: The classification result (can be dict or object with attributes)

    Returns:
        EvaluationResult object with scores and reasoning
    """
    # Handle both dict and object inputs
    if isinstance(classification, dict):
        industry = classification.get('industryCodes', classification.get('industry', 'N/A'))
        naics = classification.get('naicsCodes', classification.get('naics', 'N/A'))
        sic = classification.get('sicCodes', classification.get('sic', 'N/A'))
        crosscheck = classification.get('crosscheckValid', 'N/A')
        accuracy = classification.get('accuracyScore', 'N/A')
    else:
        industry = getattr(classification, 'industryCodes', 'N/A')
        naics = getattr(classification, 'naicsCodes', 'N/A')
        sic = getattr(classification, 'sicCodes', 'N/A')
        crosscheck = getattr(classification, 'crosscheckValid', 'N/A')
        accuracy = getattr(classification, 'accuracyScore', 'N/A')

    eval_prompt = f"""
Domain: {domain}

Classification Results:
- Industry: {industry}
- NAICS Code: {naics}
- SIC Code: {sic}
- Crosscheck Valid: {crosscheck}
- Classifier Confidence Score: {accuracy}

Please evaluate the quality of this industry classification.
"""

    messages = [{"role": "user", "content": eval_prompt}]
    result = evaluator_agent.invoke({"messages": messages})

    try:
        return result["structured_response"]
    except Exception as e:
        return str(e)


def evaluate_batch(results_list):
    """
    Evaluate a batch of classification results.

    Args:
        results_list: List of dicts with 'domain' and 'classification' keys

    Returns:
        List of dicts with domain, classification, and evaluation
    """
    evaluated_results = []

    for idx, item in enumerate(results_list, 1):
        domain = item['domain']
        classification = item['classification']

        print(f"[{idx}/{len(results_list)}] Evaluating: {domain}")
        evaluation = evaluate_classification(domain, classification)

        evaluated_results.append({
            'domain': domain,
            'classification': classification,
            'evaluation': evaluation
        })

        print(f"  Overall Score: {evaluation.overallScore}/10")
        print(f"  {evaluation.reasoning}\n")

    return evaluated_results


def print_evaluation_summary(evaluated_results):
    """Print a summary of all evaluations."""
    if not evaluated_results:
        print("No results to summarize.")
        return

    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80 + "\n")

    # Calculate averages
    total_overall = 0
    total_industry = 0
    total_naics = 0
    total_sic = 0
    total_consistency = 0
    count = len(evaluated_results)

    for result in evaluated_results:
        eval_result = result['evaluation']
        if isinstance(eval_result, EvaluationResult):
            total_overall += eval_result.overallScore
            total_industry += eval_result.industryAccuracy
            total_naics += eval_result.naicsValidity
            total_sic += eval_result.sicValidity
            total_consistency += eval_result.consistency

    print(f"Total Classifications Evaluated: {count}")
    print(f"\nAverage Scores:")
    print(f"  Overall Score:       {total_overall/count:.2f}/10")
    print(f"  Industry Accuracy:   {total_industry/count:.2f}/10")
    print(f"  NAICS Validity:      {total_naics/count:.2f}/10")
    print(f"  SIC Validity:        {total_sic/count:.2f}/10")
    print(f"  Consistency:         {total_consistency/count:.2f}/10")

    # Show best and worst
    sorted_results = sorted(evaluated_results,
                          key=lambda x: x['evaluation'].overallScore if isinstance(x['evaluation'], EvaluationResult) else 0,
                          reverse=True)

    print(f"\n{'='*80}")
    print("BEST CLASSIFICATION:")
    best = sorted_results[0]
    print(f"Domain: {best['domain']}")
    print(f"Score: {best['evaluation'].overallScore}/10")
    print(f"Reasoning: {best['evaluation'].reasoning}")

    print(f"\n{'='*80}")
    print("WORST CLASSIFICATION:")
    worst = sorted_results[-1]
    print(f"Domain: {worst['domain']}")
    print(f"Score: {worst['evaluation'].overallScore}/10")
    print(f"Reasoning: {worst['evaluation'].reasoning}")
    print("="*80 + "\n")


# Main execution for standalone usage
if __name__ == "__main__":
    print("🔍 Industry Classification Evaluator\n")
    print("This tool evaluates the quality of industry classifications.\n")

    # Example usage
    print("Example: Evaluating a sample classification\n")

    sample_domain = "salesforce.com"
    sample_classification = {
        'industryCodes': 'Software & Technology',
        'naicsCodes': '511210',
        'sicCodes': '7372',
        'crosscheckValid': 'Valid',
        'accuracyScore': 95
    }

    print(f"Domain: {sample_domain}")
    print(f"Classification: {sample_classification}\n")

    evaluation = evaluate_classification(sample_domain, sample_classification)

    print("Evaluation Results:")
    print(f"  Industry Accuracy: {evaluation.industryAccuracy}/10")
    print(f"  NAICS Validity: {evaluation.naicsValidity}/10")
    print(f"  SIC Validity: {evaluation.sicValidity}/10")
    print(f"  Consistency: {evaluation.consistency}/10")
    print(f"  Overall Score: {evaluation.overallScore}/10")
    print(f"\nReasoning: {evaluation.reasoning}")