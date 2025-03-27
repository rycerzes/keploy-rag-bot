import os
from dotenv import load_dotenv
import json
from typing import List, Dict, Any
from pydantic import BaseModel
import google.generativeai as genai
import instructor
import time
import glob
import argparse
from datetime import datetime

os.environ["DEEPEVAL_RESULTS_FOLDER"] = "./results"

from deepeval import evaluate
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM

from chat import setup_astradb, setup_llm, setup_rag_chain

load_dotenv()


class CustomGeminiPro(DeepEvalBaseLLM):
    def __init__(self):
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model_name="gemini-1.5-pro")

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        client = self.load_model()
        instructor_client = instructor.from_gemini(
            client=client,
            mode=instructor.Mode.GEMINI_JSON,
        )
        resp = instructor_client.messages.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_model=schema,
        )
        return resp

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "Gemini 1.5 Pro"


def create_test_cases(test_data_path: str) -> List[LLMTestCase]:
    """
    Create test cases from a JSON file containing test data

    Format:
    [
        {
            "question": "How does Keploy handle API mocking?",
            "expected_answer": "Keploy handles API mocking by..."
        },
        ...
    ]
    """
    with open(test_data_path, "r") as f:
        test_data = json.load(f)

    test_cases = []
    for item in test_data:
        test_case = LLMTestCase(
            input=item["question"],
            actual_output=None,  # Will be filled during evaluation
            expected_output=item["expected_answer"],
        )
        test_cases.append(test_case)

    return test_cases


def run_rag_for_test_cases(test_cases: List[LLMTestCase]) -> List[LLMTestCase]:
    """Run the RAG chain for each test case and collect retrieval contexts and answers"""
    vector_store = setup_astradb()
    llm = setup_llm()
    rag_chain, _ = setup_rag_chain(vector_store, llm)

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    for test_case in test_cases:
        retrieved_docs = retriever.invoke(test_case.input)
        retrieval_context = [doc.page_content for doc in retrieved_docs]
        test_case.retrieval_context = retrieval_context

        actual_output = rag_chain.invoke(test_case.input)
        test_case.actual_output = actual_output

    return test_cases


def format_score(score, decimal_places=4):
    """Format a score value, handling None values gracefully."""
    if score is None:
        return "N/A"
    return f"{score:.{decimal_places}f}"


def evaluate_retrieval(test_cases: List[LLMTestCase]) -> Dict[str, Any]:
    """Evaluate retrieval quality using contextual metrics"""
    precision_metric = ContextualPrecisionMetric(
        threshold=0.7, model=CustomGeminiPro(), async_mode=True
    )
    recall_metric = ContextualRecallMetric(
        threshold=0.7, model=CustomGeminiPro(), async_mode=True
    )
    relevancy_metric = ContextualRelevancyMetric(
        threshold=0.7, model=CustomGeminiPro(), async_mode=True
    )

    # Evaluate metrics on test cases
    evaluate(test_cases, [precision_metric, recall_metric, relevancy_metric])

    # Extract scores from the evaluation result
    results = {
        "contextual_precision": precision_metric.score,
        "contextual_recall": recall_metric.score,
        "contextual_relevancy": relevancy_metric.score,
    }

    return results


def evaluate_generation(test_cases: List[LLMTestCase]) -> Dict[str, Any]:
    """Evaluate generation quality using answer relevancy and faithfulness metrics"""
    relevancy_metric = AnswerRelevancyMetric(
        threshold=0.7, model=CustomGeminiPro(), async_mode=True
    )
    faithfulness_metric = FaithfulnessMetric(
        threshold=0.7, model=CustomGeminiPro(), async_mode=True
    )

    # Evaluate metrics on test cases
    evaluate(test_cases, [relevancy_metric, faithfulness_metric])

    # Extract scores from the evaluation result
    results = {
        "answer_relevancy": relevancy_metric.score,
        "faithfulness": faithfulness_metric.score,
    }

    return results


def get_latest_results_file() -> str:
    """Get the most recent results file from the results directory."""
    results_dir = os.environ.get("DEEPEVAL_RESULTS_FOLDER", "./results")
    results_files = glob.glob(os.path.join(results_dir, "*"))

    if not results_files:
        return None

    # Get the most recent file
    latest_file = max(results_files, key=os.path.getctime)
    return latest_file


def wait_for_results(timeout=60) -> str:
    """Wait for new results to appear in the results directory."""
    start_time = time.time()
    print("Waiting for DeepEval to generate results...")

    # Get the initial list of files in the results directory
    results_dir = os.environ.get("DEEPEVAL_RESULTS_FOLDER", "./results")
    initial_files = set(glob.glob(os.path.join(results_dir, "*")))
    initial_latest = max(initial_files, key=os.path.getctime) if initial_files else None

    while time.time() - start_time < timeout:
        # Check for new files
        current_files = set(glob.glob(os.path.join(results_dir, "*")))
        new_files = current_files - initial_files

        if new_files:
            # New files found
            latest_file = max(new_files, key=os.path.getctime)
            print(f"Found new results file: {os.path.basename(latest_file)}")
            return latest_file

        # If no new files, check if the latest file has been modified
        if initial_latest:
            current_latest = (
                max(current_files, key=os.path.getctime) if current_files else None
            )
            if (
                current_latest
                and current_latest == initial_latest
                and os.path.getmtime(current_latest) > time.time() - 5
            ):
                print(f"Results file updated: {os.path.basename(current_latest)}")
                return current_latest

        # Sleep for a short time before checking again
        time.sleep(1)

    # If timeout is reached, return the latest file if it exists
    latest_file = get_latest_results_file()
    if latest_file:
        print(f"Using latest results file: {os.path.basename(latest_file)}")
        return latest_file

    print("No results files found within timeout period.")
    return None


def aggregate_scores(results_file: str) -> Dict[str, Any]:
    """Aggregate evaluation scores from a results file."""
    if not results_file or not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return {}

    try:
        with open(results_file, "r", encoding="utf-8") as f:
            results_data = json.load(f)

        # Extract the metrics scores from test cases
        metrics_by_name = {}

        # Process individual test cases
        for test_case in results_data.get("testCases", []):
            print(f"Processing test case: {test_case.get('name')}")
            for metric_data in test_case.get("metricsData", []):
                metric_name = metric_data.get("name")
                print(f"  Found metric: {metric_name}")
                if metric_name not in metrics_by_name:
                    metrics_by_name[metric_name] = {
                        "scores": [],
                        "threshold": metric_data.get("threshold"),
                        "success_count": 0,
                        "total_count": 0,
                        "evaluation_model": metric_data.get("evaluationModel"),
                    }

                # Add score to the list
                score = metric_data.get("score")
                if score is not None:
                    metrics_by_name[metric_name]["scores"].append(score)

                # Count successes
                if metric_data.get("success", False):
                    metrics_by_name[metric_name]["success_count"] += 1

                metrics_by_name[metric_name]["total_count"] += 1

        # Calculate aggregate statistics
        aggregate_results = {}

        print(f"Found metrics: {list(metrics_by_name.keys())}")
        for metric_name, metric_data in metrics_by_name.items():
            scores = metric_data["scores"]
            aggregate_results[metric_name] = {
                "average_score": sum(scores) / len(scores) if scores else None,
                "min_score": min(scores) if scores else None,
                "max_score": max(scores) if scores else None,
                "success_rate": metric_data["success_count"]
                / metric_data["total_count"]
                if metric_data["total_count"] > 0
                else None,
                "threshold": metric_data["threshold"],
                "evaluation_model": metric_data["evaluation_model"],
                "test_cases_count": len(scores),
            }

        return aggregate_results

    except Exception as e:
        print(f"Error parsing results file: {e}")
        import traceback

        traceback.print_exc()
        return {}


def generate_evaluation_report(results_file: str) -> Dict[str, Any]:
    """Generate a comprehensive evaluation report."""
    # Aggregate scores from test cases
    scores = aggregate_scores(results_file)

    # Count total test cases more reliably by looking at the results file directly
    total_test_cases = 0
    try:
        with open(results_file, "r", encoding="utf-8") as f:
            results_data = json.load(f)
            total_test_cases = len(results_data.get("testCases", []))
    except Exception as e:
        print(f"Error counting test cases: {e}")

    # Create report structure
    report = {
        "metrics": scores,
        "timestamp": datetime.now().isoformat(),
        "results_file": os.path.basename(results_file) if results_file else None,
        "total_test_cases": total_test_cases,
        "retrieval_metrics": {
            "contextual_precision": scores.get("Contextual Precision", {}).get(
                "average_score"
            ),
            "contextual_recall": scores.get("Contextual Recall", {}).get(
                "average_score"
            ),
            "contextual_relevancy": scores.get("Contextual Relevancy", {}).get(
                "average_score"
            ),
        },
        "generation_metrics": {
            "answer_relevancy": scores.get("Answer Relevancy", {}).get("average_score"),
            "faithfulness": scores.get("Faithfulness", {}).get("average_score"),
        },
    }

    return report


def save_evaluation_report(report: Dict[str, Any], output_path: str = None):
    """Save the evaluation report to a JSON file."""
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(__file__), "..", "evaluation_report.json"
        )

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Evaluation report saved to: {output_path}")
    except Exception as e:
        print(f"Error saving evaluation report: {e}")


def get_latest_results_files(n=2) -> List[str]:
    """Get the n most recent results files from the results directory."""
    results_dir = os.environ.get("DEEPEVAL_RESULTS_FOLDER", "./results")
    results_files = glob.glob(os.path.join(results_dir, "*"))

    if not results_files:
        return []

    # Sort files by creation time (newest first)
    sorted_files = sorted(results_files, key=os.path.getctime, reverse=True)

    # Return the n newest files (or all if fewer than n)
    return sorted_files[: min(n, len(sorted_files))]


def aggregate_scores_from_files(results_files: List[str]) -> Dict[str, Any]:
    """Aggregate evaluation scores from multiple results files."""
    all_metrics = {}
    processed_files = 0

    for results_file in results_files:
        if not os.path.exists(results_file):
            print(f"Results file not found: {results_file}")
            continue

        try:
            print(f"Processing results file: {os.path.basename(results_file)}")
            with open(results_file, "r", encoding="utf-8") as f:
                results_data = json.load(f)

            # Process individual test cases
            for test_case in results_data.get("testCases", []):
                print(f"  Processing test case: {test_case.get('name')}")
                for metric_data in test_case.get("metricsData", []):
                    metric_name = metric_data.get("name")
                    print(f"    Found metric: {metric_name}")

                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = {
                            "scores": [],
                            "threshold": metric_data.get("threshold"),
                            "success_count": 0,
                            "total_count": 0,
                            "evaluation_model": metric_data.get("evaluationModel"),
                        }

                    # Add score to the list
                    score = metric_data.get("score")
                    if score is not None:
                        all_metrics[metric_name]["scores"].append(score)

                    # Count successes
                    if metric_data.get("success", False):
                        all_metrics[metric_name]["success_count"] += 1

                    all_metrics[metric_name]["total_count"] += 1

            processed_files += 1

        except Exception as e:
            print(
                f"Error processing results file {os.path.basename(results_file)}: {e}"
            )
            import traceback

            traceback.print_exc()

    if processed_files == 0:
        print("No results files were successfully processed.")
        return {}

    # Calculate aggregate statistics
    aggregate_results = {}

    print(f"Found metrics across {processed_files} files: {list(all_metrics.keys())}")
    for metric_name, metric_data in all_metrics.items():
        scores = metric_data["scores"]
        if not scores:
            continue

        aggregate_results[metric_name] = {
            "average_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "success_rate": metric_data["success_count"] / metric_data["total_count"]
            if metric_data["total_count"] > 0
            else None,
            "threshold": metric_data["threshold"],
            "evaluation_model": metric_data["evaluation_model"],
            "test_cases_count": len(scores),
        }

    return aggregate_results


def generate_evaluation_report_from_files(results_files: List[str]) -> Dict[str, Any]:
    """Generate a comprehensive evaluation report from multiple results files."""
    # Aggregate scores from all test cases across files
    scores = aggregate_scores_from_files(results_files)

    # Count total test cases
    total_test_cases = 0
    for results_file in results_files:
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                results_data = json.load(f)
                total_test_cases += len(results_data.get("testCases", []))
        except Exception as e:
            print(f"Error counting test cases in {os.path.basename(results_file)}: {e}")

    # Create report structure
    report = {
        "metrics": scores,
        "timestamp": datetime.now().isoformat(),
        "results_files": [os.path.basename(f) for f in results_files],
        "total_test_cases": total_test_cases,
        "retrieval_metrics": {
            "contextual_precision": scores.get("Contextual Precision", {}).get(
                "average_score"
            ),
            "contextual_recall": scores.get("Contextual Recall", {}).get(
                "average_score"
            ),
            "contextual_relevancy": scores.get("Contextual Relevancy", {}).get(
                "average_score"
            ),
        },
        "generation_metrics": {
            "answer_relevancy": scores.get("Answer Relevancy", {}).get("average_score"),
            "faithfulness": scores.get("Faithfulness", {}).get("average_score"),
        },
    }

    return report


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate RAG system with DeepEval")
    parser.add_argument(
        "--existing",
        action="store_true",
        help="Use existing results and generate report without running evaluations",
    )
    args = parser.parse_args()

    if args.existing:
        print("Using existing results to generate evaluation report...")
        results_files = get_latest_results_files(2)  # Get latest 2 files
        if results_files:
            print(f"Found {len(results_files)} latest results files:")
            for f in results_files:
                print(f"  - {os.path.basename(f)}")
            report = generate_evaluation_report_from_files(results_files)
            save_evaluation_report(report)
        else:
            print("No existing results files found.")
        return

    print("Evaluating RAG system with DeepEval...")

    test_data_path = os.path.join(os.path.dirname(__file__), "test_data.json")

    test_cases = create_test_cases(test_data_path)

    test_cases = run_rag_for_test_cases(test_cases)

    print("\nEvaluating Retrieval Quality:")
    evaluate_retrieval(test_cases)

    print("\nEvaluating Generation Quality:")
    evaluate_generation(test_cases)

    print("\nWaiting for evaluation results to be generated...")
    results_file = wait_for_results()

    if results_file:
        # Generate and save evaluation report
        report = generate_evaluation_report(results_file)
        save_evaluation_report(report)
    else:
        print("Failed to generate evaluation report: No results file found.")


if __name__ == "__main__":
    main()
