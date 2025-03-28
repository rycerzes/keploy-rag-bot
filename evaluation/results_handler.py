import os
import glob
import json
from datetime import datetime
import time
from typing import Dict, Any, List


def get_latest_results_file() -> str:
    """Get the most recent results file from the results directory."""
    results_dir = os.environ.get("DEEPEVAL_RESULTS_FOLDER", "./results")
    results_files = glob.glob(os.path.join(results_dir, "*"))

    if not results_files:
        return None

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
        "total_test_cases": total_test_cases/2,  # Divide by 2 to account for retrieval and generation
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
