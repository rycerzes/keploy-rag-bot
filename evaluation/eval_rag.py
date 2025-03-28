import os
from dotenv import load_dotenv
import json
from typing import List, Dict, Any
from pydantic import BaseModel
import google.generativeai as genai
import instructor
from results_handler import (
    wait_for_results,
    save_evaluation_report,
    get_latest_results_files,
    generate_evaluation_report_from_files,
)
import argparse

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
os.environ["DEEPEVAL_RESULTS_FOLDER"] = "./results"


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
    # Wait for both evaluation types to complete
    wait_for_results()

    results_files = get_latest_results_files(2)

    if results_files:
        print(f"Found {len(results_files)} latest results files:")
        for f in results_files:
            print(f"  - {os.path.basename(f)}")
        report = generate_evaluation_report_from_files(results_files)
        save_evaluation_report(report)
    else:
        print("Failed to generate evaluation report: No results files found.")


if __name__ == "__main__":
    main()
