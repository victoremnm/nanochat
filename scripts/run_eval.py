"""
Domain-specific evaluation script for NanoChat models.

Evaluates model quality on:
- Clickhouse SQL syntax (vs SQL Server/MySQL)
- Table/column name knowledge
- Codebase understanding
- General reasoning

Usage:
    uv run python -m scripts.run_eval --source=sft --model_tag=d34 --step=55
    uv run python -m scripts.run_eval --source=sft --model_tag=d34 --step=55 --category=clickhouse_syntax
"""

import argparse
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch

from nanochat.checkpoint_manager import CheckpointManager
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import Tokenizer


@dataclass
class EvalResult:
    id: str
    category: str
    difficulty: str
    question: str
    response: str
    expected_found: list[str]
    expected_missing: list[str]
    negative_found: list[str]
    score: float
    passed: bool


def load_model(source: str, model_tag: str, step: Optional[int] = None):
    """Load model and tokenizer."""
    print(f"Loading model: source={source}, tag={model_tag}, step={step}")

    # Load tokenizer
    tokenizer = Tokenizer()

    # Load checkpoint
    cm = CheckpointManager()
    ckpt_path, meta = cm.load_checkpoint(source, model_tag, step)

    print(f"Loaded checkpoint: {ckpt_path}")

    # Initialize model
    config = GPTConfig(**meta["model_config"])
    model = GPT(config)

    # Load weights
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    print(f"Model loaded on {device}")
    return model, tokenizer, device


def generate_response(model, tokenizer, device, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
    """Generate a response from the model."""
    # Format as chat
    chat_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"

    # Tokenize
    input_ids = tokenizer.encode(chat_prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=50,
        )[0].tolist()

    # Decode only the new tokens
    response = tokenizer.decode(output_ids[len(input_ids):])

    # Clean up response (stop at end token or next turn)
    for stop in ["<|end|>", "<|user|>", "<|assistant|>"]:
        if stop in response:
            response = response.split(stop)[0]

    return response.strip()


def evaluate_response(response: str, expected_keywords: list[str], negative_keywords: list[str]) -> tuple[list, list, list, float]:
    """Evaluate a response against expected/negative keywords."""
    response_lower = response.lower()

    # Check expected keywords
    expected_found = []
    expected_missing = []
    for kw in expected_keywords:
        if kw.lower() in response_lower:
            expected_found.append(kw)
        else:
            expected_missing.append(kw)

    # Check negative keywords (things we DON'T want)
    negative_found = []
    for kw in negative_keywords:
        if kw.lower() in response_lower:
            negative_found.append(kw)

    # Calculate score
    if len(expected_keywords) == 0:
        expected_score = 1.0
    else:
        expected_score = len(expected_found) / len(expected_keywords)

    # Penalize for negative keywords
    negative_penalty = len(negative_found) * 0.25

    score = max(0, expected_score - negative_penalty)

    return expected_found, expected_missing, negative_found, score


def run_eval(model, tokenizer, device, eval_file: str, category: Optional[str] = None, verbose: bool = True):
    """Run evaluation on all test cases."""
    results = []

    with open(eval_file, "r") as f:
        test_cases = [json.loads(line) for line in f]

    # Filter by category if specified
    if category:
        test_cases = [tc for tc in test_cases if tc["category"] == category]

    print(f"\nRunning {len(test_cases)} eval cases...")
    print("=" * 60)

    for i, tc in enumerate(test_cases):
        print(f"\n[{i+1}/{len(test_cases)}] {tc['id']} ({tc['category']}, {tc['difficulty']})")
        print(f"Q: {tc['question'][:80]}...")

        # Generate response
        start_time = time.time()
        response = generate_response(model, tokenizer, device, tc["question"])
        gen_time = time.time() - start_time

        # Evaluate
        expected_found, expected_missing, negative_found, score = evaluate_response(
            response,
            tc["expected_keywords"],
            tc.get("negative_keywords", [])
        )

        passed = score >= 0.5 and len(negative_found) == 0

        result = EvalResult(
            id=tc["id"],
            category=tc["category"],
            difficulty=tc["difficulty"],
            question=tc["question"],
            response=response,
            expected_found=expected_found,
            expected_missing=expected_missing,
            negative_found=negative_found,
            score=score,
            passed=passed,
        )
        results.append(result)

        # Print result
        status = "PASS" if passed else "FAIL"
        print(f"A: {response[:100]}..." if len(response) > 100 else f"A: {response}")
        print(f"Score: {score:.2f} [{status}] ({gen_time:.1f}s)")

        if expected_missing:
            print(f"  Missing: {expected_missing}")
        if negative_found:
            print(f"  BAD (found negative): {negative_found}")

    return results


def print_summary(results: list[EvalResult]):
    """Print evaluation summary."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    # Overall stats
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    avg_score = sum(r.score for r in results) / total if total > 0 else 0

    print(f"\nOverall: {passed}/{total} passed ({100*passed/total:.1f}%)")
    print(f"Average score: {avg_score:.2f}")

    # By category
    categories = set(r.category for r in results)
    print("\nBy Category:")
    for cat in sorted(categories):
        cat_results = [r for r in results if r.category == cat]
        cat_passed = sum(1 for r in cat_results if r.passed)
        cat_score = sum(r.score for r in cat_results) / len(cat_results)
        print(f"  {cat}: {cat_passed}/{len(cat_results)} ({100*cat_passed/len(cat_results):.0f}%), avg={cat_score:.2f}")

    # By difficulty
    difficulties = ["easy", "medium", "hard"]
    print("\nBy Difficulty:")
    for diff in difficulties:
        diff_results = [r for r in results if r.difficulty == diff]
        if diff_results:
            diff_passed = sum(1 for r in diff_results if r.passed)
            diff_score = sum(r.score for r in diff_results) / len(diff_results)
            print(f"  {diff}: {diff_passed}/{len(diff_results)} ({100*diff_passed/len(diff_results):.0f}%), avg={diff_score:.2f}")

    # Failed cases
    failed = [r for r in results if not r.passed]
    if failed:
        print(f"\nFailed cases ({len(failed)}):")
        for r in failed:
            print(f"  - {r.id}: score={r.score:.2f}, missing={r.expected_missing}, bad={r.negative_found}")


def main():
    parser = argparse.ArgumentParser(description="Run domain-specific evaluation")
    parser.add_argument("--source", type=str, default="sft", choices=["base", "mid", "sft"])
    parser.add_argument("--model_tag", type=str, default="d34")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--eval_file", type=str, default="evals/domain_eval.jsonl")
    parser.add_argument("--category", type=str, default=None, help="Filter by category")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    # Load model
    model, tokenizer, device = load_model(args.source, args.model_tag, args.step)

    # Run evaluation
    results = run_eval(model, tokenizer, device, args.eval_file, args.category)

    # Print summary
    print_summary(results)

    # Save results if requested
    if args.output:
        output_data = {
            "config": {
                "source": args.source,
                "model_tag": args.model_tag,
                "step": args.step,
                "eval_file": args.eval_file,
            },
            "results": [
                {
                    "id": r.id,
                    "category": r.category,
                    "difficulty": r.difficulty,
                    "question": r.question,
                    "response": r.response,
                    "expected_found": r.expected_found,
                    "expected_missing": r.expected_missing,
                    "negative_found": r.negative_found,
                    "score": r.score,
                    "passed": r.passed,
                }
                for r in results
            ],
            "summary": {
                "total": len(results),
                "passed": sum(1 for r in results if r.passed),
                "avg_score": sum(r.score for r in results) / len(results) if results else 0,
            }
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
