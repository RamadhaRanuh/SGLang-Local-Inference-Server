import argparse
import json
import time
import os
import statistics

import sglang as sgl
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
from sglang.utils import dump_state_text, read_jsonl

system_prompt = "You are an expert in creating software Product Requirement Document. Please provide constructive criticism and suggest improvements for the attached PDF. Be concise and limit your total output to approximately 1000 tokens."

review_dimensions = [
    "Structure and Organization: Assess the document's overall structure, section organization, and flow. Check if the requirements are logically grouped and presented in a clear hierarchy. <MAXIMUM TOKEN: 100>",
    "Completeness: Evaluate whether the PRD includes all necessary sections (overview, features, user stories, acceptance criteria, etc.) and if each requirement is fully specified. <MAXIMUM TOKEN: 100>",
    "Clarity and Specificity: Examine how clear and specific the requirements are. Requirements should be unambiguous and leave no room for interpretation. <MAXIMUM TOKEN: 100>",
    "Consistency: Check for inconsistencies in terminology, formatting, and requirements throughout the document. <MAXIMUM TOKEN: 100>",
    "Testability: Assess if the requirements are written in a way that makes them testable. Can you easily determine if a requirement has been met? <MAXIMUM TOKEN: 100>",
    "Prioritization: Evaluate how well requirements are prioritized. Are must-have features clearly distinguished from nice-to-have ones? <MAXIMUM TOKEN: 100>",
]


@sgl.function
def prd_review(s, document):
    s += system_prompt
    s += "\n```\n" + document + "\n```\n\n"

    forks = s.fork(len(review_dimensions))
    for i in range(len(review_dimensions)):
        forks[i] += (
            "USER: Please review the document based on the following dimension: "
            + review_dimensions[i]
            + " Provide specific feedback on strengths and areas for improvement. "
            + "Focus only on this dimension and be concise. "
            'End your review with the word "END"\nASSISTANT:'
        )
        forks[i] += sgl.gen("feedback", max_tokens=160, stop="END")
    forks.join()

    s += "I'll provide a comprehensive review of this PRD based on several key dimensions:\n\n"
    for i in range(len(review_dimensions)):
        dimension_name = review_dimensions[i].split(":")[0]
        s += f"**{dimension_name}**\n{forks[i]['feedback'].strip()}\n\n"

    s += "## Summary of Recommendations\n"
    s += sgl.gen("recommendations", max_tokens=300)
    
    s += "\n\nOverall Quality Rating (1-10): "
    s += sgl.gen("rating", max_tokens=2)


def count_tokens_simple(text):
    """A simple token counting function that approximates token count.
    This is just an approximation - actual token counts may vary by model."""
    # Roughly estimate tokens: 1 token ~= 4 chars for English text
    return len(text) // 4


def measure_ttft_single(args, document, backend, stream=True):
    """Measure TTFT for a single document with streaming enabled."""
    ttft_arg = {"document": document}
    
    # Create a minimal function for TTFT measurement
    @sgl.function
    def ttft_test(s, document):
        s += "Review this document: " + document[:100] + "..."
        s += sgl.gen("response", max_tokens=10)
    
    # Time the first token
    tic = time.time()
    
    if stream:
        # Use streaming mode if available to measure TTFT more accurately
        try:
            state = ttft_test.stream(
                **ttft_arg,
                temperature=0,
                backend=backend,
            )
            # Get the first token
            for chunk in state:
                ttft = time.time() - tic
                break
        except Exception:
            # Fall back if streaming not supported
            state = ttft_test.run(
                **ttft_arg,
                temperature=0,
                backend=backend,
            )
            ttft = time.time() - tic
    else:
        # Non-streaming mode
        state = ttft_test.run(
            **ttft_arg,
            temperature=0,
            backend=backend,
        )
        ttft = time.time() - tic
    
    return ttft


def main(args):
    documents_data = list(read_jsonl(args.data_path))[: args.num_documents]

    documents = [doc_dict["document"] for doc_dict in documents_data]
    arguments = [{"document": doc} for doc in documents]

    # Select backend
    backend = select_sglang_backend(args)
    
    # First, measure TTFT in a separate quick test
    ttfts = []
    for doc in documents:
        ttft = measure_ttft_single(args, doc, backend)
        ttfts.append(ttft)
    
    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0
    print(f"Average TTFT: {avg_ttft:.3f}s")
    
    # Run the main benchmark
    start_time = time.time()
    states = prd_review.run_batch(
        arguments,
        temperature=0,
        backend=backend,
        num_threads=args.parallel,
        progress_bar=True,
    )
    end_time = time.time()
    latency = end_time - start_time
    
    # Calculate approximate token count and TPS
    total_tokens = 0
    for state in states:
        tokens = count_tokens_simple(state.text)
        total_tokens += tokens
    
    tps = total_tokens / latency if latency > 0 else 0
    
    print(f"Latency: {latency:.3f}s")
    print(f"Tokens Per Second (TPS): {tps:.3f}")
    print(f"Total Tokens Generated: {total_tokens}")

    # Write results
    output_file = f"prd_review_output_{args.backend.replace('/', '_')}.txt"
    dump_state_text(output_file, states)
    print(f"Detailed outputs saved to {output_file}")

    with open(args.result_file, "a") as fout:
        value = {
            "task": "prd_review",
            "backend": args.backend,
            "num_gpus": 1,
            "latency": round(latency, 3),
            "ttft": round(avg_ttft, 3),
            "tps": round(tps, 3),
            "total_tokens": total_tokens,
            "num_requests": len(arguments),
            "other": {
                "num_documents": args.num_documents,
                "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value) + "\n")


def analyze_results(results_file="results.jsonl"):
    """Analyze all results in the results.jsonl file"""
    if not os.path.exists(results_file):
        print(f"Results file {results_file} not found")
        return
    
    results = list(read_jsonl(results_file))
    
    # Group by backend
    backend_results = {}
    for result in results:
        backend = result.get("backend", "unknown")
        if backend not in backend_results:
            backend_results[backend] = []
        backend_results[backend].append(result)
    
    # Print summary stats for each backend
    print("\n===== PERFORMANCE RESULTS SUMMARY =====")
    for backend, results in backend_results.items():
        latencies = [r.get("latency", 0) for r in results]
        ttfts = [r.get("ttft", 0) for r in results if "ttft" in r]
        tpss = [r.get("tps", 0) for r in results if "tps" in r]
        
        print(f"\nBackend: {backend}")
        print(f"  Number of runs: {len(results)}")
        
        if latencies:
            print(f"  Average Latency: {sum(latencies)/len(latencies):.3f}s (min: {min(latencies):.3f}s, max: {max(latencies):.3f}s)")
            if len(latencies) > 1:
                print(f"  Latency Std Dev: {statistics.stdev(latencies):.3f}s")
        
        if ttfts:
            print(f"  Average TTFT: {sum(ttfts)/len(ttfts):.3f}s (min: {min(ttfts):.3f}s, max: {max(ttfts):.3f}s)")
            if len(ttfts) > 1:
                print(f"  TTFT Std Dev: {statistics.stdev(ttfts):.3f}s")
        
        if tpss:
            print(f"  Average TPS: {sum(tpss)/len(tpss):.3f} (min: {min(tpss):.3f}, max: {max(tpss):.3f})")
            if len(tpss) > 1:
                print(f"  TPS Std Dev: {statistics.stdev(tpss):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="prompt.jsonl")
    parser.add_argument("--analyze", action="store_true", help="Analyze existing results file")
    args = add_common_sglang_args_and_parse(parser)
    args.num_documents = 1  # Hard-code this value
    
    if args.analyze:
        analyze_results()
    else:
        main(args)
