import argparse
import json
import time

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


def main(args):
    documents_data = list(read_jsonl(args.data_path))[: args.num_documents]

    documents = [doc_dict["document"] for doc_dict in documents_data]
    arguments = [{"document": doc} for doc in documents]

    # Select backend
    backend = select_sglang_backend(args)

    # Run requests
    tic = time.time()
    states = prd_review.run_batch(
        arguments,
        temperature=0,
        backend=backend,
        num_threads=args.parallel,
        progress_bar=True,
    )
    # Run requests with manual timing
    ttfts = []
    total_tokens = 0
    tic = time.time()
    
    # You might need to run the requests individually to measure TTFT
    states = []
    for arg in arguments:
        start_time = time.time()
        state = prd_review.run(
            document=arg["document"],
            temperature=0,
            backend=backend,
        )
        first_token_time = time.time() - start_time  # This is approximate
        ttfts.append(first_token_time)
        states.append(state)
        #print(f"State type: {type(state)}")
        #print(f"State attributes: {dir(state)}")
        text_content = state.text()
        total_tokens += len(text_content.split())
        
    latency = time.time() - tic

    # P90 TTFT
    ttfts.sort()
    p90_ttft = ttfts[int(len(ttfts) * 0.9)] if ttfts else 0

    # TPS
    total_tokens = 0
    for state in states:
        text_content = state.text()
        total_tokens += len(text_content.split())
    
    tps = total_tokens / latency if latency > 0 else 0

    print(f"P90 TTFT: {p90_ttft:.3f}")
    print(f"TPS: {tps:.3f}")
    print(f"Latency: {latency:.3f}")

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
            "p90_ttft": round(p90_ttft, 3),
            "tps": round(tps, 1),
            "num_requests": len(arguments),
            "other": {
                "num_documents": args.num_documents,
                "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="prompt.jsonl")
    # Remove the --num-documents argument since we're always using 1
    args = add_common_sglang_args_and_parse(parser)
    args.num_documents = 1  # Hard-code this value
    main(args)
