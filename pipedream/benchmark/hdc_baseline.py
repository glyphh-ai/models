#!/usr/bin/env python3
"""Run HDC benchmark on the exact same queries used in the GPT-4o benchmark."""

import json
import time
import sys
import requests

MCP_URL = "http://localhost:8002/custom/pipedream/mcp"

# Reusable session for connection pooling
_session = requests.Session()
_session.headers.update({"Content-Type": "application/json"})


def query_hdc(query: str) -> dict:
    """Send a single NL query to the HDC MCP endpoint."""
    resp = _session.post(
        MCP_URL,
        json={"tool": "nl_query", "arguments": {"query": query}},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def parse_mcp_response(resp: dict) -> tuple[str, str | None, list[str]]:
    """Parse MCP response into (state, top_action, all_candidates).

    Returns:
        state: "DONE", "ASK", or "ERROR"
        top_action: best action key (from DONE result or first ASK option)
        all_candidates: list of all action keys in the response
    """
    content = resp.get("content", [])
    if not content:
        return "ERROR", None, []

    data = content[0].get("data", {}) if isinstance(content[0], dict) else {}
    state = data.get("state", "ERROR")

    if state == "DONE":
        # Extract action_key from fact_tree → results → Match 1 → concept_text
        ft = data.get("fact_tree", {})
        candidates = []
        for child in ft.get("children", []):
            if child.get("description") == "results":
                for match in child.get("children", []):
                    val = match.get("value", {})
                    if isinstance(val, dict):
                        ct = val.get("concept_text", "")
                        if ct:
                            candidates.append(ct)
        top = candidates[0] if candidates else None
        return "DONE", top, candidates

    elif state == "ASK":
        ask = data.get("ask", {})
        options = ask.get("disambiguation_options", [])
        candidates = [opt.get("intent", "") for opt in options]
        top = candidates[0] if candidates else None
        return "ASK", top, candidates

    return state, None, []


def run_benchmark(source_file: str, output_file: str):
    with open(source_file) as f:
        data = json.load(f)

    queries = data["per_query"]
    total = len(queries)
    correct_action = 0
    correct_app = 0
    correct_any = 0  # correct action appears in candidates
    done_count = 0
    ask_count = 0
    error_count = 0
    total_ms = 0.0
    results = []

    print(f"Running {total} queries against HDC MCP endpoint...")
    print(f"Source: {source_file}")
    print(f"Endpoint: {MCP_URL}")
    print()

    for i, q in enumerate(queries):
        query_text = q["query"]
        expected = q["expected"]
        app_slug = q.get("app_slug", "")

        try:
            t0 = time.time()
            resp = query_hdc(query_text)
            latency = (time.time() - t0) * 1000

            query_time = resp.get("query_time_ms", latency)
            total_ms += query_time

            state, top_action, candidates = parse_mcp_response(resp)

            if state == "DONE":
                done_count += 1
            elif state == "ASK":
                ask_count += 1
            else:
                error_count += 1

            # Action correct: top prediction matches expected
            is_correct = top_action == expected if top_action else False
            if is_correct:
                correct_action += 1

            # Action in any candidate
            if expected in candidates:
                correct_any += 1

            # App correct: predicted action starts with the expected app slug
            if top_action and app_slug:
                if top_action.startswith(app_slug + "-") or top_action.startswith(app_slug.replace("-", "_") + "-"):
                    correct_app += 1
                else:
                    # Also check if expected and predicted share the same app prefix
                    exp_app = expected.split("-")[0]
                    pred_app = top_action.split("-")[0]
                    if exp_app == pred_app:
                        correct_app += 1

            results.append({
                "query": query_text,
                "expected": expected,
                "predicted": top_action,
                "correct": is_correct,
                "in_candidates": expected in candidates,
                "state": state,
                "candidates": candidates,
                "app_slug": app_slug,
                "latency_ms": round(query_time, 1),
                "test_type": q.get("test_type", "unknown"),
            })

        except Exception as e:
            error_count += 1
            results.append({
                "query": query_text,
                "expected": expected,
                "predicted": None,
                "correct": False,
                "in_candidates": False,
                "state": "ERROR",
                "error": str(e),
                "app_slug": app_slug,
                "latency_ms": 0,
                "test_type": q.get("test_type", "unknown"),
            })

        if (i + 1) % 100 == 0 or i == total - 1:
            pct = correct_action / (i + 1) * 100
            any_pct = correct_any / (i + 1) * 100
            app_pct = correct_app / (i + 1) * 100
            print(f"  [{i+1:>4}/{total}] top1={pct:.1f}% any={any_pct:.1f}% app={app_pct:.1f}% "
                  f"DONE={done_count} ASK={ask_count} avg={total_ms/(i+1):.1f}ms")

    avg_latency = total_ms / total if total > 0 else 0

    output = {
        "test": "HDC_same_queries",
        "source": source_file,
        "total": total,
        "correct_action": correct_action,
        "correct_app": correct_app,
        "correct_any": correct_any,
        "done_count": done_count,
        "ask_count": ask_count,
        "errors": error_count,
        "total_latency_ms": round(total_ms, 1),
        "avg_latency_ms": round(avg_latency, 1),
        "action_accuracy_top1": round(correct_action / total * 100, 1),
        "app_accuracy": round(correct_app / total * 100, 1),
        "action_accuracy_any": round(correct_any / total * 100, 1),
        "per_query": results,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"HDC Benchmark — Same {total} queries as GPT-4o")
    print(f"{'='*60}")
    print(f"Action correct (top-1):   {correct_action}/{total} ({correct_action/total*100:.1f}%)")
    print(f"Action in candidates:     {correct_any}/{total} ({correct_any/total*100:.1f}%)")
    print(f"App correct:              {correct_app}/{total} ({correct_app/total*100:.1f}%)")
    print(f"DONE (routed instantly):  {done_count} ({done_count/total*100:.1f}%)")
    print(f"ASK (clarification):      {ask_count} ({ask_count/total*100:.1f}%)")
    print(f"Errors:                   {error_count}")
    print(f"Avg latency:              {avg_latency:.1f}ms")
    print(f"Results: {output_file}")


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else "benchmark/results_A_gpt-4o.json"
    output = sys.argv[2] if len(sys.argv) > 2 else "benchmark/results_hdc_same_queries.json"
    run_benchmark(source, output)
