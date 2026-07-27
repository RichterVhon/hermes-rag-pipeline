#!/usr/bin/env python3
"""
Regression test runner for the RAG pipeline.

Runs a saved set of question/expected-keyword pairs against rag_query.py and
reports PASS/FAIL for each, so we can catch retrieval regressions automatically
after future pipeline changes instead of manually re-testing every time.

Usage:
    python3 run_regression_tests.py [path/to/regression_tests.json]

A test passes if every string in "expected_keywords" appears somewhere in the
combined retrieved content (case-insensitive), and none of the strings in
"not_expected_keywords" appear.

Note: this tests OUR retrieval pipeline (rag_query.py) directly, not Hermes's
final generated answer -- it checks that the right source material comes back,
not how Hermes phrases a response from it. A failure here means something in
ingestion, search, or reranking regressed; it does not mean Hermes's wording
was wrong.
"""
import json
import subprocess
import sys

DEFAULT_TEST_FILE = "/opt/data/regression_tests.json"
RAG_QUERY_PATH = "/opt/data/rag_query.py"


def run_query(question, source_filter=None):
    cmd = ["python3", RAG_QUERY_PATH, question]
    if source_filter:
        cmd.append(source_filter)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return result.stdout + result.stderr


def run_tests(test_file):
    with open(test_file) as f:
        tests = json.load(f)

    passed, failed = 0, 0
    for test in tests:
        test_id = test.get("id", test["question"][:40])
        output = run_query(test["question"], test.get("source_filter"))
        output_lower = output.lower()

        missing = [kw for kw in test.get("expected_keywords", []) if kw.lower() not in output_lower]
        unwanted = [kw for kw in test.get("not_expected_keywords", []) if kw.lower() in output_lower]

        if not missing and not unwanted:
            print(f"PASS  {test_id}")
            passed += 1
        else:
            print(f"FAIL  {test_id}")
            if missing:
                print(f"      missing expected keywords: {missing}")
            if unwanted:
                print(f"      found keywords that should be absent: {unwanted}")
            if test.get("note"):
                print(f"      note: {test['note']}")
            failed += 1

    print("---")
    print(f"{passed}/{passed + failed} tests passed")
    return failed == 0


if __name__ == "__main__":
    test_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEST_FILE
    ok = run_tests(test_file)
    sys.exit(0 if ok else 1)
