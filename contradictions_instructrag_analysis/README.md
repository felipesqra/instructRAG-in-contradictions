# Phase 2 Results Analysis

This directory contains only the reproducible analysis code and generated analysis outputs for the Phase 2 RAG comparison experiment.

Run the latest complete Phase 2 execution automatically:

```bash
../contradictions_instructrag_evaluation/.venv/bin/python analyze_phase2_results.py
```

Run a specific execution:

```bash
../contradictions_instructrag_evaluation/.venv/bin/python analyze_phase2_results.py \
  --run-dir ../contradictions_instructrag_evaluation/outputs/20260706T081559Z_phase2_rag_comparison
```

By default, outputs are written to:

```text
analysis_outputs/YYYYMMDDTHHMMSS_phase2_results_analysis/
```

The script does not alter the original experiment outputs.
