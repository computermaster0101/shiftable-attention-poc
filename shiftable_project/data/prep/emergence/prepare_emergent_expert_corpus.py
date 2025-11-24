#!/usr/bin/env python3
"""
prepare_emergent_expert_corpus.py

This script reads emergent-query logs produced by the SMA PoC
(emergence_log.jsonl) and assembles a text corpus for a NEW specialist
expert domain.

Typical workflow:

1. Run the SMA API and let it log emergent queries to:
       shiftable_project/outputs/shiftable/emergence_log.jsonl

2. Inspect / filter those emergent queries with this script, and
   write a corpus file for a new specialist domain into:
       shiftable_project/data/<expert_name>/

3. Call the SMA API's /specialists endpoint (or use the UI) with
   that expert name to retrain the shiftable model with the new
   specialist head.

This script is intended to live at:

    shiftable_project/data/prep/emergence/prepare_emergent_expert_corpus.py

It uses only the Python standard library.
"""

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def compute_paths() -> Dict[str, Path]:
    """
    Compute important paths relative to this script's location.

    We expect the following structure:

      shiftable_project/
        data/
          prep/
            emergence/
              prepare_emergent_expert_corpus.py
        outputs/
          shiftable/
            emergence_log.jsonl

    Returns a dict with keys:
      - base_dir: shiftable_project root
      - data_dir: shiftable_project/data
      - outputs_dir: shiftable_project/outputs
      - emergence_log: outputs/shiftable/emergence_log.jsonl
    """
    script_path = Path(__file__).resolve()
    emergence_dir = script_path.parent

    # .../shiftable_project/data/prep/emergence
    prep_dir = emergence_dir.parent
    data_dir = prep_dir.parent
    base_dir = data_dir.parent

    outputs_dir = base_dir / "outputs"
    emergence_log = outputs_dir / "shiftable" / "emergence_log.jsonl"

    return {
        "base_dir": base_dir,
        "data_dir": data_dir,
        "outputs_dir": outputs_dir,
        "emergence_log": emergence_log,
    }


def load_emergence_records(log_path: Path) -> Iterable[Dict[str, Any]]:
    """
    Stream JSON lines from the emergence log.

    Each line is expected to be a JSON object with keys like:
      - timestamp
      - prompt
      - completion
      - embedding
      - routing: { metrics, best_domain, is_unknown, reason }
      - specialists
    """
    if not log_path.exists():
        raise FileNotFoundError(f"Emergence log not found: {log_path}")

    with open(log_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[warn] Skipping malformed JSON on line {line_num}: {e}")
                continue
            yield obj


def is_emergent(record: Dict[str, Any]) -> bool:
    """
    Heuristic: treat a record as "emergent" if either:

      - There are no specialists at the time (specialists == []), or
      - routing.is_unknown is True, or
      - routing.reason is 'unknown_domain' or 'no_domain_stats'.

    This aligns with the SMA PoC's EmergenceTracker behavior.
    """
    specialists = record.get("specialists") or []
    routing = record.get("routing") or {}
    is_unknown = bool(routing.get("is_unknown", False))
    reason = routing.get("reason", "")

    if not specialists:
        return True
    if is_unknown:
        return True
    if reason in ("unknown_domain", "no_domain_stats"):
        return True
    return False


def format_record_as_text(record: Dict[str, Any]) -> str:
    """
    Turn a single emergence record into a text block suitable for training.

    We include both the prompt and the completion, separated by markers.
    """
    prompt = record.get("prompt", "")
    completion = record.get("completion", "")

    # Basic normalization
    prompt = prompt.replace("\r\n", "\n").strip()
    completion = completion.replace("\r\n", "\n").strip()

    # Ensure non-empty strings to avoid blank lines
    prompt = prompt if prompt else "[EMPTY_PROMPT]"
    completion = completion if completion else "[EMPTY_COMPLETION]"

    block = textwrap.dedent(
        f"""
        ### PROMPT ###
        {prompt}

        ### COMPLETION ###
        {completion}

        """
    ).strip("\n") + "\n\n"

    return block


def build_corpus_for_expert(
    expert_name: str,
    max_samples: Optional[int],
    min_prompt_length: int,
    paths: Dict[str, Path],
    output_filename: Optional[str] = None,
) -> Path:
    """
    Build a corpus file for a new expert domain by sampling emergent
    queries from the emergence log.

    Args:
        expert_name: Name of the new domain (e.g., "cyberrisk").
        max_samples: Optional maximum number of emergent records to include.
        min_prompt_length: Minimum length of prompt string to be included.
        paths: Dict from compute_paths().
        output_filename: Optional filename for the corpus .txt. If not
                        provided, a default like 'emergent_<expert_name>.txt'
                        will be used.

    Returns:
        Path to the written corpus file.
    """
    data_dir = paths["data_dir"]
    emergence_log = paths["emergence_log"]

    # Domain corpus directory: shiftable_project/data/<expert_name>/
    expert_dir = data_dir / expert_name
    expert_dir.mkdir(parents=True, exist_ok=True)

    if output_filename:
        corpus_path = expert_dir / output_filename
    else:
        corpus_path = expert_dir / f"emergent_{expert_name}.txt"

    print(f"[info] Base directory          : {paths['base_dir']}")
    print(f"[info] Emergence log           : {emergence_log}")
    print(f"[info] Expert corpus directory : {expert_dir}")
    print(f"[info] Corpus output file      : {corpus_path}")
    print()

    count_total = 0
    count_selected = 0

    with open(corpus_path, "w", encoding="utf-8") as f_out:
        for record in load_emergence_records(emergence_log):
            count_total += 1

            if not is_emergent(record):
                continue

            prompt = record.get("prompt", "") or ""
            if len(prompt.strip()) < min_prompt_length:
                continue

            text_block = format_record_as_text(record)
            f_out.write(text_block)
            count_selected += 1

            if max_samples is not None and count_selected >= max_samples:
                break

    print(f"[info] Total records scanned   : {count_total}")
    print(f"[info] Emergent records kept   : {count_selected}")
    print(f"[info] Corpus written to       : {corpus_path}")
    return corpus_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a corpus for a new expert domain using emergent queries."
    )
    parser.add_argument(
        "--expert-name",
        required=True,
        help="Name of the new expert/domain (e.g., 'cyberrisk'). This becomes the directory under data/.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum number of emergent records to include in the corpus (default: 1000). Use -1 for no limit.",
    )
    parser.add_argument(
        "--min-prompt-length",
        type=int,
        default=10,
        help="Minimum prompt length (characters) required to include a record (default: 10).",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="",
        help="Optional explicit filename for the corpus (default: emergent_<expert_name>.txt).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = compute_paths()

    max_samples = None if args.max_samples is not None and args.max_samples < 0 else args.max_samples
    output_filename = args.output_filename if args.output_filename else None

    build_corpus_for_expert(
        expert_name=args.expert_name.strip(),
        max_samples=max_samples,
        min_prompt_length=max(0, int(args.min_prompt_length)),
        paths=paths,
        output_filename=output_filename,
    )


if __name__ == "__main__":
    main()
