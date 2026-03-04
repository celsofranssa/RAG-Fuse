"""
Generate target_descriptions.pkl for a given dataset using OpenAI Batch API.

This script:
  1. Loads samples.pkl from resource/dataset/{DATASET}/
  2. Extracts all unique labels
  3. Selects up to 256 samples uniformly across labels
  4. For each label, sends text context to GPT-4 via Batch API (50% cheaper)
  5. Polls the batch until completion and saves {label: description} as
     resource/llm/{DATASET}/target_descriptions.pkl

Usage:
    export OPENAI_API_KEY="sk-..."

    # Batch mode (default – 50% cost discount, 24h turnaround)
    python generate_target_descriptions.py --dataset DBLP
    python generate_target_descriptions.py --dataset DBLP --model gpt-4o

    # Resume polling an existing batch
    python generate_target_descriptions.py --dataset DBLP --batch_id batch_abc123

    # Synchronous mode (immediate, full price)
    python generate_target_descriptions.py --dataset DBLP --sync

    # Dry run – estimate tokens and cost without calling the API
    python generate_target_descriptions.py --dataset DBLP --model gpt-4o --dry_run
"""

import argparse
import json
import logging
import os
import pickle
import random
import time
from collections import defaultdict
from pathlib import Path

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pricing per 1M tokens (USD) – update as needed
# ---------------------------------------------------------------------------
MODEL_PRICING = {
    # model_name: (input_cost_per_1M, output_cost_per_1M)
    "gpt-4":        (30.00, 60.00),
    "gpt-4-turbo":  (10.00, 30.00),
    "gpt-4o":       ( 2.50,  10.00),
    "gpt-4o-mini":  ( 0.15,   0.60),
    "gpt-4.1":      ( 2.00,   8.00),
    "gpt-4.1-mini": ( 0.40,   1.60),
    "gpt-4.1-nano": ( 0.10,   0.40),
    "o3-mini":      ( 1.10,   4.40),
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "resource" / "dataset"
LLM_DIR = BASE_DIR / "resource" / "llm"


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------
def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Count tokens in `text` using tiktoken.
    Falls back to a word-based approximation (1 token ≈ 0.75 words) if
    tiktoken is not installed.
    """
    if HAS_TIKTOKEN:
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    else:
        # rough approximation
        return int(len(text.split()) / 0.75)


def count_chat_tokens(system_msg: str, user_msg: str, model: str = "gpt-4o") -> int:
    """
    Estimate total input tokens for a ChatCompletion request with
    one system message and one user message.
    Adds ~7 tokens of overhead per message for chat formatting.
    """
    overhead = 7  # per-message overhead (role, name, delimiters)
    return (
        count_tokens(system_msg, model)
        + count_tokens(user_msg, model)
        + overhead * 2
        + 3  # reply priming
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_samples(dataset: str) -> list[dict]:
    """Load samples.pkl for the given dataset."""
    path = DATASET_DIR / dataset / "samples.pkl"
    if not path.exists():
        raise FileNotFoundError(f"samples.pkl not found at {path}")
    with open(path, "rb") as f:
        samples = pickle.load(f)
    logger.info(f"Loaded {len(samples)} samples from {path}")
    return samples


def extract_labels(samples: list[dict]) -> list[str]:
    """Return sorted list of unique labels across all samples."""
    labels = set()
    for sample in samples:
        for label in sample["labels"]:
            labels.add(label)
    return sorted(labels)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def select_samples_uniformly(
    samples: list[dict],
    labels: list[str],
    max_total: int = 256,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """
    Select up to `max_total` samples distributed uniformly across labels.
    Returns {label: [list of sample dicts]}.
    """
    rng = random.Random(seed)

    # Group sample indices by label
    label_to_samples: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        for label in sample["labels"]:
            label_to_samples[label].append(sample)

    num_labels = len(labels)
    per_label = max_total // num_labels  # uniform budget per label

    selected: dict[str, list[dict]] = {}
    for label in labels:
        pool = label_to_samples[label]
        k = min(per_label, len(pool))
        selected[label] = rng.sample(pool, k)
        logger.info(f"  Label '{label}': selected {k}/{len(pool)} samples")

    total = sum(len(v) for v in selected.values())
    logger.info(f"Total selected samples: {total} (budget: {max_total})")
    return selected


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_context_for_label(
    label: str,
    label_samples: list[dict],
    max_text_tokens: int = 128,
) -> str:
    """
    Build the text context block for a label.
    Each sample contributes truncated text + its labels.
    """
    lines = []
    for sample in label_samples:
        truncated_text = " ".join(sample["text"].split()[:max_text_tokens])
        sample_labels = "; ".join(sample["labels"])
        lines.append(f"  Text: {truncated_text}")
        lines.append(f"  Labels: {sample_labels}")
        lines.append("")
    return "\n".join(lines)


def build_gpt4_prompt(label: str, context_block: str) -> str:
    """
    Build the prompt sent to GPT-4 to generate a target description
    for a given label based on its associated texts.
    """
    return (
        f"You are an expert in text classification and domain analysis.\n\n"
        f"Below is a set of texts and their associated labels from an academic dataset. "
        f"Your task is to generate a concise and informative description for the label '{label}'.\n\n"
        f"The description should:\n"
        f"- Capture the essence of the label as reflected in the texts\n"
        f"- Focus on technical aspects, key concepts, techniques, and challenges\n"
        f"- Be a single paragraph (3-5 sentences)\n\n"
        f"Texts and Labels:\n"
        f"{context_block}\n"
        f"Now, provide a detailed and accurate description for the label '{label}':\n"
    )


# ---------------------------------------------------------------------------
# GPT-4 interaction
# ---------------------------------------------------------------------------
def call_gpt4(
    prompt: str,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: int = 512,
    max_retries: int = 5,
) -> str:
    """Call OpenAI ChatCompletion API with retry logic."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package not installed. Run: pip install openai"
        )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable not set. "
            "Export it before running: export OPENAI_API_KEY='sk-...'"
        )

    client = OpenAI(api_key=api_key)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a domain expert that generates precise, "
                            "technical descriptions of text classification labels."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            wait = 2**attempt
            logger.warning(
                f"GPT-4 call failed (attempt {attempt}/{max_retries}): {e}. "
                f"Retrying in {wait}s..."
            )
            time.sleep(wait)

    raise RuntimeError(f"GPT-4 call failed after {max_retries} retries")


# ---------------------------------------------------------------------------
# OpenAI client helper
# ---------------------------------------------------------------------------
SYSTEM_MSG = (
    "You are a domain expert that generates precise, "
    "technical descriptions of text classification labels."
)


def get_openai_client():
    """Return an authenticated OpenAI client."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable not set. "
            "Export it before running: export OPENAI_API_KEY='sk-...'"
        )
    return OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# Batch API functions
# ---------------------------------------------------------------------------
def prepare_batch_jsonl(
    labels: list[str],
    prompts: dict[str, str],
    model: str,
    temperature: float,
    max_tokens: int = 512,
    output_dir: Path = None,
) -> tuple[str, dict[str, str]]:
    """
    Write a .jsonl batch input file and a mapping JSON.
    Returns (jsonl_path, {custom_id: label} mapping).
    """
    id_to_label = {}
    jsonl_path = (output_dir / "batch_input.jsonl") if output_dir else Path("batch_input.jsonl")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with open(jsonl_path, "w") as f:
        for i, label in enumerate(labels):
            custom_id = f"label-{i}"
            id_to_label[custom_id] = label
            line = json.dumps({
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_MSG},
                        {"role": "user", "content": prompts[label]},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            })
            f.write(line + "\n")

    # Save mapping so we can resume with --batch_id later
    mapping_path = jsonl_path.parent / "batch_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(id_to_label, f, indent=2)

    logger.info(f"Batch input written: {jsonl_path} ({len(labels)} requests)")
    logger.info(f"Label mapping saved: {mapping_path}")
    return str(jsonl_path), id_to_label


def submit_batch(client, jsonl_path: str):
    """Upload JSONL file and create a batch job. Returns the Batch object."""
    logger.info("Uploading batch input file to OpenAI...")
    with open(jsonl_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    logger.info(f"File uploaded: id={file_obj.id}")

    logger.info("Creating batch job...")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    logger.info(f"Batch created: id={batch.id} | status={batch.status}")
    return batch


def poll_batch(client, batch_id: str, poll_interval: int = 30):
    """Poll batch status until a terminal state is reached."""
    terminal_states = {"completed", "failed", "expired", "cancelled"}
    logger.info(f"Polling batch {batch_id} every {poll_interval}s...")

    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        logger.info(
            f"  status={batch.status} | "
            f"completed={counts.completed}/{counts.total} | "
            f"failed={counts.failed}"
        )
        if batch.status in terminal_states:
            return batch
        time.sleep(poll_interval)


def download_batch_results(client, output_file_id: str) -> list[dict]:
    """Download and parse the batch output JSONL."""
    logger.info(f"Downloading results (file: {output_file_id})...")
    content = client.files.content(output_file_id)
    text = content.text
    results = []
    for line in text.strip().split("\n"):
        if line.strip():
            results.append(json.loads(line))
    logger.info(f"Downloaded {len(results)} result(s)")
    return results


def parse_batch_results(
    results: list[dict],
    id_to_label: dict[str, str],
) -> dict[str, str]:
    """Parse batch output into {label: description} dict."""
    target_descriptions = {}
    errors = []
    for result in results:
        custom_id = result["custom_id"]
        label = id_to_label.get(custom_id, custom_id)

        if result.get("error"):
            errors.append((label, result["error"]))
            logger.error(f"  ERROR [{label}]: {result['error']}")
            continue

        response = result.get("response", {})
        if response.get("status_code") == 200:
            body = response["body"]
            content = body["choices"][0]["message"]["content"].strip()
            target_descriptions[label] = content
            logger.info(f"  [{label}]: {content[:120]}...")
        else:
            errors.append((label, response))
            logger.error(
                f"  NON-200 [{label}]: status_code={response.get('status_code')}"
            )

    if errors:
        logger.warning(f"{len(errors)} request(s) failed — see errors above.")

    return target_descriptions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate target_descriptions.pkl for a dataset using GPT-4o."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., DBLP, ACM, REUTERS, OHSUMED, TWITTER)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=256,
        help="Maximum total samples to select across all labels (default: 256)",
    )
    parser.add_argument(
        "--max_text_tokens",
        type=int,
        default=128,
        help="Max words per sample text to include in the context (default: 128)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o). Can also use gpt-4-turbo, etc.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for GPT-4o (default: 0.7)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sample selection (default: 42)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, print prompts without calling GPT-4o (useful for debugging)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path override. Default: resource/llm/{dataset}/target_descriptions.pkl",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Use synchronous API calls instead of Batch API (no 50%% discount)",
    )
    parser.add_argument(
        "--poll_interval",
        type=int,
        default=30,
        help="Seconds between batch status checks (default: 30)",
    )
    parser.add_argument(
        "--batch_id",
        type=str,
        default=None,
        help="Resume polling on an existing batch ID (skips prompt building & upload)",
    )

    args = parser.parse_args()

    # ---- 1. Load data ----
    logger.info(f"=== Generating target descriptions for dataset: {args.dataset} ===")
    samples = load_samples(args.dataset)
    labels = extract_labels(samples)
    logger.info(f"Found {len(labels)} unique labels: {labels}")

    # ---- 2. Select samples uniformly ----
    logger.info(f"Selecting up to {args.max_samples} samples uniformly across labels...")
    selected = select_samples_uniformly(
        samples, labels, max_total=args.max_samples, seed=args.seed
    )

    # ---- 3. Build prompts for all labels ----
    system_msg = SYSTEM_MSG
    max_output_tokens = 512

    # accumulators for dry-run token report
    total_input_tokens = 0
    total_output_tokens_est = 0
    per_label_tokens: list[dict] = []

    if args.dry_run and not HAS_TIKTOKEN:
        logger.warning(
            "tiktoken not installed – token counts are approximate. "
            "Install with: pip install tiktoken"
        )

    prompts: dict[str, str] = {}
    for label in labels:
        label_samples = selected[label]
        context_block = build_context_for_label(
            label, label_samples, max_text_tokens=args.max_text_tokens
        )
        prompts[label] = build_gpt4_prompt(label, context_block)

        if args.dry_run:
            input_tokens = count_chat_tokens(system_msg, prompts[label], model=args.model)
            total_input_tokens += input_tokens
            total_output_tokens_est += max_output_tokens
            per_label_tokens.append({
                "label": label,
                "input_tokens": input_tokens,
                "output_tokens_est": max_output_tokens,
            })
            logger.info(f"\n{'='*60}")
            logger.info(f"[DRY RUN] Label: {label}")
            logger.info(f"Prompt ({len(prompts[label])} chars / ~{input_tokens} input tokens)")
            logger.info(f"Estimated output: up to {max_output_tokens} tokens")
            logger.info(f"{'='*60}")

    # ---- 4. Generate descriptions ----
    target_descriptions: dict[str, str] = {}

    if args.dry_run:
        # ---- Dry-run: no API calls ----
        for label in labels:
            target_descriptions[label] = f"[DRY RUN] Description for '{label}'"

    elif args.batch_id:
        # ---- Resume: poll an existing batch ----
        llm_dir = LLM_DIR / args.dataset
        mapping_path = llm_dir / "batch_mapping.json"
        if not mapping_path.exists():
            logger.error(
                f"Cannot resume: {mapping_path} not found. "
                f"Run without --batch_id first to create a new batch."
            )
            return
        with open(mapping_path) as f:
            id_to_label = json.load(f)

        client = get_openai_client()
        batch = poll_batch(client, args.batch_id, args.poll_interval)

        if batch.status == "completed":
            results = download_batch_results(client, batch.output_file_id)
            target_descriptions = parse_batch_results(results, id_to_label)
        else:
            logger.error(f"Batch ended with status: {batch.status}")
            if batch.error_file_id:
                logger.error(f"Error file ID: {batch.error_file_id}")
            return

    elif args.sync:
        # ---- Synchronous mode (old behavior, full price) ----
        logger.info("Using synchronous API calls (no batch discount)...")
        for label in labels:
            logger.info(f"Generating description for label: '{label}'...")
            description = call_gpt4(
                prompts[label],
                model=args.model,
                temperature=args.temperature,
            )
            target_descriptions[label] = description
            logger.info(f"  -> Generated ({len(description)} chars): {description[:150]}...")

    else:
        # ---- Batch mode (default – 50% cost discount) ----
        logger.info("Using Batch API (50% cost discount, up to 24h turnaround)...")
        llm_dir = LLM_DIR / args.dataset
        client = get_openai_client()

        jsonl_path, id_to_label = prepare_batch_jsonl(
            labels,
            prompts,
            model=args.model,
            temperature=args.temperature,
            max_tokens=max_output_tokens,
            output_dir=llm_dir,
        )

        batch = submit_batch(client, jsonl_path)
        logger.info(
            f"\n  >>> Batch ID: {batch.id}\n"
            f"  >>> To resume later: python {Path(__file__).name} "
            f"--dataset {args.dataset} --batch_id {batch.id}\n"
        )

        batch = poll_batch(client, batch.id, args.poll_interval)

        if batch.status == "completed":
            results = download_batch_results(client, batch.output_file_id)
            target_descriptions = parse_batch_results(results, id_to_label)
        else:
            logger.error(f"Batch ended with status: {batch.status}")
            if batch.error_file_id:
                logger.error(f"Error file ID: {batch.error_file_id}")
            logger.info(
                f"You can retry later with: python {Path(__file__).name} "
                f"--dataset {args.dataset} --batch_id {batch.id}"
            )
            return

    if not target_descriptions:
        logger.error("No descriptions generated. Aborting save.")
        return

    # ---- 5. Save target_descriptions.pkl ----
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = LLM_DIR / args.dataset / "target_descriptions.pkl"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(target_descriptions, f)

    logger.info(f"\nSaved target_descriptions.pkl at: {output_path}")
    logger.info(f"Contents: {len(target_descriptions)} label descriptions")

    # ---- 6. Summary ----
    logger.info("\n=== Summary ===")
    for label, desc in target_descriptions.items():
        logger.info(f"  [{label}]: {desc[:100]}...")

    # ---- 7. Token / cost estimation (dry-run only) ----
    if args.dry_run and per_label_tokens:
        logger.info(f"\n{'='*60}")
        logger.info("TOKEN & COST ESTIMATION")
        logger.info(f"{'='*60}")
        logger.info(f"{'Label':<35} {'Input':>10} {'Output (est)':>14}")
        logger.info(f"{'-'*35} {'-'*10} {'-'*14}")
        for entry in per_label_tokens:
            logger.info(
                f"{entry['label']:<35} {entry['input_tokens']:>10,} {entry['output_tokens_est']:>14,}"
            )
        logger.info(f"{'-'*35} {'-'*10} {'-'*14}")
        logger.info(
            f"{'TOTAL':<35} {total_input_tokens:>10,} {total_output_tokens_est:>14,}"
        )

        # Cost estimate: sync vs batch (50% discount)
        in_price, out_price = MODEL_PRICING.get(
            args.model, MODEL_PRICING.get("gpt-4")
        )
        input_cost_sync  = (total_input_tokens / 1_000_000) * in_price
        output_cost_sync = (total_output_tokens_est / 1_000_000) * out_price
        total_cost_sync  = input_cost_sync + output_cost_sync

        batch_discount = 0.5
        input_cost_batch  = input_cost_sync * batch_discount
        output_cost_batch = output_cost_sync * batch_discount
        total_cost_batch  = total_cost_sync * batch_discount

        logger.info(f"\nModel: {args.model}")
        logger.info(f"Pricing: ${in_price:.2f} / 1M input, ${out_price:.2f} / 1M output")
        if args.model not in MODEL_PRICING:
            logger.warning(
                f"Model '{args.model}' not in pricing table – using gpt-4o pricing as fallback. "
                f"Update MODEL_PRICING dict for accurate estimates."
            )
        logger.info(f"")
        logger.info(f"  {'':30} {'Sync':>12} {'Batch (-50%)':>14}")
        logger.info(f"  {'─'*30} {'─'*12} {'─'*14}")
        logger.info(f"  {'Input cost:':<30} ${input_cost_sync:>10.4f} ${input_cost_batch:>12.4f}")
        logger.info(f"  {'Output cost (est):':<30} ${output_cost_sync:>10.4f} ${output_cost_batch:>12.4f}")
        logger.info(f"  {'─'*30} {'─'*12} {'─'*14}")
        logger.info(f"  {'TOTAL:':<30} ${total_cost_sync:>10.4f} ${total_cost_batch:>12.4f}")
        logger.info(f"")
        logger.info(
            f"Note: output tokens estimated at max ({max_output_tokens}/label). "
            f"Actual usage will likely be lower."
        )
        logger.info(f"{'='*60}")

    logger.info("\nDone! You can now run prompt optimization:")
    logger.info(
        f"  python main.py tasks=[prompt_opt] data={args.dataset} data.text_features_source=TXT"
    )


if __name__ == "__main__":
    main()
