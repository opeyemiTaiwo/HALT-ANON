"""
Shared configuration, paths and helpers.

Set the input directory once, either by editing RAW_DIR below or by exporting
an environment variable before running anything:

    export HALLUBENCH_RAW=/path/to/your/files
    export HALLUBENCH_OUT=/path/to/write/outputs

Expected files in RAW_DIR (names as released):

    panel_raw_judge_labels_full.csv      per-judge H1-H6, 2 judges x 19,361
    full_labeled_dataset_full.csv        report text + ground truth (TEXT ONLY)
    embeddings_openai.npy                19,361 x 1536, optional
    embedding_index.csv                  row order for the .npy, optional
    Rater-A-scores-807.xlsx              human rater A, 130 reports
    Rater-B-scores-807.xlsx              human rater B, 130 reports
    Panel-A-scores-807.xlsx              panel labels for the same 130
    UCFCrime_Train.json / _Val / _Test   UCA annotations, optional (audit only)

IMPORTANT: full_labeled_dataset_full.csv aggregates the two judges with OR and
does not reproduce the published statistics. It is used here only as a source
of report text. All labels are rebuilt from panel_raw_judge_labels_full.csv
under the paper's rule: two-judge majority, ties broken toward no
hallucination (with two judges, AND).
"""
import os
import re
from pathlib import Path

RAW_DIR = Path(os.environ.get("HALLUBENCH_RAW", "./raw"))
OUT_DIR = Path(os.environ.get("HALLUBENCH_OUT", "./out"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

H = ["H1", "H2", "H3", "H4", "H5", "H6"]
TARGETS = H + ["any_hallucination"]
KEY = ["model", "technique", "video", "crime_type"]
JOIN = ["model", "technique", "video"]

HNAME = {"H1": "Scene Fabrication", "H2": "Crime Misclassification",
         "H3": "Crime Missed", "H4": "Severity Minimization",
         "H5": "Entity Fabrication", "H6": "Phantom Actors"}
AXIS_OF = {"H1": "fabrication", "H5": "fabrication", "H6": "fabrication",
           "H3": "omission", "H2": "distortion", "H4": "distortion"}
AXES = {"fabrication": ["H1", "H5", "H6"], "omission": ["H3"],
        "distortion": ["H2", "H4"]}

MULTI_TURN = {"Sequential", "ReAct", "Least-to-Most", "True-Iterative"}
SINGLE_TURN = {"Zero-Shot", "Chain-of-Thought", "Meta-Prompting",
               "Self-Consistency"}

JUDGE_FAIL = -1          # sentinel in the raw judge file; not a verdict
TRUNCATION_CAP = 3000    # judge input cap, in characters

BENCH_FILE = OUT_DIR / "benchmark_labels.csv.gz"

STOPWORDS = set("""a an the and or but if while of to in on at by for with from as is are
was were be been being it its this that these those there here he she they them his her
their you we i not no than then so such which who whom what when where how also into over
under after before up down out off again more most very can will just do does did done
have has had having""".split())

HEDGE_RE = re.compile(
    r"\b(may|might|could|possibly|perhaps|appears?|seems?|likely|unclear|uncertain|"
    r"presumably|apparently|suggests?|potentially|probably|cannot determine|"
    r"difficult to)\b", re.I)


def require(*names):
    """Fail early and clearly if an expected input file is absent."""
    missing = [n for n in names if not (RAW_DIR / n).exists()]
    if missing:
        raise SystemExit(
            f"Missing input file(s) in {RAW_DIR.resolve()}:\n  "
            + "\n  ".join(missing)
            + "\n\nSet HALLUBENCH_RAW to the directory holding them."
        )
    return [RAW_DIR / n for n in names]


def content_tokens(s):
    """Lowercase content words, stopwords and short tokens removed."""
    return [w for w in re.findall(r"[a-z]+", str(s).lower())
            if w not in STOPWORDS and len(w) > 2]


def load_benchmark():
    """Load the rebuilt benchmark table, with a clear error if not built yet."""
    import pandas as pd
    if not BENCH_FILE.exists():
        raise SystemExit(f"{BENCH_FILE} not found. Run 00_build_benchmark.py first.")
    df = pd.read_csv(BENCH_FILE)
    df["model_output"] = df.model_output.fillna("")
    df["ground_truth"] = df.ground_truth.fillna("")
    return df


def banner(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
