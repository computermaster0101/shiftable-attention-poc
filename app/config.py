from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------- #
# Project roots
# --------------------------------------------------------------------- #

# Root of the sma_poc project (this repo)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Root of the upstream shiftable_project (submodule / sibling repo)
SHIFTABLE_PROJECT_ROOT = PROJECT_ROOT / "shiftable_project"

# --------------------------------------------------------------------- #
# Data directories
# --------------------------------------------------------------------- #

# Root data directory for shiftable_project
DATA_ROOT = SHIFTABLE_PROJECT_ROOT / "data"

# Generalist corpus lives here (already provided by shiftable_project)
GENERAL_DATA_DIR = DATA_ROOT / "general"

# --------------------------------------------------------------------- #
# Output / checkpoint directories
# --------------------------------------------------------------------- #

OUTPUTS_ROOT = SHIFTABLE_PROJECT_ROOT / "outputs"

# Where the standalone generalist checkpoint + tokenizer are stored
GENERAL_OUTPUT_DIR = OUTPUTS_ROOT / "generalist"

# Where the shiftable (generalist + specialists) model and routing stats live
SHIFTABLE_OUTPUT_DIR = OUTPUTS_ROOT / "shiftable"

# Checkpoint paths
GENERALIST_CKPT_PATH = GENERAL_OUTPUT_DIR / "generalist.pt"
SHIFTABLE_CKPT_PATH = SHIFTABLE_OUTPUT_DIR / "shiftable.pt"
# Modular specialist artifacts (one .pt per specialist)
SHIFTABLE_BASE_PATH = SHIFTABLE_OUTPUT_DIR / "shiftable_base.pt"
SPECIALISTS_DIR = SHIFTABLE_OUTPUT_DIR / "specialists"
# Optional extension for specialist metadata sidecars
SPECIALIST_META_EXT = ".meta.json"

# Tokenizer path (shared between generalist and shiftable model)
TOKENIZER_PATH = GENERAL_OUTPUT_DIR / "tokenizer.json"

# --------------------------------------------------------------------- #
# Domain stats + emergence logs (for geometric routing)
# --------------------------------------------------------------------- #

# JSON file containing per-domain centroid + diagonal covariance:
# {
#   "dim": 256,
#   "domains": {
#     "general": {"count": ..., "centroid": [...], "var_diag": [...]},
#     ...
#   }
# }
DOMAIN_STATS_PATH = SHIFTABLE_OUTPUT_DIR / "domain_stats.json"

# JSONL log of "unknown domain" / emergent prompts for later analysis
EMERGENCE_LOG_PATH = SHIFTABLE_OUTPUT_DIR / "emergence_log.jsonl"

# --------------------------------------------------------------------- #
# Model architecture (must match what you train in shiftable_project)
# --------------------------------------------------------------------- #

# Transformer model dimension
D_MODEL = 256

# Number of attention heads
N_HEADS = 4

# Number of Transformer layers
N_LAYERS = 4

# Feedforward dimension inside each Transformer block
DIM_FEEDFORWARD = 1024

# Dropout used in Transformer blocks and embeddings
DROPOUT = 0.1

# Maximum sequence length (tokens)
MAX_SEQ_LEN = 256

# Whether specialists pool from CLS token or mean of sequence
USE_CLS_TOKEN_POOL = False

# --------------------------------------------------------------------- #
# Generalist training hyperparameters
# --------------------------------------------------------------------- #

# Minimum token frequency when building tokenizer vocab
GENERAL_MIN_FREQ = 5

# Maximum vocabulary size (including specials)
# Adjust to 20k–50k depending on corpus size and hardware limits
GENERAL_MAX_VOCAB_SIZE = 25000

# Batch size for generalist pretraining
GENERAL_BATCH_SIZE = 8

# Number of epochs for generalist training
GENERAL_EPOCHS = 3

# Learning rate for generalist optimizer
GENERAL_LR = 3e-4

# --------------------------------------------------------------------- #
# Shiftable (specialist) training hyperparameters
# --------------------------------------------------------------------- #

# Batch size for training the shiftable model (generalist + specialists)
SHIFTABLE_BATCH_SIZE = 8

# Number of epochs for shiftable training
SHIFTABLE_EPOCHS = 3

# Learning rate for shiftable optimizer
SHIFTABLE_LR = 3e-4

# --------------------------------------------------------------------- #
# Geometric router hyperparameters
# --------------------------------------------------------------------- #

# Minimum variance value for each dimension when inverting diagonal covariance.
# Prevents division by zero and overly confident distances.
ROUTER_MIN_VAR = 1e-4

# Numerical epsilon for norms / denominators
ROUTER_EPS = 1e-8

# Covariance regularization used when computing Mahalanobis distances:
# Cov_k = E[xxᵀ] − μμᵀ + λI
ROUTER_COV_LAMBDA = 1e-3

# Composite score weights:
#   R_k(q) = α s_k(q) − β m_k(q) − γ H_k(q) + δ r_k(q)
# where:
#   s_k = cosine similarity
#   m_k = Mahalanobis distance
#   H_k = entropy term (currently 0.0)
#   r_k = support term (currently 0.0)
ROUTER_ALPHA = 1.0
ROUTER_BETA = 0.25
ROUTER_GAMMA = 0.0
ROUTER_DELTA = 0.0

# Heuristic thresholds for flagging "unknown" domains.
# If the best similarity is below ROUTER_UNKNOWN_MAX_SIM and the best
# Mahalanobis distance is above ROUTER_UNKNOWN_MIN_DIST, the router will
# mark the query as is_unknown=True and send it to emergence.
ROUTER_UNKNOWN_MAX_SIM = -1.0
ROUTER_UNKNOWN_MIN_DIST = 1e9

# Optional entropy-based unknown threshold (in nats). None disables.
ROUTER_UNKNOWN_MAX_ENTROPY = None

# Support-based unknown detection (0.0 disables).
ROUTER_UNKNOWN_MIN_SUPPORT = 0.0

# Sliding-window size for support ratios r_k.
ROUTER_SUPPORT_WINDOW = 1024

# --------------------------------------------------------------------- #
# Emergent domain / expert configuration
# --------------------------------------------------------------------- #

# Prefix used for auto-created emergent experts: emergent_001, emergent_002, ...
EMERGENT_AUTO_PREFIX = "emergent"

# Minimum length of the prompt (in characters) to consider it useful for
# emergent specialist training. Short prompts are ignored.
EMERGENT_MIN_PROMPT_LENGTH = 20

# Number of emergent samples to collect before auto-registering a new specialist.
EMERGENT_MAX_SAMPLES_PER_SPECIALIST = 200

# Number of emergent samples to collect before auto-registering a new specialist.
EMERGENT_MAX_SAMPLES_PER_SPECIALIST = 200

# Minimum number of unknown-domain embeddings required in the in-memory
# emergent buffer before we consider spawning a new domain based on the
# geometric clustering condition (Definition 4 in GRCLM).
EMERGENT_MIN_CLUSTER_SIZE = 64

# Maximum allowed average squared radius (σ^2) of the emergent buffer
# around its empirical mean. If the buffer variance is below this value
# and EMERGENT_MIN_CLUSTER_SIZE is satisfied, a new specialist will be
# spawned even if EMERGENT_MAX_SAMPLES_PER_SPECIALIST has not yet been
# reached.
EMERGENT_MAX_CLUSTER_VARIANCE = 50.0

# Bound memory usage for the in-memory unknown buffer.
EMERGENT_MAX_BUFFER_SIZE = 2048

# Persistent state for the emergent expert index & sample count.
EMERGENT_STATE_PATH = SHIFTABLE_OUTPUT_DIR / "emergent_state.json"

# --------------------------------------------------------------------- #
# Learned blend correction (Policy B: geometry + small learned correction)
# --------------------------------------------------------------------- #

# Combine logits as: alpha * log(domain_prior) + beta * learned_gate_logits
# Use alpha >> beta to keep geometry dominant.
GATE_GEOMETRY_ALPHA = 6.0
GATE_LEARNED_BETA = 1.0

# How many specialist advisors (besides base) to invite per query
ROUTER_TOP_K_ADVISORS = 3
