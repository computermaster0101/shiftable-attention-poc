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

# Tokenizer path (shared between generalist and shiftable model)
TOKENIZER_PATH = GENERAL_OUTPUT_DIR / "tokenizer.json"

# --------------------------------------------------------------------- #
# Domain stats + emergence logs (for geometric routing)
# --------------------------------------------------------------------- #

# JSON file containing per-domain centroid + diagonal covariance:
# {
#   "dim": 256,
#   "domains": {
#     "general": {
#       "count": ...,
#       "centroid": [...],
#       "var_diag": [...]
#     },
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

# Composite score:
#   R_k = α * s_k - β * m_k - γ * H_k + δ * r_k
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

# ---------------------------------------------------------------------
# Router covariance regularization
# ---------------------------------------------------------------------
# Small diagonal regularization term added to each domain covariance
# matrix (Σ_k + λI) when computing Mahalanobis distance.
#
# This ensures numerical stability, keeps covariance matrices
# positive-definite, and prevents overconfident routing when
# domain sample counts are small.
#
# Matches GRCLM PDF: Cov_k = E[xxᵀ] − μμᵀ + λI
ROUTER_COV_LAMBDA = 1e-3


# --------------------------------------------------------------------- #
# Emergent expert configuration
# --------------------------------------------------------------------- #

# Prefix used for auto-created emergent experts:
# emergent_001, emergent_002, ...
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

# Maximum number of recent unknown-domain embeddings kept in memory for
# the emergent buffer. When this is exceeded, the oldest embeddings are
# dropped. This bounds memory usage while still giving enough signal to
# detect coherent clusters.
EMERGENT_MAX_BUFFER_SIZE = 2048

# Persistent state for the emergent expert index & sample count.
EMERGENT_STATE_PATH = SHIFTABLE_OUTPUT_DIR / "emergent_state.json"

# --------------------------------------------------------------------- #
# Router hyperparameters (Section 3 in GRCLM)
# --------------------------------------------------------------------- #

# Numerical stability for diagonal covariances.
ROUTER_MIN_VAR = 1e-4
ROUTER_EPS = 1e-8

# Composite score weights (Definition 2).
#   R_k(q) = α s_k(q) − β m_k(q) − γ H_k(q) + δ r_k
ROUTER_ALPHA = 1.0   # similarity weight
ROUTER_BETA = 0.25   # Mahalanobis distance weight
ROUTER_GAMMA = 0.0   # entropy weight (H_k is not modeled by DomainRouter)
ROUTER_DELTA = 0.0   # support ratio weight (set >0 to use r_k in R_k)

# Unknown-domain thresholds (Definition 3).
# max_k s_k(q) < τ_s  ∨  min_k m_k(q) > τ_m  ∨  max_k r_k(q) < τ_r
ROUTER_UNKNOWN_MAX_SIM = 0.15
ROUTER_UNKNOWN_MIN_DIST = 5.0
# If set > 0, the router will treat queries as unknown when every domain
# has support r_k below this threshold.
ROUTER_UNKNOWN_MIN_SUPPORT = 0.0

# Sliding-window size for support ratios r_k (Eq. 6).
# r_k = N_k / sum_j N_j  over the most recent ROUTER_SUPPORT_WINDOW queries.
ROUTER_SUPPORT_WINDOW = 1024

# --------------------------------------------------------------------- #
# Emergent domain clustering (Section 3.5, Definition 4)
# --------------------------------------------------------------------- #

# Minimum cluster size N_min for the emergent buffer B.
EMERGENT_MIN_CLUSTER_SIZE = 64

# Maximum allowed average squared radius σ^2_max of the emergent buffer
# around its empirical mean. See Eq. (10)–(12) in the paper.
EMERGENT_MAX_CLUSTER_VARIANCE = 50.0

# Maximum number of recent unknown-domain embeddings kept in memory for
# the emergent buffer. Older embeddings are dropped.
EMERGENT_MAX_BUFFER_SIZE = 2048

# Prefix used for auto-created emergent experts:
# emergent_001, emergent_002, ...
EMERGENT_AUTO_PREFIX = "emergent"

# Minimum length of the prompt (in characters) to consider it useful for
# emergent specialist training. Short prompts are ignored.
EMERGENT_MIN_PROMPT_LENGTH = 20

# Number of emergent samples to collect before auto-registering a new specialist.
# This provides a fallback in addition to the geometric cluster condition.
EMERGENT_MAX_SAMPLES_PER_SPECIALIST = 200

# Persistent state for the emergent expert index & sample count.
EMERGENT_STATE_PATH = SHIFTABLE_OUTPUT_DIR / "emergent_state.json"

# --------------------------------------------------------------------- #
# Geometric router hyperparameters
# --------------------------------------------------------------------- #

# Numerical safety for Mahalanobis and cosine computations
ROUTER_MIN_VAR = 1e-4
ROUTER_EPS = 1e-8

# Composite score weights:
#   R_k(q) = α s_k(q) - β m_k(q) - γ H_k(q) + δ r_k
ROUTER_ALPHA = 1.0    # similarity weight
ROUTER_BETA = 0.25    # Mahalanobis distance weight
ROUTER_GAMMA = 0.5    # entropy weight (now active via _entropy_from_similarity)
ROUTER_DELTA = 0.1    # support ratio weight (set 0 to disable r_k in R_k)

# Unknown-domain thresholds:
# - If the best similarity is below ROUTER_UNKNOWN_MAX_SIM AND the best
#   Mahalanobis distance is above ROUTER_UNKNOWN_MIN_DIST, the router
#   will mark the query as unknown.
# - Additionally, if all domains are too high-entropy (H_k > max) or
#   all have very low support, the query is also treated as unknown.
ROUTER_UNKNOWN_MAX_SIM = 0.15        # similarity threshold
ROUTER_UNKNOWN_MIN_DIST = 5.0        # distance threshold

# Optional entropy-based unknown threshold. If set to None, entropy is
# NOT used for unknown detection. A reasonable starting value is around
# the entropy of p=0.75 in bits (~0.81), but we work in nats here.
ROUTER_UNKNOWN_MAX_ENTROPY = None     # in nats; set to None to disable

# Support-based unknown detection:
# If even the most supported domain has r_k < ROUTER_UNKNOWN_MIN_SUPPORT,
# the query is treated as unknown. Set to 0.0 to disable.
ROUTER_UNKNOWN_MIN_SUPPORT = 0.00    # e.g. 2% of recent routed queries

# r_k = N_k / sum_j N_j  over the most recent ROUTER_SUPPORT_WINDOW queries.
ROUTER_SUPPORT_WINDOW = 1024

# Minimum number of routed (non-unknown) queries before we trust support-based
# unknown detection. This prevents a cold-start catch-22 where support is 0 for
# all domains and everything is permanently marked unknown.
ROUTER_SUPPORT_WARMUP_MIN_EVENTS = 128


# --- Learned blend correction (Policy B: geometry + small learned correction) ---
# Combine logits as: alpha * log(domain_prior) + beta * learned_gate_logits
# Use alpha >> beta to keep geometry dominant.
GATE_GEOMETRY_ALPHA = 6.0
GATE_LEARNED_BETA = 1.0
# How many specialist advisors (besides base) to invite per query
ROUTER_TOP_K_ADVISORS = 3
