TensorBoard Projector export

Files:
- query_embeddings.tsv / query_metadata.tsv : embeddings captured from logs/input_output_log.jsonl
- domain_centroids.tsv / domain_metadata.tsv : centroids from outputs/shiftable/domain_stats.json
- projector_config.pbtxt : TensorBoard projector config

How to run (from this folder):
1) python -m pip install tensorboard
2) tensorboard --logdir . --bind_all --port 6006

Then open the Projector tab and select:
- query_embeddings
- domain_centroids

Tip: In Projector, switch "Color by" to best_domain or domain.
