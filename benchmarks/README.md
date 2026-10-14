# Artifact upload benchmark

`artifact_uploads.py` measures the public upload API. It creates fresh random content for each case
and removes the resulting Artifacts. It does not submit Calls or use GPUs.

Each repetition uploads one large file and one directory tree with the same total size. The timer
includes client hashing, transfer, and publication. It excludes data generation and Artifact
cleanup. The benchmark alternates concurrency order between repetitions.

## Run a comparison

Use an API credential with Artifact create and delete access. Set the API URL and credential in the
environment. Do not put the credential in command arguments.

```bash
export GFAAS_API_BASE=https://gpu.example.com/api
# Supply GFAAS_API_KEY through your normal credential mechanism.
python benchmarks/artifact_uploads.py \
  --client-revision "$(git rev-parse HEAD)" \
  --service-revision SERVICE_COMMIT \
  --concurrency 1 4 --repetitions 3 --mib 64 --files 32 \
  --inventory upload-inventory.jsonl \
  > upload-results.jsonl
```

This command uploads 768 MiB in total. Each JSON line records a workload, its duration, throughput,
source revisions, client platform, Artifact identity, and cleanup result. Use the deployed service
revision for `SERVICE_COMMIT`. Include a source hash in the client revision when the checkout has
uncommitted changes.

The inventory retains publication receipts and child IDs before cleanup starts. After a lost delete
response, cleanup checks the Artifact before a retry. It permits at most three delete attempts. The
benchmark stops on an upload or unresolved cleanup failure. The exception retains the file session
ID or completed tree children. These resources remain available for diagnosis.

For a comparison with an older SDK, run the same script with that revision on `PYTHONPATH`. Use
`--concurrency 1` for revisions that do not have the concurrency configuration. Keep the client
machine, API endpoint, service revision, file sizes, and file counts the same. Alternate old and new
runs. Compare medians and retain every sample.

The results measure the complete client upload operation. They do not isolate coordinator assembly,
object-store transfer, or network bandwidth. A higher concurrency limit cannot guarantee a speedup.
