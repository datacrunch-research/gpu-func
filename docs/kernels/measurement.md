# How kernel measurements work

This chapter is for readers who need to interpret the benchmark algorithm. Start with
[tuning controls](tuning.md) for normal use. Numbers in the algorithm description use the default
policy. Configured trial and replay caps still apply.

Input rings are independent copies of tensor storage used across repeated launches. They reduce
reuse of the same memory addresses. A compiler shard is one batch of configurations compiled by one
CPU Call.

The GPU builds independent sets until their logical tensor footprint exceeds its reported L2 size.
Shared storages are counted conservatively once. This is an allocation and scheduling contract, not
a guarantee of cache eviction: kernels may access only a subset, and resets can warm cache lines.
The supplied inputs must exercise the intended working set. Ring allocation and graph padding are
bounded by `max_input_sets` (65,536) and `max_ring_bytes` (8 GiB); exceeding the initial ring limits
fails clearly. If graph padding would exceed a ring limit, final benchmarking uses direct events
with per-call reset instead. Reports include the ring footprint and allocation size.

Pilot groups (eight variants by default) enqueue 100 L2 zeroing operations, then interleave three
single-launch event trials per variant, rotating inputs without intermediate cache flushes.
Synchronize once, take minima, and validate proposed new bests before using them for pruning. All
pilot groups finish before survivors are regrouped. Refinement uses up to
`min(250, ceil(1_000 / pilot_us))` launches per variant when at least ten are required, returning
each minimum. Reset runs outside timed intervals. Refined pruning uses a validated best; every
survivor passes the optional evaluator before final timing. Without an evaluator correctness is
assumed.

Final timing handles one configuration at a time. Set
`Z = max(1, min(100, floor(1_000 / refined_us)))`. For `Z >= 10`, capture Z calls per graph and
measure `ceil(25_000 / (Z * refined_us))` replays. Cycle distinct graphs spanning the ring because
CUDA graphs fix input pointers. Pad to whole graphs so every captured call uses a distinct slot
within its graph; this also allows resets outside replay timing. Return the minimum event duration
divided by Z. For `Z < 10`, use direct events for `max(25, ceil(25_000 / refined_us))` launches.
Both paths begin with 100 L2 flushes; final results include their trials and timing method.

Initial shards report all candidates and their final timings. Client reduction applies refined
pruning against the global best, then schedules only surviving configurations for additional
final-only measurements. Every completed replica must report a different physical GPU UUID. The
initial measurement counts toward `replication_factor` (three by default); final ranking uses the
arithmetic mean. After each replica shard completes, partial means are reduced and pruned again;
this is a heuristic, and partial results are never selected as completed winners. Replication is
also re-evaluated for correctness on each device.

The public vFunc scheduler selects a GPU model but has no device-exclusion field. Replica Calls
therefore reserve `replication_factor` GPUs on one worker and measure their distinct device UUIDs
separately. Devices already represented in that candidate's measurements are skipped. This avoids
repeatedly scheduling single-GPU retries onto the same available device. Each final-only shard
requires that many free GPUs on one worker; the selected worker must provide that number of GPUs.
Calls are bounded by the normal capacity deadline and `replication_max_attempts`. Failed capacity or
lack of enough distinct devices raises `TritonBenchmarkError` with retained reports and Call IDs.
Generic single-device dispatch also verifies identities and never counts duplicate UUID placements
as independent replicas.

`quick_benchmark_variants_per_job` defaults to 256 and GPU concurrency to four; these are
configurable batching limits. All phases use the active App and Function settings. The service must
provide the image and hardware identity described in [managed kernels](index.md).
