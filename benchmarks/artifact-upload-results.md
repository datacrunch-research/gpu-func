# Artifact upload comparison

This comparison measures serial uploads against bounded concurrency through the public HTTPS API. It
uses three paired runs from one ARM64 Linux client with Python 3.12.12.

Each run contains two cases:

- One 64 MiB file.
- One tree with 32 files of 2 MiB each.

Every case uses fresh random bytes. This avoids intentional content deduplication between runs. The
service revision and endpoint remain the same. The pair order is serial/concurrent,
concurrent/serial, then serial/concurrent.

The timer includes file hashing, transfer, and publication. Data generation and cleanup are outside
the timer. These measurements use a shared dev service, so network and service load can vary.

## Versions and controls

- Serial client: `00867f263545ed4c89d241c49f433812c06ce14b`.
- Concurrent client: the SDK changes on `rd/concurrent-artifact-uploads`, with a limit of four.
- Service: `74e9bcc8d4020e1cf694724390c49e582919a8c2`.
- Request timeout: 180 seconds for both clients.
- [Reproduction instructions](README.md).

The accompanying JSON records each sample and hashes of the SDK source files used for the concurrent
runs. The CLI configuration change does not participate in this benchmark.

## Results

| Pair | Serial file | Concurrent file | Serial tree | Concurrent tree |
| ---- | ----------: | --------------: | ----------: | --------------: |
| 1    |     12.00 s |          7.95 s |     33.84 s |         17.16 s |
| 2    |     15.64 s |          7.54 s |     49.85 s |         15.39 s |
| 3    |     10.66 s |          4.65 s |     33.76 s |         14.59 s |

| Median |  Serial | Concurrent | Speedup |
| ------ | ------: | ---------: | ------: |
| File   | 12.00 s |     7.54 s |   1.59× |
| Tree   | 33.84 s |    15.39 s |   2.20× |

Concurrency improved both cases in every pair. All 768 MiB of uploads completed, and all benchmark
Artifacts were removed.

The [JSON results](artifact-upload-results.json) retain every sample and the measured SDK source
hashes. After measurement, final review added a queued-cancellation check, bounded explicit session
preparation, and deduplicated failure receipts. Regression tests cover the final code. These timings
precede those final guards.

## Cleanup and limits

A connection closed during cleanup of the second serial tree. The upload had already succeeded, and
its timer had stopped. The benchmark stopped before the third pair.

The remaining 11 child Artifacts were identified from their filenames, sizes, and creation times and
removed through the public API. The benchmark now retains publication receipts before cleanup and
reconciles ambiguous delete responses before a bounded retry. The third pair uses that cleanup
change for both clients.

The upload code stays unchanged throughout the comparison. No GPU workload, deployment, or service
configuration change is part of this benchmark.

These samples show the effect on these two workloads. They do not establish a general speedup or
separate network, coordinator assembly, and object-store costs.
