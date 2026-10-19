# Events and live logs

Every Call has a durable event stream. The stream carries lifecycle changes, preparation progress,
stdout, stderr, log truncation, and Artifact publication. Every event has a stable cursor.

If a process needs progress before the result is ready, use the event stream. If a process only
needs retained text, use `logs()`.

## Follow the logs

```python
for stream, text in job.iter_logs():
    print(f"[{stream}] {text}", end="")
```

If the process needs a bounded wait, set a total client deadline:

```python
for stream, text in job.iter_logs(timeout_s=900):
    print(f"[{stream}] {text}", end="")
```

With `timeout_s`, the SDK polls the durable event pages. Without it, the SDK follows the SSE stream.
With `follow=False`, the SDK reads the currently available event pages and returns.

The deadline starts when iteration starts. It includes requests, retries, waits, and time that your
code spends between events. Recovery does not reset it. The SDK limits HTTP socket timeouts to the
remaining time and checks the deadline between response chunks. A socket operation can return after
the deadline. The SDK then raises `TimeoutError` instead of delivering more events.

A local timeout does not change the Call's capacity wait or execution timeout. Closing the iterator
also leaves the remote Call running.

## Automatic recovery

`iter_events()` and `iter_logs()` recover from temporary connection errors. Recovery resumes after
the last delivered event cursor. It does not resubmit or cancel the Call.

The same rules apply to polling and SSE:

- Retry connection errors, network timeouts, and premature disconnects.
- Retry HTTP 408, 429, 500, 502, 503, and 504.
- Allow five retries per interruption, with delays of 0.5, 1, 2, 4, and 8 seconds.
- Limit each recovery period to 60 seconds. The total watch deadline can end recovery sooner.
- Honor `Retry-After` as seconds or an HTTP date. If its delay exceeds the recovery budget, raise
  the error without retrying early.
- Reset the recovery budget after delivery of an event or a successful polling cycle.
- Stop immediately for authentication errors, other HTTP errors, or invalid event data.

If recovery fails, the iterator raises the last transport error or `GfaasError`. HTTP errors include
`GfaasError.status_code`. If the total deadline expires, it raises `TimeoutError`. Keyboard
interrupts and iterator closure stop observation without further retries.

After SSE closes, the SDK reads Call status to distinguish completion from an interrupted stream.
For a completed Call, it drains the remaining events. This also handles a supplied cursor that
follows the terminal event.

These guarantees apply to retained events. Reconnection cannot recover output that the service
already discarded under its retention limits.

## Reconnect from your application

Automatic recovery retains its cursor within the iterator. To resume after a process restart or an
exhausted retry budget, save the Call ID and cursor in your application.

Store the cursor after you process an event:

```python
cursor = None

for event in job.iter_events(follow=True):
    print(event["type"], event.get("state", ""))
    cursor = event["cursor"]
```

After a restart or an exhausted retry budget, create a new iterator after the stored cursor:

```python
for event in job.iter_events(after=cursor, follow=True):
    print(event["type"], event.get("state", ""))
    cursor = event["cursor"]
```

The service returns events after that cursor. The stream ends when the Call reaches a terminal
state.

## Event fields

Every event contains `cursor`, `call_id`, `type`, `occurred_at`, and `attributes`. Attempt events
also contain `attempt_id`.

State events expose `state` at the top level. Output events expose `stream_data` at the top level.

A `preparation` event reports work before worker acceptance. Its `attributes` contain `phase`,
`completed_files`, and `completed_bytes`. The phase can report image resolution, bundle upload,
Artifact staging, Artifact reuse, transfer progress, or preparation completion.

Updated subscribers report changing transfer counters at most once every two seconds. The report can
arrive before the first file completes. The event includes `worker_id`, `placement_generation`,
`reporter_id`, and a reporter `sequence`. `downloaded_bytes` and `uploaded_bytes` measure transport
within that reporter lifetime. `total_files`, `total_bytes`, and `totals_complete` describe known
attached Artifact content.

Retained events describe past observations.
[Current Call status](index.md#preparation-before-assignment) reports the current preparation and
its freshness. After reconnect, clients can read status without reconstructing it from all previous
events.

The event type `retention.truncated` means that the worker discarded output after the configured log
limit.

## Without the SDK

The same durable events are available over HTTP. Replace `call_...` with the ID printed at
submission:

```bash
curl --no-buffer --fail --silent --show-error \
  -H "X-API-Key: ${GFAAS_API_KEY}" \
  "${GFAAS_API_BASE}/v1/calls/call_.../events?follow=true"
```

Use `follow=false&limit=1000` for a bounded page. The page includes a `next_cursor` field.

Use a non-null `next_cursor` as the next `after` value. A null value means that the page contains
all events currently available.
