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

Set a total client deadline if the process must stop after a bounded wait:

```python
for stream, text in job.iter_logs(timeout_s=900):
    print(f"[{stream}] {text}", end="")
```

With `timeout_s`, the SDK polls the durable event pages. The deadline includes queue and execution
time. It does not change the Call's capacity wait or execution timeout.

## Reconnect

Store the cursor after you process an event:

```python
cursor = None

for event in job.iter_events(follow=True):
    print(event["type"], event.get("state", ""))
    cursor = event["cursor"]
```

If the connection closes early, reconnect after the stored cursor:

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
Artifact staging, Artifact reuse, or preparation completion.

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
