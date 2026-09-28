# Calls

A Call is one asynchronous invocation of a Function. A Call is a durable resource. It survives
client disconnects and coordinator restarts.

## Lifecycle states

| State        | Meaning                                                      |
| ------------ | ------------------------------------------------------------ |
| `pending`    | The coordinator admits the Call.                             |
| `queued`     | The Call awaits assignment, including subscriber preparation.               |
| `starting`   | A worker accepted the Call and prepares the run.             |
| `running`    | The workload process is active.                              |
| `cancelling` | The coordinator accepted a cancellation request.             |
| `succeeded`  | The worker returned a successful result.                     |
| `failed`     | Delivery, preparation, execution, or result handling failed. |
| `timed_out`  | The worker reported an execution timeout.                    |
| `cancelled`  | Cancellation reached a terminal outcome.                     |

The final four states are terminal. The `cancelling` state is not terminal. A Call does not always
pass through every nonterminal state.

- [Submit, wait, and cancel](lifecycle.md)
- [Events and live logs](events.md)

## Preparation before assignment

A Call can remain `queued` while a subscriber transfers its inputs.
`job.status()["preparation"]` reports this work separately from the durable Call state.
The same field can appear between stages of a `running` Call.

```python
from gfaas import call_status_summary

print(call_status_summary(job.status()))
```

Example output:

```text
state=queued preparation=active phase=transferring worker=worker-1 generation=2 files=0 bytes=0 downloaded=65536 uploaded=0
```

The snapshot includes completed file and byte counts, transfer counters, known totals, and the last report time.
Transfer counters include unfinished files. They do not establish that the worker verified or retained those bytes.
Totals can grow as the subscriber reads metadata. `totals_complete` identifies complete totals.

After 30 seconds without a report, `preparation.status` becomes `stalled`.
This means that progress is unconfirmed. It does not prove that the worker is dead.
Image resolution and file verification can also have periods without reported progress.

A placement retry removes the previous snapshot. Assignment, cancellation, and completion also remove it.
A new subscriber lifetime resets its counters. The `reporter_id` and `placement_generation` identify that boundary.
The original capacity-wait deadline still applies.

Older servers omit `preparation`. The SDK still accepts their status responses.
