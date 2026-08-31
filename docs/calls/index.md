# Calls

A Call is one asynchronous invocation of a Function. A Call is a durable resource. It survives
client disconnects and coordinator restarts.

## Lifecycle states

| State        | Meaning                                                      |
| ------------ | ------------------------------------------------------------ |
| `pending`    | The coordinator admits the Call.                             |
| `queued`     | The Call waits for delivery and pool capacity.               |
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
