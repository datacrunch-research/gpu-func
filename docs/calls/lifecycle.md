# Submit, wait, and cancel

## Submit

`fn.spawn()` and `Client.submit()` create a Call. The SDK generates an idempotency key unless you
provide one.

```python
import gfaas

def add_one(value: int) -> int:
    return value + 1

if __name__ == "__main__":
    with gfaas.Client() as client:
        job = client.submit(
            image="cuda-nvcc",
            function=add_one,
            args=(42,),
            gpu_count=1,
            gpu_type="gb300",
            timeout_s=300,
            idempotency_key="my-key-1",
        )
        print(job.wait())
```

Use `fn.spawn()` for an automatically generated idempotency key. If the caller must provide and
retain that key, use `Client.submit()`.

## Idempotency

`POST /calls` requires an `Idempotency-Key`. A retry with the same key and the same request returns
the original Call. A retry with the same key and different content returns HTTP 409.

If a network error interrupts a submission, retry with the same key. Then the service returns the
original Call instead of starting a second one.

If recovery matters, provide the key before the first `Client.submit()` call. If the SDK generates
the key and submission raises, the caller cannot recover that generated value.

## Wait

```python
result = job.wait(timeout_s=150)
```

`wait()` polls the Call until it reaches a terminal state. It returns the decoded result value. If
the Call ends in a non-success state, `wait()` raises `GfaasError`. If the deadline passes first, it
raises `TimeoutError`.

## Inspect

```python
state = job.status()
attempts = client.list_attempts(job.call_id)
```

`status()` returns the public Call resource. It includes the current state, effective resources,
timestamps, and the current Attempt identity.

`list_attempts()` returns the infrastructure Attempts. A selected Attempt includes the worker, GPU
count, and GPU models.

## Cancel

```python
job.cancel(reason="user request")
```

Cancellation is a request. A queued Call can stop before assignment. If preparation is active, the
subscriber stops new transfers and removes its temporary files. For a running Call, the coordinator
sends the request to its assigned worker.

Read the returned Call state. Then use `wait()` or `status()` until the Call reaches a terminal
state. If cancellation succeeds, `wait()` raises `GfaasError` because the Call did not succeed.

## Reconnect from another process

Store the Call ID before a long wait. A new process can inspect or wait for that Call with the same
principal's API credentials:

```python
import os

import gfaas

with gfaas.Client() as client:
    call_id = os.environ["SAVED_CALL_ID"]
    print(gfaas.call_status_summary(client.get_call(call_id)))
    print(client.get_call_logs(call_id))
    result = client.wait_for_result(call_id, timeout_s=60)
```

This does not resubmit the workload. `wait_for_result()` decodes the Python result, so the
[serialization trust rules](../python-functions.md#serialization-and-trust) apply. For status-only
tools, use `get_call()` and avoid result decoding.

A timeout ends the local wait, not the remote Call. A failed Call is terminal; waiting again does
not restart it. Submit a new Call if you need another attempt. Use an application checkpoint when
resuming computation. A resumable Artifact upload is a separate operation.
