# Images

An image is the container filesystem your function runs in. You select an operator-provided image in
one of two ways.

| Kind           | Source                                                |
| -------------- | ----------------------------------------------------- |
| Prepared image | Already present on the GPU machines                   |
| Remote image   | Content-addressed image published to an object origin |

Operators install prepared images on each worker. Remote images let workers fetch filesystem objects
through a local cache when a Call reads them.

Both image types need operator preparation. Select an image that the operator provides for your
pool. SDK users do not publish images or receive storage credentials.

- [Prepared images](prepared.md)
- [Remote images](remote.md)
