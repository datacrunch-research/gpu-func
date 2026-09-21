# Images

An image is the container filesystem your function runs in. You select a prepared or remote image.

| Kind           | Source                                                |
| -------------- | ----------------------------------------------------- |
| Prepared image | Already present on the GPU machines                   |
| Remote image   | Content-addressed image published to an object origin |

Operators install prepared images on each worker. Remote images let workers fetch filesystem objects
through a local cache when a Call reads them. An operator can publish a shared remote image, and a
producer can publish a private remote image with the separate image-producer tools. The SDK never
receives storage credentials.

- [Prepared images](prepared.md)
- [Remote images](remote.md)
- [Qualify a published image](qualification.md)
