# Artifacts

An Artifact is an immutable blob or directory tree. Each blob has a SHA-256 digest. The service
makes sure that the digest matches the bytes.

You use Artifacts for data that crosses the public API: source bundles, function inputs, results,
and files your function reads.

Artifacts belong to the principal for the API key. Keys that map to the same principal share access
to those Artifacts.

- [Upload and download](upload-download.md)
- [Pass an ArtifactRef into a function](ref.md)
- [Outputs, logs, and retention](outputs.md)
