#!/usr/bin/env python3
"""Build the book and check rendered links and Python snippet syntax, without GPU work."""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


class Page(HTMLParser):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.links: list[str] = []
        self.ids: set[str] = set()
        self.feed(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if identity := values.get("id"):
            self.ids.add(identity)
        if tag == "a" and (href := values.get("href")):
            self.links.append(href)


def main() -> None:
    documents = [ROOT / "README.md", ROOT / "GUIDE.md"]
    documents.extend((ROOT / "docs").rglob("*.md"))
    snippets = 0
    for path in documents:
        if "target" in path.relative_to(ROOT).parts:
            continue
        text = path.read_text()
        for match in re.finditer(r"^```python[^\n]*\n(.*?)^```", text, re.M | re.S):
            ast.parse(match[1], filename=str(path))
            snippets += 1

    # Build from a clean copy so old docs/target output cannot mask missing pages or assets.
    with tempfile.TemporaryDirectory(prefix="vfunc-book-") as temporary:
        checkout = Path(temporary)
        shutil.copytree(ROOT / "docs", checkout / "docs", ignore=shutil.ignore_patterns("target"))
        shutil.copyfile(ROOT / "GUIDE.md", checkout / "GUIDE.md")
        subprocess.run(["mdbook", "build", str(checkout / "docs")], check=True)
        output = checkout / "docs" / "target"
        pages = {path.resolve(): Page(path.read_text()) for path in output.rglob("*.html")}
        errors = []
        for path, page in pages.items():
            # These aggregate/generated pages repeat relative links from several chapters.
            if path.name in {"print.html", "404.html", "toc.html"}:
                continue
            for href in page.links:
                url = urlsplit(href)
                if url.scheme or url.netloc:
                    continue
                target = (path.parent / unquote(url.path)).resolve() if url.path else path
                if not target.exists():
                    errors.append(f"{path.relative_to(output)}: missing target {href}")
                elif (
                    url.fragment
                    and target in pages
                    and unquote(url.fragment) not in pages[target].ids
                ):
                    errors.append(f"{path.relative_to(output)}: missing anchor {href}")
        if errors:
            raise SystemExit("\n".join(sorted(set(errors))))
    print(f"Book links and {snippets} Python snippet syntax checks passed.")


if __name__ == "__main__":
    main()
