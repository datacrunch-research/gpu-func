"""`vfunc image`: build a source file's container image on the image build service.

The CLI does not import the source file. It reads module-level assignments such as

    image = gfaas.Image.from_container("nvidia/cuda:...", python_version="3.12", pip_packages=[...])

whose arguments are literals or names bound to literals earlier in the module.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import json
import os
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any

import httpx
from argcomplete.completers import FilesCompleter

from gfaas import Image
from gfaas.client import Client

from . import auth
from .errors import CliError

IMAGE_COMMANDS = frozenset({"image", "images"})
DASHBOARD_COMMAND = "image-dashboard"
# The build service's dashboard listens on its host's loopback; operators forward it to this port.
DEFAULT_DASHBOARD_URL = "http://127.0.0.1:8091/"
_TERMINAL_STATES = {"ready", "failed"}


def add_commands(commands: Any) -> None:
    image = commands.add_parser(
        "image", aliases=["images"], help="build container images on the image build service"
    )
    image_commands = image.add_subparsers(dest="image_command", required=True)
    publish = image_commands.add_parser(
        "publish", help="build and publish the Image.from_container(...) a file declares"
    )
    target = publish.add_argument("target", help="file.py or file.py:VARIABLE")
    target.completer = FilesCompleter(("py",))
    publish.add_argument(
        "--architecture",
        choices=("amd64", "arm64"),
        help="image CPU architecture (default: the build service's architecture)",
    )
    publish.add_argument(
        "--no-wait", action="store_true", help="return after the service accepts the build"
    )
    publish.add_argument("--json", action="store_true", help="write the final build as JSON")
    status = image_commands.add_parser("status", help="show the build of a file's image")
    status_target = status.add_argument("target", help="file.py or file.py:VARIABLE")
    status_target.completer = FilesCompleter(("py",))
    status.add_argument("--json", action="store_true", help="write the build as JSON")
    listing = image_commands.add_parser("list", help="list your principal's image builds")
    listing.add_argument("--json", action="store_true", help="write the builds as JSON")
    dashboard = commands.add_parser(
        DASHBOARD_COMMAND, help="open the image build service's dashboard in a browser"
    )
    dashboard.add_argument(
        "--url",
        default=os.environ.get("GFAAS_IMAGE_DASHBOARD_URL", DEFAULT_DASHBOARD_URL),
        help=f"dashboard address (default: $GFAAS_IMAGE_DASHBOARD_URL or {DEFAULT_DASHBOARD_URL})",
    )
    dashboard.add_argument(
        "--no-browser", action="store_true", help="print the address instead of opening it"
    )


class UnparsableSourceError(CliError):
    """The local Python cannot parse the source file."""


def _image_constructor(value: ast.expr) -> bool:
    if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Attribute):
        return False
    receiver = value.func.value
    receiver_name = receiver.attr if isinstance(receiver, ast.Attribute) else None
    if isinstance(receiver, ast.Name):
        receiver_name = receiver.id
    return value.func.attr == "from_container" and receiver_name == "Image"


def _assignment(statement: ast.stmt) -> tuple[str, ast.expr] | None:
    if (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    ):
        return statement.targets[0].id, statement.value
    if (
        isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and statement.value is not None
    ):
        return statement.target.id, statement.value
    return None


def _argument(node: ast.expr, constants: dict[str, Any], location: str) -> Any:
    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id]
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as error:
        raise CliError(
            f"{location}: Image.from_container arguments must be literals or names of "
            "module-level literals"
        ) from error


def declared_images(path: Path) -> dict[str, Image]:
    """The container images a file assigns at module level, by variable name."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as error:
        raise UnparsableSourceError(f"cannot parse {path}: {error}") from error
    constants: dict[str, Any] = {}
    images: dict[str, Image] = {}
    for statement in tree.body:
        assignment = _assignment(statement)
        if assignment is None:
            continue
        variable, value = assignment
        constants.pop(variable, None)
        images.pop(variable, None)
        if not _image_constructor(value):
            with contextlib.suppress(
                ValueError, TypeError, SyntaxError, MemoryError, RecursionError
            ):
                constants[variable] = ast.literal_eval(value)
            continue
        assert isinstance(value, ast.Call)
        location = f"{path}:{value.lineno}"
        if any(isinstance(arg, ast.Starred) for arg in value.args) or any(
            keyword.arg is None for keyword in value.keywords
        ):
            raise CliError(f"{location}: Image.from_container cannot use * or ** arguments")
        args = [_argument(arg, constants, location) for arg in value.args]
        kwargs = {
            str(keyword.arg): _argument(keyword.value, constants, location)
            for keyword in value.keywords
        }
        try:
            images[variable] = Image.from_container(*args, **kwargs)
        except TypeError as error:
            raise CliError(f"{location}: {error}") from error
    return images


def declared_image(path: Path, variable: str | None = None) -> tuple[str, Image] | None:
    """The image a file declares, or None when it declares none and no variable was named."""
    images = declared_images(path)
    if variable is not None:
        if variable not in images:
            raise CliError(
                f"{path} does not assign {variable} = Image.from_container(...) at module level"
            )
        return variable, images[variable]
    if not images:
        return None
    if len(images) > 1:
        raise CliError(
            f"{path} declares several container images ({', '.join(sorted(images))}); "
            f"select one as {path}:NAME"
        )
    return next(iter(images.items()))


def _target(value: str) -> tuple[Path, str | None]:
    prefix, separator, suffix = value.rpartition(":")
    path, variable = Path(value), None
    if separator and prefix.endswith(".py"):
        path, variable = Path(prefix), suffix
    if variable is not None and not variable.isidentifier():
        raise CliError("select an image as file.py:VARIABLE")
    if not path.is_file():
        raise CliError(f"source file does not exist: {path}")
    return path, variable


def _target_image(value: str) -> tuple[str, Image]:
    path, variable = _target(value)
    selected = declared_image(path, variable)
    if selected is None:
        raise CliError(f"{path} does not assign an Image.from_container(...) at module level")
    return f"{path}:{selected[0]}", selected[1]


def _summary(build: dict[str, Any]) -> str:
    fields = [f"image={build.get('name')}", f"state={build.get('state')}"]
    if build.get("digest"):
        fields.append(f"digest={build['digest']}")
    return "[vfunc] " + " ".join(fields)


def _follow(client: Client, name: str, poll_interval_s: float) -> dict[str, Any]:
    """Copy the build log to stderr until the build is ready or failed."""
    offset = 0
    while True:
        chunk = client.read_image_build_log(name, offset=offset)
        text = str(chunk.get("text", ""))
        if text:
            sys.stderr.write(text)
            sys.stderr.flush()
        offset = int(chunk.get("offset", offset))
        if chunk.get("state") in _TERMINAL_STATES and not text:
            return client.get_image_build(name)
        if not text:
            time.sleep(poll_interval_s)


def _show(build: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(build, sort_keys=True))
    else:
        print(_summary(build), file=sys.stderr)
        if build.get("state") == "ready":
            print(build["name"])


def _publish(client: Client, args: argparse.Namespace) -> int:
    label, image = _target_image(args.target)
    build = client.request_image_build(image, architecture=args.architecture)
    if not args.json:
        print(f"[vfunc] {label} -> {image.name} state={build.get('state')}", file=sys.stderr)
    if build.get("state") not in _TERMINAL_STATES and not args.no_wait:
        try:
            build = _follow(client, image.name, max(client.cfg.poll_interval_s, 1.0))
        except KeyboardInterrupt:
            print(
                f"\n[vfunc] the build continues on the service; check it with "
                f"`vfunc image status {args.target}`",
                file=sys.stderr,
            )
            return 130
    _show(build, json_output=args.json)
    if build.get("state") == "failed":
        raise CliError(f"image build failed: {build.get('error') or 'see the build log'}")
    return 0


def _status(client: Client, args: argparse.Namespace) -> int:
    _, image = _target_image(args.target)
    build = client.get_image_build(image.name)
    _show(build, json_output=args.json)
    if build.get("error") and not args.json:
        print(f"[vfunc] error: {build['error']}", file=sys.stderr)
    return 0


def _list(client: Client, args: argparse.Namespace) -> int:
    builds = client.list_image_builds()
    if args.json:
        print(json.dumps(builds, sort_keys=True))
        return 0
    width = max([len("NAME"), *(len(str(build.get("name", ""))) for build in builds)])
    print(f"{'NAME':<{width}}  {'STATE':<10}  {'UPDATED (UTC)':<19}  DIGEST")
    for build in builds:
        updated = build.get("updated_ms")
        when = (
            time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(updated / 1000))
            if isinstance(updated, int)
            else ""
        )
        print(
            f"{str(build.get('name', '')):<{width}}  {str(build.get('state', '')):<10}  "
            f"{when:<19}  {str(build.get('digest') or '')[:19]}"
        )
    return 0


def open_dashboard(args: argparse.Namespace) -> int:
    """Open the dashboard after checking that something answers at its address."""
    url = args.url
    try:
        httpx.get(url, timeout=3.0)
    except httpx.HTTPError as error:
        port = httpx.URL(url).port or 80
        raise CliError(
            f"nothing answers at {url} ({error.__class__.__name__}). The dashboard listens only "
            f"on the build host; forward it first, for example:\n"
            f"  ssh -N -L {port}:127.0.0.1:8091 BUILD_HOST",
            exit_code=1,
        ) from error
    if args.no_browser or not auth._browser_available(os.environ) or not webbrowser.open(url):
        print(url)
        return 0
    print(f"Opened {url}")
    return 0


def dispatch(args: argparse.Namespace, client: Client) -> int:
    if args.image_command == "publish":
        return _publish(client, args)
    if args.image_command == "list":
        return _list(client, args)
    return _status(client, args)
