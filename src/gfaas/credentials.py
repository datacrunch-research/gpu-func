"""API keys stored by `vfunc login`, one named profile per login.

The file is private to the user (mode 0600) and holds a default profile name and the profiles:

    {"default_profile": "gb300dev",
     "profiles": {"gb300dev": {"api_base": "https://api.example.com/api", "api_key": "...",
                               "email": "person@example.com", "key_id": "key_...",
                               "expires_at_ms": 1790000000000}}}

A profile's key is only ever sent to that profile's API base.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

PATH_VARIABLE = "VFUNC_CREDENTIALS"
PROFILE_VARIABLE = "VFUNC_PROFILE"


@dataclass(frozen=True)
class Profile:
    """One stored login."""

    name: str
    api_base: str
    api_key: str
    email: str | None = None
    key_id: str | None = None
    expires_at_ms: int | None = None


@dataclass(frozen=True)
class Credentials:
    """Every stored profile and the name of the default one."""

    default_profile: str | None
    profiles: dict[str, Profile]


def credentials_path(env: Mapping[str, str]) -> Path | None:
    """Where this environment keeps credentials, or None without a home directory."""
    if explicit := env.get(PATH_VARIABLE):
        return Path(explicit)
    config_home = env.get("XDG_CONFIG_HOME") or (
        str(Path(env["HOME"]) / ".config") if env.get("HOME") else None
    )
    return Path(config_home) / "vfunc" / "credentials.json" if config_home else None


def load(path: Path | None) -> Credentials:
    """Read stored credentials; a missing file holds none."""
    if path is None or not path.exists():
        return Credentials(None, {})
    document = json.loads(path.read_text())
    profiles = {
        name: Profile(name=name, **{key: value for key, value in fields.items() if key != "name"})
        for name, fields in document.get("profiles", {}).items()
    }
    default = document.get("default_profile")
    return Credentials(default if default in profiles else None, profiles)


def _write(path: Path, credentials: Credentials) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    document = {
        "default_profile": credentials.default_profile,
        "profiles": {
            name: {key: value for key, value in asdict(profile).items() if key != "name"}
            for name, profile in sorted(credentials.profiles.items())
        },
    }
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".credentials-")
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(document, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def save(path: Path, profile: Profile, *, make_default: bool = True) -> None:
    """Store or replace one profile, and make it the default unless told otherwise."""
    current = load(path)
    profiles = {**current.profiles, profile.name: profile}
    default = profile.name if make_default or current.default_profile is None else None
    _write(path, Credentials(default or current.default_profile, profiles))


def remove(path: Path, name: str) -> bool:
    """Delete one profile; False when it was not stored."""
    current = load(path)
    if name not in current.profiles:
        return False
    profiles = {key: value for key, value in current.profiles.items() if key != name}
    default = current.default_profile if current.default_profile != name else None
    _write(path, Credentials(default or next(iter(sorted(profiles)), None), profiles))
    return True


def select(env: Mapping[str, str], api_base: str | None) -> Profile | None:
    """The profile to use: the one named by VFUNC_PROFILE, else one for this API base (the
    default first), else the default when no API base was requested."""
    credentials = load(credentials_path(env))
    if name := env.get(PROFILE_VARIABLE):
        return credentials.profiles.get(name)
    default = credentials.profiles.get(credentials.default_profile or "")
    if api_base is None:
        return default
    candidates = [default, *credentials.profiles.values()]
    return next(
        (profile for profile in candidates if profile and profile.api_base == api_base),
        None,
    )
