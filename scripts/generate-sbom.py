"""Generate a compact CycloneDX-compatible runtime SBOM for the installer."""
from __future__ import annotations

import argparse
import json
from importlib import metadata
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("dist/sbom.json"))
    args = parser.parse_args()
    components = []
    for distribution in sorted(metadata.distributions(), key=lambda item: (item.metadata.get("Name") or "").casefold()):
        name = distribution.metadata.get("Name")
        if not name:
            continue
        components.append({
            "type": "library", "name": name, "version": distribution.version,
            "purl": f"pkg:pypi/{name.casefold().replace('_', '-')}@{distribution.version}",
        })
    payload = {
        "bomFormat": "CycloneDX", "specVersion": "1.5", "version": 1,
        "metadata": {"component": {"type": "application", "name": "Visual Search", "version": "1.0.0"}},
        "components": components,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
