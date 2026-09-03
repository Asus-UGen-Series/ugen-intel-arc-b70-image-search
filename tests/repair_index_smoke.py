"""Local-only B70 stability run; tests/ is intentionally ignored by Git."""

import json
from pathlib import Path

from visual_search.app import create_model_registry, create_services
from visual_search.config import AppConfig
from visual_search.paths import AppPaths


paths = AppPaths.default()
config = AppConfig.load(paths.config)
roots = [Path(root) for root in config.roots if Path(root).is_dir()]
if not roots:
    raise SystemExit("No configured image root")

repository, controller, _search = create_services(paths, config, create_model_registry(paths))


def progress(current: int, total: int, name: str) -> None:
    if current % 25 == 0 or controller.last_stop_reason:
        print(f"[{current}/{total}] {name}", flush=True)
    if current >= 300:
        controller.cancel()


try:
    controller.scan(roots[0], progress)
    print(json.dumps(repository.index_coverage(), ensure_ascii=False, indent=2), flush=True)
    if controller.last_stop_reason:
        raise SystemExit(controller.last_stop_reason)
finally:
    if controller.scheduler is not None:
        controller.scheduler.shutdown()
