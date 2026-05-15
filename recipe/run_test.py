import json
import tempfile
from importlib import metadata

from tensorboard.plugins import base_plugin
from werkzeug.test import EnvironBuilder
from werkzeug.wrappers import Response

from torch_tb_profiler import __version__, consts
from torch_tb_profiler.plugin import TorchProfilerPlugin


def _tensorboard_plugin_entry_points():
    entry_points = metadata.entry_points()
    if hasattr(entry_points, "select"):
        return entry_points.select(group="tensorboard_plugins")
    return entry_points.get("tensorboard_plugins", [])


def _json_response(app, path):
    environ = EnvironBuilder(path=path, method="GET").get_environ()
    response = Response.from_app(app, environ)
    assert response.status_code == 200
    return json.loads(response.get_data(as_text=True))


def main():
    assert __version__ == "0.4.3"

    entry_points = {
        entry_point.name: entry_point for entry_point in _tensorboard_plugin_entry_points()
    }
    assert entry_points["torch_profiler"].load() is TorchProfilerPlugin

    with tempfile.TemporaryDirectory() as logdir:
        plugin = TorchProfilerPlugin(base_plugin.TBContext(logdir=logdir))
        assert plugin.plugin_name == consts.PLUGIN_NAME == "pytorch_profiler"
        assert not plugin.is_active()

        apps = plugin.get_plugin_apps()
        for route in ("/runs", "/views", "/workers", "/overview", "/index.js"):
            assert route in apps

        runs = _json_response(apps["/runs"], "/runs")
        assert runs == {"runs": [], "loading": False}

        frontend_metadata = plugin.frontend_metadata()
        assert frontend_metadata.es_module_path == "/index.js"
        assert frontend_metadata.disable_reload is True


if __name__ == "__main__":
    main()
