from pathlib import Path

import humanfriendly
from exasol_script_languages_container_tool.lib.tasks.export.export_info import ExportInfo


def test_slc_size_below_2gb(export_slc: ExportInfo):
    stat_result = Path(export_slc.cache_file).resolve().stat()
    assert stat_result.st_size < humanfriendly.parse_size(size="2GB")
