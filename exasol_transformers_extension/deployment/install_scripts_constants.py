from pathlib import Path
from importlib_resources import files

class InstallScriptsConstants:
    def __init__(self,
                 base_dir : str,
                 templates_dir : Path,
                 udf_callers_dir_suffix : str,
                 udf_callers_templates : dict,
                 ordered_columns : list):

        self.base_dir = base_dir
        self.templates_dir = templates_dir
        self.udf_callers_dir = files(f"{self.base_dir}.{udf_callers_dir_suffix}")
        self.udf_callers_templates = udf_callers_templates
        self.ordered_columns = ordered_columns
