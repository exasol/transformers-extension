from pathlib import Path

from exasol_transformers_extension.deployment.scripts_deployer import ScriptsDeployer


def write_create_script():
    print("Write create script")
    sd = ScriptsDeployer(#todo make these options?
        language_alias="PYTHON3_TE", schema="test", pyexasol_conn=None, use_spans=True, install_all_scripts=True
    )

    root_dir = Path(__file__).resolve().parent.parent
    script_path = root_dir / "deployment/create_script.sql"
    sd.write_create_sql_script(script_path)
    print("create_script written.")

if __name__ == "__main__":
    write_create_script()
