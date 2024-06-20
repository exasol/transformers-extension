FROM exasol/script-language-container:template-Exasol-all-python-3.10-release_BFRSH344TDRPT7LK2FBOJK4KBIDW6A253FFPYEUYT4O2ERFMTCNA

Run mkdir /project
COPY dependencies/requirements.txt /project/requirements.txt
RUN python3.10 -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /project/requirements.txt
