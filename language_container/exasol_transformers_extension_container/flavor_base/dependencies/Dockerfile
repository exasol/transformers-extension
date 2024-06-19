FROM exasol/script-language-container:template-Exasol-all-python-3.10-build_run_BUXX2FHQPEBXZNMS6RQRXPKRMKZREERZ5HIRVNXTV56DLNXXTGQA

Run mkdir /project
COPY dependencies/requirements.txt /project/requirements.txt
RUN python3.10 -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /project/requirements.txt
