FROM exasol/script-language-container:python-3.8-minimal-EXASOL-6.2.0-build_run_DPUTJLTDYQ7Q3JW4EDOJAWGSIXZVBTQQ4BZWHVCIPT7P5E7A3B2A

Run mkdir /project
COPY dependencies/requirements.txt /project/requirements.txt
RUN python3.8 -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /project/requirements.txt
