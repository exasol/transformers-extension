FROM {{ dependencies }}

COPY release/dist /project/dist
RUN python3.10 -m pip install --no-deps /project/dist/*.whl

RUN mkdir -p /build_info/actual_installed_packages/release && \
    /scripts/list_installed_scripts/list_installed_apt.sh > /build_info/actual_installed_packages/release/apt_get_packages && \
    /scripts/list_installed_scripts/list_installed_pip.sh python3.10 > /build_info/actual_installed_packages/release/python3_pip_packages
