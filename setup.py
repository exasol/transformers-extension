# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['exasol_transformers_extension',
 'exasol_transformers_extension.deployment',
 'exasol_transformers_extension.resources',
 'exasol_transformers_extension.udfs',
 'exasol_transformers_extension.udfs.callers',
 'exasol_transformers_extension.udfs.models',
 'exasol_transformers_extension.utils']

package_data = \
{'': ['*'], 'exasol_transformers_extension.resources': ['templates/*']}

install_requires = \
['Jinja2>=3.0.3,<4.0.0',
 'click>=8.0.4,<9.0.0',
 'exasol-bucketfs-utils-python @ '
 'git+https://github.com/exasol/bucketfs-utils-python.git@main',
 'importlib-resources>=5.4.0,<6.0.0',
 'pandas>=1.4.2,<2.0.0',
 'pyexasol>=0.17.0,<0.18.0',
 'torch>=1.11.0,<2.0.0',
 'transformers[torch]==4.21.3']

setup_kwargs = {
    'name': 'exasol-transformers-extension',
    'version': '0.1.0',
    'description': 'An Exasol extension to use state-of-the-art pretrained machine learning models via the transformers api.',
    'long_description': '# Exasol Transformers Extension\n\n**This project is at an early development stage.**\n\nAn Exasol extension to use state-of-the-art pretrained machine learning models \nvia the [transformers api](https://github.com/huggingface/transformers).\n\n\n## Table of Contents\n\n### Information for Users\n\n* [User Guide](doc/user_guide/user_guide.md)\n* [Changelog](doc/changes/changelog.md)\n* [License](LICENSE)\n\n### Information for Contributors\n\n* [Design](doc/design.md)\n* [Dependencies](doc/dependencies.md)\n\n',
    'author': 'Umit Buyuksahin',
    'author_email': 'umit.buyuksahin@exasol.com',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'https://github.com/exasol/transformers-extension',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.8,<4.0',
}


setup(**setup_kwargs)
