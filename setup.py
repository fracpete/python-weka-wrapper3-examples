# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# setup.py
# Copyright (C) 2014-2024 Fracpete (pythonwekawrapper at gmail dot com)

from setuptools import setup


setup(
    name="python-weka-wrapper3-examples",
    description="Examples for the python-weka-wrapper3 library.",
    long_description=
    "Examples for the python-weka-wrapper3 library. "
    + "Some examples are modelled after the Examples for Weka, located here: "
    + "https://git.cms.waikato.ac.nz/weka/weka/-/tree/main/trunk/wekaexamples",
    url="https://github.com/fracpete/python-weka-wrapper3-examples",
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
    ],
    license='GNU General Public License version 3.0 (GPLv3)',
    package_dir={
        '': 'src'
    },
    packages=[
        "wekaexamples",
        "wekaexamples.associations",
        "wekaexamples.attribute_selection",
        "wekaexamples.book",
        "wekaexamples.classifiers",
        "wekaexamples.core"
    ],
    version="0.3.0",
    author='Peter "fracpete" Reutemann',
    author_email='pythonwekawrapper at gmail dot com',
    install_requires=[
        "python-weka-wrapper3>=0.3.0",
    ],
)
