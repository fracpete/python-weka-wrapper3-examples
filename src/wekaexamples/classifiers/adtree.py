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

# adtree.py
# Copyright (C) 2022 Fracpete (pythonwekawrapper at gmail dot com)

import traceback
import weka.core.jvm as jvm
from weka.core.packages import is_installed, install_package
from weka.classifiers import Classifier


def main():
    """
    Installs package for ADTree if necessary (and prompts user to restart)
    and once available instantiates an ADTree instance.
    """
    pkgname = "alternatingDecisionTrees"
    if not is_installed(pkgname):
        print("Package %s not installed, attempting installation..." % pkgname)
        if install_package(pkgname):
            print("Package %s installed, please rerun script!" % pkgname)
        else:
            print("Failed to install package %s!" % pkgname)
        return
    else:
        print("Package already installed: %s" % pkgname)

    cls = Classifier(classname="weka.classifiers.trees.ADTree", options=[])
    print(cls.to_commandline())


if __name__ == "__main__":
    try:
        jvm.start(packages=True)
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
