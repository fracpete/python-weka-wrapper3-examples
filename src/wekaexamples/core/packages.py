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

# packages.py
# Copyright (C) 2021 Fracpete (pythonwekawrapper at gmail dot com)

import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
import weka.core.packages as pkgs


def main():
    """
    Runs some example code.
    """
    helper.print_info("Listing packages")
    items = pkgs.all_packages()
    for item in items:
        print(item)
        if item.name == "CLOPE":
            print(item.name + " " + item.url)

    helper.print_info("Installing CLOPE")
    pkgs.install_package("CLOPE")
    items = pkgs.installed_packages()
    for item in items:
        print(item.name + " " + item.url)

    helper.print_info("Uninstalling CLOPE")
    pkgs.uninstall_package("CLOPE")
    items = pkgs.installed_packages()
    for item in items:
        print(item.name + " " + item.url)


if __name__ == "__main__":
    try:
        jvm.start(packages=True)
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
