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

# list_files.py
# Copyright (C) 2015-2023 Fracpete (pythonwekawrapper at gmail dot com)

import tempfile
import traceback

import weka.core.jvm as jvm
from simflow.control import Flow, run_flow
from simflow.sink import Console
from simflow.source import ListFiles


def main():
    """
    Just runs some example code.
    """

    # setup the flow
    flow = Flow(name="list files")

    listfiles = ListFiles()
    listfiles.config["dir"] = str(tempfile.gettempdir())
    listfiles.config["list_files"] = True
    listfiles.config["list_dirs"] = False
    listfiles.config["recursive"] = False
    listfiles.config["regexp"] = ".*r.*"
    flow.actors.append(listfiles)

    console = Console()
    console.config["prefix"] = "Match: "
    flow.actors.append(console)

    # run the flow
    run_flow(flow, print_tree=True, cleanup=True)


if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
