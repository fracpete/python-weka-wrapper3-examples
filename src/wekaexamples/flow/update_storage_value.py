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

# update_storage_value.py
# Copyright (C) 2015-2023 Fracpete (pythonwekawrapper at gmail dot com)

import traceback

import weka.core.jvm as jvm
from simflow.control import Flow, Trigger, run_flow
from simflow.sink import Console
from simflow.source import ForLoop, Start
from simflow.transformer import InitStorageValue, UpdateStorageValue


def main():
    """
    Just runs some example code.
    """

    # setup the flow
    flow = Flow(name="update storage value")

    start = Start()
    flow.actors.append(start)

    init = InitStorageValue()
    init.config["storage_name"] = "max"
    init.config["value"] = "int(1)"
    flow.actors.append(init)

    trigger = Trigger()
    flow.actors.append(trigger)

    outer = ForLoop()
    outer.name = "outer"
    outer.config["max"] = 3
    trigger.actors.append(outer)

    trigger2 = Trigger()
    trigger.actors.append(trigger2)

    inner = ForLoop()
    inner.name = "inner"
    inner.config["max"] = "@{max}"
    trigger2.actors.append(inner)

    console = Console()
    trigger2.actors.append(console)

    update = UpdateStorageValue()
    update.config["storage_name"] = "max"
    update.config["expression"] = "{X} + 2"
    trigger.actors.append(update)

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
