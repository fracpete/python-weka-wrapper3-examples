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

# load_database.py
# Copyright (C) 2015-2023 Fracpete (pythonwekawrapper at gmail dot com)

import traceback
import weka.core.jvm as jvm
import wekaexamples.helper as helper
from simflow.control import Flow, run_flow
from weka.flow.source import LoadDatabase
from simflow.sink import Console


def main():
    """
    Just runs some example code.
    """
    """
    Loads data from a database.
    """

    # setup the flow
    helper.print_title("Load from database")

    flow = Flow(name="load from database")

    loaddatabase = LoadDatabase()
    loaddatabase.config["db_url"] = "jdbc:mysql://HOSTNAME:3306/DBNAME"
    loaddatabase.config["user"] = "DBUSER"
    loaddatabase.config["password"] = "DBPW"
    loaddatabase.config["query"] = "select * from TABLE"
    flow.actors.append(loaddatabase)

    console = Console()
    flow.actors.append(console)

    # run the flow
    run_flow(flow, print_tree=True, cleanup=True)


if __name__ == "__main__":
    try:
        mysql_jar = "/some/where/mysql-connector-java-X.Y.Z-bin.jar"
        jvm.start(class_path=[mysql_jar])
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
