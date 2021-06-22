# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import re
import json
import os
import socket
from dace.sdfg import state
from dace.sdfg import nodes


class SdfgLocation:
    def __init__(self, sdfg_id, state_id, node_ids):
        self.sdfg_id = sdfg_id
        self.state_id = state_id
        self.node_ids = node_ids

    def printer(self):
        print("SDFG {}:{}:{}".format(self.sdfg_id, self.state_id,
                                     self.node_ids))


def create_folder(path_str: str):
    """ Creates a folder if it does not yet exist
        :param path_str: location the folder will be crated at
    """
    if not os.path.exists(path_str):
        path = os.path.abspath(path_str)
        os.mkdir(path)


def create_cache(name: str) -> str:
    """ Creates the map folder in .dacecache if it
        does not yet exist
        :param name: name of the SDFG
        :return: relative path to the created folder starting from '.dacecache'
    """
    create_folder(".dacecache")
    create_folder(".dacecache/" + name)
    create_folder(".dacecache/" + name + "/map")
    return ".dacecache/" + name


def temporaryInfo(filename: str, data):
    """ Creates a temporary file that stores the json object
        in the map folder (.dacecache/functionName/map) 
        of the corresponding function
        :param filename: name of the function
        :param data: data to save
    """
    folder = create_cache(filename)
    path = os.path.abspath(folder + "/map/tmp.json")

    with open(path, "w") as writer:
        json.dump(data, writer)


def get_tmp(name: str):
    """ Returns the data saved in the temporary file
        from the corresponding function
        :param name: name of the SDFG
        :return: the parsed data in the tmp file of the SDFG. 
        If the tmp file doesn't exist returns None
    """
    path = ".dacecache/" + name + "/map/tmp.json"
    if os.path.exists(path):
        with open(path, "r") as tmp_json:
            data = json.load(tmp_json)
        return data
    else:
        return None


def remove_tmp(name: str, remove_cache: bool = False):
    """ Remove the tmp created by "temporaryInfo"
        :param name: name of the function whiches tmp will be removed
        :param remove_cache: If true, checks if the directory only contains 
        the tmp file, if this is the case, remove the SDFG cache directory.
    """
    path = ".dacecache/" + name

    if not os.path.exists(path):
        return

    os.remove(path + "/map/tmp.json")

    if (remove_cache and len(os.listdir(path)) == 1
            and len(os.listdir(path + "/map")) == 0):
        if os.path.exists(path):
            os.rmdir(os.path.join(path, 'map'))
            os.rmdir(path)


def send(data: json):
    """ Sends a json object to the port given as the env variable DACE_port.
        If the port isn't set we don't send anything.
        :param data: json object to send
    """

    if "DACE_port" not in os.environ:
        return

    HOST = socket.gethostname()
    PORT = os.environ["DACE_port"]

    data_bytes = bytes(json.dumps(data), "utf-8")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, int(PORT)))
        s.sendall(data_bytes)


# TODO sdfg.build_folder()
def save(language: str, name: str, map: dict) -> str:
    """ Saves the mapping in the map folder of
        the corresponding SDFG
        :param language: used for the file name to save to: py -> map_py.json
        :param name: name of the SDFG
        :param map: the map object to be saved
        :return: absolute path to the cache folder of the SDFG
    """
    folder = create_cache(name)
    path = os.path.abspath(os.path.join(folder, 'map', 'map_' + language + '.json'))

    with open(path, "w") as json_file:
        json.dump(map, json_file, indent=4)

    return os.path.abspath(folder)


def create_py_map(sdfg):
    """ Creates the mapping from the python source lines to the SDFG nodes.
        The mapping gets saved at: .dacecache/{sdfg.name}/map/map_py.json
        :param sdfg: The SDFG for which the mapping will be created
    """
    py_mapper = MapPython(sdfg.name)
    py_mapper.mapper(sdfg)
    save("py", sdfg.name, py_mapper.map)


def create_cpp_map(code: str, sdfg_name: str, target_name: str):
    """ Creates the mapping from the SDFG nodes to the C++ code lines.
        The mapping gets saved at: .dacecache/{sdfg_name}/map/map_cpp.json
        :param code: C++ code containing the identifiers '////__DACE:0:0:0'
        :param name: The name of the SDFG
        :param target_name: The target type, example: 'cpu'
    """
    cpp_mapper = MapCpp(code, sdfg_name, target_name)
    cpp_mapper.mapper()
    folder = save("cpp", sdfg_name, cpp_mapper.map)

    tmp = get_tmp(sdfg_name)
    remove_tmp(sdfg_name)

    # Send information about the SDFG to VSCode
    if tmp is not None:
        send({
            "type": "registerFunction",
            "name": sdfg_name,
            "path_cache": folder,
            "path_file": tmp.get("src_file"),
            "target_name": target_name
        })


class MapCpp:
    """ Creates the mapping between the SDFG nodes and
        the generated C++ code lines.
    """
    def __init__(self, code: str, name: str, target_name: str):
        self.name = name
        self.code = code
        self.map = {'target': target_name}

    def mapper(self):
        """ For each line of code retrieve the corresponding nodes
            and map the nodes to this line
        """
        for line_num, line in enumerate(self.code.split("\n"), 1):
            nodes = self.get_nodes(line)
            for node in nodes:
                self.create_mapping(node, line_num)

    def create_mapping(self, node: SdfgLocation, line_num: int):
        """ Adds a C++ line number to the mapping
            :param node: A node which will map to the line number
            :param line_num: The line number to add to the mapping
        """
        if node.sdfg_id not in self.map:
            self.map[node.sdfg_id] = {}
        if node.state_id not in self.map[node.sdfg_id]:
            self.map[node.sdfg_id][node.state_id] = {}

        state = self.map[node.sdfg_id][node.state_id]

        for node_id in node.node_ids:
            if node_id not in state:
                state[node_id] = {'from': line_num, 'to': line_num}
            elif state[node_id]['to'] + 1 == line_num:
                # If the current node maps to the previous line
                # (before "line_num"), then extend the range this node maps to
                state[node_id]['to'] += 1

    def get_nodes(self, line_code: str):
        """ Retrive all identifiers set at the end of the line of code.
            Example: x = y ////__DACE:0:0:0 ////__DACE:0:0:1
                Returns [SDFGL(0,0,0), SDFGL(0,0,1)]
            :param line_code: a single line of code
            :return: list of SDFGLocation
        """
        line_identifiers = self.get_identifiers(line_code)
        nodes = []
        for identifier in line_identifiers:
            ids_split = identifier.split(":")
            nodes.append(
                SdfgLocation(
                    ids_split[1],
                    ids_split[2],
                    # node might be an edge
                    ids_split[3].split(",")))
        return nodes

    def get_identifiers(self, line: str):
        """ Retruns a list of identifiers found in the code line
            :param line: line of C++ code with identifiers
            :return: list of identifers
        """
        line_identifier = re.findall(
            r'(\/\/\/\/__DACE:[0-9]:[0-9]:[0-9](,[0-9])*)', line)
        # remove tuples
        return [x or y for (x, y) in line_identifier]


class MapPython:
    """ Creates the mapping between the source code and
        the SDFG nodes
    """
    def __init__(self, name):
        self.name = name
        self.map = {}
        self.debuginfo = {}

    def mapper(self, sdfg) -> dict:
        """ Creates the source to SDFG node mapping
            :param sdfg: SDFG to create the mapping for
        """
        self.debuginfo = self.sdfg_debuginfo(sdfg)
        self.sorter()
        line_info = get_tmp(self.name)

        # If we haven't saved line and/or src info
        # return after creating the map
        # Happens when only using the SDFG API
        if (line_info is None or "start_line" not in line_info
                or "end_line" not in line_info):
            self.create_mapping()
            return

        # Store the start and end line as a tuple
        # for each function/sub-function in the SDFG
        ranges = [(line_info["start_line"], line_info["end_line"])]

        if "other_sdfgs" in line_info:
            for other_sdfg_name in line_info["other_sdfgs"]:
                other_tmp = get_tmp(other_sdfg_name)
                # SDFGs created with the API don't have tmp files
                if other_tmp is not None:
                    remove_tmp(other_sdfg_name, True)
                    ranges.append(
                        (other_tmp["start_line"], other_tmp["end_line"]))

        self.create_mapping(ranges)

    def sorter(self):
        """ Prioritizes smaller ranges over larger ones """
        self.debuginfo = sorted(
            self.debuginfo,
            key=lambda n:
            (n['debuginfo']['start_line'], n['debuginfo']['start_column'], n[
                'debuginfo']['end_line'], n['debuginfo']['end_column']))

    def make_info(self, debuginfo, node_id: int, state_id: int,
                  sdfg_id: int) -> dict:
        """ Creates an object for the current node with
            the most important information
            :param debuginfo: JSON object of the debuginfo of the node
            :param node_id: ID of the node
            :param state_id: ID of the state
            :param sdfg_id: ID of the sdfg
            :return: Dictionary with a debuginfo JSON object and the identifiers
        """
        return {
            "debuginfo": debuginfo,
            "sdfg_id": sdfg_id,
            "state_id": state_id,
            "node_id": node_id
        }

    def sdfg_debuginfo(self, graph, sdfg_id: int = 0, state_id: int = 0):
        """ Recursively retracts all debuginfo from the nodes
            :param graph: An SDFG or SDFGState to check for nodes
            :param sdfg_id: Id of the current SDFG/NestedSDFG
            :param state_id: Id of the current SDFGState
            :return: list of debuginfo with the node identifiers
        """
        if sdfg_id is None:
            sdfg_id = 0

        mapping = []
        for id, node in enumerate(graph.nodes()):
            # Node has Debuginfo, no recursive call
            if isinstance(node,
                          (nodes.AccessNode, nodes.Tasklet, nodes.LibraryNode,
                           nodes.Map)) and node.debuginfo is not None:

                dbinfo = node.debuginfo.to_json()
                mapping.append(self.make_info(dbinfo, id, state_id, sdfg_id))

            elif isinstance(node, (nodes.MapEntry, nodes.MapExit)):
                dbinfo = node.map.debuginfo.to_json()
                mapping.append(self.make_info(dbinfo, id, state_id, sdfg_id))

            # State no debuginfo, recursive call
            elif isinstance(node, state.SDFGState):
                mapping += self.sdfg_debuginfo(node, sdfg_id,
                                               graph.node_id(node))

            # Sdfg not using debuginfo, recursive call
            elif isinstance(node, nodes.NestedSDFG):
                mapping += self.sdfg_debuginfo(node.sdfg, node.sdfg.sdfg_id,
                                               state_id)

        return mapping

    def create_mapping(self, ranges=None):
        """ Creates the actual mapping by using the debuginfo list
            :param ranges: list of tuples containing a start and 
            end line of a DaCe program
        """
        for node in self.debuginfo:
            for line in range(node["debuginfo"]["start_line"],
                              node["debuginfo"]["end_line"] + 1):
                # Maps a python line to a list of nodes
                # The nodes have been sorted by priority
                if not str(line) in self.map:
                    self.map[str(line)] = []

                self.map[str(line)].append({
                    "sdfg_id": node["sdfg_id"],
                    "state_id": node["state_id"],
                    "node_id": node["node_id"]
                })

        if ranges:
            # Mapping lines that don't occur in the debugInfo of the SDFG
            # These might be lines that don't have any code on them or
            # no debugInfo correspond directly to them
            for start, end in ranges:
                for line in range(start, end + 1):
                    if not str(line) in self.map:
                        # Set to the same node as the previous line
                        # If the previous line doesn't exist
                        # (line - 1 < f_start_line) then search the next lines
                        # until a mapping can be found
                        if str(line - 1) in self.map:
                            self.map[str(line)] = self.map[str(line - 1)]
                        else:
                            for line_after in range(line + 1, end + 1):
                                if str(line_after) in self.map:
                                    self.map[str(line)] = self.map[str(
                                        line_after)]
