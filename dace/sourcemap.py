# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import collections
import re
import json
import os
import socket
from dace import Config, dtypes
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
        os.makedirs(path, exist_ok=True)


def create_cache(name: str, folder: str) -> str:
    """ Creates the map folder in the build path if it
        does not yet exist
        :param name: name of the SDFG
        :param folder: the build folder
        :return: relative path to the created folder
    """
    if (folder is not None):
        create_folder(os.path.join(folder, "map"))
        return folder
    else:
        build_folder = Config.get('default_build_folder')
        cache_folder = os.path.join(build_folder, name)
        create_folder(os.path.join(cache_folder, "map"))
        return cache_folder


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


def save(language: str, name: str, map: dict, build_folder: str) -> str:
    """ Saves the mapping in the map folder of
        the corresponding SDFG
        :param language: used for the file name to save to: py -> map_py.json
        :param name: name of the SDFG
        :param map: the map object to be saved
        :param build_folder: build folder
        :return: absolute path to the cache folder of the SDFG
    """
    folder = create_cache(name, build_folder)
    path = os.path.abspath(
        os.path.join(folder, 'map', 'map_' + language + '.json'))

    with open(path, "w") as json_file:
        json.dump(map, json_file, indent=4)

    return os.path.abspath(folder)


def get_src_files(sdfg):
    """ Search all nodes for debuginfo to find the source filenames
        :param sdfg: An SDFG to check for source files
        :return: list of unique source filenames
    """
    sourcefiles = []
    for node, _ in sdfg.all_nodes_recursive():
        if (isinstance(node, (nodes.AccessNode, nodes.Tasklet,
                              nodes.LibraryNode, nodes.Map, nodes.NestedSDFG))
                and node.debuginfo is not None):

            filename = node.debuginfo.filename
            if not filename in sourcefiles:
                sourcefiles.append(filename)

        elif (isinstance(node, (nodes.MapEntry, nodes.MapExit))
              and node.map.debuginfo is not None):

            filename = node.map.debuginfo.filename
            if not filename in sourcefiles:
                sourcefiles.append(filename)

    return sourcefiles


def create_py_map(sdfg):
    """ Creates the mapping from the python source lines to the SDFG nodes.
        The mapping gets saved at: <SDFG build folder>/map/map_py.json
        :param sdfg: The SDFG for which the mapping will be created
        :return: an object with the build_folder, src_files and made_with_api
    """
    py_mapper = MapPython(sdfg.name)
    made_with_api = py_mapper.mapper(sdfg)
    folder = sdfg.build_folder
    save("py", sdfg.name, py_mapper.map, folder)
    sourceFiles = get_src_files(sdfg)
    return (folder, sourceFiles, made_with_api)


def create_cpp_map(code: str, name: str, target_name: str, build_folder: str,
                   sourceFiles: [str], made_with_api: bool):
    """ Creates the mapping from the SDFG nodes to the C++ code lines.
        The mapping gets saved at: <SDFG build folder>/map/map_cpp.json
        :param code: C++ code containing the identifiers '////__DACE:0:0:0'
        :param name: The name of the SDFG
        :param target_name: The target type, example: 'cpu'
        :param build_folder: The build_folder of the SDFG
        :param sourceFiles: A list of source files of to the SDFG
        :param made_with_api: true if the SDFG was created just with the API
    """
    codegen_debug = Config.get_bool('compiler', 'codegen_lineinfo')
    cpp_mapper = MapCpp(code, name, target_name)
    cpp_mapper.mapper(codegen_debug)

    folder = save("cpp", name, cpp_mapper.map, build_folder)

    if codegen_debug:
        save("codegen", name, cpp_mapper.codegen_map, build_folder)

    # Send information about the SDFG to VSCode
    send({
        "type": "registerFunction",
        "name": name,
        "path_cache": folder,
        "path_file": sourceFiles,
        "target_name": target_name,
        "made_with_api": made_with_api,
        "codegen_map": codegen_debug
    })


def create_maps(sdfg, code: str, target_name: str):
    """ Creates the C++, Py and Codegen mapping
        :param sdfg: The sdfg to create the mapping for
        :param code: The generated code
        :param target_name: The target name
    """
    build_folder, sourceFiles, made_with_api = create_py_map(sdfg)
    create_cpp_map(code, sdfg.name, target_name, build_folder, sourceFiles,
                   made_with_api)


class MapCpp:
    """ Creates the mapping between the SDFG nodes and
        the generated C++ code lines.
    """
    def __init__(self, code: str, name: str, target_name: str):
        self.name = name
        self.code = code
        self.map = {'target': target_name}
        self.codegen_map = {}
        self.cpp_pattern = re.compile(
            r'(\/\/\/\/__DACE:[0-9]+:[0-9]+:[0-9]+(,[0-9])*)')
        self.codegen_pattern = re.compile(
            r'(\/\/\/\/__CODEGEN;([A-z]:)?(\/|\\)([A-z0-9-_+]+(\/|\\))*([A-z0-9]+\.[A-z0-9]+);[0-9]+)'
        )

    def mapper(self, codegen_debug: bool = False):
        """ For each line of code retrieve the corresponding identifiers
            and create the mapping
            :param codegen_debug: if the codegen mapping should be created
        """
        for line_num, line in enumerate(self.code.split("\n"), 1):
            nodes = self.get_nodes(line)
            for node in nodes:
                self.create_mapping(node, line_num)
            if codegen_debug:
                self.codegen_mapping(line, line_num)

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

    def codegen_mapping(self, line: str, line_num: int):
        """ Searches the code line for the first ////__CODEGEN identifier
            and adds the information to the codegen_map
            :param line: code line to search for identifiers
            :param line_num: corresponding line number
        """
        codegen_identifier = self.get_identifiers(line, findall=False)
        if codegen_identifier:
            codegen_debuginfo = codegen_identifier.split(';')
            self.codegen_map[line_num] = {
                'file': codegen_debuginfo[1],
                'line': codegen_debuginfo[2]
            }

    def get_identifiers(self, line: str, findall: bool = True):
        """ Retruns a list of identifiers found in the code line
            :param line: line of C++ code with identifiers
            :param findall: if it should return all finds or just the first one
            :return: if findall is true return list of identifers 
            otherwise a single identifier
        """
        if findall:
            line_identifiers = re.findall(self.cpp_pattern, line)
            # The regex expression returns multiple groups (a tuple).
            # We are only interested in the first element of the tuple (the entire match).
            # Example tuple in the case of an edge ('////__DACE:0:0:2,6', ',6')
            return [groups[0] for groups in line_identifiers]
        else:
            line_identifier = re.search(self.codegen_pattern, line)
            return line_identifier.group(0) if line_identifier else None


class MapPython:
    """ Creates the mapping between the source code and
        the SDFG nodes
    """
    def __init__(self, name):
        self.name = name
        self.map = {}
        self.debuginfo = {}

    def mapper(self, sdfg) -> bool:
        """ Creates the source to SDFG node mapping
            :param sdfg: SDFG to create the mapping for
            :return: if the sdfg was created only by the API
        """
        self.debuginfo = self.sdfg_debuginfo(sdfg)
        self.debuginfo = self.divide()
        self.sorter()

        # Store the start and end line as a tuple
        # for each function/sub-function in the SDFG
        range_dict = collections.defaultdict(list)

        for nested_sdfg in sdfg.all_sdfgs_recursive():
            # NOTE: SDFGs created with the API may not have debuginfo
            debuginfo: dtypes.DebugInfo = nested_sdfg.debuginfo
            if debuginfo.filename:
                range_dict[debuginfo.filename].append(
                    (debuginfo.start_line, debuginfo.end_line))

        self.create_mapping(range_dict)
        return len(range_dict.items()) == 0

    def divide(self):
        """ Divide debuginfo into an array where each entry
            corresponds to the debuginfo of a diffrent sourcefile.
        """
        divided = []
        for dbinfo in self.debuginfo:
            source = dbinfo['debuginfo']['filename']
            exists = False
            for src_infos in divided:
                if len(src_infos) > 0 and src_infos[0]['debuginfo'][
                        'filename'] == source:
                    src_infos.append(dbinfo)
                    exists = True
                    break
            if not exists:
                divided.append([dbinfo])

        return divided

    def sorter(self):
        """ Prioritizes smaller ranges over larger ones """
        db_sorted = []
        for dbinfo_source in self.debuginfo:
            db_sorted.append(
                sorted(dbinfo_source,
                       key=lambda n: (n['debuginfo']['start_line'], n[
                           'debuginfo']['start_column'], n['debuginfo'][
                               'end_line'], n['debuginfo']['end_column'])))
        return db_sorted

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

            elif isinstance(node,
                            (nodes.MapEntry,
                             nodes.MapExit)) and node.map.debuginfo is not None:
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

    def create_mapping(self, range_dict=None):
        """ Creates the actual mapping by using the debuginfo list
            :param range_dict: for ech file a list of tuples containing a start and 
            end line of a DaCe program
        """
        for file_dbinfo in self.debuginfo:
            for node in file_dbinfo:
                src_file = node["debuginfo"]["filename"]
                if not src_file in self.map:
                    self.map[src_file] = {}
                for line in range(node["debuginfo"]["start_line"],
                                  node["debuginfo"]["end_line"] + 1):
                    # Maps a python line to a list of nodes
                    # The nodes have been sorted by priority
                    if not str(line) in self.map[src_file]:
                        self.map[src_file][str(line)] = []

                    self.map[src_file][str(line)].append({
                        "sdfg_id":
                        node["sdfg_id"],
                        "state_id":
                        node["state_id"],
                        "node_id":
                        node["node_id"]
                    })

        if range_dict:
            # Mapping lines that don't occur in the debugInfo of the SDFG
            # These might be lines that don't have any code on them or
            # no debugInfo correspond directly to them
            for src_file, ranges in range_dict.items():

                src_map = self.map.get(src_file)
                if src_map is None:
                    src_map = {}

                for start, end in ranges:
                    for line in range(start, end + 1):
                        if not str(line) in src_map:
                            # Set to the same node as the previous line
                            # If the previous line doesn't exist
                            # (line - 1 < f_start_line) then search the next lines
                            # until a mapping can be found
                            if str(line - 1) in src_map:
                                src_map[str(line)] = src_map[str(line - 1)]
                            else:
                                for line_after in range(line + 1, end + 1):
                                    if str(line_after) in src_map:
                                        src_map[str(line)] = src_map[str(
                                            line_after)]
                self.map[src_file] = src_map
