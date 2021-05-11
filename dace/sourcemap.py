import re
import json
import os
from pathlib import Path
from dace.config import Config

class SdfgLocation:
    def __init__(self, sdfg_id, state_id, node_ids):
        self.sdfg_id = sdfg_id
        self.state_id = state_id
        self.node_ids = node_ids
    
    def printer(self):
        print(
            "SDFG {}:{}:{}".format(
                self.sdfg_id,
                self.state_id,
                self.node_ids
            )
        )

def create_folder(path_str:str):
    if not os.path.exists(path_str):
        path = os.path.abspath(path_str)
        os.mkdir(path)

def create_cache(name):
    create_folder(".dacecache")
    create_folder(".dacecache/" + name)
    create_folder(".dacecache/" + name + "/map")
    return ".dacecache/" + name

def temporaryInfo(name, data):
    """ Creates a temporary file that stores start and end line 
    of the function for sourcemap processing later on
        :param file: filename
        :param start_line: first line of the function
        :param end_line: last line of the function
    """
    folder = create_cache(name)
    path = os.path.abspath(folder + "/map/tmp.json")

    with open(path, "w") as writer:
        json.dump(data, writer)

def send(folder, name):
    import socket

    HOST = socket.gethostname()
    PORT = os.getenv('DACE_port')

    if PORT is None:
        return
        
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, int(PORT)))
        data = json.dumps(
                {
                    "folder": folder,
                    "name": name
                }
            )
        s.sendall(bytes(data, "utf-8"))

class MapCreater:
    def __init__(self, name):
        self.name = name
        self.map = {}

    def save(self, language:str):
        folder = create_cache(self.name)
        path = os.path.abspath(
            folder +
            "/map_{language}.json"
        )

        with open(path ,"w") as json_file:
            json.dump(self.map, json_file, indent=4)
        
        return os.path.abspath(folder)
        


class MapCpp(MapCreater):
    def __init__(self, code, name):
        super(MapCpp, self).__init__(name)
        self.code = code
        self.mapper()
        folder = self.save("cpp")
        send(
            folder,
            self.name
        )

    def mapper(self):
        for line_num, line in enumerate(self.code.split("\n"),1):
            nodes = self.get_node(line)
            for node in nodes:
                self.create_mapping(node, line_num)

    def create_mapping(self, node, line_num):
        if node.sdfg_id not in self.map:
            self.map[node.sdfg_id] = {}
        if node.state_id not in self.map[node.sdfg_id]:
            self.map[node.sdfg_id][node.state_id] = {}
        for node_id in node.node_ids:
            if node_id not in self.map[node.sdfg_id][node.state_id]:
                self.map[node.sdfg_id][node.state_id][node_id] = line_num

    def get_node(self, line):
        line_identifiers = self.get_identifiers(line)
        nodes = []
        for identifier in line_identifiers:
            ids_split = identifier.split(":")
            nodes.append(
                SdfgLocation(
                    ids_split[1],
                    ids_split[2],
                    # node might be an edge
                    ids_split[3].split(",")
                )
            )
        return nodes
    
    def get_identifiers(self, line):
        line_identifier = re.findall(
            r'(\/\/\/\/__DACE:[0-9]:[0-9]:[0-9](,[0-9])*)',
            line
        )
        # remove tuples
        return [x or y for (x, y) in line_identifier]

class MapPython(MapCreater):
    def __init__(self, sdfg, name):
        super(MapPython, self).__init__(name)
        self.mapper(sdfg)

    def mapper(self, sdfg):
        self.map = self.sdfg_debuginfo(sdfg, None)

        self.sorter()

        line_info = self.get_tmp()
        if line_info is not None:
            self.create_mapping(
                line_info["start_line"],
                line_info["end_line"]
            )
        else:
            self.create_mapping()

        self.save("py")

    def sorter(self):
        self.map = sorted(
            self.map, 
            key = lambda n: (
                n['debuginfo']['start_line'], 
                n['debuginfo']['start_column'],
                n['debuginfo']['end_line'],
                n['debuginfo']['end_column']
            )
        )

    def make_mapping(self, node, sdfg_id, state_id):
        return {
            "type": node["type"],
            "label": node["label"],
            "debuginfo": node["attributes"]["debuginfo"],
            "sdfg_id": sdfg_id,
            "state_id": state_id,
            "node_id": node["id"]
        }

    def sdfg_debuginfo(self, graph, sdfg_id=0, state_id=0):
        """ 
        Returns an array of Json objects with sdfg nodes with their debuginfo
        as also the sdfg, state and node id
        """
        mapping = []
        for node in graph["nodes"]:
            # If the node contains debugInfo, add it to the mapping
            if ("attributes" in node) & ("debuginfo" in node["attributes"]):
                mapping.append(self.make_mapping(node, sdfg_id, state_id))

            # If node has sub nodes, recursivly call
            if("nodes" in node):
                mapping += self.sdfg_debuginfo(node, state_id=node["id"])

            # If the node is a SDFG, recursivly call
            if(("attributes" in node) & ("sdfg" in node["attributes"])):
                mapping += self.sdfg_debuginfo(
                    node["attributes"]["sdfg"], 
                    sdfg_id=node["id"]
                )

        return mapping
    
    def get_tmp(self):
        path = ".dacecache/" + self.name + "/map/tmp.json"
        if os.path.exists(path):
            with open(path, "r") as tmp_json:
                data = json.load(tmp_json)
            os.remove(path)
            return data
        else:
            return None
        
    def create_mapping(self, f_start_line = None, f_end_line = None):
        mapping = {}
        for node in self.map:
            for line in range(
                node["debuginfo"]["start_line"], 
                node["debuginfo"]["end_line"] + 1
                ):
                if not str(line) in mapping:
                    mapping[str(line)] = {
                        "sdfg_id": node["sdfg_id"],
                        "state_id": node["state_id"],
                        "node_id": node["node_id"]
                    }
        if (f_start_line is not None) & (f_end_line is not None):
            for line in range(f_start_line, f_end_line + 1):
                if not str(line) in mapping:
                        mapping[str(line)] = {
                            "sdfg_id": 0,
                            "state_id": 0,
                            "node_id": 0
                        }
        self.map = mapping
