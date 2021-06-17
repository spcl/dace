import re
import json
import os
import socket
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


def create_folder(path_str: str):
    """ Creates a folder if it does not yet exist """
    if not os.path.exists(path_str):
        path = os.path.abspath(path_str)
        os.mkdir(path)


def create_cache(name):
    """ Creates the map folder in .dacecache if it
        does not yet exist
    """
    create_folder(".dacecache")
    create_folder(".dacecache/" + name)
    create_folder(".dacecache/" + name + "/map")
    return ".dacecache/" + name


def temporaryInfo(filename: str, data: json):
    """ Creates a temporary file that stores the json object
    in a temporary file in the map folder of the function
        :param filename: name of the function
        :param data: data to save
    """
    folder = create_cache(filename)
    path = os.path.abspath(folder + "/map/tmp.json")

    with open(path, "w") as writer:
        json.dump(data, writer)


def get_tmp(name):
    path = ".dacecache/" + name + "/map/tmp.json"
    if os.path.exists(path):
        with open(path, "r") as tmp_json:
            data = json.load(tmp_json)
        return data
    else:
        return None

def remove_tmp(name, remove_cache=False):
    ''' Remove the tmp file of the function 'name'.
        If remove_cache is true then check if the 
        directory only contains the tmp file,
        if this is the case, remove the cache directory
        of this function.
    '''
    path = ".dacecache/" + name

    if not os.path.exists(path):
        return

    try:
        os.remove(path + "/map/tmp.json")
    except OSError:
        print("removing error 1")
        return
    
    if (
        remove_cache and
        len(os.listdir(path)) == 1 and
        len(os.listdir(path + "/map")) == 0
    ):
        if os.path.exists(path):
            try:
                os.rmdir(path + "/map")
                os.rmdir(path)
            except OSError:
                print("removing error 2")
                return
        

def send(data: json):
    """ Sends a json object to the port given as the 
        env variable DACE_port. If the port isn't set
        we won't send anything
            :param data, json object to send
    """

    if("DACE_port" not in os.environ):
        return

    HOST = socket.gethostname()
    PORT = os.environ["DACE_port"]

    data_bytes = bytes(json.dumps(data), "utf-8")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, int(PORT)))
        s.sendall(data_bytes)


class MapCreater:
    def __init__(self, name):
        self.name = name
        self.map = {}

    def save(self, language: str):
        folder = create_cache(self.name)
        path = os.path.abspath(
            folder +
            "/map/map_" + language + ".json"
        )

        with open(path, "w") as json_file:
            json.dump(self.map, json_file, indent=4)

        return os.path.abspath(folder)



class MapCpp(MapCreater):
    def __init__(self, code, name, target_name):
        super(MapCpp, self).__init__(name)
        self.code = code
        self.mapper()
        self.map['target'] = target_name

        folder = self.save("cpp")

        tmp = get_tmp(self.name)
        remove_tmp(self.name)

        if tmp is not None:
            send(
                {
                    "type": "registerFunction",
                    "name": self.name,
                    "path_cache": folder,
                    "path_file": tmp.get("src_file"),
                    "target_name": target_name
                }
            )

    def mapper(self):
        for line_num, line in enumerate(self.code.split("\n"), 1):
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
                self.map[node.sdfg_id][node.state_id][node_id] = {
                    'from': line_num,
                    'to': line_num
                }
            elif self.map[node.sdfg_id][node.state_id][node_id]['to'] + 1 == line_num:
                self.map[node.sdfg_id][node.state_id][node_id]['to'] += 1

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

        line_info = get_tmp(self.name)

        # If we haven't save line and/or src info return
        # after creating the map and saving it
        # Happens when using the SDFG API
        if (
            line_info is None or
            "start_line" not in line_info or
            "end_line" not in line_info
        ):
            self.create_mapping()
        else:
            # Store the start and end line as a tuple 
            # for every function in the SDFG
            ranges = [
                (line_info["start_line"], line_info["end_line"])
            ]

            if "other_sdfgs" in line_info:
                for other_sdfg_name in line_info["other_sdfgs"]:
                    other_tmp = get_tmp(other_sdfg_name)
                    # SDFG's created with the API don't have tmp files
                    if other_tmp is not None:
                        remove_tmp(other_sdfg_name, True)
                        ranges.append(
                            (other_tmp["start_line"], other_tmp["end_line"])
                        )

                self.create_mapping(ranges)

        self.save("py")

    def sorter(self):
        self.map = sorted(
            self.map,
            key=lambda n: (
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
        Returns an array of Json objects of sdfg nodes with their debuginfo
        as also the sdfg, state and node id
        """
        if(sdfg_id is None):
            sdfg_id = 0

        mapping = []
        for node in graph["nodes"]:
            # If the node contains debugInfo, add it to the mapping
            if ("attributes" in node) and ("debuginfo" in node["attributes"]):
                mapping.append(self.make_mapping(node, sdfg_id, state_id))

            # If node has sub nodes, recursively call
            if("nodes" in node):
                mapping += self.sdfg_debuginfo(
                    node, 
                    sdfg_id = sdfg_id,
                    state_id = node["id"]
                )

            # If the node is a SDFG, recursively call
            if(("attributes" in node) and ("sdfg" in node["attributes"])):
                mapping += self.sdfg_debuginfo(
                    node["attributes"]["sdfg"],
                    sdfg_id=node["attributes"]["sdfg"]["sdfg_list_id"]
                )

        return mapping

    def create_mapping(self, ranges=None):
        mapping = {}
        for node in self.map:
            for line in range(
                node["debuginfo"]["start_line"],
                node["debuginfo"]["end_line"] + 1
            ):
                # If we find a line for the first time then we map
                # it to it's corresponding node
                # We expect that the nodes are sorted by priority
                if not str(line) in mapping:
                    mapping[str(line)] = [{
                        "sdfg_id": node["sdfg_id"],
                        "state_id": node["state_id"],
                        "node_id": node["node_id"]
                    }]
                else:
                    mapping[str(line)].append({
                        "sdfg_id": node["sdfg_id"],
                        "state_id": node["state_id"],
                        "node_id": node["node_id"]
                    })

        if ranges:
            # Mapping lines that don't occur in the debugInfo of the SDFG
            for start, end in ranges:
                for line in range(start, end + 1):
                    if not str(line) in mapping:
                        # Set to the same node as the previous line
                        # If the previous line doesn't exist 
                        # (line - 1 < f_start_line) then search the next lines
                        # until a mapping can be found
                        if str(line-1) in mapping:
                            mapping[str(line)] = mapping[str(line-1)]
                        else:
                            for line_after in range(line+1, end+1):
                                if str(line_after) in mapping:
                                    mapping[str(line)] = mapping[str(line_after)]

        self.map = mapping
