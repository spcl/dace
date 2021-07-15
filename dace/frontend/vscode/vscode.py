# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import os
import json
import socket
import time
import dace


def is_available() -> bool:
    """ Checks if vscode is listening to messages
        :return: if VSCode is listening to messages
    """
    return "DACE_port" in os.environ


def sdfg_edit() -> bool:
    """ If the debugger is in sdfg_edit mode
        :return: Returns true if the debugger is in sdfg_edit mode
    """
    return ("DACE_sdfg_edit" in os.environ
            and os.environ["DACE_sdfg_edit"] != "none")


def load_or_transform() -> bool:
    """ Checks if the user want's to apply transformations before the
        codegen runs on the SDFG
        :return: true if the env variable DACE_sdfg_edit is set
        to transform or load
    """
    if "DACE_sdfg_edit" in os.environ:
        return (os.environ["DACE_sdfg_edit"] == 'transform'
                or os.environ["DACE_sdfg_edit"] == 'load')
    return False


def stop_and_load(sdfg):
    """ Stops and loads an SDFG file to create the code with
        :param sdfg: The current SDFG
        :return: The loaded SDFG
    """
    reply = send_bp_recv({'type': 'loadSDFG'})
    if reply and 'filename' in reply:
        return dace.SDFG.from_file(reply['filename'])
    return sdfg


def stop_and_save(sdfg):
    """ Stops and save an SDFG to the chosen location
        :param sdfg: The current SDFG
    """
    reply = send_bp_recv({'type': 'saveSDFG'})
    if reply and 'filename' in reply:
        sdfg.save(reply['filename'])


def stop_and_transform(sdfg):
    """ Stops the compilation to allow transformations to the sdfg
        :param sdfg: The sdfg to allow transformations on
        :return: Transformed sdfg
    """
    filename = os.path.abspath(os.path.join(sdfg.build_folder, 'program.sdfg'))
    sdfg.save(filename)
    send_bp_recv({'type': 'stopAndTransform', 'filename': filename})
    return dace.SDFG.from_file(filename)


def pre_codegen_action(sdfg):
    """ Before the code generation we either load a saved SDFG
        or apply transformations o the current one
        :param sdfg: The current SDFG
        :return: The newly loaded SDFG 
    """
    if os.environ["DACE_sdfg_edit"] == 'load':
        return stop_and_load(sdfg)
    elif os.environ["DACE_sdfg_edit"] == 'transform':
        return stop_and_transform(sdfg)


def send(data: json):
    """ Sends a json object to the port given as the env variable DACE_port.
        If the port isn't set we don't send anything.
        :param data: json object to send
    """

    if not is_available():
        return

    HOST = socket.gethostname()
    PORT = os.environ["DACE_port"]

    data_bytes = bytes(json.dumps(data), "utf-8")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, int(PORT)))
        s.sendall(data_bytes)


def recv(data: json):
    """ Sends a json object to the port given as the env variable DACE_port
        and returns the received json object
        If the port isn't set we don't send anything.
        :param data: json object with request data
        :return: returns the object received
    """

    if not is_available():
        return

    HOST = socket.gethostname()
    PORT = os.environ["DACE_port"]

    data_bytes = bytes(json.dumps(data), "utf-8")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, int(PORT)))
        s.sendall(data_bytes)
        result = s.recv(1024)
        return json.loads(result)
    return None


def send_bp_recv(data: json) -> json:
    """ Sends a json object to the port given as the env variable DACE_port.
        After the data is sent a breakpint is set. As soon as the debugger
        continues, return the received data
        :param data: json object with request data
        :return: the received data as a json object
    """

    if not is_available():
        return

    HOST = socket.gethostname()
    PORT = os.environ["DACE_port"]

    data_bytes = bytes(json.dumps(data), "utf-8")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, int(PORT)))
        s.sendall(data_bytes)
        result = s.recv(1024)
        return json.loads(result)
    return None