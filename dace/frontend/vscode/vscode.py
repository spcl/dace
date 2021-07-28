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


def stop_and_load():
    """ Stops and loads an SDFG file to create the code with
        :return: The loaded SDFG or None in the case that no SDFG was picked
    """
    reply = send_bp_recv({'type': 'loadSDFG'})
    if reply and 'filename' in reply:
        return dace.SDFG.from_file(reply['filename'])
    return None


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
    send_bp_recv({
        'type': 'stopAndTransform',
        'filename': filename,
        'sdfgName': sdfg.name
    })
    # Continues as soon as vscode sends an answer
    sdfg = dace.SDFG.from_file(filename)
    sdfg.name = sdfg.name + '_t'
    return sdfg


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


def sdfg_remove_instrumentations(sdfg: dace.sdfg.SDFG):
    sdfg.instrument = dace.dtypes.InstrumentationType.No_Instrumentation
    for state in sdfg.nodes():
        state.instrument = dace.dtypes.InstrumentationType.No_Instrumentation
        for node in state.nodes():
            node.instrument = dace.dtypes.InstrumentationType.No_Instrumentation
            if isinstance(node, dace.sdfg.nodes.NestedSDFG):
                sdfg_remove_instrumentations(node.sdfg)


def sdfg_has_instrumentation(sdfg: dace.sdfg.SDFG) -> bool:
    if sdfg.instrument != dace.dtypes.InstrumentationType.No_Instrumentation:
        return True
    for state in sdfg.nodes():
        if state.instrument != dace.dtypes.InstrumentationType.No_Instrumentation:
            return True
        for node in state.nodes():
            if (hasattr(node, 'instrument') and node.instrument !=
                    dace.dtypes.InstrumentationType.No_Instrumentation):
                return True
            if isinstance(node, dace.sdfg.nodes.NestedSDFG):
                return sdfg_has_instrumentation(node.sdfg)
    return False