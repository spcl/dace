# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import os
import json
import socket
import numpy as np
import dace
from dace.sdfg import state as st
from dace.sdfg import nodes as nd
from dace import memlet
from dace import data as dt, dtypes, symbolic


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
    ''' Removed the instrumentations set on the sdfg
        :param sdfg: The sdfg from which the instrumentsation is removed
    '''
    sdfg.instrument = dace.dtypes.InstrumentationType.No_Instrumentation
    for state in sdfg.nodes():
        state.instrument = dace.dtypes.InstrumentationType.No_Instrumentation
        for node in state.nodes():
            node.instrument = dace.dtypes.InstrumentationType.No_Instrumentation
            if isinstance(node, dace.sdfg.nodes.NestedSDFG):
                sdfg_remove_instrumentations(node.sdfg)


def sdfg_has_instrumentation(sdfg: dace.sdfg.SDFG) -> bool:
    ''' Checks if the SDFG has instrumentation set on it
        :param sdfg: The sdfg to check for instrumentation
        :return: True if the sdfg has instrumentation set on it
    '''
    if sdfg.instrument != dace.dtypes.InstrumentationType.No_Instrumentation:
        return True

    for node in sdfg.all_nodes_recursive():
        if (hasattr(node, 'instrument') and node.instrument !=
                dace.dtypes.InstrumentationType.No_Instrumentation):
            return True
    return False


def an_of_array(nested_sdfg: dace.sdfg.SDFG, array_name: str,
                state_id: int) -> [int]:
    ''' Gets all access node IDs that use the array in a specific state
        :param graph: The SDFG to check for the access node
        :param array_name: The name of the array to search for the access node
        :param state_id: The state ID in which the access node is located
        :return: An array of IDs of access nodes in the state that use the array
    '''
    ids = []
    for i, state in enumerate(nested_sdfg.nodes()):
        if i == state_id:
            for n_id, n in enumerate(state.nodes()):
                if isinstance(n, nd.AccessNode) and n.label == array_name:
                    ids.append(n_id)
            break
    return ids


def create_report(parent_sdfg: dace.sdfg.SDFG, sdfg_name: str, foldername1: str,
                  foldername2: str) -> dace.sdfg.SDFG:
    ''' Compare all matching arrays between both reports and send the 
        the comparisions to VSCode
        :param parent_sdfg: The parent SDFG of the second report
        :param sdfg_name: The sdfg name of the second report
        :param foldername1: The folder path of the referenced report
        :param foldername2: The folder path of the current report
        :return: The new sdfg if it has been modified, otherwise None
    '''
    reports = []

    all_arrays = [i for i in parent_sdfg.arrays_recursive()]

    for filename in os.listdir(foldername1):
        filename_split = filename.split('.')[0].split('_')
        array_name = '_'.join(filename_split[:-2])
        state_id = int(filename_split[-1])
        sdfg_id = int(filename_split[-2])

        # Search for the corresponding array in the sdfg
        curr_array = None
        access_node_ids = []
        for sdfg, name, array in all_arrays:
            if name == array_name:
                if sdfg.sdfg_id == sdfg_id:
                    curr_array = array
                    access_node_ids = an_of_array(sdfg, array_name, state_id)
                    break

        # Skip this array if it is not part of the SDFG anymore
        if curr_array is None or len(access_node_ids) == 0:
            continue

        dtype = curr_array.dtype
        filename1 = os.path.join(foldername1, filename)
        filename2 = os.path.join(foldername2, filename)

        # Compare both arrays
        msg = ''
        diff = None
        if os.path.isfile(filename2):
            nparray1 = np.fromfile(filename1, dtype=dtype.type)
            nparray2 = np.fromfile(filename2, dtype=dtype.type)

            if len(nparray1) == len(nparray2):
                diff = np.linalg.norm(nparray1 - nparray2)
                msg = 'Difference: ' + str(diff)
            else:
                msg = 'The array length doesn\'t match: {len1} vs. {len2}'.format(
                    len1=len(nparray1), len2=len(nparray2))
        else:
            msg = '''
                    The array {array_name} at SDFG: {sdfg_id} State: {state_id} 
                    isn\'t available in both reports
                '''.format(array_name=array_name,
                           state_id=state_id,
                           sdfg_id=sdfg_id)

        for an_id in access_node_ids:
            reports.append({
                'array_name': array_name,
                'sdfg_id': sdfg_id,
                'state_id': state_id,
                'node_id': an_id,
                'diff': str(diff),
                'msg': msg
            })

    filename = os.path.abspath(
        os.path.join(parent_sdfg.build_folder, 'program.sdfg'))
    send_bp_recv({
        'type': 'correctness_report',
        'sdfgName': sdfg_name,
        'filename': filename,
        'reports': reports
    })

    # Load the new SDFG if it has been changed
    new_sdfg = dace.SDFG.from_file(filename)
    if new_sdfg.hash_sdfg() != parent_sdfg.hash_sdfg():
        return new_sdfg
    return None


def pre_report(sdfg: dace.sdfg.SDFG):
    ''' Prepare the sdfg and arrays to run and create the report
        :param sdfg: SDFG to prepare for report
    '''
    if sdfg_has_instrumentation(sdfg):
        sdfg_remove_instrumentations(sdfg)

    # Make every array persistent so that we can access the data
    for _, _, array in sdfg.arrays_recursive():
        if array.storage != dtypes.StorageType.Register:
            array.lifetime = dtypes.AllocationLifetime.Persistent

    for node, nested_sdfg in sdfg.all_nodes_recursive():
        # Create an accuracy instrumentation for each SDFG state to
        # retrieve the data
        if isinstance(node, st.SDFGState):
            node.instrument = dtypes.InstrumentationType.Accuracy

            # If an Accessnode is not accessable from the CPU
            # then create a copy on the CPU and create an edge
            for an in node.data_nodes():
                name = an.data
                array = nested_sdfg.data(name)

                if (isinstance(array, dt.Array)
                        and array.storage != dtypes.StorageType.Register):
                    if not dtypes.can_access(dtypes.ScheduleType.Default,
                                             array.storage):
                        arr_name, _ = nested_sdfg.add_array(
                            name + '__cpu',
                            array.shape,
                            array.dtype,
                            lifetime=dtypes.AllocationLifetime.Persistent)

                        arr_node = node.add_access(arr_name)
                        mm = memlet.Memlet(data=an.data)
                        node.add_edge(an, None, arr_node, None, mm)