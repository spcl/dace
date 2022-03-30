import re
import dace

from collections import Counter
from pathlib import Path

from dace.transformation import subgraph as sg
from dace.transformation.subgraph import helpers
from dace.transformation import helpers as xfh
from dace.optimization import SubgraphFusionTuner

def is_state_line(line):
    words = line.split()
    if len(words) != 2:
        return False

    state_id, state_name = words
    return state_id.isdigit()

def extract_patterns(i, lines):
    patterns = []
    while i < len(lines):
        line = lines[i]
        if is_state_line(line):
            break

        # 1. Waiting for "Finished"-line
        words = line.split()
        if len(words) > 1 or words[0] != "Finished":
            i = i + 1
            continue

        # State machine: Line == "Finished"

        # 2. Check for runtimes line
        i = i + 1
        line = lines[i]
        words = line.split()
        if len(words) != 2:
            continue

        base_runtime, new_runtime = words
        base_runtime = float(base_runtime)
        new_runtime = float(new_runtime)
        if new_runtime > base_runtime:
            i = i + 1
            continue

        # 3. Check for patterns
        i = i + 1
        line = lines[i]
        subgraph_pattern = re.search(r"\{(.*?)\}", line).groups()
        assert len(subgraph_pattern) == 1
        subgraph_pattern = "{" + subgraph_pattern[0] + "}"
        subgraph_pattern = eval(subgraph_pattern)
        subgraph_pattern = Counter(subgraph_pattern)

        # 4. "Fusing subgraph line..."
        i = i + 1
        line = lines[i]

        patterns.append(subgraph_pattern)

        # Onto the next line
        i = i + 1

    return patterns, i

sdfg_path = Path(__file__).parent / "hG-prepared.sdfg"
sdfg = dace.SDFG.from_file(sdfg_path)

states = [state for nsdfg in sdfg.all_sdfgs_recursive() for state in nsdfg]
state_names = set(map(lambda s: s.label, states))

applied = 0

fusions = {}
extracted_state_names = {}

log_path = Path(__file__).parent / "slurm-37284526_sgf.out"
with open(log_path, "r") as handle:
    def nonblank_lines(f):
        for l in f:
            if l.strip():
                line = l.strip()
                if line == "'NoneType' object has no attribute 'startswith'":
                    continue

                yield line

    lines = list(nonblank_lines(handle))

    i = 0
    while i < len(lines):
        line = lines[i]
        words = line.split()
        if len(words) != 2:
            i = i + 1
            continue

        state_id, state_name = words
        if not is_state_line(line):
            i = i + 1
            continue

        print("State: ", state_name)

        state_id = int(state_id)

        i = i + 1
        subgraph_patterns, i = extract_patterns(i, lines)
        # i already pointing to the next line

        if len(subgraph_patterns) == 0:
            continue

        fusions[state_id] = subgraph_patterns
        extracted_state_names[state_id] = state_name

for state_id, state in enumerate(states):
    if state_id not in fusions:
        continue

    if state.label != extracted_state_names[state_id]:
        raise ValueError("Out of sync")

    subgraph_patterns = fusions[state_id]
    for pattern in subgraph_patterns:
        maps = []
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry) and xfh.get_parent_map(state, node) is None:
                maps.append(node)

        maps_desc = {}
        state_desc = Counter()
        for map_entry in maps:
            map_desc = SubgraphFusionTuner.map_descriptor(state, map_entry)
            state_desc.update({map_desc: 1})

            if not map_desc in maps_desc:
                maps_desc[map_desc] = []

            maps_desc[map_desc].append(map_entry)

        included = True
        for key in pattern:
            if not key in state_desc or pattern[key] > state_desc[key]:
                included = False
                break

        if not included:
            raise ValueError("Out of sync")

        # Construct subgraph greedily
        subgraph_maps = []
        for desc in pattern:
            num = pattern[desc]
            subgraph_maps.extend(maps_desc[desc][:num])

        nsdfg = state.parent
        subgraph = helpers.subgraph_from_maps(sdfg=nsdfg, graph=state, map_entries=subgraph_maps)
        subgraph_fusion = sg.CompositeFusion(subgraph, nsdfg.sdfg_id, nsdfg.node_id(state))
        subgraph_fusion.allow_tiling = True
        subgraph_fusion.schedule_innermaps = dace.ScheduleType.GPU_Device
        subgraph_fusion.apply(nsdfg)

        print(state.label, pattern)
        applied += 1

k = 0
for key, item in fusions.items():
    k += len(item)
print(k)

print(applied)

sdfg.save(Path(__file__).parent / "hG-log-transfered.sdfg")
