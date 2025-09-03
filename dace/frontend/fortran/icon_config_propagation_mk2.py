# Copyright 2023 ETH Zurich and the DaCe authors. All rights reserved.

import os
import re
from typing import List, Set, Tuple
import sys
from pathlib import Path

from fparser.common.readfortran import FortranFileReader as ffr
from fparser.two.parser import ParserFactory as pf

from dace.frontend.fortran.fortran_parser import ParseConfig, create_fparser_ast

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from dace.frontend.fortran import fortran_parser

import dace.frontend.fortran.ast_components as ast_components
import dace.frontend.fortran.ast_transforms as ast_transforms
import dace.frontend.fortran.ast_internal_classes as ast_internal


def find_path_recursive(base_dir):
    dirs = os.listdir(base_dir)
    fortran_files = []
    for path in dirs:
        if os.path.isdir(os.path.join(base_dir, path)):
            fortran_files.extend(find_path_recursive(os.path.join(base_dir, path)))
        if os.path.isfile(os.path.join(base_dir, path)) and (path.endswith(".F90") or path.endswith(".f90")):
            fortran_files.append(os.path.join(base_dir, path))
    return fortran_files


def read_lines_between(file_path: str, start_str: str, end_str: str) -> list[str]:
    lines_between = []
    with open(file_path, 'r') as file:
        capture = False
        for line in file:
            if start_str in line:
                capture = True
                continue
            if end_str in line:
                if capture:
                    capture = False
                    break
            if capture:
                lines_between.append(line.strip())
    return lines_between[1:]


def parse_assignments(assignments: list[str]) -> list[tuple[str, str]]:
    parsed_assignments = []
    for assignment in assignments:
        # Remove comments
        assignment = assignment.split('!')[0].strip()
        if '=' in assignment:
            a, b = assignment.split('=', 1)
            if b.strip()=="F":
                b=".FALSE."
            elif b.strip()=="T":
                b=".TRUE."
            parsed_assignments.append((a.strip().replace("_type","").replace("_obj","").replace(".","%"), b.strip()))
    return parsed_assignments

# Function to iterate over all the files in the given directory that are of the pattern "<name>_obj<number>.text". Name and directory are inputs.
# The function checks line by line that the files are identical and returns two lists:
# 1. Lines that are identical across all files.
# 2. Lines that are different across all files, in which case the list item will be a set of all different possibilities for that line.
# Most lines are independent, except if the line contains the string "_a =  F". If it does, consecutive lines following it that contain the pattern "_d<number>_s =" or the pattern "_o<number>_s =" are ignored fomr the comparison until a line that does not contain such a pattern is encountered.


def compare_files_in_directory(directory: str, name: str) -> Tuple[List[str], List[Set[str]]]:
    pattern = re.compile(rf"{name}_obj\d+\.txt")
    files = [os.path.join(directory, f) for f in os.listdir(directory) if pattern.match(f)]
    
    if not files:
        raise ValueError("No files matching the pattern found in the directory")

    identical_lines = []
    different_lines = []
    
    # Read all files into lists of lines
    file_lines = [open(file).readlines() for file in files]
    
    in_unallocated=False
    # Transpose the list of lists to compare lines at the same index
    for lines in zip(*file_lines):
        # Check if the line contains "_a =  F"
        if "_a =  F" in lines[0] and all(line == lines[0] for line in lines):
            identical_lines.append(lines[0].strip())
            in_unallocated=True
            continue
        
        # Check if the line should be ignored based on the patterns
        if in_unallocated:
            if any(re.search(r"_d\d+_s =", line) or re.search(r"_o\d+_s =", line) for line in lines):
                continue
            else:
                in_unallocated=False
        
        # Check if all lines are identical
        if all(line == lines[0] for line in lines):
            identical_lines.append(lines[0].strip())
        else:
            different_lines.append(set(line.strip() for line in lines))
    
    return identical_lines, different_lines


def replace_strings_in_file(filename: str, replacements: List[Tuple[str, str]]) -> None:
    # Read the file content
    with open(filename, 'r') as file:
        lines = file.readlines()

    def find_and_replace(line: str, old: str, new: str) -> str:
        # Remove spaces for comparison
        line_no_spaces = line.replace(" ", "")
        old_no_spaces = old.replace(" ", "")
        
        # Find all start indices of matches in the line without spaces
        start_indices = [m.start() for m in re.finditer(re.escape(old_no_spaces), line_no_spaces)]
        
        # Replace each match in the original line
        for start_index in start_indices:
            # Match character for character in the original line
            i, j = 0, 0
            finding = False
            new_start_index = -1
            while i < len(line) and j < len(old_no_spaces):
                if line[i]==' ':
                    i+=1
                    continue

                elif line[i] == old_no_spaces[j]:
                    if not finding:
                        new_start_index = i
                        finding = True
                    j += 1
                    i += 1
                else:
                    finding = False
                    new_start_index =-1
                    j = 0 
                    i += 1

                if j == len(old_no_spaces):
                    if i< len(line):
                        if line[i]=='_':
                            j=0
                            start_index=-1
            if start_index == -1:
                continue    
            end_index = i
            
            
            # Replace the matched portion in the original line
            line = line[:new_start_index] + new + line[end_index:]
            # Update line_no_spaces for subsequent matches
            line_no_spaces = line.replace(" ", "")
        
        return line

    # Process each line
    new_lines = []
    for line in lines:
        new_line = line
        for old, new in replacements:
            new_line = find_and_replace(new_line, old, new)
        new_lines.append(new_line)

    # Create the new filename with the suffix _prop before the file extension
    base, ext = os.path.splitext(filename)
    new_filename = f"{base}_prop{ext}"

    # Write the modified content to the new file
    with open(new_filename, 'w') as file:
        file.writelines(new_lines)


if __name__ == "__main__":

    base_icon_path = sys.argv[1]
    icon_file = sys.argv[2]
    sdfgs_dir = sys.argv[3]
    if len(sys.argv) > 4:
        already_parsed_ast = sys.argv[4]
    else:
        already_parsed_ast = None

    directory = "/home/alex/backup-icon-model/experiments/exclaim_ape_R2B09"
    id={}
    dif={}
    assigns={}
    all_assigns=[]
    for name in ["config_type","single_level","thermodynamics","gas","cloud","aerosol","flux"]:
        
        identical, different = compare_files_in_directory(directory, name)
        id[name]=identical
        dif[name]=different
        assert len(different)==0
        tmp=parse_assignments(identical)  
        assigns[name]=tmp
        all_assigns.extend(tmp)

    replace_strings_in_file(already_parsed_ast,all_assigns)          
       

    print("done")

    # base_dir_ecrad = f"{base_icon_path}/externals/ecrad"
    # base_dir_icon = f"{base_icon_path}/src"
    # fortran_files = find_path_recursive(base_dir_ecrad)
    # inc_list = [f"{base_icon_path}/externals/ecrad/include"]

    # # Construct the primary ECRad AST.
    # parse_cfg = ParseConfig(
    #     main=Path(f"{base_icon_path}/{icon_file}"),
    #     sources=[Path(f) for f in fortran_files],
    #     entry_points=[('radiation_interface', 'radiation')],
    # )
    #already_parsed_ast=None
    #ecrad_ast = create_fparser_ast(parse_cfg)
    # mini_parser=pf().create(std="f2008")
    # ecrad_ast = mini_parser(ffr(file_candidate=already_parsed_ast))
    # already_parsed_ast_bool = False       
    # ast = correct_for_function_calls(ast) 
    # cfg = fortran_parser.FindUsedFunctionsConfig(
    #     root='radiation',
    #     needed_functions=[['radiation_interface', 'radiation']],
    #     skip_functions=['radiation_monochromatic', 'radiation_cloudless_sw',
    #                     'radiation_tripleclouds_sw', 'radiation_homogeneous_sw']
    # )

    # generate_propagation_info(propagation_info)

    # previous steps were used to generate the initial list of assignments for ECRAD
    # this includes user config and internal enumerations of ICON
    # the previous ASTs can be now disregarded
    # we only keep the list of assignments and propagate it to ECRAD parsing.
    

    
