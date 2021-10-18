
if { !($::argc == 7 || $::argc == 8) } {
    puts "Error: Program \"$::argv0\" requires 7-8 arguments.\n"
    puts "Usage: $::argv0 <src_dir> <top_file> <build_dir> <lib_dir> <gen_dir> <board_part> <user_ip_repo> (elaborate)\n"
    exit
}

set src_dir    [lindex $::argv 0]
set top_file   [lindex $::argv 1]
set build_dir  [lindex $::argv 2]
set lib_dir    [lindex $::argv 3]
set gen_dir    [lindex $::argv 4]
set board_part [lindex $::argv 5]
set user_repo  [lindex $::argv 6]

create_project batch_synthesis $build_dir/synthesis -part $board_part -force
add_files [glob $src_dir/*.*v $lib_dir/*.*v $gen_dir/*.*v]
set_property top $top_file [current_fileset]
set_property top_file {$src_dir/$top_file} [current_fileset]
if {$user_repo != ""} {
    set_property ip_repo_paths $user_repo [current_project]
    update_ip_catalog -rebuild
}
create_ip -name dp_kernel -vendor xilinx.com -library hls -version 1.0 -module_name dp_kernel_0
create_ip -name axis_clock_converter -vendor xilinx.com -library ip -version 1.1 -module_name clock_sync_A_pipe
set_property -dict [list CONFIG.TDATA_NUM_BYTES {4} CONFIG.SYNCHRONIZATION_STAGES {8}] [get_ips clock_sync_A_pipe]
#create_ip -name axis_dwidth_converter -vendor xilinx.com -library ip -version 1.1 -module_name data_issue_pack_A_pipe
#set_property -dict [list CONFIG.S_TDATA_NUM_BYTES {4} CONFIG.M_TDATA_NUM_BYTES {2}] [get_ips data_issue_pack_A_pipe]
create_ip -name axis_clock_converter -vendor xilinx.com -library ip -version 1.1 -module_name clock_sync_B_pipe
set_property -dict [list CONFIG.TDATA_NUM_BYTES {32} CONFIG.SYNCHRONIZATION_STAGES {8}] [get_ips clock_sync_B_pipe]
create_ip -name axis_dwidth_converter -vendor xilinx.com -library ip -version 1.1 -module_name data_issue_pack_B_pipe
set_property -dict [list CONFIG.S_TDATA_NUM_BYTES {32} CONFIG.M_TDATA_NUM_BYTES {16}] [get_ips data_issue_pack_B_pipe]
create_ip -name axis_clock_converter -vendor xilinx.com -library ip -version 1.1 -module_name clock_sync_C_pipe
set_property -dict [list CONFIG.TDATA_NUM_BYTES {32} CONFIG.SYNCHRONIZATION_STAGES {8}] [get_ips clock_sync_C_pipe]
create_ip -name axis_dwidth_converter -vendor xilinx.com -library ip -version 1.1 -module_name data_issue_pack_C_pipe
set_property -dict [list CONFIG.S_TDATA_NUM_BYTES {16} CONFIG.M_TDATA_NUM_BYTES {32}] [get_ips data_issue_pack_C_pipe]
generate_target all [get_ips]

update_compile_order -fileset sources_1
update_compile_order -fileset sources_1
set_msg_config -id "HDL" -new_severity "ERROR"
check_syntax
reset_msg_config -id "HDL" -default_severity
synth_ip [get_ips]
if { $::argc == 7 } {
    synth_design -top $top_file -rtl
} else {
    synth_design -top $top_file
}

close_project
