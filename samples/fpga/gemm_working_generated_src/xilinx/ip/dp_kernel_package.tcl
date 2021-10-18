
#
# Argument parsing
#
if { $::argc != 8 } {
    puts "Error: Program \"$::argv0\" requires 7 arguments.\n"
    puts "Usage: $::argv0 <xoname> <kernel_name> <build_dir> <rtl_src_dir> <library_dir> <generate_dir> <board_part> <user_ip_repo>\n"
    exit
}

set xoname      [lindex $::argv 0]
set kernel_name [lindex $::argv 1]
set build_dir   [lindex $::argv 2]
set src_dir     [lindex $::argv 3]
set lib_dir     [lindex $::argv 4]
set gen_dir     [lindex $::argv 5]
set board_part  [lindex $::argv 6]
set user_repo   [lindex $::argv 7]

set tmp_dir "$build_dir/tmp"
set pkg_dir "$build_dir/pkg"

#
# Build the kernel
#
create_project kernel_packing $tmp_dir -part $board_part -force
add_files [glob $src_dir/*.*v $lib_dir/*.*v $gen_dir/*.*v]
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
update_compile_order -fileset sim_1
set_property top $kernel_name [current_fileset]
set_property top_file {$src_dir/$kernel_name} [current_fileset]
set_msg_config -id "HDL" -new_severity "ERROR"
check_syntax
reset_msg_config -id "HDL" -default_severity
ipx::package_project -root_dir $pkg_dir -vendor xilinx.com -library RTLKernel -taxonomy /KernelIP -import_files -set_current false
ipx::unload_core $pkg_dir/component.xml
ipx::edit_ip_in_project -upgrade true -name tmp_project -directory $pkg_dir $pkg_dir/component.xml

set core [ipx::current_core]

set_property core_revision 2 $core
foreach up [ipx::get_user_parameters] {
    ipx::remove_user_parameter [get_property NAME $up] $core
}
ipx::associate_bus_interfaces -busif s_axi_control -clock ap_clk $core
ipx::associate_bus_interfaces -busif s_axis_A_pipe -clock ap_clk $core
ipx::associate_bus_interfaces -busif s_axis_B_pipe -clock ap_clk $core
ipx::associate_bus_interfaces -busif m_axis_C_pipe -clock ap_clk $core


::ipx::infer_bus_interface "ap_clk_2"   "xilinx.com:signal:clock_rtl:1.0" $core
::ipx::infer_bus_interface "ap_rst_n_2" "xilinx.com:signal:reset_rtl:1.0" $core


# Specify the freq_hz parameter
set clkbif      [::ipx::get_bus_interfaces -of $core "ap_clk"]
set clkbifparam [::ipx::add_bus_parameter -quiet "FREQ_HZ" $clkbif]
# Set desired frequency
set_property value 250000000 $clkbifparam
# set value_resolve_type 'user' if the frequency can vary.
set_property value_resolve_type user $clkbifparam
# set value_resolve_type 'immediate' if the frequency cannot change.
# set_property value_resolve_type immediate $clkbifparam

# Specify the freq_hz parameter
set clkbif      [::ipx::get_bus_interfaces -of $core "ap_clk_2"]
set clkbifparam [::ipx::add_bus_parameter -quiet "FREQ_HZ" $clkbif]
# Set desired frequency
set_property value 250000000 $clkbifparam
# set value_resolve_type 'user' if the frequency can vary.
set_property value_resolve_type user $clkbifparam
# set value_resolve_type 'immediate' if the frequency cannot change.
# set_property value_resolve_type immediate $clkbifparam

set mem_map    [::ipx::add_memory_map -quiet "s_axi_control" $core]
set addr_block [::ipx::add_address_block -quiet "reg0" $mem_map]

# Set the control registers
set reg [::ipx::add_register "CTRL" $addr_block]
    set_property description          "Control signals" $reg
    set_property address_offset       0x000             $reg
    set_property size                 32                $reg
set field [ipx::add_field AP_START $reg]
    set_property ACCESS               {read-write}                              $field
    set_property BIT_OFFSET           {0}                                       $field
    set_property BIT_WIDTH            {1}                                       $field
    set_property DESCRIPTION          {Control signal Register for 'ap_start'.} $field
    set_property MODIFIED_WRITE_VALUE {modify}                                  $field
set field [ipx::add_field AP_DONE $reg]
    set_property ACCESS               {read-only}                              $field
    set_property BIT_OFFSET           {1}                                      $field
    set_property BIT_WIDTH            {1}                                      $field
    set_property DESCRIPTION          {Control signal Register for 'ap_done'.} $field
    set_property READ_ACTION          {modify}                                 $field
set field [ipx::add_field AP_IDLE $reg]
    set_property ACCESS               {read-only}                              $field
    set_property BIT_OFFSET           {2}                                      $field
    set_property BIT_WIDTH            {1}                                      $field
    set_property DESCRIPTION          {Control signal Register for 'ap_idle'.} $field
    set_property READ_ACTION          {modify}                                 $field
set field [ipx::add_field AP_READY $reg]
    set_property ACCESS               {read-only}                               $field
    set_property BIT_OFFSET           {3}                                       $field
    set_property BIT_WIDTH            {1}                                       $field
    set_property DESCRIPTION          {Control signal Register for 'ap_ready'.} $field
    set_property READ_ACTION          {modify}                                  $field
set field [ipx::add_field AP_RESERVED_1 $reg]
    set_property ACCESS               {read-only}              $field
    set_property BIT_OFFSET           {4}                      $field
    set_property BIT_WIDTH            {3}                      $field
    set_property DESCRIPTION          {Reserved.  0s on read.} $field
    set_property READ_ACTION          {modify}                 $field
set field [ipx::add_field AUTO_RESTART $reg]
    set_property ACCESS               {read-write}                                  $field
    set_property BIT_OFFSET           {7}                                           $field
    set_property BIT_WIDTH            {1}                                           $field
    set_property DESCRIPTION          {Control signal Register for 'auto_restart'.} $field
    set_property MODIFIED_WRITE_VALUE {modify}                                      $field
set field [ipx::add_field RESERVED_2 $reg]
    set_property ACCESS               {read-only}              $field
    set_property BIT_OFFSET           {8}                      $field
    set_property BIT_WIDTH            {24}                     $field
    set_property DESCRIPTION          {Reserved.  0s on read.} $field
    set_property READ_ACTION          {modify}                 $field

# Set the interrupt registers
set reg [::ipx::add_register "GIER" $addr_block]
    set_property description    "Global Interrupt Enable Register" $reg
    set_property address_offset 0x004                              $reg
    set_property size           32                                 $reg
set reg [::ipx::add_register "IP_IER" $addr_block]
    set_property description    "IP Interrupt Enable Register" $reg
    set_property address_offset 0x008                          $reg
    set_property size           32                             $reg
set reg [::ipx::add_register "IP_ISR" $addr_block]
    set_property description    "IP Interrupt Status Register" $reg
    set_property address_offset 0x00C                          $reg
    set_property size           32                             $reg

# Set the IP registers of the core




set_property slave_memory_map_ref "s_axi_control" [::ipx::get_bus_interfaces -of $core "s_axi_control"]

# Set the final project properties
set_property xpm_libraries             {XPM_CDC XPM_MEMORY XPM_FIFO} $core
set_property sdx_kernel                true                          $core
set_property sdx_kernel_type           rtl                           $core
set_property supported_families        { }                           $core
set_property auto_family_support_level level_2                       $core

# Save and close the project
ipx::create_xgui_files       $core
ipx::update_checksums        $core
ipx::check_integrity -kernel $core
ipx::save_core               $core
close_project

#
# Package the kernel
#
package_xo -xo_path ${xoname} -kernel_name $kernel_name -ip_directory $pkg_dir -force
