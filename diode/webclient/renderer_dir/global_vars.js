var LINEHEIGHT = 10;
var TASKLET_LOD = 0.35; // Points-per-pixel threshold for drawing tasklet contents
var SCOPE_LOD = 1.5; // Points-per-pixel threshold for simple version of map nodes (label only)
var EDGE_LOD = 8; // Points-per-pixel threshold for not drawing memlets/interstate edges
var NODE_LOD = 5; // Points-per-pixel threshold for not drawing node shapes and labels
var STATE_LOD = 50; // Pixel threshold for not drawing state contents

var targetsection_ignore_error = true; // Set to true to ignore errors occurring when no valid targetsection was found
var toplevel_use_mean = false;
var toplevel_use_median = true;
var cache_graphs = true; // Cache graphs, such they are not re-layouted on every draw call
var global_disable_verifier = false; // Disable verification

var max_overhead_percentage = 1e6;

var auto_compensate_overhead = true; // If set to true, the minimal collected overhead times are subtracted from every measurement.

var display_memory_correlation = false;

var target_dp_flops_per_cycle_per_thread = 8.0; // http://www.crc.nd.edu/~rich/CRC_EPYC_Cluster_Build_Feb_2018/Installing%20and%20running%20HPL%20on%20AMD%20EPYC%20v2.pdf
var target_sp_flops_per_cycle_per_thread = 16.0; // https://en.wikipedia.org/wiki/FLOPS#FLOPs_per_cycle_for_various_processors (Double the SP flops)

var all_analyses_global = false; // Specifies whether displayed results should be local (false) or global (true). local is easier for debugging, while global can be used to differentiate overall runtimes