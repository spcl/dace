`default_nettype none
`timescale 1 ns / 1 ps

module dp_kernel_top #(
    parameter integer C_S_AXI_CONTROL_ADDR_WIDTH = 4,
    parameter integer C_S_AXI_CONTROL_DATA_WIDTH = 32,
    parameter integer C_AXIS_TDATA_WIDTH         = 32
)
(
    input wire ap_clk,
    input wire ap_clk_2,
    input wire ap_rst_n,
    input wire ap_rst_n_2,

    input  wire                            s_axis_A_pipe_tvalid,
    input  wire [C_AXIS_TDATA_WIDTH-1:0]   s_axis_A_pipe_tdata,
    output wire                            s_axis_A_pipe_tready,
    input  wire [C_AXIS_TDATA_WIDTH/8-1:0] s_axis_A_pipe_tkeep,
    input  wire                            s_axis_A_pipe_tlast,

    input  wire                                 s_axis_B_pipe_tvalid,
    input  wire [7:0][C_AXIS_TDATA_WIDTH-1:0]   s_axis_B_pipe_tdata,
    output wire                                 s_axis_B_pipe_tready,
    input  wire      [C_AXIS_TDATA_WIDTH/8-1:0] s_axis_B_pipe_tkeep,
    input  wire                                 s_axis_B_pipe_tlast,

    output wire                                 m_axis_C_pipe_tvalid,
    output wire [7:0][C_AXIS_TDATA_WIDTH-1:0]   m_axis_C_pipe_tdata,
    input  wire                                 m_axis_C_pipe_tready,
    output wire      [C_AXIS_TDATA_WIDTH/8-1:0] m_axis_C_pipe_tkeep,
    output wire                                 m_axis_C_pipe_tlast,

    // Control AXI-Lite bus
    input  wire                                    s_axi_control_awvalid,
    output wire                                    s_axi_control_awready,
    input  wire [C_S_AXI_CONTROL_ADDR_WIDTH-1:0]   s_axi_control_awaddr,
    input  wire                                    s_axi_control_wvalid,
    output wire                                    s_axi_control_wready,
    input  wire [C_S_AXI_CONTROL_DATA_WIDTH-1:0]   s_axi_control_wdata,
    input  wire [C_S_AXI_CONTROL_DATA_WIDTH/8-1:0] s_axi_control_wstrb,
    input  wire                                    s_axi_control_arvalid,
    output wire                                    s_axi_control_arready,
    input  wire [C_S_AXI_CONTROL_ADDR_WIDTH-1:0]   s_axi_control_araddr,
    output wire                                    s_axi_control_rvalid,
    input  wire                                    s_axi_control_rready,
    output wire [C_S_AXI_CONTROL_DATA_WIDTH-1:0]   s_axi_control_rdata,
    output wire [2-1:0]                            s_axi_control_rresp,
    output wire                                    s_axi_control_bvalid,
    input  wire                                    s_axi_control_bready,
    output wire [2-1:0]                            s_axi_control_bresp
);

(* DONT_TOUCH = "yes" *)
reg  areset = 1'b0;
(* DONT_TOUCH = "yes" *)
reg  areset_2 = 1'b0;

wire ap_idle;
reg  ap_idle_r = 1'b1;
wire ap_done;
reg  ap_done_r = 1'b0;
wire ap_done_w;
wire ap_start;
reg  ap_start_r = 1'b0;
wire ap_start_pulse;


always @(posedge ap_clk) begin
    areset <= ~ap_rst_n;
end
always @(posedge ap_clk_2) begin
    areset_2 <= ~ap_rst_n_2;
end

always @(posedge ap_clk) begin
    begin
        ap_start_r <= ap_start;
    end
end
assign ap_start_pulse = ap_start & ~ap_start_r;

always @(posedge ap_clk) begin
    if (areset) begin
        ap_idle_r <= 1'b1;
    end else begin
        ap_idle_r <= ap_done ? 1'b1 : ap_start_pulse ? 1'b0 : ap_idle;
    end
end
assign ap_idle = ap_idle_r;

always @(posedge ap_clk) begin
    if (areset) begin
        ap_done_r <= 1'b0;
    end else begin
        ap_done_r <= ap_done ? 1'b0 : ap_done_w;
    end
end
assign ap_done = ap_done_r;

dp_kernel_control #(
    .C_S_AXI_ADDR_WIDTH ( C_S_AXI_CONTROL_ADDR_WIDTH ),
    .C_S_AXI_DATA_WIDTH ( C_S_AXI_CONTROL_DATA_WIDTH )
)
inst_dp_kernel_control (
    .ACLK       ( ap_clk ),
    .ARESET     ( areset ),
    .ACLK_EN    ( 1'b1 ),
    .AWVALID    ( s_axi_control_awvalid ),
    .AWREADY    ( s_axi_control_awready ),
    .AWADDR     ( s_axi_control_awaddr ),
    .WVALID     ( s_axi_control_wvalid ),
    .WREADY     ( s_axi_control_wready ),
    .WDATA      ( s_axi_control_wdata ),
    .WSTRB      ( s_axi_control_wstrb ),
    .ARVALID    ( s_axi_control_arvalid ),
    .ARREADY    ( s_axi_control_arready ),
    .ARADDR     ( s_axi_control_araddr ),
    .RVALID     ( s_axi_control_rvalid ),
    .RREADY     ( s_axi_control_rready ),
    .RDATA      ( s_axi_control_rdata ),
    .RRESP      ( s_axi_control_rresp ),
    .BVALID     ( s_axi_control_bvalid ),
    .BREADY     ( s_axi_control_bready ),
    .BRESP      ( s_axi_control_bresp ),
    .ap_start   ( ap_start ),
    .ap_done    ( ap_done ),
    .ap_ready   ( ap_done ),
    .ap_idle    ( ap_idle ),

    .interrupt  ( )
);


    wire        axis_A_pipe_clk_tvalid;
    wire [(1*C_AXIS_TDATA_WIDTH)-1:0] axis_A_pipe_clk_tdata;
    wire        axis_A_pipe_clk_tready;

    clock_sync_A_pipe clock_sync_A_pipe_inst (
        .s_axis_aclk(ap_clk),
        .s_axis_aresetn(ap_rst_n),
        .m_axis_aclk(ap_clk_2),
        .m_axis_aresetn(ap_rst_n_2),

        .s_axis_tvalid(s_axis_A_pipe_tvalid),
        .s_axis_tdata( s_axis_A_pipe_tdata),
        .s_axis_tready(s_axis_A_pipe_tready),

        .m_axis_tvalid(axis_A_pipe_clk_tvalid),
        .m_axis_tdata( axis_A_pipe_clk_tdata),
        .m_axis_tready(axis_A_pipe_clk_tready)
    );

    wire        axis_B_pipe_clk_tvalid;
    wire [(8*C_AXIS_TDATA_WIDTH)-1:0] axis_B_pipe_clk_tdata;
    wire        axis_B_pipe_clk_tready;

    clock_sync_B_pipe clock_sync_B_pipe_inst (
        .s_axis_aclk(ap_clk),
        .s_axis_aresetn(ap_rst_n),
        .m_axis_aclk(ap_clk_2),
        .m_axis_aresetn(ap_rst_n_2),

        .s_axis_tvalid(s_axis_B_pipe_tvalid),
        .s_axis_tdata( s_axis_B_pipe_tdata),
        .s_axis_tready(s_axis_B_pipe_tready),

        .m_axis_tvalid(axis_B_pipe_clk_tvalid),
        .m_axis_tdata( axis_B_pipe_clk_tdata),
        .m_axis_tready(axis_B_pipe_clk_tready)
    );

    wire        axis_C_pipe_clk_tvalid;
    wire [(8*C_AXIS_TDATA_WIDTH)-1:0] axis_C_pipe_clk_tdata;
    wire        axis_C_pipe_clk_tready;

    wire        axis_C_pipe_data_tvalid;
    wire [(4*C_AXIS_TDATA_WIDTH)-1:0] axis_C_pipe_data_tdata;
    wire        axis_C_pipe_data_tready;

    clock_sync_C_pipe clock_sync_C_pipe_inst (
        .s_axis_aclk(ap_clk_2),
        .s_axis_aresetn(ap_rst_n_2),
        .m_axis_aclk(ap_clk),
        .m_axis_aresetn(ap_rst_n),

        .s_axis_tvalid(axis_C_pipe_clk_tvalid),
        .s_axis_tdata( axis_C_pipe_clk_tdata),
        .s_axis_tready(axis_C_pipe_clk_tready),

        .m_axis_tvalid(m_axis_C_pipe_tvalid),
        .m_axis_tdata( m_axis_C_pipe_tdata),
        .m_axis_tready(m_axis_C_pipe_tready)
    );

    wire        axis_B_pipe_data_tvalid;
    wire [(4*C_AXIS_TDATA_WIDTH)-1:0] axis_B_pipe_data_tdata;
    wire        axis_B_pipe_data_tready;

    data_issue_pack_B_pipe data_issue_pack_B_pipe_inst (
        .aclk(ap_clk_2),
        .aresetn(ap_rst_n_2),

        .s_axis_tvalid(axis_B_pipe_clk_tvalid),
        .s_axis_tdata( axis_B_pipe_clk_tdata),
        .s_axis_tready(axis_B_pipe_clk_tready),

        .m_axis_tvalid(axis_B_pipe_data_tvalid),
        .m_axis_tdata( axis_B_pipe_data_tdata),
        .m_axis_tready(axis_B_pipe_data_tready)
    );

    data_issue_pack_C_pipe data_issue_pack_C_pipe_inst (
        .aclk(ap_clk_2),
        .aresetn(ap_rst_n_2),

        .s_axis_tvalid(axis_C_pipe_data_tvalid),
        .s_axis_tdata( axis_C_pipe_data_tdata),
        .s_axis_tready(axis_C_pipe_data_tready),

        .m_axis_tvalid(axis_C_pipe_clk_tvalid),
        .m_axis_tdata( axis_C_pipe_clk_tdata),
        .m_axis_tready(axis_C_pipe_clk_tready)
    );

// Free running kernel
assign ap_done_w = 1;

dp_kernel_0 inst_dp_kernel (
    .ap_clk   ( ap_clk_2 ),
    .ap_rst_n ( ap_rst_n_2 ),

    .A_pipe_in_V_TVALID ( axis_A_pipe_clk_tvalid ),
    .A_pipe_in_V_TDATA  ( axis_A_pipe_clk_tdata  ),
    .A_pipe_in_V_TREADY ( axis_A_pipe_clk_tready ),
    .B_pipe_in_V_TVALID ( axis_B_pipe_data_tvalid ),
    .B_pipe_in_V_TDATA  ( axis_B_pipe_data_tdata  ),
    .B_pipe_in_V_TREADY ( axis_B_pipe_data_tready ),
    .C_pipe_out_V_TVALID ( axis_C_pipe_data_tvalid ),
    .C_pipe_out_V_TDATA  ( axis_C_pipe_data_tdata  ),
    .C_pipe_out_V_TREADY ( axis_C_pipe_data_tready )
);

endmodule
`default_nettype wire
