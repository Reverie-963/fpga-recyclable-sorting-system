# 双核统一集成：Zynq PS7 + 3×AXI DMA + mask_geometry_accel + visual_evidence_accel
# 运行: vivado -mode batch -source build/integrate_bd.tcl
# 前置: PYNQ_BOARD_REPO 指向含 pynq-z2 的 board_files；两核 IP 已导出到 prj_*/sol1/impl/ip
# 架构:
#   dma_mask    MM2S->mask_0/input_r,      mask_0/output_r -> dma_mask/S2MM
#   dma_vis_a   MM2S->visual_0/inputA,     visual_0/output_r-> dma_vis_a/S2MM
#   dma_vis_b   MM2S->visual_0/inputB
#   5 个 AXI-Lite 从(3×DMA lite + 2×核 control) 挂 GP0；DMA 数据 5 主端口走 HP0 DDR

if {![info exists ::env(KM1_DUAL_BUILD)]} {
    error "Run tools/build_dual_core.ps1 or set KM1_DUAL_BUILD to a fresh build directory"
}
set root [file normalize $::env(KM1_DUAL_BUILD)]
cd $root
if {![info exists ::env(PYNQ_BOARD_REPO)]} {
    error "Set PYNQ_BOARD_REPO to directory containing PYNQ-Z2 board files"
}
set_param board.repoPaths [list $::env(PYNQ_BOARD_REPO)]

create_project mask_bd [file join $root vivado_project] -part xc7z020clg400-1
set boards [get_board_parts -quiet *:pynq-z2:part0:*]
if {[llength $boards] == 0} {error "PYNQ-Z2 board preset not found"}
set_property board_part [lindex $boards end] [current_project]

set_property ip_repo_paths [list \
    [file join $root mask_geometry_accel hls impl ip] \
    [file join $root visual_evidence_accel hls impl ip]] [current_project]
update_ip_catalog

create_bd_design design_1
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable"} \
    [get_bd_cells ps7]
set_property -dict [list \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100}] [get_bd_cells ps7]

# --- AXI DMA 配置 ---
set dma_props [dict create \
    CONFIG.c_include_sg {0} \
    CONFIG.c_sg_length_width {26} \
    CONFIG.c_m_axis_mm2s_tdata_width {32} \
    CONFIG.c_s_axis_s2mm_tdata_width {32} \
    CONFIG.c_include_mm2s_dre {1} \
    CONFIG.c_include_s2mm_dre {1}]
foreach d {dma_mask dma_vis_a} {
    create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 $d
    set_property -dict $dma_props [get_bd_cells $d]
}
# dma_vis_b 仅用于 MM2S 提供 inputB，S2MM 通道移除（避免孤儿端口）
set dma_props_b [dict merge $dma_props [dict create CONFIG.c_include_s2mm {0}]]
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 dma_vis_b
set_property -dict $dma_props_b [get_bd_cells dma_vis_b]

# --- mask 核 ---
create_bd_cell -type ip -vlnv xilinx.com:hls:mask_geometry_accel:1.0 mask_0
connect_bd_intf_net [get_bd_intf_pins dma_mask/M_AXIS_MM2S] [get_bd_intf_pins mask_0/input_r]
connect_bd_intf_net [get_bd_intf_pins mask_0/output_r] [get_bd_intf_pins dma_mask/S_AXIS_S2MM]

# --- visual 核（双输入）---
create_bd_cell -type ip -vlnv xilinx.com:hls:visual_evidence_accel:1.0 visual_0
connect_bd_intf_net [get_bd_intf_pins dma_vis_a/M_AXIS_MM2S] [get_bd_intf_pins visual_0/inputA]
connect_bd_intf_net [get_bd_intf_pins dma_vis_b/M_AXIS_MM2S] [get_bd_intf_pins visual_0/inputB]
connect_bd_intf_net [get_bd_intf_pins visual_0/output_r] [get_bd_intf_pins dma_vis_a/S_AXIS_S2MM]

# --- AXI-Lite：5 个从挂 GP0 ---
foreach slave {dma_mask/S_AXI_LITE mask_0/s_axi_control \
                dma_vis_a/S_AXI_LITE dma_vis_b/S_AXI_LITE visual_0/s_axi_control} {
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config \
        {Master "/ps7/M_AXI_GP0" Clk_master "/ps7/FCLK_CLK0 (100 MHz)" \
         Clk_slave "/ps7/FCLK_CLK0 (100 MHz)" Clk_xbar "/ps7/FCLK_CLK0 (100 MHz)"} \
        [get_bd_intf_pins $slave]
}

# --- DDR：先建 HP0 interconnect，再把 5 个 DMA 主端口都接到同一 interconnect ---
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config \
    {Master "/dma_mask/M_AXI_MM2S" Slave "/ps7/S_AXI_HP0" intc_ip "New AXI Interconnect" \
     Clk_master "/ps7/FCLK_CLK0 (100 MHz)" Clk_slave "/ps7/FCLK_CLK0 (100 MHz)" \
     Clk_xbar "/ps7/FCLK_CLK0 (100 MHz)"} [get_bd_intf_pins ps7/S_AXI_HP0]
foreach m {dma_mask/M_AXI_S2MM dma_vis_a/M_AXI_MM2S dma_vis_a/M_AXI_S2MM dma_vis_b/M_AXI_MM2S} {
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list \
        Master "/$m" Slave "/ps7/S_AXI_HP0" intc_ip "/axi_mem_intercon" \
        Clk_master "/ps7/FCLK_CLK0 (100 MHz)" Clk_slave "/ps7/FCLK_CLK0 (100 MHz)" \
        Clk_xbar "/ps7/FCLK_CLK0 (100 MHz)"] [get_bd_intf_pins $m]
}

assign_bd_address
validate_bd_design

# 检查所有 DMA 数据空间都映射到 DDR
foreach d {dma_mask dma_vis_a dma_vis_b} {
    foreach space {Data_MM2S Data_S2MM} {
        if {$d eq "dma_vis_b" && $space eq "Data_S2MM"} {continue}
        set segs [get_bd_addr_segs -of_objects [get_bd_addr_spaces $d/$space]]
        if {![string match *HP0_DDR_LOWOCM* $segs]} {
            error "$d/$space missing DDR mapping"
        }
    }
}

save_bd_design
generate_target all [get_files design_1.bd]
make_wrapper -files [get_files design_1.bd] -top
add_files -norecurse [file join $root vivado_project/mask_bd.gen/sources_1/bd/design_1/hdl/design_1_wrapper.v]
set_property top design_1_wrapper [current_fileset]

launch_runs impl_1 -to_step write_bitstream -jobs 1
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "Implementation failed"}
open_run impl_1
file mkdir [file join $root reports]
report_timing_summary -file [file join $root reports timing.rpt]
report_utilization -file [file join $root reports utilization.rpt]
report_drc -file [file join $root reports drc.rpt]
report_cdc -file [file join $root reports cdc.rpt]
report_methodology -file [file join $root reports methodology.rpt]
foreach delay {max min} {
    set paths [get_timing_paths -delay_type $delay -max_paths 1]
    if {[llength $paths] == 0 || [get_property SLACK [lindex $paths 0]] < 0} {
        error "$delay timing failed"
    }
    puts "DUALCORE_${delay}_SLACK=[get_property SLACK [lindex $paths 0]]"
}
if {[llength [get_drc_violations -filter {SEVERITY == Error}]] > 0} {error "DRC errors"}
file mkdir [file join $root board]
file copy -force [file join $root vivado_project/mask_bd.runs/impl_1/design_1_wrapper.bit] [file join $root board vision.bit]
file copy -force [file join $root vivado_project/mask_bd.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh] [file join $root board vision.hwh]
exit
