# Historical entry point: intentionally disabled to prevent stale artifact export.
error {Use tools/build_dual_core.ps1 with a fresh output directory; see pynq_km1_vision/board/README_board_test.md}

# mask_geometry_accel 集成：Zynq PS7 + AXI DMA + mask 核（单核最小闭环）
# 运行: vivado -mode batch -source build/integrate_mask_bd.tcl
# 前置: PYNQ_BOARD_REPO 指向含 pynq-z2 的 board_files 目录；两个核 IP 已导出到
#       prj_*/sol1/impl/ip

set root [file normalize [file join [file dirname [info script]] .. ..]]
if {![info exists ::env(PYNQ_BOARD_REPO)]} {
    error "Set PYNQ_BOARD_REPO to directory containing PYNQ-Z2 board files"
}
set_param board.repoPaths [list $::env(PYNQ_BOARD_REPO)]

create_project -force mask_bd build/vivado_project_mask -part xc7z020clg400-1
set boards [get_board_parts -quiet *:pynq-z2:part0:*]
if {[llength $boards] == 0} {error "PYNQ-Z2 board preset not found"}
set_property board_part [lindex $boards end] [current_project]

# 两个 HLS 核的 IP 仓库都注册（本次只用 mask，visual 备用）
set_property ip_repo_paths [list \
    [file join $root prj_mask_geom_accel sol1 impl ip] \
    [file join $root prj_visual_evid_accel sol1 impl ip]] [current_project]
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

# AXI DMA：MM2S 供核输入流，S2MM 回收核输出流（避免背压挂起）
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_m_axis_mm2s_tdata_width {32} \
    CONFIG.c_s_axis_s2mm_tdata_width {32} \
    CONFIG.c_include_mm2s_dre {1} \
    CONFIG.c_include_s2mm_dre {1}] [get_bd_cells axi_dma_0]

# mask 核
create_bd_cell -type ip -vlnv xilinx.com:hls:mask_geometry_accel:1.0 mask_0
set axis_in [get_bd_intf_pins -of_objects [get_bd_cells mask_0] \
    -filter {MODE == Slave && VLNV =~ "*:axis_rtl:*"}]
set axis_out [get_bd_intf_pins -of_objects [get_bd_cells mask_0] \
    -filter {MODE == Master && VLNV =~ "*:axis_rtl:*"}]
if {[llength $axis_in] != 1 || [llength $axis_out] != 1} {
    error "mask_0 应有恰好 1 个 AXIS 输入和 1 个 AXIS 输出"
}
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] $axis_in
connect_bd_intf_net $axis_out [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]

# AXI-Lite：DMA 配置 + 核控制寄存器 挂 GP0
foreach slave {axi_dma_0/S_AXI_LITE mask_0/s_axi_control} {
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config \
        {Master "/ps7/M_AXI_GP0" Clk_master "/ps7/FCLK_CLK0 (100 MHz)" \
         Clk_slave "/ps7/FCLK_CLK0 (100 MHz)" Clk_xbar "/ps7/FCLK_CLK0 (100 MHz)"} \
        [get_bd_intf_pins $slave]
}

# DDR：先建 HP0 的 interconnect，再把 MM2S 和 S2MM 都接到同一 interconnect
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config \
    {Master "/axi_dma_0/M_AXI_MM2S" Slave "/ps7/S_AXI_HP0" intc_ip "New AXI Interconnect" \
     Clk_master "/ps7/FCLK_CLK0 (100 MHz)" Clk_slave "/ps7/FCLK_CLK0 (100 MHz)" \
     Clk_xbar "/ps7/FCLK_CLK0 (100 MHz)"} [get_bd_intf_pins ps7/S_AXI_HP0]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config \
    {Master "/axi_dma_0/M_AXI_S2MM" Slave "/ps7/S_AXI_HP0" intc_ip "/axi_mem_intercon" \
     Clk_master "/ps7/FCLK_CLK0 (100 MHz)" Clk_slave "/ps7/FCLK_CLK0 (100 MHz)" \
     Clk_xbar "/ps7/FCLK_CLK0 (100 MHz)"} [get_bd_intf_pins axi_dma_0/M_AXI_S2MM]

assign_bd_address
validate_bd_design

# 检查 DMA 是否映射到 DDR
foreach space {Data_MM2S Data_S2MM} {
    set segments [get_bd_addr_segs -of_objects [get_bd_addr_spaces axi_dma_0/$space]]
    if {![string match *HP0_DDR_LOWOCM* $segments]} {
        error "DMA $space 未映射到 DDR 地址段"
    }
    puts "KM1_DMA_${space}_DDR=$segments"
}

save_bd_design
generate_target all [get_files design_1.bd]
make_wrapper -files [get_files design_1.bd] -top
add_files -norecurse [file join $root build/vivado_project_mask mask_bd.gen/sources_1/bd/design_1/hdl/design_1_wrapper.v]
set_property top design_1_wrapper [current_fileset]

launch_runs impl_1 -to_step write_bitstream -jobs 1
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "Implementation failed"}

open_run impl_1
report_timing_summary -file report/timing_mask.rpt
report_utilization -file report/utilization_mask.rpt
report_drc -file report/drc_mask.rpt
set paths [get_timing_paths -delay_type max -max_paths 1]
if {[llength $paths] == 0 || [get_property SLACK [lindex $paths 0]] < 0} {
    error "Timing not met"
}

file mkdir -force board
file copy -force build/vivado_project_mask/mask_bd.runs/impl_1/design_1_wrapper.bit board/vision.bit
file copy -force build/vivado_project_mask/mask_bd.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh board/vision.hwh
open_bd_design [get_files design_1.bd]
write_bd_tcl -force build/generated_mask_bd.tcl
exit
