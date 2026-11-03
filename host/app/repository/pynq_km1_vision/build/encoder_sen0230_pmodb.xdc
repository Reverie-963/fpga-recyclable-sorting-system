# SEN0230: NPN open-collector A/B, external pull-ups to 3.3 V.
# Sensor power is 5 V; never pull the FPGA inputs up to 5 V.
# Check the sensor's own wire label and the board connector orientation.
# PMODB physical pin 3 = T11 (A); physical pin 4 = T10 (B).
# PMODB pin 5 = GND; pin 6 = 3.3 V for signal pull-ups only.
# Keep PMODB free of other peripherals and load the vision overlay before use.
# Sources checked 2026-09-23:
# https://raw.githubusercontent.com/Xilinx/PYNQ/master/boards/Pynq-Z2/base/vivado/constraints/base.xdc
# TUL PYNQ-Z2 Reference Manual v1.0, section 16, table 14.
# https://wiki.dfrobot.com/sen0230/
# These pin constraints are source-checked, not board-tested or timing sign-off.
set_property -dict {PACKAGE_PIN T11 IOSTANDARD LVCMOS33} [get_ports {encoder_a}]
set_property -dict {PACKAGE_PIN T10 IOSTANDARD LVCMOS33} [get_ports {encoder_b}]
