# Joint vision/motion methodology and timing resolution

The first 100 MHz route failed setup timing: WNS -0.051 ns, TNS -0.589 ns, 14 endpoints. The critical vision counter-enable path spent 77.1% of its data delay in routing and included fanout 777.

Post-route `phys_opt_design -directive Explore` followed by `route_design -directive Explore` closed setup and hold: WNS +0.423 ns, WHS +0.051 ns, zero failing endpoints. No RTL, IP parameters, DDR preset, clock intent or timing exceptions were edited. Logical constraint commands are byte-identical; the text diff changes only the generated output filename comment.

| Aspect | Constraint | Period | Waveform | Clock Definition Object |
|--------|------------|--------|----------|-------------------------|
| Before | `create_clock -period 10.000 -name clk_fpga_0 [get_pins {PS7_i/FCLKCLK[0]}]` in `current_instance design_1_i/ps7/inst` | 10.000 ns | {0 5.000} | PS7_i/FCLKCLK[0] |
| After | `create_clock -period 10.000 -name clk_fpga_0 [get_pins {PS7_i/FCLKCLK[0]}]` in `current_instance design_1_i/ps7/inst` | 10.000 ns | {0 5.000} | PS7_i/FCLKCLK[0] |

| Check | Before | After | Resolution |
|-------|--------|-------|------------|
| Setup failing endpoints | 14 | 0 | Physical implementation optimization, same 10 ns requirement |
| Hold failing endpoints | 0 | 0 | Verified after reroute |
| LUTAR-1 | 11 warnings | 11 warnings | Retained vendor AXI protocol-converter FIFO async-reset LUT warnings; unsupported RTL check, no waiver |

DRC: 0 errors, 57 warnings (DPIP-1 9, DPOP-1 12, DPOP-2 24, PDCN-1569 11, RTSTAT-10 1). No waivers were created. Board-preset negative DDR skew warnings remain in original construction logs. The report does not claim a warning-free design.

Evidence: `results/methodology.json`, `results/methodology_after_fix.json`, `results/constraints.xdc`, `results/constraints_after_fix.xdc`, `results/constraints_diff.html`, `results/agent.log`, `results/timing_after.rpt`, `results/drc_after.rpt`, `run.tcl`, `native-vivado.log`. Hardware concurrency, pixel and route selftests still required before activation. Physical bitstream and HWH share the same logical joint construction and frozen IP exports, with the post-route checkpoint recorded by SHA.
