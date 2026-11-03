# Retained only to give old commands an actionable error.
# The supported 2026.1 Windows flow is tools/build_windows.ps1 with generated
# .cfg files, vitis-run --csim, v++ synthesis, --cosim, and --package.
# The classic Tcl entry point omitted the design from csim.mk on this host,
# even with absolute paths and a fresh project. Do not resume the failed flow.
error "Use tools/build_windows.ps1 for the 2026.1 config-based HLS flow; see docs/vivado_2026_1.md. The legacy HLS Tcl entry point is disabled."
