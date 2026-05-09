#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# scripts/run_eda.sh
#
# Run OpenROAD-flow-scripts EDA pipeline for a single design.
# Produces: synthesized netlist, placed DEF, timing report, power report.
#
# Usage:
#   bash scripts/run_eda.sh <verilog_file> <design_id> <clock_period_ns>
#
# Prerequisites:
#   - OpenROAD-flow-scripts installed
#   - OPENROAD_BIN, YOSYS_BIN, PDK_ROOT environment variables set
#   - ASAP7 PDK available at $PDK_ROOT/asap7
#
# Example:
#   bash scripts/run_eda.sh designs/ibex/ibex_core.v ibex_3ns 3.0
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Arguments ─────────────────────────────────────────────────────────────────
VERILOG_FILE="${1:?Usage: run_eda.sh <verilog_file> <design_id> <clock_period_ns>}"
DESIGN_ID="${2:?Missing design_id}"
CLOCK_PERIOD="${3:?Missing clock_period_ns}"

# ── Tool paths ────────────────────────────────────────────────────────────────
OPENROAD="${OPENROAD_BIN:-openroad}"
YOSYS="${YOSYS_BIN:-yosys}"
PDK="${PDK_ROOT:-/opt/openroad/flow/platforms}/asap7"

# ── Output directories ────────────────────────────────────────────────────────
WORK_DIR="runs/${DESIGN_ID}"
mkdir -p "${WORK_DIR}"

echo "═══════════════════════════════════════════════════════════"
echo " Design:  ${DESIGN_ID}"
echo " Verilog: ${VERILOG_FILE}"
echo " Clock:   ${CLOCK_PERIOD} ns"
echo " PDK:     ${PDK}"
echo "═══════════════════════════════════════════════════════════"

# ── Validate tools ────────────────────────────────────────────────────────────
command -v "${YOSYS}" >/dev/null 2>&1 || { echo "ERROR: Yosys not found at ${YOSYS}"; exit 1; }
command -v "${OPENROAD}" >/dev/null 2>&1 || { echo "ERROR: OpenROAD not found at ${OPENROAD}"; exit 1; }
[ -d "${PDK}" ] || { echo "ERROR: ASAP7 PDK not found at ${PDK}"; exit 1; }

# ── Step 1: Synthesis (Yosys) ─────────────────────────────────────────────────
echo ""
echo "▸ Step 1: Synthesis with Yosys"
SYNTH_SCRIPT="${WORK_DIR}/synth.tcl"
cat > "${SYNTH_SCRIPT}" <<EOF
read_verilog ${VERILOG_FILE}
read_liberty ${PDK}/lib/asap7sc7p5t_AO_RVT_FF_nldm_220122.lib
synth -top $(basename ${VERILOG_FILE} .v)
dfflibmap -liberty ${PDK}/lib/asap7sc7p5t_AO_RVT_FF_nldm_220122.lib
abc -liberty ${PDK}/lib/asap7sc7p5t_AO_RVT_FF_nldm_220122.lib
write_verilog ${WORK_DIR}/synth.v
EOF

"${YOSYS}" -s "${SYNTH_SCRIPT}" > "${WORK_DIR}/yosys.log" 2>&1
echo "  ✓ Synthesized netlist: ${WORK_DIR}/synth.v"

# ── Step 2: Placement (OpenROAD) ──────────────────────────────────────────────
echo ""
echo "▸ Step 2: Global + Detailed Placement"
PLACE_SCRIPT="${WORK_DIR}/place.tcl"
cat > "${PLACE_SCRIPT}" <<EOF
read_lef ${PDK}/lef/asap7_tech_1x_201209.lef
read_lef ${PDK}/lef/asap7sc7p5t_28_R_1x_220121a.lef
read_verilog ${WORK_DIR}/synth.v
link_design $(basename ${VERILOG_FILE} .v)

# Create floorplan
initialize_floorplan -die_area "0 0 200 200" \
  -core_area "10 10 190 190" \
  -site asap7sc7p5t

# Power planning
source ${PDK}/openroad/pdn.tcl

# Global placement
global_placement -density 0.7

# Detailed placement
detailed_placement

write_def ${WORK_DIR}/placed.def
EOF

"${OPENROAD}" "${PLACE_SCRIPT}" > "${WORK_DIR}/openroad_place.log" 2>&1
echo "  ✓ Placed DEF: ${WORK_DIR}/placed.def"

# ── Step 3: CTS + STA ────────────────────────────────────────────────────────
echo ""
echo "▸ Step 3: Clock Tree Synthesis + Static Timing Analysis"
STA_SCRIPT="${WORK_DIR}/sta.tcl"
cat > "${STA_SCRIPT}" <<EOF
read_lef ${PDK}/lef/asap7_tech_1x_201209.lef
read_lef ${PDK}/lef/asap7sc7p5t_28_R_1x_220121a.lef
read_def ${WORK_DIR}/placed.def
read_liberty ${PDK}/lib/asap7sc7p5t_AO_RVT_FF_nldm_220122.lib

create_clock -name clk -period ${CLOCK_PERIOD} [get_ports clk]

report_checks -path_delay max -fields {slew trans cap input_pins} \
  -format full_clock_expanded \
  > ${WORK_DIR}/timing.rpt

report_checks -path_delay min \
  >> ${WORK_DIR}/timing.rpt

report_tns >> ${WORK_DIR}/timing.rpt
report_wns >> ${WORK_DIR}/timing.rpt
EOF

"${OPENROAD}" "${STA_SCRIPT}" > "${WORK_DIR}/openroad_sta.log" 2>&1
echo "  ✓ Timing report: ${WORK_DIR}/timing.rpt"

# ── Step 4: Power Analysis ────────────────────────────────────────────────────
echo ""
echo "▸ Step 4: Power Analysis"
POWER_SCRIPT="${WORK_DIR}/power.tcl"
cat > "${POWER_SCRIPT}" <<EOF
read_lef ${PDK}/lef/asap7_tech_1x_201209.lef
read_lef ${PDK}/lef/asap7sc7p5t_28_R_1x_220121a.lef
read_def ${WORK_DIR}/placed.def
read_liberty ${PDK}/lib/asap7sc7p5t_AO_RVT_FF_nldm_220122.lib

create_clock -name clk -period ${CLOCK_PERIOD} [get_ports clk]

report_power > ${WORK_DIR}/power.rpt
EOF

"${OPENROAD}" "${POWER_SCRIPT}" > "${WORK_DIR}/openroad_power.log" 2>&1
echo "  ✓ Power report: ${WORK_DIR}/power.rpt"

# ── Step 5: Parse outputs ─────────────────────────────────────────────────────
echo ""
echo "▸ Step 5: Parsing reports"
python scripts/extract_timing.py \
  --input "${WORK_DIR}" \
  --output "${WORK_DIR}/timing_features.csv" 2>/dev/null || true

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " Complete: ${DESIGN_ID}"
echo " Outputs:  ${WORK_DIR}/"
echo "═══════════════════════════════════════════════════════════"
