#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${1:-/home/masif-neosurf/PDBbind_v2020_refined/refined-set}"
MASIF_SOURCE="${MASIF_SOURCE:-/home/masif-neosurf/masif/source}"
WORKROOT="${WORKROOT:-/tmp/masif_ply_work}"

echo "[INFO] DATASET_ROOT = ${DATASET_ROOT}"
echo "[INFO] MASIF_SOURCE  = ${MASIF_SOURCE}"
echo "[INFO] WORKROOT      = ${WORKROOT}"
echo

# Fail fast if wrong paths
[[ -d "${DATASET_ROOT}" ]] || { echo "[FATAL] Dataset root not found: ${DATASET_ROOT}"; exit 2; }
[[ -f "${MASIF_SOURCE}/data_preparation/01-pdb_extract_and_triangulate.py" ]] || {
  echo "[FATAL] Missing: ${MASIF_SOURCE}/data_preparation/01-pdb_extract_and_triangulate.py"
  exit 3
}

export PYTHONPATH="${PYTHONPATH:-}:${MASIF_SOURCE}"
mkdir -p "${WORKROOT}"

# Find pocket files (this ensures we don't silently do nothing)
mapfile -t pocket_files < <(find "${DATASET_ROOT}" -maxdepth 2 -type f -name "*_pocket.pdb" | sort)
echo "[INFO] Found ${#pocket_files[@]} pocket PDBs."
echo
if [[ ${#pocket_files[@]} -eq 0 ]]; then
  echo "[FATAL] No '*_pocket.pdb' found under ${DATASET_ROOT}."
  exit 4
fi

# Detect first chain in a PDB
detect_chain_py='
import sys
with open(sys.argv[1], "r", errors="ignore") as f:
  for line in f:
    if line.startswith(("ATOM","HETATM")) and len(line) >= 22:
      c=line[21].strip()
      if c:
        print(c); raise SystemExit
print("A")
'

ok_count=0
fail_count=0

for pocket_pdb in "${pocket_files[@]}"; do
  pdb_dir="$(dirname "${pocket_pdb}")"
  pdbid="$(basename "${pdb_dir}")"

  chain="$(python -c "${detect_chain_py}" "${pocket_pdb}")"
  name_chain="${pdbid}_${chain}"

  echo "============================================================"
  echo "[INFO] Processing ${pdbid}"
  echo "[INFO] pocket_pdb = ${pocket_pdb}"
  echo "[INFO] chain      = ${chain}"
  echo "[INFO] name_chain = ${name_chain}"
  echo "============================================================"

  outdir="${WORKROOT}/${name_chain}"
  rm -rf "${outdir}"
  mkdir -p "${outdir}/data_preparation/00-raw_pdbs"

  # MaSIF expects this filename
  cp "${pocket_pdb}" "${outdir}/data_preparation/00-raw_pdbs/${pdbid}.pdb"

  # Run triangulation; capture logs
  set +e
  ( cd "${outdir}" && python -W ignore "${MASIF_SOURCE}/data_preparation/01-pdb_extract_and_triangulate.py" "${name_chain}" ) \
    2>&1 | tee "${outdir}/triangulate.log"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ "${rc}" -ne 0 ]]; then
    echo "[ERROR] Triangulation failed for ${name_chain} (exit ${rc})."
    echo "        Log: ${outdir}/triangulate.log"
    ((fail_count+=1))
    continue
  fi

  # MaSIF can write the PLY in a few possible subfolders; search broadly
  ply_file="$(find "${outdir}/data_preparation" -type f -name "*.ply" | head -n 1 || true)"

  if [[ -z "${ply_file}" ]]; then
    echo "[ERROR] Triangulation succeeded but no .ply was found."
    echo "        Keeping workdir for inspection: ${outdir}"
    ((fail_count+=1))
    continue
  fi

  dest="${pdb_dir}/${name_chain}.ply"
  cp -vf "${ply_file}" "${dest}"
  echo "[OK] Saved: ${dest}"

  rm -rf "${outdir}"
  ((ok_count+=1))
done

echo
echo "[DONE] Completed. OK=${ok_count} FAIL=${fail_count}"
