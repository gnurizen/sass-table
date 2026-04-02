#!/usr/bin/env python3
"""
generate.py — Full pipeline to build SASS opcode lookup tables.

Usage:
    python3 generate.py [--archs sm_75,sm_80,...] [--nvcc /path/to/nvcc] [--nvdisasm /path/to/nvdisasm]

If --archs is omitted, auto-detects all supported SM >= 75 from nvcc.
Outputs:
    sass_opcode_tables.go  — per-arch opcode→mnemonic maps
    sass_decode.go         — decoder function
    sass_decode_test.go    — test file with embedded test vectors
"""

import argparse
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CORPUS_CU = SCRIPT_DIR / "corpus.cu"

# Architectures we care about (SM75+). The pipeline will skip any that
# the local nvcc doesn't support.
ALL_ARCHS = [
    "sm_75", "sm_80", "sm_86", "sm_87", "sm_89", "sm_90",
    "sm_100", "sm_120", "sm_121",
]

GO_PACKAGE = "sasstable"

# The opcode is bits [0:11] of the lower 64-bit word of the 128-bit instruction.
# 12 bits, no shift needed. Empirically verified: zero conflicts across SM75-SM90.
OPCODE_BITS = 12
OPCODE_MASK = (1 << OPCODE_BITS) - 1  # 0xfff

# Host compilers to try, in preference order, when nvcc rejects the default.
CCBIN_CANDIDATES = ["gcc-11", "gcc-12", "clang-14", "clang-15", "clang-16"]


def find_ccbin(nvcc):
    """Try to find a host compiler that nvcc accepts."""
    # Write a trivial .cu file to test with
    test_cu = Path(tempfile.mktemp(suffix=".cu"))
    test_cu.write_text("__global__ void k(){}\n")
    test_out = Path(tempfile.mktemp(suffix=".cubin"))

    try:
        # First try without -ccbin (default host compiler).
        result = subprocess.run(
            [nvcc, "-arch=sm_75", "-cubin", "-o", str(test_out), str(test_cu), "-w"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return None  # default compiler works

        for cc in CCBIN_CANDIDATES:
            cc_path = shutil.which(cc)
            if cc_path is None:
                continue
            result = subprocess.run(
                [nvcc, f"-ccbin={cc_path}", "-arch=sm_75", "-cubin",
                 "-o", str(test_out), str(test_cu), "-w"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                return cc_path
    finally:
        test_cu.unlink(missing_ok=True)
        test_out.unlink(missing_ok=True)

    return None


def find_tool(name, override=None):
    if override and os.path.isfile(override):
        return override
    path = shutil.which(name)
    if path is None:
        sys.exit(f"error: {name} not found in PATH")
    return path


def detect_supported_archs(nvcc):
    """Ask nvcc which gpu architectures it supports, return those >= sm_75."""
    try:
        out = subprocess.check_output([nvcc, "--list-gpu-arch"],
                                      stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError:
        # Older nvcc may not support --list-gpu-arch; fall back to trying each.
        return None
    supported = set()
    for line in out.strip().splitlines():
        line = line.strip()
        # Lines look like "compute_75" or "sm_75"
        m = re.match(r"(?:compute|sm)_(\d+)", line)
        if m:
            sm = int(m.group(1))
            if sm >= 75:
                supported.add(f"sm_{sm}")
    return sorted(supported)


def compile_cubin(nvcc, arch, corpus_cu, out_dir, ccbin=None):
    """Compile corpus.cu → cubin for a single arch. Returns path or None."""
    cubin = out_dir / f"corpus_{arch}.cubin"
    cmd = [nvcc, f"-arch={arch}", "-cubin", "-o", str(cubin), str(corpus_cu),
           "-w",  # suppress warnings
           ]
    if ccbin:
        cmd.insert(1, f"-ccbin={ccbin}")
    print(f"  compiling {arch}...", end=" ", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED\n    {result.stderr.strip().splitlines()[-1] if result.stderr.strip() else 'unknown error'}")
        return None
    print(f"OK ({cubin.stat().st_size} bytes)")
    return cubin


def disassemble(nvdisasm, cubin):
    """Run nvdisasm -hex -c and return the combined output text."""
    result = subprocess.run([nvdisasm, "-hex", "-c", str(cubin)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    nvdisasm failed: {result.stderr.strip()}")
        return None
    return result.stdout


def parse_disassembly(hex_text):
    """
    Parse nvdisasm -hex -c output.

    Each instruction spans two lines:
      /*0000*/   MOV R1, c[0x0][0x28] ;   /* 0x00000a0000017a02 */
                                           /* 0x000fc40000000f00 */

    The first line has the offset, mnemonic, and lower 64 bits.
    The second line has the upper 64 bits (the control/scheduling word).
    Together they form the 128-bit instruction: lower | (upper << 64).

    Returns list of (offset, raw_16_bytes, base_mnemonic).
    """
    # Match instruction lines: offset + mnemonic + hex encoding
    instr_re = re.compile(
        r'/\*([0-9a-fA-F]+)\*/\s+'       # /*offset*/
        r'(?:@!?P\d+\s+)?'               # optional predicate guard
        r'([A-Z_][A-Z0-9_.]*)'           # mnemonic (e.g. IMAD.WIDE)
        r'.*?/\*\s*(0x[0-9a-fA-F]+)\s*\*/'  # /* 0x... */ encoding
    )
    # Match continuation line (upper 64 bits): just /* 0x... */
    upper_re = re.compile(
        r'^\s+/\*\s*(0x[0-9a-fA-F]+)\s*\*/\s*$'
    )

    records = []
    pending = None  # (offset, mnemonic, lower_u64)

    for line in hex_text.splitlines():
        m = instr_re.search(line)
        if m:
            # If we had a pending instruction without upper bits, skip it
            off = int(m.group(1), 16)
            mnem_full = m.group(2)
            lower = int(m.group(3), 16)
            base = mnem_full.split(".")[0]
            pending = (off, base, lower)
            continue

        if pending is not None:
            mu = upper_re.match(line)
            if mu:
                upper = int(mu.group(1), 16)
                off, base, lower = pending
                # Pack as 128-bit little-endian: lower 8 bytes + upper 8 bytes
                raw = struct.pack("<QQ", lower, upper)
                records.append((off, raw, base))
                pending = None

    return records


def build_opcode_table(records):
    """Build opcode_value → mnemonic mapping.

    The opcode is the low 12 bits of the lower 64-bit word of the 128-bit
    instruction. This was determined empirically: 12 bits yield zero
    conflicts across SM75-SM90 with our corpus.
    """
    table = {}
    conflicts = []
    for off, raw, mnem in records:
        lower = struct.unpack_from("<Q", raw, 0)[0]
        opcode = lower & OPCODE_MASK
        if opcode in table:
            if table[opcode] != mnem:
                conflicts.append((opcode, table[opcode], mnem, off))
        else:
            table[opcode] = mnem
    return table, conflicts


def generate_go_tables(arch_tables):
    """Generate sass_opcode_tables.go content."""
    lines = []
    lines.append(f"// Code generated by generate.py. DO NOT EDIT.")
    lines.append(f"")
    lines.append(f"package {GO_PACKAGE}")
    lines.append(f"")
    lines.append(f"// Opcode field: bits [0:{OPCODE_BITS-1}] of the lower 64-bit word")
    lines.append(f"// of the 128-bit SASS instruction (little-endian).")
    lines.append(f"const opcodeMask = 0x{OPCODE_MASK:x}")
    lines.append(f"")

    # Build a merged superset table and per-arch tables
    merged = {}
    for arch, table in sorted(arch_tables.items()):
        for opc, mnem in table.items():
            if opc not in merged:
                merged[opc] = mnem

    # Check if all arches agree — if so, just emit one table.
    all_agree = True
    for arch, table in arch_tables.items():
        for opc, mnem in table.items():
            if merged.get(opc) != mnem:
                all_agree = False
                break

    if all_agree:
        lines.append(f"// All architectures share the same opcode encoding.")
        lines.append(f"// Merged from: {', '.join(sorted(arch_tables.keys()))}")
        lines.append(f"var opcodeTable = map[uint16]string{{")
        for opc in sorted(merged.keys()):
            lines.append(f'\t0x{opc:03x}: "{merged[opc]}",')
        lines.append(f"}}")
        lines.append(f"")
        lines.append(f"// ArchTables maps SM version → opcode table.")
        lines.append(f"// Since all supported architectures share the same encoding,")
        lines.append(f"// they all point to the same table.")
        lines.append(f"var ArchTables = map[int]map[uint16]string{{")
        for arch in sorted(arch_tables.keys()):
            sm_num = int(arch.replace("sm_", ""))
            lines.append(f"\t{sm_num}: opcodeTable,")
        lines.append(f"}}")
    else:
        # Per-arch tables
        lines.append(f"// Per-architecture opcode tables (encodings differ between arches).")
        for arch in sorted(arch_tables.keys()):
            sm_num = int(arch.replace("sm_", ""))
            table = arch_tables[arch]
            varname = f"opcodeTableSM{sm_num}"
            lines.append(f"")
            lines.append(f"var {varname} = map[uint16]string{{")
            for opc in sorted(table.keys()):
                lines.append(f'\t0x{opc:03x}: "{table[opc]}",')
            lines.append(f"}}")

        lines.append(f"")
        lines.append(f"// ArchTables maps SM version → opcode table.")
        lines.append(f"var ArchTables = map[int]map[uint16]string{{")
        for arch in sorted(arch_tables.keys()):
            sm_num = int(arch.replace("sm_", ""))
            lines.append(f"\t{sm_num}: opcodeTableSM{sm_num},")
        lines.append(f"}}")

    lines.append(f"")
    return "\n".join(lines)


def generate_go_decoder():
    """Generate sass_decode.go content."""
    return f"""// Code generated by generate.py. DO NOT EDIT.

package {GO_PACKAGE}

import "encoding/binary"

// DecodeMnemonic extracts the base SASS mnemonic from a 16-byte instruction
// word. archSM is the SM version number (e.g. 75, 80, 90). Returns "" if
// the opcode is unknown or the architecture is unsupported.
func DecodeMnemonic(archSM int, instrBytes [16]byte) string {{
\ttable := ArchTables[archSM]
\tif table == nil {{
\t\treturn ""
\t}}
\t// The opcode is in bits [0:11] of the lower 64-bit word (little-endian).
\tlo := binary.LittleEndian.Uint64(instrBytes[0:8])
\topcode := uint16(lo & opcodeMask)
\treturn table[opcode]
}}

// DecodeMnemonicFromSlice is a convenience wrapper that accepts a byte slice.
func DecodeMnemonicFromSlice(archSM int, instr []byte) string {{
\tif len(instr) < 16 {{
\t\treturn ""
\t}}
\tvar buf [16]byte
\tcopy(buf[:], instr[:16])
\treturn DecodeMnemonic(archSM, buf)
}}
"""


def generate_go_test(test_vectors):
    """Generate sass_decode_test.go with embedded test vectors."""
    lines = []
    lines.append(f"// Code generated by generate.py. DO NOT EDIT.")
    lines.append(f"")
    lines.append(f"package {GO_PACKAGE}")
    lines.append(f"")
    lines.append(f'import "testing"')
    lines.append(f"")
    lines.append(f"func TestDecodeMnemonic(t *testing.T) {{")
    lines.append(f"\ttests := []struct {{")
    lines.append(f"\t\tarchSM int")
    lines.append(f"\t\tinstr  [16]byte")
    lines.append(f"\t\twant   string")
    lines.append(f"\t}}{{")

    for arch_sm, raw_bytes, mnemonic in test_vectors:
        byte_str = ", ".join(f"0x{b:02x}" for b in raw_bytes)
        lines.append(f'\t\t{{{arch_sm}, [16]byte{{{byte_str}}}, "{mnemonic}"}},')

    lines.append(f"\t}}")
    lines.append(f"")
    lines.append(f"\tfor _, tt := range tests {{")
    lines.append(f"\t\tgot := DecodeMnemonic(tt.archSM, tt.instr)")
    lines.append(f'\t\tif got != tt.want {{')
    lines.append(f'\t\t\tt.Errorf("DecodeMnemonic(%d, %x) = %q, want %q",')
    lines.append(f"\t\t\t\ttt.archSM, tt.instr, got, tt.want)")
    lines.append(f"\t\t}}")
    lines.append(f"\t}}")
    lines.append(f"}}")
    lines.append(f"")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate SASS opcode lookup tables")
    parser.add_argument("--archs", help="Comma-separated arch list (e.g. sm_75,sm_80)")
    parser.add_argument("--nvcc", help="Path to nvcc")
    parser.add_argument("--nvdisasm", help="Path to nvdisasm")
    parser.add_argument("--ccbin", help="Host compiler for nvcc (auto-detected if omitted)")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep temp build dir")
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR), help="Output directory for Go files")
    args = parser.parse_args()

    nvcc = find_tool("nvcc", args.nvcc)
    nvdisasm = find_tool("nvdisasm", args.nvdisasm)

    # Find a working host compiler
    ccbin = args.ccbin or find_ccbin(nvcc)

    print(f"nvcc:     {nvcc}")
    print(f"nvdisasm: {nvdisasm}")
    print(f"ccbin:    {ccbin or '(default)'}")

    # Determine architectures
    if args.archs:
        archs = [a.strip() for a in args.archs.split(",")]
    else:
        detected = detect_supported_archs(nvcc)
        if detected is None:
            archs = ALL_ARCHS  # try them all
        else:
            archs = [a for a in ALL_ARCHS if a in detected]
        print(f"Detected supported archs: {', '.join(archs)}")

    if not archs:
        sys.exit("No architectures to build for!")

    # Create temp build dir
    tmp_dir = Path(tempfile.mkdtemp(prefix="sass_table_"))
    print(f"Build dir: {tmp_dir}")

    # --- Step 1 & 2: Compile cubins ---
    print("\n=== Compiling cubins ===")
    cubins = {}
    for arch in archs:
        cubin = compile_cubin(nvcc, arch, CORPUS_CU, tmp_dir, ccbin=ccbin)
        if cubin:
            cubins[arch] = cubin

    if not cubins:
        sys.exit("No cubins compiled successfully!")

    # --- Step 3: Disassemble ---
    print("\n=== Disassembling ===")
    arch_records = {}  # arch → list of (offset, raw_bytes, mnemonic)
    for arch, cubin in sorted(cubins.items()):
        print(f"  {arch}...", end=" ", flush=True)
        hex_text = disassemble(nvdisasm, cubin)
        if hex_text is None:
            print("FAILED")
            continue
        records = parse_disassembly(hex_text)
        arch_records[arch] = records
        mnemonics = set(r[2] for r in records)
        print(f"{len(records)} instructions, {len(mnemonics)} unique mnemonics")

    if not arch_records:
        sys.exit("No disassembly succeeded!")

    # --- Step 4+5: Build per-arch opcode tables ---
    print(f"\n=== Building opcode tables (bits [0:{OPCODE_BITS-1}], mask=0x{OPCODE_MASK:x}) ===")
    arch_tables = {}
    for arch, records in sorted(arch_records.items()):
        table, conflicts = build_opcode_table(records)
        arch_tables[arch] = table
        print(f"  {arch}: {len(table)} opcodes", end="")
        if conflicts:
            print(f", {len(conflicts)} CONFLICTS:")
            for opc, m1, m2, off in conflicts[:5]:
                print(f"    0x{opc:03x}: {m1} vs {m2} (at offset 0x{off:x})")
        else:
            print()

    # --- Check cross-arch consistency ---
    print("\n=== Cross-architecture comparison ===")
    all_opcodes = set()
    for table in arch_tables.values():
        all_opcodes.update(table.keys())

    diffs = 0
    for opc in sorted(all_opcodes):
        mnems = {}
        for arch, table in arch_tables.items():
            if opc in table:
                mnems[arch] = table[opc]
        unique = set(mnems.values())
        if len(unique) > 1:
            diffs += 1
            print(f"  DIFF 0x{opc:03x}: {mnems}")

    if diffs == 0:
        print("  All architectures agree on opcode encoding!")
    else:
        print(f"  {diffs} opcode(s) differ between architectures")

    # --- Step 6: Generate Go code ---
    print("\n=== Generating Go code ===")
    out_dir = Path(args.out_dir)

    # Tables
    tables_go = generate_go_tables(arch_tables)
    tables_path = out_dir / "sass_opcode_tables.go"
    tables_path.write_text(tables_go)
    print(f"  wrote {tables_path}")

    # Decoder
    decoder_go = generate_go_decoder()
    decoder_path = out_dir / "sass_decode.go"
    decoder_path.write_text(decoder_go)
    print(f"  wrote {decoder_path}")

    # Test vectors: one per unique (arch, opcode) pair to cover every table entry
    test_vectors = []
    for arch, records in sorted(arch_records.items()):
        sm_num = int(arch.replace("sm_", ""))
        seen_opcodes = set()
        for off, raw, mnem in records:
            lower = struct.unpack_from("<Q", raw, 0)[0]
            opc = lower & OPCODE_MASK
            if opc not in seen_opcodes:
                seen_opcodes.add(opc)
                test_vectors.append((sm_num, raw, mnem))

    test_go = generate_go_test(test_vectors)
    test_path = out_dir / "sass_decode_test.go"
    test_path.write_text(test_go)
    print(f"  wrote {test_path}")

    # Summary
    all_mnemonics = set()
    for table in arch_tables.values():
        all_mnemonics.update(table.values())
    print(f"\n=== Done ===")
    print(f"  Architectures: {', '.join(sorted(arch_tables.keys()))}")
    print(f"  Total unique mnemonics: {len(all_mnemonics)}")
    print(f"  Mnemonics: {', '.join(sorted(all_mnemonics))}")

    # Cleanup
    if not args.keep_tmp:
        shutil.rmtree(tmp_dir)
        print(f"  Cleaned up {tmp_dir}")
    else:
        print(f"  Kept build dir: {tmp_dir}")


if __name__ == "__main__":
    main()
