# Plan: Build SASS opcode→mnemonic lookup tables from SM75+

## Goal

Build a static lookup table that maps raw instruction bytes to SASS mnemonic
strings (e.g. `FFMA`, `LDG`, `IMAD.MOV.U32`) for SM75 through SM121+. This
enables the backend to resolve instruction types from cubin PC offsets without
shelling out to nvdisasm at query time.

## Approach

Compile a large corpus of CUDA code for each target architecture, disassemble
with nvdisasm, and correlate raw instruction bytes with disassembled mnemonics
to discover the opcode bit positions and build the mapping table.

## Step 1: Gather CUDA source corpus

Collect as many distinct CUDA kernels as possible to maximize instruction
coverage. Sources:

- **CUDA samples** (`/usr/local/cuda/samples/` or github.com/NVIDIA/cuda-samples)
- **cuBLAS, cuFFT, cuDNN** test/example kernels
- **Thrust/CUB** — template-heavy, generates diverse instruction sequences
- **CUTLASS** (github.com/NVIDIA/cutlass) — matrix multiply kernels exercise
  tensor core instructions (HMMA, IMMA, etc.)
- **PyTorch/TensorRT** compiled kernels if available
- **Our own microbenchmarks** — write small kernels that deliberately exercise
  uncommon instructions:
  - Atomic operations (ATOM, ATOMG, RED)
  - Shared memory (LDS, STS, LDSM)
  - Texture/surface ops (TEX, TLD, SUST)
  - Warp-level (SHFL, VOTE, MATCH, REDUX)
  - Control flow (BRA, BRX, BSSY, BSYNC, EXIT, CALL, RET)
  - Uniform datapath (UMOV, UIADD3, ULOP3, etc.)
  - Double precision (DFMA, DADD, DMUL)
  - Half precision (HFMA2, HADD2)
  - Tensor core (HMMA, IMMA)
  - Special function (MUFU — sin, cos, rsqrt, etc.)

## Step 2: Compile for each target architecture

For each source file, compile a cubin per SM version:

```bash
ARCHS="sm_75 sm_80 sm_86 sm_87 sm_89 sm_90"
# sm_100+ requires CUDA 13.x toolkit
ARCHS_13="sm_100 sm_120 sm_121"

for arch in $ARCHS; do
  nvcc -arch=$arch -cubin -o corpus_${arch}.cubin corpus.cu
done
for arch in $ARCHS_13; do
  /usr/local/cuda-13.1/bin/nvcc -arch=$arch -cubin -o corpus_${arch}.cubin corpus.cu
done
```

Use `-cubin` (not `-fatbin`) to get a single-arch ELF per file.

## Step 3: Disassemble and extract raw bytes + mnemonics

For each cubin, produce two outputs:

**a) Disassembly text (mnemonics + offsets):**
```bash
nvdisasm corpus_sm_75.cubin > corpus_sm_75.sass
```

Output format:
```
/*0000*/  IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
/*0010*/  S2R R0, SR_CTAID.X ;
/*0020*/  IMAD R0, R0, c[0x0][0x170], R2 ;
```

**b) Raw hex dump of .text sections:**
```bash
nvdisasm --print-code corpus_sm_75.cubin > corpus_sm_75.hex
```

Or extract .text bytes with:
```bash
readelf -x .text._Z... corpus_sm_75.cubin
```

## Step 4: Correlate bytes to mnemonics

For each instruction at offset N:
1. Read 16 bytes from the .text section at offset N
2. Read the mnemonic from nvdisasm output at offset N
3. Record the pair: `(raw_bytes[0:16], mnemonic_base)`

Strip modifiers to get the base mnemonic: `IMAD.MOV.U32` → `IMAD`,
`LDG.E.128` → `LDG`, `FFMA.FTZ` → `FFMA`.

Build a dataset of `(architecture, raw_16_bytes, base_mnemonic)` tuples.

## Step 5: Discover opcode bit positions

With thousands of (bytes, mnemonic) pairs per architecture:

1. **Group by mnemonic** — all `FFMA` instructions should share the same
   bits in the opcode field
2. **Find the bits that are constant within each mnemonic group but vary
   between groups** — these are the opcode bits
3. **Verify across architectures** — check if the opcode bit positions are
   the same for SM75 through SM90+

Expected result based on community reverse engineering: the opcode is encoded
in approximately bits 108-119 of the 128-bit instruction word (the upper
portion of the second 64-bit half). The exact mask may differ slightly per
generation.

Script pseudocode:
```python
from collections import defaultdict

# For each architecture
for arch in architectures:
    groups = defaultdict(list)  # mnemonic → list of 128-bit instruction values
    for offset, raw_bytes, mnemonic in corpus[arch]:
        val = int.from_bytes(raw_bytes, 'little')
        groups[mnemonic].append(val)

    # For each bit position 0..127, check if it's constant within every group
    # but varies between groups
    opcode_bits = []
    for bit in range(128):
        is_opcode = True
        for mnemonic, vals in groups.items():
            bit_vals = set((v >> bit) & 1 for v in vals)
            if len(bit_vals) > 1:
                is_opcode = False  # This bit varies within the same mnemonic
                break
        if is_opcode:
            opcode_bits.append(bit)

    # The opcode mask is the set of bits that are constant per mnemonic
    print(f"{arch}: opcode bits = {opcode_bits}")
```

## Step 6: Build the lookup table

Once opcode bit positions are known:

```python
# Extract opcode value for each instruction
opcode_mask = compute_mask(opcode_bits)

table = {}  # opcode_value → mnemonic
for offset, raw_bytes, mnemonic in corpus[arch]:
    val = int.from_bytes(raw_bytes, 'little')
    opcode = (val & opcode_mask) >> min(opcode_bits)
    if opcode in table:
        assert table[opcode] == mnemonic, f"Conflict: {opcode:#x} → {table[opcode]} vs {mnemonic}"
    table[opcode] = mnemonic
```

Generate the table as a Go map or a C array:

```go
// Generated from SM75 corpus. Opcode extracted from bits [108:119].
var sm75Opcodes = map[uint16]string{
    0x042: "FFMA",
    0x066: "FADD",
    0x068: "FMUL",
    0x024: "IMAD",
    0x381: "LDG",
    0x385: "STG",
    0x202: "MOV",
    // ...
}
```

## Step 7: Implement the decoder

Minimal Go function (~20 lines):

```go
func decodeSASSMnemonic(archSM int, instrBytes [16]byte) string {
    val := binary.LittleEndian.Uint64(instrBytes[8:16])  // upper 64 bits
    opcode := (val >> opcodeBitShift) & opcodeBitMask

    table := opcodeTablesPerArch[archSM]
    if table == nil {
        return ""
    }
    return table[uint16(opcode)]
}
```

The arch SM version is read from the cubin ELF `e_flags` (bits 8-15).

## Step 8: Validate

1. Compile the test toy with debug info
2. Disassemble with nvdisasm (ground truth)
3. Run our decoder on the same .text bytes
4. Diff — every mnemonic should match

## Deliverables

- `sass_opcode_tables.go` — per-arch opcode→mnemonic maps
- `sass_decode.go` — the decoder function
- `sass_decode_test.go` — validation against nvdisasm output
- `cmd/gen-sass-tables/main.go` — the corpus compiler + table generator tool

## Open questions

- **Are opcode bits truly identical across SM75-SM121?** If yes, one table
  with a superset of mnemonics works for all. If not, one table per SM family.
- **Modifier encoding** — can we also decode `.E`, `.128`, `.FTZ` etc. or
  just the base mnemonic? Base mnemonic is likely sufficient for profiling.
- **Blackwell (SM100+)** — may have new encoding. Validate with corpus.

## Estimated effort

- Step 1-3 (corpus + disassembly): ~2 hours, mostly scripting
- Step 4-5 (bit analysis): ~2 hours, one-time Python script
- Step 6-7 (table + decoder): ~1 hour
- Step 8 (validation): ~1 hour

Total: ~1 day of focused work. The result is a ~200-line Go package with
zero external dependencies that decodes SASS mnemonics in nanoseconds.
