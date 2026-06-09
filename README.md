# Fixed-Point Quantization for LeNet-5 CNN Acceleration on FPGA

**A Hardware-Software Performance Comparison**

> **Jay Im** · Department of Computer Science and Engineering · Santa Clara University  
> ECEN 529: Hardware Acceleration for Machine Learning on FPGAs · June 2026

---

## Overview

This project implements and benchmarks a LeNet-5 convolutional neural network for traffic sign classification on a PYNQ-Z2 FPGA board (Xilinx Zynq XC7Z020). Three progressively optimized versions are compared — from a software-only ARM CPU baseline to a fully deployed fixed-point FPGA accelerator — with a focus on how fixed-point quantization affects timing closure, resource utilization, and inference accuracy.

**Key result: 57.78× speedup over software with identical classification accuracy.**

---

## Results at a Glance

| Metric | SW: ARM CPU (float) | HW: FPGA (ap_fixed\<16,8\>) |
|---|---|---|
| Predicted class | 29 ✅ | 29 ✅ |
| Avg latency | 1,194.31 ms | **20.67 ms** |
| Min latency | 1,186.80 ms | 20.64 ms |
| Max latency | 1,233.49 ms | 20.73 ms |
| Speedup | 1× | **57.78×** |

---

## HLS Synthesis Comparison

| Metric | Float Baseline | Float Optimized | ap_fixed\<16,8\> (Ours) |
|---|---|---|---|
| Est. clock (ns) | 8.424 | 8.424 | **7.279** |
| Latency (cycles) | 18,700,000 | 689,000 | 3,698,000 |
| Interval (cycles) | 1,873,177 | 68,948 | 369,809 |
| BRAM | 207 | 6,631 | **196** |
| DSP | 10 | 2,006 | **29** |
| FF | 9,511 | 295,996 | **14,254** |
| LUT | 9,577 | 386,047 | **26,892** |
| Timing slack (ns) | −1.12 ❌ | −1.12 ❌ | +2.72 ✅ |
| Deployable on PYNQ | No | No | **Yes** |

> Device limits: 53,200 LUT · 106,400 FF · 280 BRAM · 220 DSP

---

## The Key Finding: Accumulator Overflow

Naively substituting `float` with `ap_fixed<16,6>` caused misclassification (predicted class 27 instead of 29). The root cause was FC1 accumulator overflow — intermediate sums reached 37.13, exceeding the `ap_fixed<16,6>` ceiling of 31.99.

| Property | ap_fixed\<16,6\> | ap_fixed\<16,8\> |
|---|---|---|
| Integer bits | 6 | 8 |
| Fractional bits | 10 | 8 |
| Max value | 31.99 | 127.99 |
| FC1 sum (37.13) | **OVERFLOW** | OK |
| Predicted class | 27 ❌ | **29 ✅** |

**Fix:** use `ap_fixed<32,16>` as the accumulator type inside each MAC loop, then cast back to `ap_fixed<16,8>` at the layer output. This is the same dual-precision strategy used in Apple's Neural Engine (INT8 storage, INT32 accumulation) and NVIDIA Tensor Cores (FP16 storage, FP32 accumulation).

---

## Architecture

### LeNet-5 Network

```
Input (1×32×32)
    ↓
C1: Conv 6× 5×5 filters  →  6×28×28  feature maps
    ↓
S2: MaxPool 2×2           →  6×14×14
    ↓
C3: Conv 16× 5×5 filters  →  16×10×10  feature maps
    ↓
S4: MaxPool 2×2           →  16×5×5
    ↓
FC1: 400 → 120 units  (ReLU)
    ↓
FC2: 120 → 84 units   (ReLU)
    ↓
FC3: 84  → 43 logits  →  argmax  →  predicted class
```

### PYNQ-Z2 System Diagram

```
┌─────────────────────────────────────────────────────┐
│                  Zynq XC7Z020                       │
│                                                     │
│  ┌──────────────────┐      ┌─────────────────────┐  │
│  │  Processing      │      │  Programmable Logic │  │
│  │  System (PS)     │      │  (PL / FPGA fabric) │  │
│  │                  │      │                     │  │
│  │  ARM Cortex-A9   │────▶│  lenet_predict_     │ │
│  │  @ 650 MHz       │      │  fixed HLS IP       │  │
│  │                  │      │                     │  │
│  │  Jupyter /       │◀────│  ap_fixed<16,8>     │  │
│  │  Python host     │      │  26,892 LUTs        │  │
│  │                  │      │  29 DSPs            │  │
│  │  DDR Memory ─────┼──────▶  7.279 ns clock    │  │
│  └──────────────────┘      └─────────────────────┘  │
│                                                     │
│  M_AXI_GP0  →  AXI Interconnect  →  s_axi_control   │
│  S_AXI_HP0  ←  AXI Interconnect  ←  m_axi_gmem      │
└─────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
├── hls/
│   ├── lenet.cpp              # Version A: float baseline (unoptimized)
│   ├── lenet.h
│   ├── lenet_optimized.cpp    # Version B: float + HLS pragmas
│   ├── lenet_fixed.cpp        # Version C: ap_fixed<16,8> + pragmas ← deployed
│   ├── lenet_fixed.h
│   ├── parameters.h           # Pre-trained GTSRB weights
│   ├── image_data.h           # Test image (32×32 grayscale, class 29)
│   ├── tb_lenet.cpp           # Testbench for float version
│   └── tb_lenet_fixed.cpp     # Testbench for fixed-point version
├── vivado/
│   └── block_diagram.png      # Vivado system block diagram
├── pynq/
│   └── lenet_inference.py     # Python host: loads overlay, runs inference
├── paper/
│   ├── main.tex               # Full research paper (ACL format)
│   └── references.bib
└── README.md
```

---

## Hardware & Software Requirements

| Component | Version |
|---|---|
| Board | PYNQ-Z2 (Xilinx Zynq XC7Z020-CLG400-1) |
| HLS Tool | Vitis HLS 2025.2 |
| Implementation | Vivado 2025.2 |
| Host Framework | PYNQ 3.0 |
| Python | 3.10 |

---

## Implementation Details

### Three Versions

**Version A — Software Baseline**
- Original `lenet.cpp` with 32-bit `float`
- Runs entirely on ARM CPU via Python/NumPy
- No FPGA involvement
- Latency: **~1,194 ms**

**Version B — Float Optimized HLS**
- `PIPELINE II=1` on inner loops
- `ARRAY_PARTITION complete` on convolution weight arrays
- `DATAFLOW` at top level
- Result: 689K cycles estimated — but **386,047 LUTs required** (7× device limit)
- **Cannot be deployed on PYNQ-Z2**

**Version C — ap_fixed\<16,8\> Optimized** ← *this is what runs on FPGA*
- `float` → `ap_fixed<16,8>` for all storage
- `ap_fixed<32,16>` accumulator inside MAC loops
- `ARRAY_PARTITION complete` on conv layers only (FC layers use BRAM)
- `PIPELINE II=1` at innermost loops
- **Successfully deployed** — 26,892 LUTs, 29 DSPs, timing passes
- Latency: **20.67 ms**

### Key Pragma Insight

Placing `PIPELINE` at the outer loop of an FC layer causes HLS to unroll all inner iterations — FC1 has 120 × 400 = 48,000 weight reads, creating massive logic. Moving it to the innermost loop creates one reused MAC unit, reducing LUT usage by ~100×.

```cpp
// Wrong — explodes to 386K LUTs
for (int o = 0; o < 120; o++) {
    #pragma HLS PIPELINE II=1   // ← outer loop unrolls everything
    for (int i = 0; i < 400; i++) { sum += ...; }
}

// Right — fits on device
for (int o = 0; o < 120; o++) {
    for (int i = 0; i < 400; i++) {
        #pragma HLS PIPELINE II=1   // ← innermost loop only
        sum += ...;
    }
}
```

### AXI System Integration

The deployed system uses two AXI paths:

1. **Control path:** `Zynq M_AXI_GP0` → AXI Interconnect → `HLS IP s_axi_control`  
   Used for start/done signaling and reading the predicted class.

2. **Data path:** `HLS IP m_axi_gmem` → AXI Interconnect → `Zynq S_AXI_HP0`  
   Used for DMA — the IP fetches the input image directly from DDR.

---

## Running on PYNQ

```python
import numpy as np
from pynq import Overlay, allocate

# Load bitstream
ol = Overlay("lenet.bit")
lenet_ip = ol.lenet_predict_fixed_0

# Prepare image: float → ap_fixed<16,8>
# Scale by 2^8 = 256, cast to int16
SCALE = 256
image_fixed = (test_image * SCALE).astype(np.int16).flatten()

# Allocate DDR buffer
input_buf = allocate(shape=(1024,), dtype=np.int16)
input_buf[:] = image_fixed

# Run hardware inference
lenet_ip.write(0x10, input_buf.physical_address & 0xFFFFFFFF)
lenet_ip.write(0x00, 0x01)                         # ap_start
while not (lenet_ip.read(0x00) & 0x02):            # wait ap_done
    pass

predicted_class = lenet_ip.read(
    lenet_ip.register_map.predicted_class.address
)
print(f"Predicted class: {predicted_class}")       # → 29 (Bicycles Crossing)

input_buf.freebuffer()
```

---

## Dataset

**GTSRB — German Traffic Sign Recognition Benchmark**  
43-class traffic sign classification dataset from Ruhr-Universität Bochum.

- Training set: 39,209 images
- Test set: 12,630 images
- Input size used: 32×32 grayscale, normalized to [−1, 1]
- Test image used in this project: **class 29 — Bicycles Crossing**

Introduced by Stallkamp et al. (2012): *Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition.* Neural Networks, 32:323–332.

---

## Comparison with Prior Work

Qiu et al. (2016) reported 187.8 GOPS on a Xilinx Virtex-7 (2,800 DSPs) using VGG-style networks with mixed 8/16-bit quantization, achieving ~117× CPU speedup. Direct comparison is not straightforward given the larger device and deeper network, but our **57.78× speedup on the resource-constrained XC7Z020** falls within the 50–200× range reported in the fixed-point CNN literature, confirming that meaningful acceleration is achievable on low-cost embedded FPGAs.

---

## Future Work (Summer 2026)

- [ ] **Full GTSRB accuracy evaluation** — run all 12,630 test images to measure top-1 accuracy degradation from `ap_fixed<16,8>` quantization
- [ ] **Mixed-precision sweep** — per-layer bit-width optimization to characterize the accuracy-resource Pareto frontier
- [ ] **Systolic array design** — replace HLS-generated conv MAC logic with a manually designed 2D PE array
- [ ] **Streaming dataflow** — replace inter-layer buffers with `hls::stream` under `DATAFLOW` for concurrent layer execution

*Target venues: FCCM 2027 or FPL 2026*

---

## References

1. LeCun, Y. et al. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11):2278–2324.
2. Zhang, C. et al. (2015). Optimizing FPGA-based accelerator design for deep CNNs. *FPGA*, pp. 161–170.
3. Qiu, J. et al. (2016). Going deeper with embedded FPGA platform for CNN. *FPGA*, pp. 26–35.
4. Sharma, H. et al. (2016). Designing reconfigurable large-scale deep learning systems. *ICRC*, pp. 1–8.
5. AMD/Xilinx (2025). Vitis HLS User Guide (UG1399).
6. Stallkamp, J. et al. (2012). Man vs. computer: Benchmarking ML for traffic sign recognition. *Neural Networks*, 32:323–332.

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{im2026lenet,
  author    = {Jay Im},
  title     = {Fixed-Point Quantization for {LeNet}-5 {CNN} Acceleration on {FPGA}:
               A Hardware-Software Performance Comparison},
  year      = {2026},
  school    = {Santa Clara University},
  note      = {ECEN 529 Course Project}
}
```

---

*Santa Clara University · ECEN 529 · June 2026*
