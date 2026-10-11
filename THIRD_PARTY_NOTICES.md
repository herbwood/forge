# Third-party notices

The release does not vendor full copies of EasyR1, verl, Transformers, PEFT, PyTorch, NumPy or Pillow. Qwen weights and generation engines are obtained separately.

| Material | Source / verified notice | Treatment |
|---|---|---|
| Image-area processing in `model_io.process_image` | `EasyR1/verl/utils/dataset.py`, Copyright 2024 Bytedance Ltd. and/or its affiliates; Apache License 2.0 | Small adapted fragment; processing order and rounding preserved. License included at `licenses/Apache-2.0.txt`. |
| EasyR1 / verl integration | Local EasyR1 `LICENSE`: Apache-2.0; checkout `602a12820f69cf45ea98de08a63b3509313a7d02` | External dependency and framework contracts; no full trainer blocks copied. The minimum position/loss hooks are adapted from the research extension. |
| PyTorch | Installed package metadata: BSD-3-Clause | External dependency; retain its distribution notices. |
| Transformers and PEFT | Installed package metadata: Apache-2.0 | External optional dependencies; no library code copied. |
| NumPy | Installed package license begins with NumPy Developers BSD terms and includes notices for bundled libraries | External dependency; its distribution notices govern bundled material. |
| PyYAML | Installed package metadata: MIT | External dependency. |
| Pillow | Installed package metadata: MIT-CMU | External dependency. |
| Qwen3-VL actor | Caller obtains weights separately under their model-card/license terms | No weights, model license grant or checkpoint availability claimed. |

The adapted image-processing fragment is provided under its original Apache-2.0 terms, with its existing warranty disclaimer. All external dependencies, model weights, and frameworks remain subject to their respective upstream licenses and terms.
