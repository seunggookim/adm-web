# Model files

`encoder.onnx`, `context.onnx` and `mlpstep.onnx` are the weights of

- **pa-audioic** (perceptually aligned autoregressive rectified flow; `pa_rectflow.pt`):
  Bjare, M. R., Cantisani, G., Pasini, M., Lattner, S., & Widmer, G. (2026). Perceptually aligning
  representations of music via noise-augmented autoencoders. *European Signal Processing Conference
  (EUSIPCO 2026)*. https://github.com/CPJKU/pa-audioic
- **pa-music2latent** (encoder; `music2latent.pt`), after Pasini, M., Lattner, S., & Fazekas, G.
  (2024). Music2Latent: Consistency autoencoders for latent audio compression. *ISMIR 2024*.
  https://github.com/CPJKU/pa-music2latent

by the Institute of Computational Perception, Johannes Kepler University Linz, licensed under the
Creative Commons Attribution-NonCommercial 4.0 International licence (CC BY-NC 4.0,
https://creativecommons.org/licenses/by-nc/4.0/). Source: Hugging Face CPJKU/pa-audioic, revision
917beb5fe491cb2af0414db7d4b2cf425e5deeb6.

**Changes:** converted to ONNX with `tools/admonnx.py` (the weights are unchanged). `mlpstep.onnx`
evaluates the diffusion MLP together with Jacobian-vector products for the Skilling-Hutchinson trace
estimate (forward mode, written out explicitly). The Music2Latent STFT representation, the Euler
integration and the Monte Carlo sampling are done in `adm.js`.

These files may be used for non-commercial purposes only, with attribution as above.
