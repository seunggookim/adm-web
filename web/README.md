# Musical surprise and uncertainty in the browser

`index.html` computes the information content (IC) and entropy of the ADM (pa-audioic) for audio
files, frame by frame (about 10.8 Hz), entirely in the browser: audio files are never uploaded.
It runs ONNX Runtime Web on WebGPU (Chrome, Edge; WebAssembly as a slower fallback). Files or
folders can be dropped; results come as one CSV per file (`Frame, Time, IC_<t>, Entropy_<t>`, as
the Python tool `admcompute.py`), as a ZIP, or written to a chosen folder, with a JSON log.

- `adm.js`: the computation (silence trimming, Music2Latent STFT representation, encoder in 30-s
  windows inside pa-audioic's 60-s chunks, 100 Euler steps with the Skilling-Hutchinson trace estimate for IC,
  32 Monte Carlo samples for entropy, with a variance reduction).
- `app.js`: the page (model loading and caching, decoding, batch, output).
- `models/`: the ONNX files and their licence (CC BY-NC 4.0; see `models/LICENSE.md`).

Checked against pa-audioic on a stimulus of Cheung et al. (2023): with the same latents and random
numbers, IC and entropy agree to within 0.0003 (r = 1.000000); with the page's own encoder windows
and the same random numbers, IC correlates r = .9998 with pa-audioic's, which is much closer than two
runs with different random numbers (r = .978-.980).

To run it locally: `python3 -m http.server -d web` and open http://localhost:8000/.
To rebuild the ONNX files: `python tools/admonnx.py export --out-dir web/models`.
