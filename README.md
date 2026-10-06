# MRI Scan Things

## Geometric distortion calibration (Esaote G-scan Brio, 0.25 T)

Interactive results: https://vgroenhuis.github.io/MRI_Scan_Things/

The scanner software corrects geometric distortion only in 2D (within slices), not in 3D.
`calibration_cube.py` measures the 3D distortion from a scan of a calibration cube with
dots on a 20 mm grid and fits 5th order polynomial correction functions.

```
pip install -r requirements.txt
python calibration_cube.py "<folder with DICOM files>" --out docs
```

Outputs:

- `docs/polynomials.json`: the `correction` polynomial maps distorted image coordinates (mm) to true
  coordinates; `forward` maps true coordinates to distorted image coordinates. Both are given in
  DICOM patient coordinates (`lps`: +x left, +y posterior, +z superior) and in `ras`.
  Each coordinate is `sum_k c_k (x/s)^a_k (y/s)^b_k (z/s)^c_k` with `s = 100 mm`.
  The web page has usage examples for Python, MATLAB, JavaScript and C.
- `docs/`: data for the web page (GitHub Pages).

Method: blob segmentation (bars and cubes for the scanner's own calibration are rejected by size),
grid identification growing outwards from the dot nearest the scanner origin, rigid registration
of the ideal grid to the dots within 60 mm of the isocentre, and least-squares polynomial fits.

## AI disclaimer

Claude Code (Opus 5.5 Medium) was used to generate the code and most documentation.
