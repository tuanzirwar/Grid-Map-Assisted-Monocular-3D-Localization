# Distortion parameters for the 3 lenses
# [k1, k2, k3, p1, p2]
ulter_wide = [-0.010966, 0.0013977, -4.17E-05, 7.51E-06, -5.90E-10]
wide = [0.0016357, 0.000064971, -1.80E-06, -1.94E-05, -9.21E-10]
tele = [-0.0025488, 0.00010349, -9.92E-07, -2.08E-05, -5.40E-11]
DISTORTION_MAP = {
    "ultra-wide": ulter_wide,
    "wide": wide,
    "tele": tele
}
