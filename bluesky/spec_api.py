"""
Make instances of Scans and give them names matching SPEC.
"""
import bluesky.simple_scans as bss


# Instantiate scans that imitate SPEC API.
ct = count = bss.Count()
ascan = bss.AbsScan()
mesh = bss.OuterProductAbsScan()
a2scan = a3scan = bss.InnerProductAbsScan()
dscan = lup = bss.DeltaScan()
d2scan = d3scan = bss.InnerProductDeltaScan()
th2th = bss.ThetaTwoThetaScan()
hscan = bss.HScan()
kscan = bss.KScan()
lscan = bss.LScan()
tscan = bss.AbsTemperatureScan()
dtscan = bss.DeltaTemperatureScan()
hklscan = bss.InnerProductHKLScan()
hklmesh = bss.OuterProductHKLScan()
