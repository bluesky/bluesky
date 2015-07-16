"""
Make instances of Scans and give them names matching SPEC.
"""
import bluesky.simple_scans as bss


# Instantiate scans that imitate SPEC API.
ct = count = bss.Count()
ascan = bss.AbsoluteScan()
mesh = bss.OuterProductAbsoluteScan()
a2scan = a3scan = bss.InnerProductAbsoluteScan()
dscan = lup = bss.DeltaScan()
d2scan = d3scan = bss.InnerProductDeltaScan()
th2th = bss.ThetaTwoThetaScan()
hscan = bss.HScan()
kscan = bss.KScan()
lscan = bss.LScan()
tscan = bss.AbsoluteTemperatureScan()
dtscan = bss.DeltaTemperatureScan()
hklscan = bss.OuterProductHKLScan()
hklmesh = bss.InnerProductHKLScan()
