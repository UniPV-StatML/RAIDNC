import orfipy_core
from Bio.Seq import Seq
fwd = "CC" + "ATG" + "GCT" * 60 + "TAA" + "GG"
rc = str(Seq(fwd).reverse_complement())
print("fwd:", [(a, b, s) for a, b, s, _ in orfipy_core.orfs(fwd, minlen=100, starts=['ATG'], stops=['TAA', 'TAG', 'TGA'])])
print("rc :", [(a, b, s) for a, b, s, _ in orfipy_core.orfs(rc, minlen=100, starts=['ATG'], stops=['TAA', 'TAG', 'TGA'])])
a, b, s, _ = orfipy_core.orfs(rc, minlen=100, starts=['ATG'], stops=['TAA', 'TAG', 'TGA'])[0]
print("slice rc[a:b] starts with:", rc[a:b][:9], "ends with:", rc[a:b][-6:])
