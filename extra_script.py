import os

# Gunakan jumlah core CPU untuk paralel build
Import("env")
env.Append(CPPDEFINES=[("CORE_DEBUG_LEVEL", 0)])
env.Execute("pio run -j {}".format(os.cpu_count()))
