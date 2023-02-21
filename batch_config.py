import os
import sys
from socket import gethostname


NUM_RUNS = 10  # number of independent simulations

# choose network class in NetworkClass.py

variants = {'NetworkIntDriven': 'IntDrivenLog80', 'NetworkExtDriven': 'ExtDrivenLog100', 'NetworkIndep': 'IndepLog80'}
# variants |= {'PlaceFields': 'HighThreshold'}  # comment this out only when using NetworkIndep


# DEFINE PATHS (change according to your file system!)
# This is where pickles and results will be saved (potentially several GB).
# Currently, the code checks in which machine I am, and whether my external drives are mounted or not.
# You can comment that out and simply define "pickles_path" and "figures_path" to whatever your like.

host_name = gethostname()
drive_names = ["INTENSO", "TOSHIBA"]

if host_name == "lux39":
    drive_parent_path = "/media/eparra"
    local_path = "/local"
elif host_name == "etp":
    drive_parent_path = "/media/eloy"
    local_path = "/c/DATA/Simulations"
else:
    sys.exit("host not recognized")

root_path = local_path
for drive_name in drive_names:
    drive_path = f"{drive_parent_path}/{drive_name}"
    if os.path.exists(drive_path):
        root_path = drive_path
        break

root_path = f"{root_path}/BDSweeps"
pickles_path = f"{root_path}/pickles"  # Path where the pickles will be saved
figures_path = f"{root_path}/figures"  # Path where the figures will be saved
