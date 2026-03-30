

import os

for guy in sorted(os.listdir("/home/moses/EPIC-KITCHENS")):
    print(guy, os.listdir(os.path.join("/home/moses/EPIC-KITCHENS/", guy)))
    os.system(
        f"rm -r {os.path.join('/home/moses/EPIC-KITCHENS', guy, 'videos')}")
