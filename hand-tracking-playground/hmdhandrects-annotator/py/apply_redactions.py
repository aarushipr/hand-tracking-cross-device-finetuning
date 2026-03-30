import os
import json
import argparse
import cv2



parser = argparse.ArgumentParser()
parser.add_argument('--euroc_path', help='the path to the EuRoC dataset')
parser.add_argument('--dry_run', help="Don't actually overwrite the files")
args = parser.parse_args()

root = args.euroc_path
dry_run = args.dry_run
ann = "redactions.json"

with open(os.path.join(root, ann)) as f:
    j_file = json.load(f)

# For every frame in the array called "frames" in redactions.json...
for idx, j in enumerate(j_file["frames"]):
    # For the left and right stereo view...
    for idx, side_name in enumerate(["left", "right"]):
        side = j[side_name]
        redactions = side["redactions"]
        # If this image has redactions associated with it, let's apply them.
        if len(redactions) != 0:
            # Load the un-redacted image
            p = os.path.join(root, side["filename"])
            asdf = cv2.imread(p)

            # For every bounding box we want to redact:
            for redaction in redactions:
                # Figure out the rect we want to put a black box over
                cx = redaction[0]
                cy = redaction[1]
                w = redaction[2]
                h = redaction[3]

                top = int(max(0, cy - (h / 2)))
                bottom = int(min(asdf.shape[0]-1, cy + (h / 2)))
                left = int(max(0, cx - (w/2)))
                right = int(min(asdf.shape[1]-1, cx + (w/2)))

                # Put the black box there
                asdf[top:bottom, left:right] = 0

                # If we're not a dry run, overwrite the original image
                if not dry_run:
                  print("Saving!")
                  cv2.imwrite(p, asdf)

            # Show the image so we can see what's happening
            cv2.imshow(side_name, asdf)
            cv2.waitKey(1)