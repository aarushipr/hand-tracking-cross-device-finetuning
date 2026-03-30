Forearm stuff

- Get forearm data in more than free synthetic data
- Turns out chasing down forearm *direction relative to the wrist* was not the way to go. Wrist annotations in our real data are all over the place (hey, I didn't annotate that stuff ;) I just scraped it) so the wrist moves a few cm as I roll my hand around. Wasn't so obvious with just 26 keypoints, but it translates to a *lot* of wrongness when viewed down the whole forearm.
    - Also, having good residuals for smoothness here is annoying when one of your optimized keypoints doesn't correspond to an observed keypoint. If you do it wrong, the optimizer will have the forearm "drag behind" the wrist instead of doing what it's supposed to do. I did figure out a way around this, but it'd just be easier to do keypoints everything.
- The wrist bone doesn't actually exist. It's actually two layers of carpal bones. The back row rotates 40ish degrees on the X and Y axes, and the front row rotates another 40ish degrees only on the X axis to add the extra up/down range of motion. This video shows how weird the articulation is, and shows that it can't correspond to simple "parenting" kinematics that we'd want.

This whole thing makes me want to annotate some real data where I instead annotate *three* keypoints to replace wrist/forearm:

    - elbow_or_forearm_at_image_edge (self-explanatory - either the visual midpoint of the forearm at the elbow, or the visual midpoint of the forearm at the image edge, depending on if the elbow is visible.
    - distal_end_of_forearm (visual midpoint of the forearm at where it meets the carpal bones. Reasonably close approximation to the old "wrist" joint.
    - estimated_center_of_distal_carpals (Annotator's best guess as to the center of the *distal* row of carpals. Also reasonable approximation to the old "wrist" joint!)

It also really makes me want to write some visualization code that mimics how the carpals *actually* articulate with reasonable accuracy, maybe port that to our optimizer, and definitely try porting it to our Blender rigging - I very prominently had trouble with the wrist joint because... it's actually not just one joint.
