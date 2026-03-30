This is not a definitive "Here is the answer, I am just waiting to have time to implement it" document. I don't know what's the right thing to do but I'm getting some ideas out.

- Big annoying one: hand going outside of the image edge, getting re-detected juust inside the image edge, going back outside.
  - For this case, remember the ROI at which the hand went untracked, and refuse to re-track it if the new ROI overlaps too much.

- I have had ideas about making a "hand_score"
  - dropping *tracking* when it goes below a certain value
  - also keep an euro filtered version of this, and have a higher value that the euro-filtered version has to hit to have is_active set in OpenXR
  - The first part is reasonableish but the second part wouldn't help for debouncing (I don't think?)
  - Well maybe we should have a (for euro filtering) higher "start_track" value and lower "stop_track" value!
  - Okay this makes sense now actually
  - What if we do something 

```cpp
enum HAND_UNTRACKED_REASON
{
    KEYPOINT_ESTIMATOR_SAID_NO,
    KEYPOINT_OUTPUTS_TOO_DARK,
    REPROJECTION_ERROR_TOO_HIGH,
    OUTSIDE_IMAGE_EDGE
};
```

Well I don't like this because it doesn't let us do scoring, which I still suspeeect would help? Like, a bad hand will have both a high not-hand score, dark keypoint outputs, and high reprojection error. Each individual factor doesn't tell the same story.

So maybe: Refuse to re-track a hand in the same spot for 0.5 seconds after it went untracked.

But what about overlapping hands?
...Right, overlapping hands going jittery wasn't really a problem because we started suppressing double detections. Maybe refusing to re-track in a similar ROI is a good move across the board?


- In general I'm kind of inspired by Mateo's
  - Autoexposure state machine
  - JSON builder state machine
I can be better at writing state machines!
