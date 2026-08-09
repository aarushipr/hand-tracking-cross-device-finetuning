"""
Compatibility shim for reading Quest 3 HOT3D sequences through Meta's
AriaDataProvider / HandBox2dDataProvider (facebookresearch/hot3d,
commit 146b34a as of 2026-08-09).

THE PROBLEM
-----------
AriaDataProvider.__init__ unconditionally calls
`self.get_sequence_timestamps(stream_id, TimeDomain.TIME_CODE)` for every
image stream, to precompute a sorted timestamp list. For Aria recordings
this works. For Quest 3 recordings it raises:

    RuntimeError: Timedomain TimeCode not supported by stream Camera Data (SLAM) #1

This is not a bug in a specific file or a corrupt recording -- verified
against two different Quest sequences (P0013_0ec32d10, P0013_3f269bab),
both fail identically. It's also not a case where the SDK can convert
around it: `vrs_data_provider.convert_from_device_time_to_timecode_ns(ts)`
returns -1 (the SDK's own sentinel for "no TimeCode reference exists") for
real Quest DEVICE_TIME values. Quest recordings in this dataset simply
have no TimeCode track at all -- there's no hardware multi-device sync
signal to convert to, unlike Aria.

WHY THE FIX BELOW IS SAFE, NOT JUST A WORKAROUND
-------------------------------------------------
1. `box2d_hands.csv`'s `timestamp[ns]` column, for Quest sequences, is
   itself in DEVICE_TIME domain. Confirmed by direct comparison, not
   assumed: querying `vp.get_timestamps_ns(stream_id, TimeDomain.DEVICE_TIME)`
   on P0013_0ec32d10's recording.vrs returns 45080933333333 as the first
   timestamp; the exact same value appears as a `timestamp[ns]` entry in
   that sequence's box2d_hands.csv. Not approximately equal -- identical
   to the nanosecond.
2. `HandBox2dDataProvider.get_bbox_at_timestamp`'s `time_domain` parameter
   is used ONLY for a guard clause (`if time_domain is not
   TimeDomain.TIME_CODE: raise ValueError(...)`) -- read the method body
   in hot3d/hot3d/data_loaders/HandBox2dDataProvider.py directly: after
   that check, `time_domain` is never referenced again. The actual lookup
   (`lookup_timestamp(...)`) matches `timestamp_ns` against the CSV's raw
   integer values with no domain-specific conversion. So passing
   TimeDomain.TIME_CODE as the label while the actual value is a
   DEVICE_TIME timestamp does not corrupt the lookup -- the guard is
   satisfied, the numeric match is still correct, because both sides
   (image timestamp and CSV timestamp) are consistently DEVICE_TIME for
   Quest.

So: fall back to DEVICE_TIME when TIME_CODE isn't available, and keep
passing TimeDomain.TIME_CODE downstream to get_bbox_at_timestamp as
before -- don't change that call site. This module only patches the one
place that actually needs different real behavior (AriaDataProvider's
internal timestamp query), not the box2d lookup, which needs no change.

WHAT THIS DOES NOT COVER
-------------------------
This has only been verified for the detection ground truth
(box2d_hands.csv) path. It has NOT been checked for the keypoint /
UmeTrack-format ground truth path -- if that uses its own timestamp
matching against a TIME_CODE assumption, it needs the same verification
before trusting Quest keypoint labels, don't assume this shim covers it.

USAGE
-----
Call `patch()` explicitly, AFTER `hot3d_repo_root` has been added to
sys.path (patch() imports `data_loaders.AriaDataProvider`, which only
exists on sys.path once the hot3d repo checkout has been inserted -- this
module deliberately does NOT patch on import, since the correct sys.path
state depends on the caller's hot3d_repo_root, which isn't known yet at
import time):

    sys.path.insert(0, hot3d_repo_root)
    from py.training.common.hot3d_timecode_compat import patch
    patch()
    from data_loaders.AriaDataProvider import AriaDataProvider  # now safe
"""
import sys


_PATCHED = False


def patch():
    """Idempotent -- safe to call more than once."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    from data_loaders.AriaDataProvider import AriaDataProvider
    from projectaria_tools.core.sensor_data import TimeDomain

    original_get_sequence_timestamps = AriaDataProvider.get_sequence_timestamps

    def get_sequence_timestamps_with_device_time_fallback(self, stream_id, time_domain=TimeDomain.TIME_CODE):
        try:
            return original_get_sequence_timestamps(self, stream_id, time_domain)
        except RuntimeError as e:
            if time_domain == TimeDomain.TIME_CODE and "TimeCode" in str(e):
                print(
                    f"[hot3d_timecode_compat] stream {stream_id} has no TimeCode "
                    f"reference (expected for Quest) -- falling back to DEVICE_TIME. "
                    f"See this module's docstring for why that's safe here.",
                    file=sys.stderr,
                )
                return self._vrs_data_provider.get_timestamps_ns(stream_id, TimeDomain.DEVICE_TIME)
            raise

    AriaDataProvider.get_sequence_timestamps = get_sequence_timestamps_with_device_time_fallback
