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
before -- don't change that call site. This module patches the two
places that actually need different real behavior:

- AriaDataProvider.get_sequence_timestamps (used internally by __init__
  to precompute the per-stream timestamp list)
- AriaDataProvider.get_image (used per-sample to actually fetch pixel
  data) -- unlike get_sequence_timestamps, this one hardcodes
  TimeDomain.TIME_CODE directly into the call to
  vrs_data_provider.get_image_data_by_time_ns with no parameter to
  override it, so there's no way to influence it from the caller side;
  it has to be patched. Confirmed this is a REAL domain-aware lookup, not
  a decorative check like HandBox2dDataProvider's guard -- it raises
  RuntimeError rather than silently returning a mismatched frame if given
  the wrong domain, so the fallback path here is a genuine correct-domain
  retry (the timestamp really is DEVICE_TIME for Quest, from the
  get_sequence_timestamps fallback above), not another "satisfy the
  label" trick.

The box2d lookup (get_bbox_at_timestamp) needs no patch at all -- see
point 2 above.

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

    def _no_timecode_streams(provider):
        """
        Per-provider record of which streams have already been found to carry
        no TimeCode track.

        Whether a recording has a TimeCode track is a property of the file, so
        it needs discovering exactly once per stream, not once per read. Before
        this was cached, every image read on a Quest sequence raised, logged and
        retried -- roughly 300k times per training epoch, which buried the real
        log output and paid for an exception on every sample.
        """
        cache = getattr(provider, "_hot3d_no_timecode_streams", None)
        if cache is None:
            cache = set()
            provider._hot3d_no_timecode_streams = cache
        return cache

    original_get_sequence_timestamps = AriaDataProvider.get_sequence_timestamps

    def get_sequence_timestamps_with_device_time_fallback(self, stream_id, time_domain=TimeDomain.TIME_CODE):
        known_missing = _no_timecode_streams(self)
        if time_domain == TimeDomain.TIME_CODE and str(stream_id) in known_missing:
            return self._vrs_data_provider.get_timestamps_ns(stream_id, TimeDomain.DEVICE_TIME)
        try:
            return original_get_sequence_timestamps(self, stream_id, time_domain)
        except RuntimeError as e:
            if time_domain == TimeDomain.TIME_CODE and "TimeCode" in str(e):
                if str(stream_id) not in known_missing:
                    known_missing.add(str(stream_id))
                    print(
                        f"[hot3d_timecode_compat] stream {stream_id} has no TimeCode "
                        f"reference (expected for Quest) -- using DEVICE_TIME for this "
                        f"recording. Logged once per stream. See this module's "
                        f"docstring for why that's safe here.",
                        file=sys.stderr,
                    )
                return self._vrs_data_provider.get_timestamps_ns(stream_id, TimeDomain.DEVICE_TIME)
            raise

    AriaDataProvider.get_sequence_timestamps = get_sequence_timestamps_with_device_time_fallback

    from projectaria_tools.core.sensor_data import TimeQueryOptions

    def get_image_with_device_time_fallback(self, timestamp_ns, stream_id):
        known_missing = _no_timecode_streams(self)

        # Once this stream is known to have no TimeCode track, go straight to
        # DEVICE_TIME. Retrying TIME_CODE first would raise on every sample --
        # per-sample exception handling and per-sample logging, for an answer
        # that cannot change within a recording.
        if str(stream_id) in known_missing:
            image = self._vrs_data_provider.get_image_data_by_time_ns(
                stream_id, timestamp_ns, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST,
            )
            return image[0].to_numpy_array() if image is not None else None

        try:
            image = self._vrs_data_provider.get_image_data_by_time_ns(
                stream_id, timestamp_ns, TimeDomain.TIME_CODE, TimeQueryOptions.CLOSEST,
            )
        except RuntimeError as e:
            if "TimeCode" in str(e):
                known_missing.add(str(stream_id))
                print(
                    f"[hot3d_timecode_compat] stream {stream_id} image lookup has no "
                    f"TimeCode reference (expected for Quest) -- using DEVICE_TIME for "
                    f"this recording. Logged once per stream. See this module's "
                    f"docstring for why that's the correct domain here, not just a "
                    f"fallback.",
                    file=sys.stderr,
                )
                image = self._vrs_data_provider.get_image_data_by_time_ns(
                    stream_id, timestamp_ns, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST,
                )
            else:
                raise
        return image[0].to_numpy_array() if image is not None else None

    AriaDataProvider.get_image = get_image_with_device_time_fallback
