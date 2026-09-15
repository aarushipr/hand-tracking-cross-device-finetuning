"""
Lets Quest 3 HOT3D sequences be read through Meta's AriaDataProvider, which asks
every stream for TIME_CODE timestamps that Quest recordings do not have. Falling
back to DEVICE_TIME is correct rather than a workaround: box2d_hands.csv's
timestamps are themselves DEVICE_TIME for Quest, confirmed to the nanosecond. Call
patch() after hot3d_repo_root is on sys.path. Verified for the detection path only.
"""
import sys


_PATCHED = False


def patch():
    """Idempotent, safe to call more than once."""
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
        retried, roughly 300k times per training epoch, which buried the real
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

        # Go straight to DEVICE_TIME once known: retrying would raise on every sample.
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
