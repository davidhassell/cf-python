class H5netcdfArray:
    """A netCDF array accessed with `h5netcdf` using the `h5py` backend.

    .. versionadded:: 3.16.3

    """

    def __init__(self, *args, **kwargs):
        class DeprecationError(Exception):
            """Deprecation error."""

        raise DeprecationError(
            f"{self.__class__.__name__} was deprecated at version NEXTVERSION "
            "and is no longer available. Use XnetcdfArray instead."
        )
