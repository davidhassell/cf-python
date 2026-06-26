class ScipyNetcdfFileArray:
    """A netCDF-3 array accessed with `scipy.io.netcdf_file`.

    .. versionadded:: 3.20.0

    """

    def __init__(self, *args, **kwargs):
        class DeprecationError(Exception):
            """Deprecation error."""

        raise DeprecationError(
            f"{self.__class__.__name__} was deprecated at version NEXTVERSION "
            "and is no longer available. Use XnetcdfArray instead."
        )
