class PyfiveArray:
    """A netCDF array accessed with `pyfive`.

    .. versionadded:: 3.20.0

    """

    def __init__(self, *args, **kwargs):
        class DeprecationError(Exception):
            """Deprecation error."""

        raise DeprecationError(
            f"{self.__class__.__name__} was deprecated at version NEXTVERSION "
            "and is no longer available. Use XnetcdfArray instead."
        )
