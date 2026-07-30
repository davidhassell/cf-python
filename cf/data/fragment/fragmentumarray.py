class FragmentUMArray:
    """A fragment of aggregated data in a PP or UM file.

    .. versionadded:: 3.14.0

    """

    def __init__(self, *args, **kwargs):
        class DeprecationError(Exception):
            """Deprecation error."""

        raise DeprecationError(
            f"{self.__class__.__name__} was deprecated at version NEXTVERSION "
            "and is no longer available. Use FragmentFileArray instead."
        )
