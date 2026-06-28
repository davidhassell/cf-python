import cfdm

from ...mixin_container import Container
from .mixin import ActiveStorageMixin


class XnetcdfArray(
    ActiveStorageMixin,
    Container,
    cfdm.XnetcdfArray,
):
    """A netCDF array accessed with `xnetcdf`.

    .. versionadded:: NEXTVERSION

    """
