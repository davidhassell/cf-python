import cfdm

from ...mixin_container import Container
from ..array.mixin import ActiveStorageMixin


class FragmentFileArray(
    ActiveStorageMixin, Container, cfdm.data.fragment.FragmentFileArray
):
    """Fragment of aggregated data in a file.

    .. versionadded:: 3.17.0

    """
