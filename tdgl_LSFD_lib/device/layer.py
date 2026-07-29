from typing import Union
import h5py


class Layer:
    """A superconducting thin film.

    Args:
        london_lambda: The London penetration depth of the film.
        coherence_length: The superconducting coherence length of the film.
        thickness: The thickness of the film.
        conductivity: The normal state conductivity of the superconductor in Siemens / length_unit.
    """
    def __init__(
        self,
        *,
        coherence_length: float,
        london_lambda: Union[float, None] = None,
        thickness: Union[float, None] = None,
        conductivity: Union[float, None] = None,
    ):
        self.london_lambda = london_lambda
        self.coherence_length = coherence_length
        self.thickness = thickness
        self.conductivity = conductivity

    def copy(self) -> "Layer":
        """Create a deep copy of the :class:`tdgl.Layer`."""
        return Layer(
            london_lambda=self.london_lambda,
            coherence_length=self.coherence_length,
            thickness=self.thickness,
            conductivity=self.conductivity,
        )

    def to_hdf5(self, h5_group: h5py.Group) -> None:
        """Save the :class:`tdgl.Layer` to an :class:`h5py.Group`.

        Args:
            h5_group: An open :class:`h5py.Group` to which to save the layer.
        """
        h5_group.attrs["coherence_length"] = self.coherence_length
        if self.thickness is not None:
            h5_group.attrs["london_lambda"] = self.london_lambda
        if self.thickness is not None:
            h5_group.attrs["thickness"] = self.thickness
        if self.conductivity is not None:
            h5_group.attrs["conductivity"] = self.conductivity

    @staticmethod
    def from_hdf5(h5_group: h5py.Group) -> "Layer":
        """Load a :class:`tdgl.Layer` from an :class:`h5py.Group`.

        Args:
            h5_group: An open :class:`h5py.Group` from which to load the layer.

        Returns:
            A new :class:`tdgl.Layer` instance.
        """

        def get(key, default=None):
            if key in h5_group.attrs:
                return h5_group.attrs[key]
            return default

        return Layer(
            london_lambda=get("london_lambda"),
            coherence_length=get("coherence_length"),
            thickness=get("thickness"),
            conductivity=get("conductivity"),
        )

    def __eq__(self, other):
        if self is other:
            return True

        if not isinstance(other, Layer):
            return False

        return (
            self.london_lambda == other.london_lambda
            and self.coherence_length == other.coherence_length
            and self.thickness == other.thickness
            and self.conductivity == other.conductivity
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"london_lambda={self.london_lambda}, "
            f"coherence_length={self.coherence_length}, "
            f"thickness={self.thickness}, "
            f"conductivity={self.conductivity}, "
            f")"
        )
