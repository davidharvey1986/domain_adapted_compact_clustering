"""
ConvNeXtCluster model with clustering capabilities.
"""
from typing import Any

import numpy as np
from netloader import layers
from netloader.models import convnext


class ConvNeXtCluster(convnext.ConvNeXtTiny):
    """
    A ConvNeXtTiny network with clustering capabilities.
    Inherits from ConvNeXtTiny and adds clustering functionality.
    """
    def __init__(
            self,
            in_shape: list[int],
            out_shape: list[int],
            latent_dim: int = 7,
            max_drop_path: float = 0,
            layer_scale: float = 1e-6) -> None:
        """
        Parameters
        ----------
        in_shape : list[int] | list[list[int]] | tuple[int, ...]
            Shape of the input tensor(s), excluding batch size
        out_shape : list[int]
            shape of the output tensor, excluding batch size
        latent_dim : int, default = 7
            Dimensionality of the latent space
        max_drop_path : float, default = 0
            Maximum drop path fraction
        layer_scale : float, default = 1e-6
            Default value for learnable parameters to scale the convolutional filters in a ConvNeXt
            block
        """
        self._latent_dim: int = latent_dim
        super().__init__(
            in_shape,
            out_shape,
            max_drop_path=max_drop_path,
            layer_scale=layer_scale,
        )

    def __getstate__(self) -> dict[str, Any]:
        """
        Returns a dictionary containing the state of the network for pickling

        Returns
        -------
        dict[str, Any]
            Dictionary containing the state of the network
        """
        return super().__getstate__() | {'latent_dim': self._latent_dim}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Sets the state of the network for pickling

        Parameters
        ----------
        state : dict[str, Any]
            Dictionary containing the state of the network
        """
        self._latent_dim = state['latent_dim'] if 'latent_dim' in state else 7
        super().__setstate__(state)

    def _build_net(self, out_shape: list[int]) -> None:
        """
        Constructs the ConvNeXt network layers

        Parameters
        ----------
        out_shape : list[int]
            shape of the output tensor, excluding batch size
        """
        assert isinstance(self.shapes, list)
        assert isinstance(self.check_shapes, list)
        drop_paths: list[float]
        depths: list[int] = np.cumsum(self._depths).tolist()
        kwargs: dict[str, Any] = {'idx': 0, 'check_shapes': []}
        drop_paths = np.linspace(0, self._max_drop_path, depths[-1]).tolist()

        # Stem
        self.net.append(layers.Conv(
            out_shape,
            self.shapes,
            filters=self._dims[0],
            kernel=4,
            stride=4,
            norm='layer',
            **kwargs,
        ))

        # Main body
        for dim, in_depth, out_depth in zip(self._dims, [0, *depths], depths):
            # Downscaling
            self.net.extend([
                layers.LayerNorm(dims=1, shapes=self.shapes, **kwargs),
                layers.Conv(
                    out_shape,
                    self.shapes,
                    filters=dim,
                    kernel=2,
                    stride=2,
                    **kwargs,
                ),
            ] if dim != self._dims[0] else [])

            # ConvNeXt blocks
            self.net.extend([
                *[convnext.ConvNeXtBlock(
                    out_shape,
                    self.shapes,
                    drop_path,
                    layer_scale=self._layer_scale,
                ) for drop_path in drop_paths[in_depth:out_depth]],
            ])

        # Head
        self.net.extend([
            layers.AdaptivePool(1, self.shapes, **kwargs),
            layers.Reshape([-1], shapes=self.shapes, **kwargs),
            layers.LayerNorm(dims=1, shapes=self.shapes, **kwargs),
            layers.Linear(out_shape, self.shapes, features=self._latent_dim, **kwargs),
            layers.OrderedBottleneck(self.shapes, min_size=1, **kwargs),
            layers.Checkpoint(self.shapes, **kwargs),
            layers.Activation('GELU', shapes=self.shapes, **kwargs),
            layers.Linear(out_shape, self.shapes, factor=1, **kwargs),
        ])
