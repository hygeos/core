#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Block processing framework for xarray datasets with lazy evaluation support.

This module provides a framework for creating chainable block processors that can 
work efficiently with xarray's chunked arrays and xarray's map_blocks functionality.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import xarray as xr

from core.tools import Var
from core.tools import raiseflag


class BlockProcessor(ABC):
    """
    Abstract base class for block processing operations. Processors can be chained to
    minimize dask graph management overhead. Processors are intended to be executed
    by `map_blockwise`.

    Each processor defines:
    - process_block: Method that performs the actual processing on a data block, by
      adding or modifying Dataset variables in place.
    - input_vars: List of variable names required as input (optional)
    - modified_vars: List of variable names that will be modified (optional)
    - created_vars: List of Var objects describing newly created variables (optional)
    - created_dims: Dictionary of new dimensions to be created by this processor (optional)
    - global_attrs: Dictionary of the global attributes set by this processor (optional)

    Examples
    --------
    >>> class AddProcessor(BlockProcessor):
    ...     def input_vars(self):
    ...         return ['a', 'b']
    ...
    ...     def created_vars(self):
    ...         return [Var('sum', 'float64', ('x', 'y'))]
    ...
    ...     def process_block(self, block, **kwargs):
    ...         block['sum'] = block['a'] + block['b']
    ...
    >>> processors = [AddProcessor()]
    >>> result = map_blockwise(ds, processors)
    """

    def input_vars(self) -> List[str]:
        """Return list of input variable names required by this processor."""
        return []

    def modified_vars(self) -> List[Var]:
        """
        Return list of variables that will be modified in-place.
        """
        return []

    def created_vars(self) -> List[Var]:
        """Return list of Var objects describing newly created variables."""
        return []

    def created_dims(self) -> Dict[str, Any]:
        """
        Return dictionary of new dimensions to be created by this processor.

        Each dimension is defined either by:
        - An int: specifying the dimension size
        - An array-like: providing coordinates for the dimension

        Returns
        -------
        Dict[str, int or array-like]
            Dictionary mapping dimension names to their size (int) or coordinates (array-like)

        Examples
        --------
        >>> def created_dims(self):
        ...     return {'z': 10, 'wavelength': [400, 500, 600, 700]}
        """
        return {}

    def global_attrs(self) -> Dict[str, Any]:
        """
        Return dictionary of global attributes to be set by this processor.
        For variable attributes, use the `attrs` argument of the created/modified variables.
        """
        return {}

    def raiseflag(
        self, block: xr.Dataset, var_name: str, flag_name: str, condition: xr.DataArray
    ):
        """
        Apply a flag to a variable in the block where condition is True.

        This is a convenience wrapper around core.tools.raiseflag that automatically
        looks up the flag bit position from the processor's flag definitions.

        Parameters
        ----------
        block : xr.Dataset
            Dataset block containing the flag variable
        var_name : str
            Name of the flag variable (e.g., 'flags', 'quality_flags')
        flag_name : str
            Name of the flag to raise (e.g., 'CLOUDY', 'BAD_DATA')
        condition : xr.DataArray
            Boolean array indicating where to raise the flag

        Raises
        ------
        ValueError
            If the variable has no flags defined or the flag name is not defined

        Examples
        --------
        >>> def process_block(self, block, **kwargs):
        ...     cloud_mask = block['reflectance'] > 0.8
        ...     self.raiseflag(block, 'flags', 'CLOUDY', cloud_mask)
        """

        # Get flag mappings (cached after first call)
        flag_mappings = self.get_flag_mappings()

        if var_name not in flag_mappings:
            raise ValueError(
                f"No flags defined for variable '{var_name}'. "
                f"Define flags in modified_vars() or created_vars()."
            )
        if flag_name not in flag_mappings[var_name]:
            available = list(flag_mappings[var_name].keys())
            raise ValueError(
                f"Flag '{flag_name}' not defined for variable '{var_name}'. "
                f"Available flags: {available}"
            )

        flag_value = flag_mappings[var_name][flag_name]
        raiseflag(block[var_name], flag_name, flag_value, condition)

    def get_flag_mappings(self) -> Dict[str, Dict[str, int]]:
        """Extract flag bit mappings from modified_vars() and created_vars(). Results are cached."""
        if not hasattr(self, "_flag_mappings"):
            flag_mappings = {}
            for var in self.modified_vars() + self.created_vars():
                if hasattr(var, "flags") and var.flags:
                    flag_mappings[var.name] = var.flags
            self._flag_mappings = flag_mappings
        return self._flag_mappings

    @abstractmethod
    def process_block(self, block: xr.Dataset, **kwargs):
        """
        Process a single block of data.

        Parameters
        ----------
        block : xr.Dataset
            Input data block containing the required input variables. This block is
            modified in place.
        **kwargs : dict
            Additional parameters for processing (e.g., algorithm parameters)
        """
        pass

    def describe(self) -> Dict[str, Any]:
        """Return a description of this processor for logging/debugging."""
        return {
            "name": self.__class__.__name__,
            "inputs": self.input_vars(),
            "creates": [
                {"name": v.name, "dtype": v.dtype, "dims": v.dims}
                for v in self.created_vars()
            ],
            "modifies": self.modified_vars,
        }


def process_single_block(
    ds: xr.Dataset, processors: list[BlockProcessor], kwargs: dict
) -> xr.Dataset:
    """
    Apply a sequence of processors to a single data block.
    
    This function is designed to be used with xr.map_blocks for chunk-wise
    processing. It applies each processor in sequence to the input dataset,
    modifying it in place.
    
    Parameters
    ----------
    ds : xr.Dataset
        Input dataset block to process
    processors : list[BlockProcessor]
        List of processors to apply sequentially
    kwargs : dict
        Additional keyword arguments passed to each processor's process_block method
        
    Returns
    -------
    xr.Dataset
        Processed dataset with modifications from all processors applied
    """
    for p in processors:
        p.process_block(ds, **kwargs)
        
        # Validate that all advertised created variables exist with correct specs
        for var in p.created_vars():
            if var.name not in ds.data_vars:
                raise ValueError(
                    f"Processor {p.__class__.__name__} failed to create variable '{var.name}'"
                )
            
            actual_var = ds[var.name]
            if actual_var.dtype != var.dtype:
                raise ValueError(
                    f"Processor {p.__class__.__name__} created variable '{var.name}' "
                    f"with dtype {actual_var.dtype}, expected {var.dtype}"
                )
            
            var_dims = var.getdims(ds)
            if actual_var.dims != var_dims:
                raise ValueError(
                    f"Processor {p.__class__.__name__} created variable '{var.name}' "
                    f"with dims {actual_var.dims}, expected {var_dims}"
                )
    
    return ds
    

def map_blockwise(
    ds: xr.Dataset, 
    processors: List[BlockProcessor],
    **kwargs
) -> xr.Dataset:
    """
    Apply a chain of processors to a dataset, using xr.map_blocks
    
    This function validates all processors and applies them sequentially
    while preserving chunking structure when possible.
    
    Parameters
    ----------
    ds : xr.Dataset
        Input dataset to process
    processors : List[BlockProcessor]
        List of processors to apply sequentially
    **kwargs : dict
        Additional keyword arguments passed to all processors
        
    Returns
    -------
    xr.Dataset
        Processed dataset with new/modified variables
    """

    # Validate all processors inputs
    current_vars = set(ds.data_vars.keys())
    for i, processor in enumerate(processors):
        # Check inputs are available
        modified_var_names = [v.name for v in processor.modified_vars()]
        missing_vars = [
            var
            for var in processor.input_vars() + modified_var_names
            if var not in current_vars
        ]
        if missing_vars:
            raise ValueError(
                f"Processor {i} ({processor.__class__.__name__}) requires "
                f"variables {missing_vars} which are not available. "
                f"Current variables: {list(current_vars)}"
            )
        
        # Add created variables to available vars for next processor
        for var_def in processor.created_vars():
            current_vars.add(var_def.name)

    # Create a subset of input, with only the relevant variables, and apply
    # xr.map_blocks
    sub = _subset(ds, processors)
    template = _create_template(sub, processors)
    
    # Ensure coordinates from the template are added to the subset
    # This is necessary because xr.map_blocks doesn't preserve coordinates
    # from the template when they're not present in the input dataset
    for coord_name in template.coords:
        if coord_name not in sub.coords:
            sub = sub.assign_coords({coord_name: template.coords[coord_name]})
    
    result = xr.map_blocks(
        process_single_block,
        sub,
        template=template,
        kwargs={"processors": processors, "kwargs": kwargs},
    )
    return result


def _subset(ds: xr.Dataset, processors: List[BlockProcessor]) -> xr.Dataset:
    """
    Extract only the variables needed by the processors.
    
    Returns a dataset containing only variables that are either inputs
    or will be modified by any of the processors.
    
    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    processors : List[BlockProcessor]
        List of processors to determine needed variables
        
    Returns
    -------
    xr.Dataset
        Subset of input dataset with only needed variables
    """
    input_vars = set()
    
    for processor in processors:
        input_vars.update(processor.input_vars())
        modified_var_names = [v.name for v in processor.modified_vars()]
        input_vars.update(modified_var_names)
    
    # Keep only the needed variables that actually exist in the dataset
    return ds[list(input_vars & set(ds.data_vars.keys()))]


def _create_template(ds: xr.Dataset, processors: List[BlockProcessor]) -> xr.Dataset:
    """
    Create output template for xr.map_blocks.

    This function generates a template dataset that defines the structure of the output
    from applying the given processors. It starts with a shallow copy of the input dataset
    and adds new variables created by each processor, along with their associated flags.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset to base the template on. This provides the existing variables
        and dimensions that will be preserved in the output.
    processors : List[BlockProcessor]
        List of processors that will create new variables. Each processor's created_vars()
        and flag mappings are used to populate the template.

    Returns
    -------
    xr.Dataset
        A template dataset containing all original variables plus newly created variables
        with their proper dimensions, dtypes, and flag definitions. This template is used
        by xr.map_blocks to determine the output structure.
    """
    template_ds = ds.copy(deep=False)

    # Add/overwrite with created_vars from each processor
    for processor in processors:
        flags = processor.get_flag_mappings()
        for var in processor.created_vars():
            assert var.name not in template_ds
            template_ds[var.name] = var.to_dataarray(
                template_ds, new_dims=processor.created_dims()
            )
        
        # Check that modified variables do not define dims/dtype
        for var in processor.modified_vars():
            assert var.dims is None
            assert var.dtype is None
        
        # Register flags in the template
        for varname, flags_mapping in flags.items():
            for flag_name, flag_value in flags_mapping.items():
                raiseflag(template_ds[varname], flag_name, flag_value)

        # Update global attributes
        template_ds.attrs.update(processor.global_attrs())
    
    return template_ds

