import cftime
import xarray as xr
import pandas as pd
import rioxarray


def add_model_and_scenario_encoding_dicts(ds, models, scenarios):
    """
    Add encoding dictionaries to the dataset attributes for model and scenario.
    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to add the encoding dictionaries to.
    models : dict
        A dictionary of models and their corresponding encoding values (usually from the luts.py file).
    scenarios : dict
        A dictionary of scenarios and their corresponding encoding values (usually from the luts.py file).
    Returns
    -------
    ds : xarray.Dataset
        The dataset with the encoding dictionaries added to the attributes.
    """
    # reverse the model and scenario dictionaries
    model_dict = {v: k for k, v in models.items()}
    scenario_dict = {v: k for k, v in scenarios.items()}
    # then drop any keys that are not in the dataset
    model_dict = {k: v for k, v in model_dict.items() if k in ds["model"].values}
    scenario_dict = {
        k: v for k, v in scenario_dict.items() if k in ds["scenario"].values
    }
    # then add the encoding dictionaries, units, and name to the attributes
    ds["model"].attrs["long_name"] = "model"
    ds["model"].attrs["units"] = " "
    ds["model"].attrs["encoding"] = str(model_dict)

    ds["scenario"].attrs["long_name"] = "scenario"
    ds["scenario"].attrs["units"] = " "
    ds["scenario"].attrs["encoding"] = str(scenario_dict)
    return ds


def map_model_and_scenario_to_int(ds, models, scenarios):
    """
    Map the model and scenario to integers using the lookup dictionaries.
    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to map the model and scenario to integers.
    models : dict
        A dictionary of models and their corresponding encoding values (usually from the luts.py file).
    scenarios : dict
        A dictionary of scenarios and their corresponding encoding values (usually from the luts.py file).
    Returns
    -------
    ds : xarray.Dataset
        The dataset with the model and scenario mapped to integers.
    """
    s_ints = [scenarios[s] for s in ds.scenario.values if s in scenarios]
    ds = ds.assign_coords({"scenario": ("scenario", s_ints)})

    m_ints = [models[m] for m in ds.model.values if m in models]
    ds = ds.assign_coords({"model": ("model", m_ints)})
    return ds


def refresh_lat_lon_metadata(ds):
    # wipe any existing attributes
    ds["lat"].attrs = {}
    ds["lon"].attrs = {}

    # update the coordinate metadata
    # include min_value and max_value attributes derived from the data
    # this is distinct from the valid_min and valid_max attributes,
    # which are the full range of values allowed by the datatype

    ds["lat"].attrs.update(
        {
            "standard_name": "latitude",
            "long_name": "latitude",
            "units": "degrees_north",
            "axis": "Y",
            "min_value": ds["lat"].min().values,
            "max_value": ds["lat"].max().values,
        }
    )
    ds["lon"].attrs.update(
        {
            "standard_name": "longitude",
            "long_name": "longitude",
            "units": "degrees_east",
            "axis": "X",
            "min_value": ds["lon"].min().values,
            "max_value": ds["lon"].max().values,
        }
    )

    return ds


def refresh_variable_metadata_and_enforce_dtype(ds, var_info):

    for var in ds.data_vars:
        # check if the variable is in the var_info dictionary
        if var in var_info:
            # wipe the existing attributes
            ds[var].attrs = {}
            # update the variable attributes with the metadata
            ds[var].attrs.update(var_info[var])
            # enforce the data type defined in the metadata
            ds[var] = ds[var].astype(var_info[var]["data_type"])

    return ds


def transpose_dims(ds):
    # transpose all vars = to have the order: model, scenario, time, lat, lon
    ds = ds.transpose("model", "scenario", "time", "lat", "lon")
    return ds


def add_global_attrs(ds, global_attributes):
    # wipe any existing attributes
    ds.attrs = {}
    ds.attrs.update(global_attributes["attributes"])
    return ds


def add_crs(ds, crs):
    ds = ds.rio.set_spatial_dims("lon", "lat")
    ds = ds.rio.write_crs(crs)  # this creates the "spatial_ref" coordinate
    return ds
