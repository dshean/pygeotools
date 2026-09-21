"""Tests for warplib output grid when res is specified by a dataset"""
import numpy as np
import pytest
from osgeo import gdal, osr

from pygeotools.lib import warplib

def make_ds(nl, ns, xres, yres, ulx=500000., uly=4100000., epsg=32610):
    ds = gdal.GetDriverByName('MEM').Create('', ns, nl, 1, gdal.GDT_Float32)
    ds.SetGeoTransform([ulx, xres, 0, uly, 0, -yres])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds.SetProjection(srs.ExportToWkt())
    y, x = np.mgrid[0:nl, 0:ns]
    a = (100 + 50*np.sin(x*xres/300.) + 50*np.cos(y*yres/200.)).astype(np.float32)
    b = ds.GetRasterBand(1)
    b.SetNoDataValue(-9999)
    b.WriteArray(a)
    return ds

def grid(ds):
    return (ds.RasterYSize, ds.RasterXSize), ds.GetGeoTransform()

def test_res_extent_dataset_nonsquare():
    #Same extent, different grids: 19 x 21 m and 20 x 20 m
    dem = make_ds(200, 300, 19., 21.)
    other = make_ds(210, 285, 20., 20.)
    out = warplib.memwarp_multi([other,], res=dem, extent=dem, t_srs=dem, verbose=False)[0]
    assert grid(out) == grid(dem)

def test_res_extent_dataset_square():
    dem = make_ds(210, 285, 20., 20.)
    other = make_ds(200, 300, 19., 21.)
    out = warplib.memwarp_multi([other,], res=dem, extent=dem, t_srs=dem, verbose=False)[0]
    assert grid(out) == grid(dem)

def test_nonsquare_not_passed_through_unwarped():
    #Mean res of 19 x 21 m pixels is 20 m, but this must still be warped to a 20 m grid
    nonsquare = make_ds(200, 300, 19., 21.)
    square = make_ds(210, 285, 20., 20.)
    out = warplib.memwarp_multi([nonsquare, square], res=20., extent='intersection', verbose=False)
    assert grid(out[0])[0] == grid(out[1])[0]
    assert out[0].GetGeoTransform()[1] == 20. and out[0].GetGeoTransform()[5] == -20.

def test_res_first_nonsquare_common_grid():
    nonsquare = make_ds(200, 300, 19., 21.)
    square = make_ds(210, 285, 20., 20.)
    out = warplib.memwarp_multi([nonsquare, square], verbose=False)
    assert grid(out[0]) == grid(out[1]) == grid(nonsquare)

@pytest.mark.parametrize('res', [30., 'min', 'max', 'mean'])
def test_scalar_res_common_grid(res):
    a = make_ds(210, 285, 20., 20.)
    b = make_ds(140, 190, 30., 30.)
    out = warplib.memwarp_multi([a, b], res=res, verbose=False)
    assert grid(out[0]) == grid(out[1])
    gt = out[0].GetGeoTransform()
    assert gt[1] == -gt[5]

@pytest.mark.parametrize('yres', [np.nextafter(np.nextafter(20., 21.), 21.), 20.0001])
def test_nearsquare_uses_mean_res(yres):
    #Pixels that are square to within warp_multi precision keep the single mean res
    a = make_ds(210, 285, 20., yres)
    b = make_ds(140, 190, 30., 30.)
    out = warplib.memwarp_multi([a, b], res='first', extent='first', verbose=False)
    gt = out[1].GetGeoTransform()
    assert gt[1] == -gt[5] == np.mean([20., yres])
