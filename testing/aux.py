import numpy as np


# Rescale legendre coefficents
def rescale(x0,x1,gx,gw):
  gx = (x1 - x0)/2.0 * gx + (x1 + x0)/2.0
  gw = gw * (x1 - x0) / 2.0
  return gx, gw