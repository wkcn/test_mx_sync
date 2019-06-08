import ctypes
import mxnet as mx
from mxnet.base import _LIB

def get_pointer(v):
    cp = ctypes.c_void_p()
    _LIB.MXNDArrayGetData(v.handle, ctypes.byref(cp))
    return cp


a = mx.nd.array([1,2,3,4,5])
a_data = get_pointer(a)
lib = ctypes.CDLL('./lib.so')
lib.SetMXEnginePushSyncND.argtypes = [ctypes.c_void_p]
lib.SetMXEnginePushSyncND(_LIB.MXEnginePushSyncND)
lib.SetMXEnginePushAsyncND.argtypes = [ctypes.c_void_p]
lib.SetMXEnginePushAsyncND(_LIB.MXEnginePushAsyncND)

lib.AddOne.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.AddOneAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

print(a)
lib.AddOne(a.handle, a_data, a.size)
mx.nd.waitall()
print(a)
lib.AddOneAsync(a.handle, a_data, a.size)
mx.nd.waitall()
print(a)
