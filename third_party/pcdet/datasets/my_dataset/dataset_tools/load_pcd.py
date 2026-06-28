#!/usr/bin/env python3
import warnings
import struct
import numpy as np
import re
import lzf

numpy_pcd_type_mappings = [(np.dtype('float32'), ('F', 4)),
                           (np.dtype('float64'), ('F', 8)),
                           (np.dtype('uint8'), ('U', 1)),
                           (np.dtype('uint16'), ('U', 2)),
                           (np.dtype('uint32'), ('U', 4)),
                           (np.dtype('uint64'), ('U', 8)),
                           (np.dtype('int16'), ('I', 2)),
                           (np.dtype('int32'), ('I', 4)),
                           (np.dtype('int64'), ('I', 8))]
numpy_type_to_pcd_type = dict(numpy_pcd_type_mappings)
pcd_type_to_numpy_type = dict((q, p) for (p, q) in numpy_pcd_type_mappings)


def parse_ascii_pc_data(f, dtype, metadata):
    """ Use numpy to parse ascii pointcloud data.
    """
    return np.loadtxt(f, dtype=dtype, delimiter=' ')


def parse_binary_pc_data(f, dtype, metadata):
    rowstep = metadata['points'] * dtype.itemsize
    # for some reason pcl adds empty space at the end of files
    buf = f.read(rowstep)
    return np.frombuffer(buf, dtype=dtype)


def parse_binary_compressed_pc_data(f, dtype, metadata):
    """ Parse lzf-compressed data.
    Format is undocumented but seems to be:
    - compressed size of data (uint32)
    - uncompressed size of data (uint32)
    - compressed data
    - junk
    """
    fmt = 'II'
    compressed_size, uncompressed_size = \
        struct.unpack(fmt, f.read(struct.calcsize(fmt)))
    compressed_data = f.read(compressed_size)
    # TODO what to use as second argument? if buf is None
    # (compressed > uncompressed)
    # should we read buf as raw binary?
    buf = lzf.decompress(compressed_data, uncompressed_size)
    if len(buf) != uncompressed_size:
        raise IOError('Error decompressing data')
    # the data is stored field-by-field
    pc_data = np.zeros(metadata['width'], dtype=dtype)
    ix = 0
    for dti in range(len(dtype)):
        dt = dtype[dti]
        bytes = dt.itemsize * metadata['width']
        column = np.fromstring(buf[ix:(ix + bytes)], dt)
        pc_data[dtype.names[dti]] = column
        ix += bytes
    return pc_data


def parse_header(lines):
    """ Parse header of PCD files.
    """
    metadata = {}
    for ln in lines:
        if ln.startswith('#') or len(ln) < 2:
            continue
        match = re.match('(\w+)\s+([\w\s\.]+)', ln)
        if not match:
            warnings.warn("warning: can't understand line: %s" % ln)
            continue
        key, value = match.group(1).lower(), match.group(2)
        if key == 'version':
            metadata[key] = value
        elif key in ('fields', 'type'):
            metadata[key] = value.split()
        elif key in ('size', 'count'):
            metadata[key] = map(int, value.split())
        elif key in ('width', 'height', 'points'):
            metadata[key] = int(value)
        elif key == 'viewpoint':
            metadata[key] = map(float, value.split())
        elif key == 'data':
            metadata[key] = value.strip().lower()
        # TODO apparently count is not required?
    # add some reasonable defaults
    if 'count' not in metadata:
        metadata['count'] = [1] * len(metadata['fields'])
    if 'viewpoint' not in metadata:
        metadata['viewpoint'] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    if 'version' not in metadata:
        metadata['version'] = '.7'
    return metadata


def _build_dtype(metadata):
    """ Build numpy structured array dtype from pcl metadata.
    Note that fields with count > 1 are 'flattened' by creating multiple
    single-count fields.
    *TODO* allow 'proper' multi-count fields.
    """
    fieldnames = []
    typenames = []
    idx = 0
    for f, c, t, s in zip(metadata['fields'],
                          metadata['count'],
                          metadata['type'],
                          metadata['size']):
        # if f=='_':
        #     continue
        idx += 1
        np_type = pcd_type_to_numpy_type[(t, s)]
        if c == 1:
            fieldnames.append(f)
            typenames.append(np_type)
        else:
            if f == '_':
                fieldnames.extend(['%s%s_%04d_%d' % (f, c, i, idx) for i in range(c)])
                typenames.extend([np_type] * c)
            else:
                fieldnames.extend(['%s%s_%04d' % (f, c, i) for i in range(c)])
                typenames.extend([np_type] * c)
    dtype = np.dtype(list(zip(fieldnames, typenames)))
    return dtype


def points_from_fileobj(f):
    """ Parse pointcloud coming from file object f
    """
    header = []
    while True:
        ln = f.readline().strip()
        ln_str = ln.decode(encoding='utf-8', errors='strict')
        header.append(ln_str)
        if ln_str.startswith('DATA'):
            metadata = parse_header(header)
            dtype = _build_dtype(metadata)
            break
    if metadata['data'] == 'ascii':
        pc_data = parse_ascii_pc_data(f, dtype, metadata)
    elif metadata['data'] == 'binary':
        pc_data = parse_binary_pc_data(f, dtype, metadata)
    elif metadata['data'] == 'binary_compressed':
        pc_data = parse_binary_compressed_pc_data(f, dtype, metadata)
    else:
        raise ValueError('DATA field is neither "ascii" or "binary" or\
                "binary_compressed"')
    return pc_data, metadata


def get_points_from_pcd_file(file_path, num_features=3):
    points, metadata = get_points(file_path)

    if num_features == 3:
        points_ = np.hstack((points['x'].reshape(-1, 1), points['y'].reshape(-1, 1), points['z'].reshape(-1, 1)))
    else:
        points_ = np.hstack((points['x'].reshape(-1, 1), points['y'].reshape(-1, 1), points['z'].reshape(-1, 1)))
        num_features_remain = num_features - 3
        for field in metadata['fields']:
            if num_features_remain <= 0:
                break
            if field not in ['x', 'y', 'z', '__']:
                points_ = np.hstack((points_, points[field].reshape(-1, 1)))
                num_features_remain -= 1
    return points_


def get_points(file_path):
    with open(file_path, 'rb') as f:
        points, metadata = points_from_fileobj(f)
    return points, metadata
