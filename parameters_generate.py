# network_generate.py
# Alessio Burrello <alessio.burrello@unibo.it>
# 
# Copyright (C) 2019-2020 University of Bologna
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import torch
import torch.nn.functional as F
from Ne16 import *

def license(filename):
    return \
"""/*
 * {filename}
 * Alessio Burrello <alessio.burrello@unibo.it>
 * Luka Macan <luka.macan@unibo.it>
 *
 * Copyright (C) 2019-2020 University of Bologna
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 */

""".format(filename=filename)

def header_guard_begin(filename):
    guard = filename.replace('.', '_')
    return \
"""#ifndef __{GUARD}__
#define __{GUARD}__

""".format(GUARD=guard.upper())

def header_guard_end(filename):
    guard = filename.replace('.', '_')
    return "#endif  // __{GUARD}__\n".format(GUARD=guard.upper())

def includes():
    return "#include <pmsis.h>\n\n"

def define(name, value):
    return '#define {name} ({value})\n'.format(name=name, value=value)

def vector_end():
    return ';\n\n'

def vector_definition(name, size):
    retval = ""
    retval += define(f'{name.upper()}_SIZE', size)
    retval += f"PI_L1 uint8_t {name}[{name.upper()}_SIZE]"
    return retval

def vector_initial_value(data, elements_per_row=10, spaces=4):
    indent = ' ' * spaces
    size = vector_size(data)

    if hasattr(data, 'flatten'):
        data = data.flatten()

    retval = ""
    retval += " = {"
    for i, element in enumerate(data):
        if i % elements_per_row == 0:
            retval += '\n' + indent
        retval += '{value:#04x}'.format(value=int(element))
        if i < size - 1:
            retval += ', '
    retval += '\n}'
    return retval

def empty_vector(name, size):
    retval = ""
    retval += define('{NAME}_SIZE'.format(NAME=name.upper()), size)
    retval += "PI_L1 uint8_t {name}[{NAME}_SIZE];\n\n".format(name=name, NAME=name.upper())
    return retval

def check(name):
    return \
f"""static void check_{name}() {{
    printf("Checking the {name} vector:\\n");

    int n_err = 0;
    for (int i = 0; i < {name.upper()}_SIZE; i++) {{
        if ({name}[i] != golden_{name}[i]) {{
            printf("ERROR: wrong value of {name} @ %d: %d vs. golden: %d\\n", i, {name}[i], golden_{name}[i]);
            n_err++;
        }}
    }}

    if (n_err == 0)
        printf("> Success! No errors found.\\n");
    else
        printf("> Failure! Found %d/%d errors.\\n", n_err, {name.upper()}_SIZE);
}}

"""

def vector_size(data):
    if hasattr(data, 'numel'):
        return data.numel()
    elif hasattr(data, 'size'):
        return data.size
    else:
        return len(data)

def render_vector(name, init=None, size=None, elements_per_row=10, spaces=4):
    size_ = vector_size(init) if init is not None else size
    retval = ""
    retval += vector_definition(name, size_)
    if init is not None:
        retval += vector_initial_value(init, elements_per_row, spaces)
    retval += vector_end()
    return retval

def generate_vector_header(data, name, golden=None):
    filename = name + '.h'
    filepath = os.path.join('inc', 'data', filename)
    print('Generating vector header file: {name} -> {filepath}'.format(name=name, filepath=filepath))

    filerender = ""
    filerender += license(filename) + header_guard_begin(filename) + includes()

    filerender += render_vector(name, init=data, size=vector_size(golden) if golden is not None else None)

    if golden is not None:
        filerender += render_vector('golden_' + name, init=golden)
        filerender += check(name)
        
    filerender += header_guard_end(filename)

    with open(filepath, 'w') as file:
        file.write(filerender)

def vector_dims(vector):
    retval = ""
    name = vector["name"]
    for dim_name, dim_value in zip(vector["dim_names"], vector["shape"]):
        retval += define('{NAME}_{DIM_NAME}'.format(NAME=name.upper(), DIM_NAME=dim_name.upper()), dim_value)
    return retval

def render_dims(vectors, filename):
    filerender = ""
    filerender += license(filename) + header_guard_begin(filename)
    for vector in vectors:
        filerender += vector_dims(vector)
        filerender += '\n'
    filerender += header_guard_end(filename)
    return filerender

def generate_dims_header(vectors, filename='dims.h'):
    filepath = os.path.join('inc', 'data', filename)
    print('Generating dimensions header file: -> {filepath}'.format(filepath=filepath))
    filerender = render_dims(vectors, filename)
    with open(filepath, 'w') as file:
        file.write(filerender)

def borders(bits, signed = False):
    low = -(2 ** (bits-1)) if signed else 0
    high = 2 ** (bits-1) - 1 if signed else 2 ** bits - 1
    return low, high

def clip(x, bits, signed=False):
    low, high = borders(bits, signed)
    x[x > high] = high
    x[x < low] = low
    return x

def create_input(channels, spatial_dim):
    size = (1, channels, spatial_dim, spatial_dim)
    return torch.randint(low=0, high=100, size=size, dtype=torch.int32)

def create_weight(channels, kernel_shape):
    size = (channels, channels , kernel_shape, kernel_shape)
    return torch.randint(low=0, high=5, size=size, dtype=torch.int32)

def create_layer(channels, spatial_dim, kernel_shape, ne16):
    x = create_input(channels, spatial_dim + kernel_shape - 1)
    x_save = x.permute(0, 2, 3, 1).type(torch.int32)
    generate_vector_header(x_save, "input")

    w = create_weight(channels, kernel_shape)
    if not ne16:
        w_save = w.permute(0, 2, 3, 1).type(torch.int32)
    else:
        w_save = Ne16().conv_unroll(w, 8, layout="CoutCinK", dw=False)
    generate_vector_header(w_save, "weights")

    #norm_scale = torch.ones((1, channels, 1, 1), dtype=torch.int32)
    norm_scale = np.ones((1, channels, 1, 1), dtype='<i4')
    generate_vector_header(norm_scale.tobytes(), "normalization_scale")
    
    y = F.conv2d(x, w).type(torch.int32)
    y = torch.from_numpy(norm_scale) * y
    y = clip(y >> 8, 8)
    y_save = y.permute(0, 2, 3, 1).type(torch.int32)
    generate_vector_header(None, "output", golden=y_save)

    generate_dims_header([
                             {"name": "input",   "shape": x_save.shape[1:], "dim_names": ["height", "width", "channel"]},
                             {"name": "output",  "shape": y_save.shape[1:], "dim_names": ["height", "width", "channel"]},
                             {"name": "weights", "shape": w.shape,          "dim_names": ["channel_out", "channel_in", "kernel_height", "kernel_width"]}
                         ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel-shape', '-ks', dest='kernel_shape', type=int, choices=[1, 3], default=1,
                        help='Shape of the kernel. Choices: 1 or 3. Default: 1')
    parser.add_argument('--channels', '-c', type=int, default=1,
                        help='Number of input and output channels. Default: 1')
    parser.add_argument('--output-spatial-dimensions', '-osd', dest='spatial_dimensions', type=int, default=1,
                        help='Output spatial dimension. Default 1')
    parser.add_argument('--ne16', default=False, action='store_true',
                        help='Use the NE16 accelerator.')
    args = parser.parse_args()
    create_layer(channels = args.channels, spatial_dim = args.spatial_dimensions, kernel_shape=args.kernel_shape, ne16=args.ne16)
