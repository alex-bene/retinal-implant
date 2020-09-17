"""
CustomImplant.py

MIT License

Copyright (c) 2020 Alexandros Benetatos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
DESCRIPTION

A pulse2percept.implants.ProsthesisSystem subclass for pulse2percept to create a custom
square retinal implant with circular electrodes (Disk Electrodes) and the desired
    * number of electrodes in each side - e_num_side
    * radius of each electrode          - e_radius
    * and spacing between them          - spacing
    * or total area of implant          - total_area
"""

import numpy as np
from skimage.transform import resize
from pulse2percept.implants import ProsthesisSystem, ElectrodeGrid, DiskElectrode

class CustomImplant(ProsthesisSystem):
    def __init__(self, e_num_side, e_radius, spacing=None, total_area=None, stim=None, eye='RE', name = None):
        # if there is no name assigned the derive it from the desired characteristics of the implant
        if name is None:
            self.name = f"e_num_side={e_num_side}-e_radius={e_radius}-spacing={spacing}-total_area={total_area}"
        else:
            self.name = name

        # if total area is set the derive the spacing between each electrod from it
        if total_area is not None:
            spacing = total_area/(e_num_side - 1)
        elif spacing is None:
            raise Exception("Provide a spacing or total_area parameter in microns")

        self.earray = ElectrodeGrid((e_num_side, e_num_side), x=0, y=0, z=0, rot=0,
                                    r=e_radius, spacing=spacing, etype=DiskElectrode,
                                    names=('A', '1'))
        self.stim   = stim
        self.eye    = eye

    # plot implant on an axon map
    def plot_on_axon_map(self, annotate_implant=False, annotate_quadrants=True):
        plot_implant_on_axon_map(self, annotate_implant=annotate_implant, annotate_quadrants=annotate_quadrants)

    # # take the flattened stimuli (from img2stim) and return an 2-D array representing and image
    # def img2implant_img(self, img):
    #     return np.reshape(self.img2stim(img), self.earray.shape)