"""
CustomImplants.py

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

import warnings
import numpy as np
from skimage.transform import resize
from pulse2percept.models import AxonMapModel
from pulse2percept.implants import ProsthesisSystem, ElectrodeGrid, DiskElectrode

class SquareImplant(ProsthesisSystem):
    def __init__(self, e_num_side, e_radius, spacing=None, side_size=None, stim=None, eye='RE', name=None):
        """
            e_num_side : the number of electrodes in the squares implant's side
            e_radius   : the radius of each electrode in the square implant [microns]
            spacing    : the spacing between to electrodes in the squares implant's [microns]
            side_size  : the size of the squares implant's side in [microns]
            stim       : stimuli signal for each electrode [one dimensional array]
            eye        : the eye where it is implanted ['RE', 'LE']
            name       : name of the implant [string]
        """

        # if there is no name assigned the derive it from the desired characteristics of the implant
        if name is None:
            self.name = f"e_num_side={e_num_side}-e_radius={e_radius}-spacing={spacing}-side_size={side_size}"
        else:
            self.name = name

        # if side size is set the derive the spacing between each electrod from it
        if side_size is not None:
            spacing = (side_size - 2*e_radius)/(e_num_side - 1)
        elif spacing is None:
            raise Exception("Provide a 'spacing' or 'side_size' parameter in microns")

        if spacing < 2*e_radius:
            warnings.warn('Either the electrode radius (e_radius) is too big or the side size (side_size) ' +
                          'is too small and there is electrode overlap', stacklevel=2)

        self.earray = ElectrodeGrid((e_num_side, e_num_side), x=0, y=0, z=0, rot=0,
                                    r=e_radius, spacing=spacing, etype=DiskElectrode,
                                    names=('A', '1'))
        self.stim   = stim
        self.eye    = eye

    # plot implant on an axon map
    def plot_on_axon_map(self, annotate_implant=False, annotate_quadrants=True, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 10))
        AxonMapModel().plot(annotate=annotate_quadrants, ax=ax)
        self.earray.plot(annotate=annotate_implant, ax=ax)

    # # take the flattened stimuli (from img2stim) and return an 2-D array representing and image
    # def img2implant_img(self, img):
    #     return np.reshape(self.img2stim(img), self.earray.shape)