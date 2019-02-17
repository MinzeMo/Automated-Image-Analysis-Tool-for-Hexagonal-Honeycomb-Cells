# Automated-Image-Analysis-Tool-for-Hexagonal-Honeycomb-Cells
Summary of the project:
The structures of hexagonal honeycombs are of great interest in the field of structural research. Honeycomb panels are utilized for its high strength to weight ratio. To gain insights into distributed topology optimization performed by honeybees, we developed two automated visual-based methods to study natural honeycombs. The first one is called the Three-Fold Symmetry Algorithm, which is based on triangle compact-hexagon geometry. The second one is called the Center Peak Gradient Algorithm, which is based on the gradient intensities of processed images. To encompass user interaction, a GUI was added, allowing users to edit and modify the inaccurate vertices and/or cell walls detected and drawn by the algorithms. The statistical data graphs of cell angles, cell walls and cell areas can be acquired depending on the specific user needs.

User Manual

It is important to note that before the program is utilized, the correct packages and Python version need to be set up and verified. This entire Python script is written in Python 3.5 and thus should always be executed and modified based on Python version 3.

In the beginning of the script, a list of every single package that is required to execute the script is imported. If any of these packages are missing the software will run into issues. These packages need to be installed following the installation manuals online for each different operating system and specific package. It is recommended that the user utilizing this script do some independent research into how his/her specific computer will need to install these packages. 

import cv2
import math
import sys
from math import sqrt
import skimage
from skimage.feature import blob_log
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
from matplotlib import collections as mc
import seaborn as sns 

The majority of Python distributions already have sys, math, matplotlib, numpy, scipy, and seaborn packages installed. The two Python packages that most users will have to install independently are OpenCV and skimage.

The OpenCV package can be installed on general Linux distributions using the command:

sudo apt-get install libopencv-dev python-opencv

The skimage package can be installed on general Linux distributions using the command:

sudo apt-get install python-skimage

Once all these packages are installed and verified, the script should run without any issues. If any specific packages are not installed properly, do some specified research into that specific package issue.

The script is broken up into three overall modes that will be explained in detail below. The initial user GUI screen can be seen below.



Once the script is executed, it will initially prompt the user for three options; option 1 is to load a new picture and process through the algorithms mentioned previously; option 2 is to reload an old image with old image data and continue working on it; option 3 allows a user to start from a blank template without any blob detection being used.

1. Automated Blob Detection and Image Analysis Mode
![picture1](https://user-images.githubusercontent.com/43077669/52919978-cc0aa180-32bc-11e9-9110-6c3783468df1.png)

Once the user enters the specific image name, the user will be able to choose which thresholding method is used (or not used).
![picture2](https://user-images.githubusercontent.com/43077669/52919981-d62ca000-32bc-11e9-81f0-cf78d05b2100.png)


The next step, after the thresholding is performed, will involve the script running blob detection on the preprocessed image. The resulting image can be seen below. This will take the majority of the time execution time.

![picture3](https://user-images.githubusercontent.com/43077669/52919984-db89ea80-32bc-11e9-969f-144c8383262e.png)

                                  Fig. 1 LoG blob detection on an example comb image.

![picture4](https://user-images.githubusercontent.com/43077669/52919985-e0e73500-32bc-11e9-8fd3-1766410c9c5a.png)

                              Fig. 2 LoG blob detection on an example comb image (zoomed in).

The script will now enter the phase 1 adjustment interface that allows the users to manually add or remove these detected red cell center points.
![picture5](https://user-images.githubusercontent.com/43077669/52919986-e3498f00-32bc-11e9-919b-8d6b44cbd63b.png)


In the case that the script fails to automatically detect/determine the center points, users can add their centers manually through this interface. For some incorrect detection, users will also be able to remove these points and then save the modified blob detection result to a file for future use. 

Option 4. Done will allow the user to proceed to the next step in the script.

The script will then perform either the Center Peak Gradient algorithm or the Three-Fold Symmetry algorithm depending on which script is executed. The resulting image can be seen below. This specific example uses the gradient method.

![picture6](https://user-images.githubusercontent.com/43077669/52919991-e93f7000-32bc-11e9-9762-1cce7f439828.png)

                              Fig. 3 Center Peak Gradient algorithm on an example comb image.

![picture8](https://user-images.githubusercontent.com/43077669/52920001-f9efe600-32bc-11e9-8519-252884eea98b.png)

                            Fig. 4 Center Peak Gradient algorithm on an example comb image (zoomed in).

The script will now enter the phase 2 adjustment interface that allows the users to manually add vertice points, remove vertice points, add cell wall lines, or remove cell wall lines after an initial automated process is complete.

![picture9](https://user-images.githubusercontent.com/43077669/52920004-fe1c0380-32bc-11e9-95ff-94a74f4ee9fa.png)


Users will be able to have full control over the vertice points and cell wall lines on the image. For example, users can add missing vertices and cell walls manually (options 1 and 3); they can also remove the incorrectly calculated vertices and cell wall lines (options 2 and 4). Option 5 allows them save the data (with the modifications) into a file that can be used in the Old Image Data Mode.
 
In the final step, the program can give the distribution graphs of specific data collected from the honeycomb image. This includes cell angles, cell areas and cell wall lengths.

The user prompt can be seen below.

![picture10](https://user-images.githubusercontent.com/43077669/52920005-02e0b780-32bd-11e9-82cf-6224734da79f.png)


Example data distribution graphs can be seen below.

![picture11](https://user-images.githubusercontent.com/43077669/52920009-07a56b80-32bd-11e9-8722-2f3e5508ab95.png)

                                    Fig. 5 Cell Area distribution results data.

![picture12](https://user-images.githubusercontent.com/43077669/52920011-0b38f280-32bd-11e9-9e10-59b6b410a3d8.png)

                                    Fig. 6 Wall length distribution results data.

![picture13](https://user-images.githubusercontent.com/43077669/52920014-0e33e300-32bd-11e9-8e14-dd700b437aca.png)

                                    Fig. 7 Cell Angular distribution results data.


2. Old Image Data Mode

![picture14](https://user-images.githubusercontent.com/43077669/52920017-112ed380-32bd-11e9-9f7b-1471737a0c15.png)

In this mode, the program does not perform blob detection. It asks users to load the previous center points data file instead.

Once these old image center points data file is loaded, the user is brought back to the User Blob RED Point Adjustment Interface, in which everything functions exactly as Mode 1.

In phase 2 for this script, the requirements are the same. The script prompts the user to enter the specific data files and continues from the interface manual modification step.

![picture15](https://user-images.githubusercontent.com/43077669/52920020-14c25a80-32bd-11e9-8570-8b8a37ec708a.png)


Once these old image data files are loaded, the user is brought back to the User Vertices Line Adjustment Interface, in which everything functions exactly as Mode 1.

The resulting data from this mode is in the exact same format as Mode 1.

3. BLANK Slate Mode

In this mode, the program does not perform blob detection. Users will be able to start from a completely unmarked comb image so that they are able to have absolute control over the starting center points locations. They will be allowed to manually add all the initial red center points and then modify the vertices these center points generate afterwards.

![picture16](https://user-images.githubusercontent.com/43077669/52920024-1b50d200-32bd-11e9-8e0c-d1b9ffafbd8f.png)

                          Fig. 8 Completely unmarked example comb image.

![picture17](https://user-images.githubusercontent.com/43077669/52920026-1ee45900-32bd-11e9-9b6f-d606b16abdfa.png)

                    Fig. 9 Example center point markings for a completely unmarked example comb image.

![picture18](https://user-images.githubusercontent.com/43077669/52920030-2277e000-32bd-11e9-8ee2-8eb473fe96b2.png)

                    Fig. 10 Default (automated) results based on the manual markings in figure 25.

Users can adjust the vertices and cell walls manually by manually adding in vertice points and cell wall (green) lines.

![picture19](https://user-images.githubusercontent.com/43077669/52920031-260b6700-32bd-11e9-9733-ce61f43abd9e.png)

                    Fig. 11 Example of modified results based on the manual markings in figure 25.

The resulting data from this mode is in the exact same format as Mode 1 and Mode 2.
