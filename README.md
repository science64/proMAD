# proMAD <img src='https://github.com/theia-dev/proMAD/raw/master/proMAD/data/templates/logo.png' align="right"/>
Semiquantitative densitometric measurement of protein microarrays


[![PyPi](https://img.shields.io/pypi/v/proMAD.svg?style=for-the-badge)](https://pypi.org/project/proMAD/)
[![Status](https://img.shields.io/pypi/status/proMAD.svg?style=for-the-badge)](https://pypi.org/project/proMAD/)
[![License](https://img.shields.io/github/license/theia-dev/proMAD.svg?style=for-the-badge)](https://github.com/theia-dev/proMAD/blob/master/LICENSE.txt)

[![Github issues](https://img.shields.io/github/issues/theia-dev/proMAD.svg?style=for-the-badge)](https://github.com/theia-dev/proMAD/issues)
[![Coverage](https://img.shields.io/coveralls/github/theia-dev/proMAD?style=for-the-badge)](https://coveralls.io/github/theia-dev/proMAD)
[![Build](https://img.shields.io/travis/theia-dev/proMAD.svg?style=for-the-badge)](https://travis-ci.org/theia-dev/proMAD)


[![DOI](https://img.shields.io/badge/DOI-10.1186%2Fs12859--020--3402--4-green?style=for-the-badge)](https://doi.org/10.1186/s12859-020-3402-4)
[![Documentation](https://img.shields.io/readthedocs/promad.svg?style=for-the-badge)](https://promad.readthedocs.io)



## Setup
    pip install proMAD
    
You can also install the latest version directly from GitHub.

    pip install -U git+https://github.com/theia-dev/proMAD.git#egg=proMAD

    
## Usage
**ArrayAnalyse**
```python
from proMAD import ArrayAnalyse
aa = ArrayAnalyse('ARY022B')  # set array type
aa.load_collection('tests/cases/prepared', rotation=90)  # set input folder

aa.evaluate("A6")  # get result dictionary
aa.get_spot("A6")  # get underlying image data
aa.evaluate()  # get result dictionary for all spots

aa.report('report.xlsx')  # export the results
```
**Cutter**

* interactive
```python
from proMAD import Cutter
c = Cutter()

c.load_collection('tests/cases/raw')  # set input folder
c.set_shape()  # ask for the shape
c.guess_positions()  # use a simple guess as a starting point
c.preview()  # display guess (uses the last loaded image as default)

c.set_positions()  # ask for refined cut positions
c.set_names()  # ask for names
c.preview()  # check in the preview
c.save_images('test/cases/formatted_image_folder')  # save to folder (will be created if it does not exist)
```

* direct
```python
from proMAD import Cutter

c = Cutter()

c.load_collection('tests/cases/raw')  # set input folder
c.shape = (2, 3)
c.cut_positions = [[20, 225, 445], [40, 130, 217, 315]]
c.names = [['OL', 'ML', 'UL'], [None, 'MR', 'UR']]
c.preview()
c.save_images('test/cases/formatted_image_folder')  # save to folder (will be created if it does not exist)
```

## Citation
If you use proMAD in your work please cite the following [article](https://doi.org/10.1186/s12859-020-3402-4).


> Jaeschke, A., Eckert, H. & Bray, L.J.
> proMAD: semiquantitative densitometric measurement of protein microarrays. 
> *BMC Bioinformatics* **21**, 72 (2020). 
> doi: [10.1186/s12859-020-3402-4](https://doi.org/10.1186/s12859-020-3402-4)

***

The full source code can be accessed on [GitHub](https://github.com/theia-dev/proMAD) with the corresponding documentation hosted at [Read the docs](https://promad.readthedocs.io).
