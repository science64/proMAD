# proMAD
Semiquantitative densitometric measurement of protein microarrays


[![PyPi](https://img.shields.io/pypi/v/proMAD.svg?style=for-the-badge)](https://pypi.org/project/proMAD/)
[![Status](https://img.shields.io/pypi/status/proMAD.svg?style=for-the-badge)](https://pypi.org/project/proMAD/)
[![Github issues](https://img.shields.io/github/issues/theia-dev/proMAD.svg?style=for-the-badge)](https://github.com/theia-dev/proMAD/issues)
[![Build](https://img.shields.io/travis/theia-dev/proMAD.svg?style=for-the-badge)](https://travis-ci.org/theia-dev/proMAD)
[![License](https://img.shields.io/github/license/theia-dev/proMAD.svg?style=for-the-badge)](https://github.com/theia-dev/proMAD/blob/master/LICENSE.txt)


## Setup
    pip install proMAD
    
## Usage
**ArrayAnalyse**
```python
from proMAD import ArrayAnalyse
aa = ArrayAnalyse('ARY022B')  # set array type
aa.load_collection('test_cases/exp_data')  # set input folder

aa.evaluate("A6")  # get result dictionary
aa.get_spot("A6")  # get raw data
aa.evaluate()  # get result dictionary for all spots
```
**Cutter**

* interactive
```python
from proMAD import Cutter
c = Cutter()

c.load_collection('test_cases/raw_image_folder')  # set input folder
c.set_shape()  # ask for the shape
c.guess_positions()  # use a simple guess as a starting point
c.preview()  # display guess (uses the last loaded image as default)

c.set_positions()  # ask for refined cut positions
c.set_names()  # ask for names
c.preview()  # check in preview
c.save_images('test_cases/formated_image_folder')  # save to folder (will be created if it does not exist
```

* direct
```python
from proMAD import Cutter

c = Cutter()

c.load_collection('test_cases/raw_image_folder')  # set input folder
c.shape = (2, 3)
c.cut_positions = [[20, 225, 445], [40, 130, 217, 315]]
c.names = [['OL', 'ML', 'UL'], [None, 'MR', 'UR']]
c.preview()
c.save_images('test_cases/formated_image_folder')  # save to folder (will be created if it does not exist
```

