Usage
==============

Array Analyse
--------------

.. code-block:: python

   from proMAD import ArrayAnalyse
   aa = ArrayAnalyse('ARY022B')  # set array type
   aa.load_collection('tests/cases/prepared', rotation=90)  # set input folder

   aa.evaluate("A6")  # get result dictionary
   aa.get_spot("A6")  # get underlying image data
   aa.evaluate()  # get result dictionary for all spots

Documentation of the `core module`_.

.. _`core module`: core.html

Cutter
--------------

* interactive

The cutter module can guess positions based on user-input of the general membrane shape.

.. code-block:: python

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

* direct

Alternatively, positions can be directly provided.

.. code-block:: python

   from proMAD import Cutter

   c = Cutter()

   c.load_collection('tests/cases/raw')  # set input folder
   c.shape = (2, 3)
   c.cut_positions = [[20, 225, 445], [40, 130, 217, 315]]
   c.names = [['OL', 'ML', 'UL'], [None, 'MR', 'UR']]
   c.preview()
   c.save_images('test/cases/formatted_image_folder')  # save to folder (will be created if it does not exist)

Documentation of the `cut module`_.

.. _`cut module`: cut.html

Report
----------

Reports can be exported in a variety of file formats (xlsx, csv, tex, json).

.. code-block:: python

   aa.report('report.xlsx')  # export the results

Documentation of the `report module`_.

.. _`report module`: report.html