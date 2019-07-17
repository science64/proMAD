import json
import re
import string
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from scipy import stats
from scipy.optimize import curve_fit, minimize_scalar
from skimage import io, img_as_float, img_as_ubyte
from skimage import measure
from skimage import transform
from skimage.color import rgb2gray
from skimage.external import tifffile
from skimage.filters import threshold_otsu
from skimage.morphology import reconstruction
from skimage.transform import rotate
from skimage.util import invert

from . import config


class ArrayAnalyse(object):
    """
    Analyse a set of micro arrays with variable exposure.

    Notes
    -----

    To enable an more detailed output *debug* can be set to 'basic' or 'plot'.

    >>> aa = ArrayAnalyse('ARY022B')
    >>> aa.debug = 'plot'
    >>> aa.debug = 'basic'

    This program will test images to ensure they match certain expected properties.
    If one of the tests are failed the image is not added to the collection.
    To disable this function set *strict* to False.

    >>> aa.strict = False

    To modify the test the conditions can be modified.
    The following parameters are available:

    - `aa.test_warp_length_ration`: deviation from the x/y ratio (0.1)
    - `aa.test_warp_angle`: deviation from the angle in radians (0.05)
    - `aa.test_warp_distance`: allowed (distance/short edge) between guessed anchor positions and nearest found contour (0.275)
    - `aa.test_quality_resolution`: minimal used portion of the brightness spectra (0.1)
    - `aa.test_quality_exposure`: maximal allowed number of full exposed pixels (1)

    """
    
    def __init__(self, array_type):
        """
        Initialize ArrayAnalyse object

        Notes
        -----
        You can list all available options for `array_type` with *list_types()*

        >>> ArrayAnalyse.list_types()
        ARY022B
            Name: Human XL Cytokine Array Kit
            Type: Proteome Profiler Array
            Company: R&D Systems, Inc
            Source: https://resources.rndsystems.com/pdfs/datasheets/ary022b.pdf
        ...


        Parameters
        ----------
        array_type: str
            id of the used micro array type
        """
        self.array_types = dict()
        for file_path in config.array_data_folder.iterdir():
            if file_path.is_file():
                if file_path.suffix == '.json':
                    new_array = json.loads(file_path.read_text())
                    self.array_types[new_array['id']] = new_array
        self.array_data = self.array_types[array_type]
        self.source_images = []
        self.raw_images = []
        self.backgrounds = []
        self.foregrounds = []
        self.bg_parameters = []
        self.exposure = []
        self.meta_data = []
        self.debug = None
        self.strict = True

        # Reaction fit parameters
        self.start_time = 60  # s
        self.k_reac = 1.4E6  # 1/(mol/L * s)
        self.kappa_fit_count = 10
        self._kappa = 1
        self._fit_selection = []

        # Test parameters
        self.test_warp_length_ration = 0.1
        self.test_warp_angle = 0.05
        self.test_warp_distance = 0.275
        self.test_quality_resolution = 0.1
        self.test_quality_exposure = 1

        self.is_finalized = False
        self.has_exposure = False

        self.grid_position = np.zeros((sum(self.array_data['net_layout_x']), sum(self.array_data['net_layout_y']), 2))

        px, py = 0, 0
        for xc, x_block in enumerate(self.array_data['net_layout_x']):
            for x in range(x_block):
                self.grid_position[px, :, 0] = (config.scale / 2.0 +
                                                (px + xc * self.array_data['net_skip_size']) * config.scale)
                px += 1

        for yc, y_block in enumerate(self.array_data['net_layout_y']):
            for y in range(y_block):
                self.grid_position[:, py, 1] = (config.scale / 2.0 +
                                                (py + yc * self.array_data['net_skip_size']) * config.scale)
                py += 1

    @staticmethod
    def _angle_between(v1, v2):
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    @staticmethod
    def _failed_passed(value):
        if value:
            return '✅'
        else:
            return '❌'

    @staticmethod
    def _add_2d_tuple(a, b):
        return a[0] + b[0], a[1] + b[1]

    @staticmethod
    def list_types():
        """
        List all available array types.

        """
        array_types = dict()
        for file_path in config.array_data_folder.iterdir():
            if file_path.is_file():
                if file_path.suffix == '.json':
                    new_array = json.load(file_path.open('r'))
                    array_types[new_array['id']] = new_array
        for array_id, array in array_types.items():
            print(array_id)
            print(f'\tName: {array["name"]}')
            print(f'\tType: {array["array_type"]}')
            print(f'\tCompany: {array["company"]}')
            print(f'\tSource: {array["docs_source"]}\n')

    def reset_collection(self):
        """
        Empties the image collections

        """
        self.source_images = []
        self.raw_images = []
        self.backgrounds = []
        self.foregrounds = []
        self.bg_parameters = []
        self.meta_data = []
        self.exposure = []
        self.is_finalized = False
        self.has_exposure = False

    def load_collection(self, data_input, rotation=None, finalize=True):
        """
        Load multiple files into the collection.
        Includes just files that are directly in the folders listed in *data_input* and does not search subfolders.

        Parameters
        ----------
        data_input: list(str) or list(Path) or Path or str
            a single input folder or list of folders / files
        rotation: int or float
            apply a rotation to all images
        finalize: bool
            defines if a collection should be directly finalized

        """

        if self.is_finalized:
            print("Data is already finalized. You can use reset_collection to start over.")
            return None

        if isinstance(data_input, str):
            input_paths = [Path(data_input)]
        elif isinstance(data_input, Path):
            input_paths = [data_input]
        else:
            input_paths = []
            for input_entry in data_input:
                input_paths.append(Path(input_entry))
        input_file_paths = []
        for input_path in input_paths:
            if input_path.is_dir():
                for folder_file_path in input_path.iterdir():
                    if folder_file_path.is_file():
                        input_file_paths.append(folder_file_path)
            elif input_path.is_file():
                input_file_paths.append(input_path)
        if not input_file_paths:
            print('Found no images')
            return None
        for file_path in input_file_paths:
            if file_path.name[0] == '.':
                continue
            self.load_image(file_path, rotation=rotation)

        if finalize and self.raw_images:
            self.finalize_collection()

    def finalize_collection(self):
        """
        Orders the image collections and convert them into numpy arrays.
        The ordering is either derived from the background intensity extracted from a histogram or exposure time in image meta data.

        """
        if self.is_finalized:
            print("Data is already finalized. You can use reset_collection to start over.")
            return None

        if len(self.raw_images) == 0:
            print("There is no data to be finalized.")
            return None

        self.bg_parameters = np.array(self.bg_parameters)
        self.exposure = np.array(self.exposure)
        if self.exposure.size == self.bg_parameters.size:
            order = np.argsort(self.exposure)
            self.exposure = self.exposure[order]
            self.has_exposure = True
        else:
            order = np.argsort(self.bg_parameters)
            self.exposure = []
        raw_images_array = np.zeros(shape=(self.raw_images[0].shape + (len(order),)),
                                    dtype=self.raw_images[0].dtype)
        backgrounds_array = np.zeros(shape=(self.backgrounds[0].shape + (len(order),)),
                                     dtype=self.backgrounds[0].dtype)
        foregrounds_array = np.zeros(shape=(self.foregrounds[0].shape + (len(order),)),
                                     dtype=self.foregrounds[0].dtype)
        source_images = []
        meta = []
        for n, i in enumerate(order):
            raw_images_array[:, :, n] = self.raw_images[i]
            backgrounds_array[:, :, n] = self.backgrounds[i]
            foregrounds_array[:, :, n] = self.foregrounds[i]
            source_images.append(self.source_images[i])
            meta = self.meta_data[i]
        self.raw_images = raw_images_array
        self.backgrounds = backgrounds_array
        self.foregrounds = foregrounds_array
        self.bg_parameters = self.bg_parameters[order]
        self.meta_data = meta
        self.source_images = source_images
        self.is_finalized = True
        if self.has_exposure:
            self.minimize_kappa()

        if self.debug == 'plot':
            self.control_alignment()
            self.contact_sheet()

    def load_image(self, file_path, rotation=None):
        """
        Load a single image file into the collection.


        Parameters
        ----------
        file_path: Path or str
        rotation: int or float
            apply a rotation to the images

        """

        if self.is_finalized:
            print("Data is already finalized. You can use reset_collection to start over.")
            return None

        file_path = Path(file_path)
        print(f'Load image: {file_path}')
        source_image = io.imread(file_path.absolute(), plugin='imageio')
        meta_data = None
        if file_path.suffix.lower() == '.tif':
            with tifffile.TiffFile(str(file_path.absolute())) as tif_data:
                tags = [page.tags for page in tif_data.pages]
            if tags:
                try:
                    meta_data = json.loads(tags[0]['image_description'].value)
                except (KeyError, json.decoder.JSONDecodeError):
                    pass

        if source_image.ndim == 3:
            print("\tLoad RGB image. Converting to greyscale alters the result. Use raw grayscale if possible.")
            source_image = rgb2gray(source_image)
            # if min value is black invert image
            if np.average(source_image) > np.max(source_image)//2:
                source_image = invert(source_image)
        if rotation:
            source_image = rotate(source_image, rotation, resize=True)

        source_image = img_as_float(source_image)
        raw_image = self.warp_image(source_image, rotation=rotation)

        if raw_image is not None:
            used_resolution = np.max(raw_image) - np.min(raw_image)
            over_exposed = np.sum(raw_image >= 1.0)
            if self.debug:
                print('Quality checks')
                print(f'\tused resolution: {self._failed_passed(not (np.isnan(used_resolution)) and not (used_resolution < self.test_quality_resolution))} ({used_resolution*100:.1f}%)')
                print(f'\tfull exposed pixels: {self._failed_passed(not (np.isnan(over_exposed)) and not (over_exposed > self.test_quality_exposure))} ({over_exposed})')

            if self.strict:
                if (np.isnan(used_resolution) or np.isnan(over_exposed) or
                        (used_resolution < self.test_quality_resolution) or
                        (over_exposed > self.test_quality_exposure)):
                    print('\tImage skipped - quality check failed.')
                    return None

            self.source_images.append(source_image)
            self.raw_images.append(raw_image)
            seed = np.copy(raw_image)
            seed[1:-1, 1:-1] = raw_image.max()
            mask = raw_image
            filled = reconstruction(seed, mask, method='erosion')
            seed = np.copy(filled)
            seed[1:-1, 1:-1] = filled.min()
            mask = filled
            dilated = reconstruction(seed, mask, method='dilation')
            image = filled - dilated
            self.backgrounds.append(dilated)
            self.foregrounds.append(image)
            self.bg_parameters.append(self.background_histogram(raw_image))
            self.meta_data.append(meta_data)
            if meta_data:
                if 'exposure_time' in meta_data:
                    self.exposure.append(meta_data['exposure_time'])

    def warp_image(self, image, rotation=None):
        """
        Warp image into an defined shape based on a contour search.


        Parameters
        ----------
        image: array_like
            docs_source image
        rotation: int or float
            apply a rotation to the images

        Returns
        -------
        array_like
            warped image (64bit float [0-1])

        """
        x_length = (sum(self.array_data['net_layout_x']) - 1 + (len(self.array_data['net_layout_x']) - 1) *
                    self.array_data['net_skip_size']) * config.scale
        y_length = (sum(self.array_data['net_layout_y']) - 1 + (len(self.array_data['net_layout_y']) - 1) *
                    self.array_data['net_skip_size']) * config.scale

        if rotation is None:
            rotation = 0

        # find bright spots and calculate their center
        contours = measure.find_contours(image, np.max(image) * 0.60)
        x = []
        y = []
        for n, contour in enumerate(contours):
            y.append(sum(contour[:, 0]) / len(contour))
            x.append(sum(contour[:, 1]) / len(contour))

        x = np.array(x)
        y = np.array(y)

        # TODO: use array_data to generate the initial guesses
        ref_1_guess = (min(x), min(y))
        ref_2_guess = (min(x), max(y))
        ref_3_guess = (max(x), min(y))

        ref_1_idx = np.argsort((x - ref_1_guess[0]) ** 2 + (y - ref_1_guess[1]) ** 2)[0]
        ref_2_idx = np.argsort((x - ref_2_guess[0]) ** 2 + (y - ref_2_guess[1]) ** 2)[0]
        ref_3_idx = np.argsort((x - ref_3_guess[0]) ** 2 + (y - ref_3_guess[1]) ** 2)[0]

        # test if found points make sense
        y_length_test = np.sqrt((x[ref_2_idx] - x[ref_1_idx]) ** 2 + (y[ref_2_idx] - y[ref_1_idx]) ** 2)
        x_length_test = np.sqrt((x[ref_2_idx] - x[ref_3_idx]) ** 2 + (y[ref_2_idx] - y[ref_3_idx]) ** 2)
        length_test = (x_length_test/y_length_test) / (x_length/y_length)

        angle_test = self._angle_between(np.array([x[ref_3_idx] - x[ref_1_idx], y[ref_3_idx] - y[ref_1_idx]]),
                                   np.array([x[ref_2_idx] - x[ref_1_idx], y[ref_2_idx] - y[ref_1_idx]]))/np.pi*2

        dis_1 = np.min(np.sqrt((x - ref_1_guess[0]) ** 2 + (y - ref_1_guess[1]) ** 2))
        dis_2 = np.min(np.sqrt((x - ref_2_guess[0]) ** 2 + (y - ref_2_guess[1]) ** 2))
        dis_3 = np.min(np.sqrt((x - ref_3_guess[0]) ** 2 + (y - ref_3_guess[1]) ** 2))
        dis_test = max(dis_1, dis_2, dis_3)/min(y_length_test, x_length_test)

        if self.debug == 'plot':
            fig, ax = plt.subplots()
            ax.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
            ax.scatter(x, y, zorder=2, marker='o', c=np.arange(len(x))+1, cmap=plt.cm.viridis, label='detected contours')
            ax.scatter((x[ref_1_idx], x[ref_2_idx], x[ref_3_idx]),
                       (y[ref_1_idx], y[ref_2_idx], y[ref_3_idx]),
                       zorder=4, color='red', marker='+', label='selected positions')
            ax.scatter((min(x), min(x), max(x)), (min(y), max(y), min(y)),
                       zorder=5, color='yellow', marker='x', label='guessed positions')
            ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                      ncol=3, mode="expand", borderaxespad=0.)
            fig.show()

        if self.debug:
            print('Warp checks')
            print(f'\tlength ratio: {self._failed_passed(not (np.isnan(length_test)) and not (abs(length_test-1.0) > self.test_warp_length_ration))} ({length_test:.4f})')
            print(f'\tangle: {self._failed_passed(not (np.isnan(angle_test)) and not (abs(angle_test-1.0) > self.test_warp_angle))} ({angle_test*90:.2f})')
            print(f'\tguess distance: {self._failed_passed(not (np.isnan(dis_test)) and not (dis_test > self.test_warp_distance))} ({dis_test:.4f})')

        if self.strict:
            if np.isnan(length_test) or np.isnan(angle_test) or np.isnan(dis_test) or abs(length_test-1.0) > self.test_warp_length_ration or abs(angle_test-1.0) > self.test_warp_angle or dis_test > self.test_warp_distance:
                print('\tImage skipped - position guess failed - image maybe not rotated correctly at %i degrees,\n'
                      '\t\tor the contrast is not good enough for the feature extraction.' % rotation)
                return None

        # define the reference spots locations and transform the image
        src = np.float32([[x[ref_1_idx], y[ref_1_idx]],
                          [x[ref_2_idx], y[ref_2_idx]],
                          [x[ref_3_idx], y[ref_3_idx]]])
        dst = np.float32([[config.scale, config.scale],
                          [config.scale, y_length + config.scale],
                          [config.scale + x_length, config.scale]])
        tform = transform.estimate_transform('affine', dst, src)
        warped = transform.warp(image, tform, output_shape=[int(y_length + config.scale * 2),
                                                            int(x_length + config.scale * 2)])

        return warped

    @staticmethod
    def background_histogram(raw_image, full=False):
        """
        Generate histogram and find background split parameter

        Parameters
        ----------
        raw_image: array_like
            input image

        full: bool
            if True return also derivation

        """
        image_reduced = np.round(np.array(list(raw_image.flat))*2**11).astype('uint16')
        histo_counter = Counter(image_reduced)
        peak = histo_counter.most_common(1)[0][0]
        background_data = image_reduced[image_reduced < (int(peak * 2))]
        background_parameter = stats.norm.fit(background_data, loc=peak)
        if not full:
            return background_parameter[0]/(2**11)
        else:
            return background_parameter

    @staticmethod
    def get_position_coordinates(position):
        """
        Convert position string to coordinates.

        Parameters
        ----------
        position: str or (int, int)
            position string  ('A6', 'C4', ...)

        Returns
        -------
        (int, int)
            coordinates

        """
        if isinstance(position, str):
            match = re.match(r"([a-z]+)([0-9]+)", position, re.I)
            if match:
                items = match.groups()
            else:
                return None
            num = 0
            for c in items[0]:
                if c in string.ascii_letters:
                    num = num * 26 + (ord(c.upper()) - ord('A')) + 1
            coordinates = (num - 1, int(items[1]) - 1)
        elif len(position) == 2:
            coordinates = position
        else:
            coordinates = None
        return coordinates

    def get_spot(self, position, double=None):
        """
        Return image data for a specific spot.

        Parameters
        ----------
        position: str or (int, int)
            position string  ('A6', 'C4', ...) or coordinates
        double: bool
            double spot switch
        Returns
        -------
        dict
            Dictionary with the three data sets for the requested spot. ('raw', 'background', 'raw')


        """

        if not self.is_finalized:
            print('Data collection needs to be finalized to generate spot data.')
            return None
        coordinates = self.get_position_coordinates(position)
        if coordinates is None:
            return None
        if double:
            coordinates = self._add_2d_tuple(coordinates, self.array_data['double_spot'])
        x_raw, y_raw = (self.grid_position[coordinates[1], coordinates[0]])
        result = dict()
        result['foreground'] = self.foregrounds[int(round(y_raw)):int(round(y_raw))+config.scale,
                                                int(round(x_raw)):int(round(x_raw))+config.scale, :]
        result['background'] = self.backgrounds[int(round(y_raw)):int(round(y_raw))+config.scale,
                                                int(round(x_raw)):int(round(x_raw))+config.scale, :]
        result['raw'] = self.raw_images[int(round(y_raw)):int(round(y_raw))+config.scale,
                                        int(round(x_raw)):int(round(x_raw))+config.scale, :]
        return result

    def light_reaction(self, t, c_enzyme):
        """
        Calculate the accumulated light intensity.

        Notes
        -----
        Function based on findings presented in "An Investigation of the Mechanism of the Luminescent Peroxidation
        of Luminol by Stopped Flow Techniques" **Milton J. Cormier and Philip M. Prichard** - *J. Biol. Chem.* 1968 243: 4706.


        Parameters
        ----------
        t: float
            time in seconds

        c_enzyme: float
            enzyme concentration in mol/L

        Returns
        -------
        float
            accumulated light intensity

        """
        return self._kappa * (np.exp(-self.k_reac * c_enzyme * self.start_time)
                              - np.exp(-self.k_reac * c_enzyme * (t + self.start_time)))

    def kappa_error(self, kappa):
        """
        Calculate the error for a given kappa for a subset of spots in `_fit_selection`.

        Parameters
        ----------
        kappa: float
            combined parameter

        Returns
        -------
        float
            absolute error sum over all selected samples

        """

        self._kappa = kappa
        err_sum = 0
        for spot_raw in self._fit_selection:
            popt, pcov = curve_fit(self.light_reaction, self.exposure, spot_raw, p0=1E-10)
            err_sum += sum([np.abs(self.light_reaction(t, popt[0]) - spot_raw[n]) for n, t in enumerate(self.exposure)])
        return err_sum

    def minimize_kappa(self):
        """
        Finds the optimal kappa for a subset of `kappa_fit_count` numbers of brightest spots.

        """
        selection = []
        for entry in self.array_data['spots']:
            spot = self.get_spot(entry['position'])
            selection.append((entry['position'], np.max(spot['raw'])))
        self._fit_selection = [np.average(self.get_spot(entry[0])['raw'], (1, 0)) - self.bg_parameters
                               for entry in sorted(selection, key=lambda x: x[1], reverse=True)[:self.kappa_fit_count]]
        result = minimize_scalar(self.kappa_error, bounds=(1E-5, 1))
        self._kappa = result.x

    @staticmethod
    def evaluate_spots_raw(spot):
        return np.average(spot['raw'], (0, 1))

    def evaluate_spots_raw_bg_corrected(self, spot):
        return np.average(spot['raw'], (0, 1))-self.bg_parameters

    @staticmethod
    def evaluate_spot_local_bg(spot):
        ratio = spot['raw'] / spot['background']
        return np.mean(ratio), np.std(ratio)

    def evaluate_spot_hist_foreground(self, spot):
        averages = np.average(spot['foreground'], (0, 1))
        data = stats.linregress(self.bg_parameters, averages)
        return data[0], data[1], data[2]**2

    def evaluate_spot_hist_raw(self, spot):
        averages = np.average(spot['raw'], (0, 1))
        data = stats.linregress(self.bg_parameters, averages)
        return data[0], data[1], data[2]**2

    def evaluate_spot_reac(self, spot):
        """

        Parameters
        ----------
        spot: dict
            spot data from `get_spot()`

        Returns
        -------
        (float, float)
            enzyme concentration in mol/L and standard deviation

        """
        popt, pcov = curve_fit(self.light_reaction, self.exposure, np.average(spot['raw']-self.bg_parameters,
                                                                              (0, 1)), p0=1E-10)
        return popt[0], np.sqrt(pcov[0])

    def evaluate(self, position=None, norm='hist_raw', just_value=False, double_spot=False):
        """
        Evaluate a spot or the complete array.

        Notes
        -----
        You can select between different evaluation modes, by setting `norm`.

        - raw: list of averages for all timesteps based on the original image
        - raw_bg: as raw but reduced by the histogram based background value
        - local_bg: mean of the ratios between the original images and the extracted backgrounds
        - hist_fg: linear correlation between background (histogramm) evolution to the average forground value
        - hist_raw: as hist_fg but compared to the original image
        - reac: estimate of the catalytic enzyme concentration

        The reaction based model is just available when the exposure times can be accessed in the loaded images.
        While resulting value is in mol/L it is important to note that it is just an estimation based on the reaction
        constants published in "An Investigation of the Mechanism of the Luminescent Peroxidation of Luminol by
        Stopped Flow Techniques" **Milton J. Cormier and Philip M. Prichard** - *J. Biol. Chem.* 1968 243: 4706.


        Parameters
        ----------
        position: str or (int, int)
            position string  ('A6', 'C4', ...) or coordinates
        norm: str
            evaluation strategy selection
        just_value: bool
        double_spot: bool

        Returns
        -------
        dict or array_like
            if `just_value` an array of values is returned else a dictionary including spot information

        """

        if not self.is_finalized:
            print('Data collection needs to be finalized to evaluate.')
            return None

        if norm == 'raw':
            evaluate_spots = self.evaluate_spots_raw
        elif norm == 'raw_bg':
            evaluate_spots = self.evaluate_spots_raw_bg_corrected
        elif norm == 'local_bg':
            evaluate_spots = self.evaluate_spot_local_bg
        elif norm == 'hist_fg':
            evaluate_spots = self.evaluate_spot_hist_foreground
        elif norm == 'hist_raw':
            evaluate_spots = self.evaluate_spot_hist_raw
        elif norm == 'reac':
            if not self.has_exposure:
                print('The reaction model needs exposure time date. Please select a different modus.')
                return None
            evaluate_spots = self.evaluate_spot_reac
        else:
            return None
        data = []
        if position is None:
            for entry in self.array_data['spots']:
                spot = self.get_spot(entry['position'])
                value = evaluate_spots(spot)
                if just_value:
                    data.append(value[0])
                else:
                    data.append(dict(position=entry['position'], info=entry['info'], value=value))
                if 'double_spot' in self.array_data:
                    spot = self.get_spot(entry['position'], double=self.array_data['double_spot'])
                    value = evaluate_spots(spot)
                    if just_value:
                        data.append(value[0])
                    else:
                        position = [entry['position'][0] + self.array_data['double_spot'][0],
                                    entry['position'][1] + self.array_data['double_spot'][1]]
                        data.append(dict(position=position, info=entry['info'], value=value))

        else:
            position = self.get_position_coordinates(position)
            spot = self.get_spot(position)
            value = evaluate_spots(spot)
            info = ''
            for entry in self.array_data['spots']:
                if list(entry['position']) == list(position):
                    info = entry['info']
                    break
                if (entry['position'][0] == position[0] - self.array_data['double_spot'][0] and
                        entry['position'][1] == position[1] - self.array_data['double_spot'][1]):
                    info = entry['info']
                    break

            if just_value:
                data.append(value)
            else:
                data.append(dict(position=position, info=info, value=value))
            if 'double_spot' in self.array_data and double_spot:
                spot = self.get_spot(position, double=True)
                value = evaluate_spots(spot)
                if just_value:
                    data.append(value[0])
                else:
                    position = (position[0] + self.array_data['double_spot'][0],
                                position[1] + self.array_data['double_spot'][1])
                    data.append(dict(position=position, info=info, value=value))

        return data

    def control_alignment(self, file_name=None):
        if not self.is_finalized:
            print('Data collection needs to be finalized to plot the alignment data.')
            return None
        align_img = np.zeros(self.raw_images[:, :, 0].shape)
        for i in range(self.raw_images.shape[2]):
            thresh = threshold_otsu(self.raw_images[:, :, i])
            align_img += self.raw_images[:, :, i] > thresh
        align_img /= self.raw_images.shape[2]
        if file_name is None:
            fig, ax = plt.subplots()
            ax.set_title('Alignment check')
            ax.imshow(align_img,  interpolation='nearest', cmap=plt.cm.nipy_spectral)
            fig.show()
        else:
            align_img = plt.cm.nipy_spectral(align_img)
            io.imsave(str(Path(file_name).absolute()), align_img)

    def control_reac_fit(self, position, c_enzyme=None, file_name=None):
        spot = self.get_spot(position)
        y_raw = self.evaluate_spots_raw_bg_corrected(spot)
        if c_enzyme is None:
            c_enzyme = self.evaluate_spot_reac(spot)[0]
        print(c_enzyme)
        y = []
        for t in self.exposure:
            y.append(self.light_reaction(t, c_enzyme))

        fig, ax = plt.subplots()
        ax.set_title(f'Reaction fit spot {position} [{c_enzyme:.3e}]')
        ax.scatter(self.exposure, y_raw)
        ax.plot(self.exposure, y, c='red')
        if file_name is None:
            fig.show()

    def contact_sheet_spot(self, position, file_name=None):
        spot = self.get_spot(position)
        pad = 2
        sx, sy, spot_count = spot['raw'].shape
        y_count = int(np.ceil(np.sqrt(spot_count)))
        x_count = int(np.ceil(spot_count / y_count))
        y_max = int(y_count * sy + (y_count - 1) * pad)
        x_max = int(x_count * sx + (x_count - 1) * pad)

        css_raw = np.ones((x_max, y_max))
        css_fg = np.ones((x_max, y_max))
        css_bg = np.ones((x_max, y_max))
        for x_c in range(x_count):
            for y_c in range(y_count):
                x = x_c * pad + x_c * sx
                y = y_c * pad + y_c * sy
                try:
                    css_raw[x:x + sx, y:y + sy] = spot['raw'][:, :, x_c * y_count + y_c]
                    css_bg[x:x + sx, y:y + sy] = spot['background'][:, :, x_c * y_count + y_c]
                    css_fg[x:x + sx, y:y + sy] = spot['foreground'][:, :, x_c * y_count + y_c]
                except IndexError:
                    pass

        for name, image in (('raw', css_raw), ('foreground', css_fg), ('background', css_bg)):
            if file_name is None:
                fig, ax = plt.subplots()
                ax.set_title(f'Contact spot sheet {name}')
                ax.imshow(image, interpolation='nearest')
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                fig.show()

            else:
                file_name = Path(file_name)
                suffix = file_name.suffix
                f_name = file_name.with_suffix('').name
                file_path = str(file_name.with_name(f_name+'_' + name).with_suffix(suffix).absolute())
                image = plt.cm.gist_ncar(image)[:, :, :3]
                image = img_as_ubyte(image)
                io.imsave(file_path, image)

    def contact_sheet(self, file_name=None):
        if not self.is_finalized:
            print('Data collection needs to be finalized to plot the contact sheet.')
            return None
        pad = 20
        y_count = int(np.ceil(np.sqrt(self.raw_images.size) / self.raw_images.shape[1]))
        x_count = int(np.ceil(self.raw_images.shape[2]/y_count))
        y_max = int(y_count * self.raw_images.shape[1] + (y_count-1) * pad)
        x_max = int(x_count * self.raw_images.shape[0] + (x_count-1) * pad)

        cs_raw = np.ones((x_max, y_max))
        cs_bg = np.ones((x_max, y_max))
        cs_fg = np.ones((x_max, y_max))
        for x_c in range(x_count):
            for y_c in range(y_count):
                x = x_c * pad + x_c * self.raw_images.shape[0]
                y = y_c * pad + y_c * self.raw_images.shape[1]
                try:
                    cs_raw[x:x + self.raw_images.shape[0], y:y + self.raw_images.shape[1]] = self.raw_images[:, :, x_c * y_count + y_c]
                    cs_bg[x:x + self.raw_images.shape[0], y:y + self.raw_images.shape[1]] = self.backgrounds[:, :, x_c * y_count + y_c]
                    cs_fg[x:x + self.raw_images.shape[0], y:y + self.raw_images.shape[1]] = self.foregrounds[:, :, x_c * y_count + y_c]
                except IndexError:
                    pass

        for name, image in (('raw', cs_raw), ('foreground', cs_fg), ('background', cs_bg)):
            if file_name is None:
                fig, ax = plt.subplots()
                ax.set_title(f'Contact sheet {name}')
                ax.imshow(image, interpolation='nearest', cmap=plt.cm.gist_ncar, vmin=0, vmax=1)
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                fig.show()

            else:
                file_name = Path(file_name)
                suffix = file_name.suffix
                f_name = file_name.with_suffix('').name
                file_path = str(file_name.with_name(f_name+'_' + name).with_suffix(suffix).absolute())
                image = plt.cm.gist_ncar(image)[:, :, :3]
                image = img_as_ubyte(image)
                io.imsave(file_path, image)

    def report(self, file_path, norm='hist_raw', report_type='xlsx'):
        if not self.is_finalized:
            print('Data collection needs to be finalized to generate a report.')
            return None
        wb = Workbook()
        ws = wb.create_sheet()
        ws.title = "Results"
        data = self.evaluate(norm=norm)
        row_offset = 3
        column_offset = 2
        for entry in data:
            if entry['value'] is float:
                ws.cell(column=entry['position'][1]+column_offset, row=entry['position'][0]+row_offset,
                        value=round(entry['value'], 2))
            else:
                ws.cell(column=entry['position'][1] + column_offset, row=entry['position'][0] + row_offset,
                        value=round(entry['value'][0], 2))

        for column in range(sum(self.array_data['net_layout_x'])):
            ws.cell(column=column + column_offset, row=row_offset - 1, value=column+1)

        for row in range(sum(self.array_data['net_layout_y'])):
            ws.cell(column=column_offset - 1, row=row + row_offset, value=get_column_letter(row+1))

        wb.save(file_path)
