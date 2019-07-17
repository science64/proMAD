from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xmltodict
from skimage import io


def scn_file(file_path, out_path=None):
    """
    Read image and metadata from .scn (Image Lab) files.
    If a out_path is given the extracted data is stored as .tif file instead.

    Parameters
    ----------
    file_path: Path or str
    out_path: Path or str

    Returns
    -------
    image: array_like
        image array (16bit uint)
    metadata: dict

    """
    file_path = Path(file_path)
    bin_data = file_path.read_bytes()
    split = bin_data.partition(b'boundary="')[2].partition(b'"')[0]
    x = x_mm = 0
    y = y_mm = 0
    exposure_time = None
    image_date = None
    data = None
    for part in bin_data.split(b"--"+split):
        if b'boundary="' in part:
            if part.partition(b'boundary="')[2].partition(b'"')[0] != split:
                second_split = part.partition(b'boundary="')[2].partition(b'"')[0]
                for sub_part in part.split(b"--"+second_split):
                    if b'<!DOCTYPE XML>' in sub_part:
                        xml = (b'<!DOCTYPE XML>' + sub_part.partition(b'<!DOCTYPE XML>')[2]).decode().strip()
                        xml_dict = xmltodict.parse(xml, process_namespaces=True)
                        x = int(xml_dict['root']['size_pix']['@height'])
                        y = int(xml_dict['root']['size_pix']['@width'])
                        x_mm = x/float(xml_dict['root']['size_mm']['@height'])
                        y_mm = y/float(xml_dict['root']['size_mm']['@width'])
                        exposure_time = float(xml_dict['root']['scan_attributes']['exposure_time']['@value'])
                        image_date = xml_dict['root']['scan_attributes']['image_date']['@value']
                    elif b'Content-Description: ImageData' in sub_part:
                        length = int(part.partition(b'Content-Length:')[2].partition(b'\r\n')[0].decode())
                        data = np.frombuffer(sub_part[-length:], dtype='<u2')
                break

    if data is not None:
        data = data.reshape(x, y)
        meta_data = {'exposure_time': exposure_time, 'image_date': image_date, 'pixel_mm': (x_mm, y_mm)}
        if out_path:
            io.imsave(out_path, data, metadata=meta_data)
            return None, None
        else:
            return data, meta_data
    else:
        return None, None


class Cutter(object):
    """
    Cut a stack of images into sub images.
    """
    def __init__(self):
        self.shape = np.zeros(2, dtype='uint8')
        self.cut_positions = None
        self.rotation = 0
        self.names = None
        self.images = []
        pass

    @staticmethod
    def _ask(question, type_func=int):
        """
        Helper to gather answers and convert them with the type_func.

        Parameters
        ----------
        question: str
        type_func: function

        """
        while True:
            raw_input = None
            try:
                raw_input = input(f'{question}:').strip()
                value = type_func(raw_input)
                break
            except ValueError:
                print(f'Please try again - {raw_input} is not a valid {type_func.__name__}')
        return value

    @staticmethod
    def _make_ordinal(n):
        """

        Parameters
        ----------
        n: int

        Returns
        -------
        str

        """
        n = int(n)
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
        if 11 <= (n % 100) <= 13:
            suffix = 'th'
        return str(n) + suffix

    def load_image(self, file_path):
        """
        Load a single image file into the collection.


        Parameters
        ----------
        file_path: Path or str

        """
        file_path = Path(file_path)
        print(file_path)
        print(f'Load image from "{file_path}"')
        if file_path.suffix == '.scn':
            source_image, meta_data = scn_file(file_path.absolute())
        else:
            source_image = io.imread(file_path.absolute(), plugin='imageio')
            meta_data = None
        self.images.append(dict(name=file_path.name, data=source_image, meta_data=meta_data))

    def load_collection(self, data_input):
        """
        Load multiple files into the collection.
        Includes just files that are directly in the folders listed in *data_input* and does not search subfolders.

        Parameters
        ----------
        data_input: Path or str or list(Path) or list(str)
            a single input folder or list of folders / files

        """
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

        for file_path in input_file_paths:
            if file_path.name[0] == '.':
                continue
            self.load_image(file_path)

    def set_shape(self):
        """
        Set the arrangement of the arrays interactively.
        """
        x = self._ask('Number of arrays next to each other (X)')
        y = self._ask('Number of arrays on top of each other (Y)')
        self.shape = (x, y)

    def set_names(self):
        """
        Set the names of the arrays interactively.
        """
        self.names = []
        print(f'Set names for file path generation. (leave empty or type None to skip)')
        for nx in range(self.shape[0] + 1):
            print(f'Names in {self._make_ordinal(nx + 1)} column')
            self.names.append([])
            for ny in range(self.shape[1] + 1):
                self.names[nx].append(self._ask(f'Name for A{nx}_{ny}', type_func=str))

    def set_positions(self):
        """
        Set the cut positions interactively.
        """
        self.cut_positions = [[], []]
        print('Set cut positions (boarder included)')
        for n in range(self.shape[0]+1):
            self.cut_positions[0].append(
                self._ask(f"Distance from left edge in pixels for {self._make_ordinal(n + 1)} x cut"))

        for n in range(self.shape[1]+1):
            self.cut_positions[1].append(
                self._ask(f"Distance from top edge in pixels for {self._make_ordinal(n + 1)} y cut"))

    def guess_positions(self):
        """
        Generate a initial cut pattern with a simple guess.
        """
        ishape = self.images[-1]['data'].shape
        x_space = [0]+[ishape[1]/self.shape[0]] * (self.shape[0])
        x_list = [int(x) for x in np.cumsum(x_space)]
        y_space = [0] + [ishape[0] / self.shape[1]] * (self.shape[1])
        y_list = [int(y) for y in np.cumsum(y_space)]
        print('The following coordinates are guessed for equal distribution')
        print([x_list, y_list])
        self.cut_positions = [x_list, y_list]

    def preview(self, example=-1):
        """
        Show a preview of the set cut pattern through matplotlib.

        Parameters
        ----------
        example: int
            collection index of the image to be used as example

        """
        preview_data = np.copy(self.images[example]['data'])
        plt.imshow(preview_data, cmap='nipy_spectral')
        if self.cut_positions is not None:
            plt.vlines(self.cut_positions[0], 0, preview_data.shape[0], color='white')
            plt.hlines(self.cut_positions[1], 0, preview_data.shape[1], color='white')

            for nx in range(self.shape[0]):
                for ny in range(self.shape[1]):
                    if self.names is None:
                        p_name = f'A{nx}_{ny}'
                    else:
                        p_name = self.names[nx][ny]
                    if (p_name != 'None') and (p_name is not None) and p_name:
                        plt.text((self.cut_positions[0][nx]+self.cut_positions[0][nx+1])*0.5,
                                 (self.cut_positions[1][ny]+self.cut_positions[1][ny+1])*0.5, p_name,
                                 ha='center', va='center', wrap=True, color='black',
                                 bbox=dict(facecolor='white', alpha=0.7))
        plt.xlim((0, preview_data.shape[1]))
        plt.ylim((preview_data.shape[0], 0))
        plt.show()

    def save_images(self, out_folder_parent):
        """
        Write cut image stacks in named subfolder under *out_folder*.
        If possible the file type of the original files is preserved.

        Parameters
        ----------
        out_folder_parent: Path or str
            Output folder is created if it does not exist
        """
        out_folder_parent = Path(out_folder_parent)
        out_folder_parent.mkdir(parents=True, exist_ok=True)
        for im_raw in self.images:
            for nx in range(self.shape[0]):
                for ny in range(self.shape[1]):
                    if self.names is None:
                        p_name = f'A{nx},{ny}'
                    else:
                        p_name = self.names[nx][ny]
                    if (p_name != 'None') and (p_name is not None) and p_name:
                        left = self.cut_positions[0][nx]
                        right = self.cut_positions[0][nx + 1]
                        top = self.cut_positions[1][ny]
                        bottom = self.cut_positions[1][ny+1]
                        im_part = im_raw['data'][top:bottom, left:right]
                        meta_data = im_raw['meta_data']
                        if meta_data is not None:
                            meta_data['system_name'] = p_name
                        name = Path(im_raw["name"])
                        suffix = name.suffix
                        name = str(name).partition(suffix)[0]
                        if suffix == ".scn":
                            suffix = '.tif'
                        out_folder = out_folder_parent / f'{p_name}'
                        out_folder.mkdir(exist_ok=True, parents=True)
                        save_path = str((out_folder/f'{p_name}_{name}{suffix}').absolute())
                        print(f'Save image under "{save_path}"')
                        io.imsave(save_path, im_part, metadata=meta_data)
