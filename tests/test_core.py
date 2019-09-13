import io
import shutil
import unittest
import warnings
from pathlib import Path

from proMAD import ArrayAnalyse

from helper import hash_file, hash_mem, hash_array
from helper import get_stdout


class TestARY022B(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.aa = ArrayAnalyse('ARY022B')
        cls.cases = Path(__file__).absolute().resolve().parent / 'cases'

    @classmethod
    def tearDownClass(cls):
        del cls.aa


class TestARY022BCollection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.aa = ArrayAnalyse('ARY022B')
        cls.cases = Path(__file__).absolute().resolve().parent / 'cases'
        cls.out_folder = cls.cases / 'testing_collection'
        cls.out_folder.mkdir(exist_ok=True, parents=True)
        cls.aa.load_collection(cls.cases / 'prepared', rotation=90)

        cls.compare_raw_bg = {'position': (0, 0), 'info': ['Reference Spots', 'N/A', 'RS'],
                              'value': [0.02435921, 0.02744768, 0.03043349, 0.03340112, 0.0363117,
                                        0.03901506, 0.04165865, 0.04422421, 0.04678247, 0.04915831,
                                        0.05159785, 0.05389296, 0.05621142, 0.05833439, 0.06049102,
                                        0.06279641, 0.06493878, 0.06679878, 0.06875748, 0.07078687,
                                        0.07264834, 0.07467496, 0.07659154, 0.07851298, 0.0802008,
                                        0.08157714, 0.08358657]}

    @classmethod
    def tearDownClass(cls):
        del cls.aa
        shutil.rmtree(cls.out_folder)


class LoadFromFile(unittest.TestCase):

    def test_load(self):
        cases = Path(__file__).absolute().resolve().parent / 'cases'
        aa = ArrayAnalyse.load(cases / 'save' / 'dump.tar')
        self.assertEqual(hash_array(aa.foregrounds),
                         '993f84db0f1211cfd9859571e9d1db8dc2443d179c5199c60fd3774057f27f0f')
        self.assertEqual(hash_array(aa.raw_images),
                         '46ee47b580e20c10cd9c50c598944929887f41a86f64ecca8071085ae5dde93c')


class TestArrays(unittest.TestCase):
    cases = Path(__file__).absolute().resolve().parent / 'cases' / 'array_test'

    def test_ARY007(self):
        aa = ArrayAnalyse('ARY007')
        aa.strict = False
        aa.load_image(self.cases / 'ARY007.tif')
        aa.finalize_collection()
        save_mem = io.BytesIO()
        aa.figure_alignment(file=save_mem)
        self.assertEqual(hash_mem(save_mem),
                         '991c54036fd3a11cf5a8763c0a8cd8fe0638415df4bc84392bb15da6daeacc78')

    def test_ARY015(self):
        aa = ArrayAnalyse('ARY015')
        aa.strict = False
        aa.load_image(self.cases / 'ARY015.tif')
        aa.finalize_collection()
        save_mem = io.BytesIO()
        aa.figure_alignment(file=save_mem)
        self.assertEqual(hash_mem(save_mem),
                         '95ac01b73c0714c2d39571deadc8e569e2fab5ca17e530df5086890d86c1b390')

    def test_ARY028(self):
        aa = ArrayAnalyse('ARY028')
        aa.strict = False
        aa.load_image(self.cases / 'ARY028.tif')
        aa.finalize_collection()
        save_mem = io.BytesIO()
        aa.figure_alignment(file=save_mem)
        self.assertEqual(hash_mem(save_mem),
                         '3573ce7aad79c56d098c99bb8523def6fc1f670068c72624af4d75bfa5abe7fb')


class LoadImagesWrongRotation(TestARY022B):
    def test_load_image_wrong_rotation(self):
        self.aa.debug = True
        self.aa.load_image(self.cases / 'prepared/prepared_00020.tif')
        output = get_stdout(self.aa.load_image, args=[self.cases / 'prepared/prepared_00025.tif', ])
        self.aa.load_image(self.cases / 'prepared/prepared_00030.tif')
        self.aa.finalize_collection()
        self.assertIn('❌', output)
        assert self.aa.is_finalized is False


class LoadImages(TestARY022B):
    def test_finalize_warnings(self):

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            self.aa.get_spot('A1')
            self.aa.evaluate()
            self.aa.save(None)

            self.assertEqual(len(w), 3)
            for i in range(3):
                self.assertEqual(w[i].category, RuntimeWarning)
                self.assertIn("Data collection needs to be finalized", str(w[i].message))

    def test_finalize_warnings_figures(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            self.aa.figure_alignment()
            self.aa.figure_reaction_fit()
            self.aa.figure_contact_sheet()
            self.aa.figure_contact_sheet_spot('A1')

            self.assertEqual(len(w), 4)
            for i in range(4):
                self.assertEqual(w[i].category, RuntimeWarning)
                self.assertIn("Data collection needs to be finalized to generate the figure.", str(w[i].message))

    def test_load_image(self):
        self.aa.debug = True
        self.aa.load_image(self.cases / 'prepared/prepared_00020.tif', rotation=90)
        load_output = get_stdout(self.aa.load_image, args=[self.cases / 'prepared/prepared_00025.tif'],
                                 kwargs=dict(rotation=90))
        self.aa.load_image(self.cases / 'prepared/prepared_00025.tif', rotation=90)

        content = (self.cases / 'prepared/prepared_00030.tif').read_bytes()
        mem_im = io.BytesIO(content)
        self.aa.load_image(mem_im, rotation=90, suffix='.tif')

        with (self.cases / 'prepared/prepared_00032.tif').open('rb') as fo:
            self.aa.load_image(fo, rotation=90, suffix='.tif')

        self.aa.finalize_collection()

        self.assertEqual(self.aa.is_finalized, True)
        self.assertIn('✅', load_output)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            self.aa.finalize_collection()

            self.assertEqual(len(w), 1)
            self.assertEqual(w[-1].category, RuntimeWarning)
            self.assertIn("Data is already finalized.", str(w[-1].message))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            self.aa.load_image(self.cases/'prepared/prepared_00025.tif', rotation=90)

            self.assertEqual(len(w), 1)
            self.assertEqual(w[-1].category, RuntimeWarning)
            self.assertIn("Data is already finalized.", str(w[-1].message))
        self.aa.reset_collection()
        self.assertEqual(self.aa.source_images, [])


class LoadCollection(TestARY022BCollection):
    def test_basics(self):
        self.assertEqual(self.aa.is_finalized, True)
        self.assertEqual(self.aa.raw_images.shape[2], 27)

    def test_bg_parameters(self):
        self.assertEqual(round(self.aa.bg_parameters[0], 4), 0.0144)
        self.assertEqual(round(self.aa.bg_parameters[6], 4), 0.0193)
        self.assertEqual(round(self.aa.bg_parameters[12], 4), 0.024)
        self.assertEqual(round(self.aa.bg_parameters[18], 4), 0.0288)

    def test_evaluate(self):
        data = self.aa.evaluate()
        compare_11 = {'position': [0, 11], 'info': ['Angiopoietin-2', '285', 'Ang-2, ANGPT2'],
                      'value': (0.9603999914272608, 8.169726417758122e-05, 0.9997357674683341)}
        compare_206 = {'position': [9, 0], 'info': ['Reference Spots', 'N/A', 'RS'],
                       'value': (3.7074421357106, -0.017640873154213907, 0.9996119545117066)}
        for i in range(3):
            self.assertAlmostEqual(data[11]['value'][i], compare_11['value'][i], places=7)
            self.assertAlmostEqual(data[206]['value'][i], compare_206['value'][i], places=7)

    def test_evaluate_double_spot(self):
        data = self.aa.evaluate(double_spot=True)
        compare_11 = {'position': [0, 22], 'info': ['Reference Spots', None, 'RS'], 'value': 3.2279057930907777}
        compare_100 = {'position': [8, 16], 'info': ['TNF-α', '7124', 'TNFSF1A'], 'value': 0.9304401202054406}

        self.assertAlmostEqual(data[11]['value'], compare_11['value'], places=7)
        self.assertAlmostEqual(data[100]['value'], compare_100['value'], places=7)

    def test_evaluate_just_value(self):
        data = self.aa.evaluate(double_spot=True, just_value=True)
        self.assertAlmostEqual(data[11], 3.2279057930907777, places=7)
        self.assertAlmostEqual(data[100], 0.9304401202054406, places=7)

        data = self.aa.evaluate(just_value=True, norm='raw')
        self.assertAlmostEqual(data[11], 0.03318000141411234, places=7)
        self.assertAlmostEqual(data[206], 0.1101721369996391, places=7)

    def test_evaluate_spot(self):
        compare_normal = [{'position': (0, 0), 'info': ['Reference Spots', None, 'RS'],
                           'value': (3.9027168252815465, -0.014834386535689317, 0.9976046377512487)}]
        compare_just = [(1.15796238349657, -0.001719920301677249, 0.9997724465008571)]
        compare_double = [{'position': (0, 0), 'info': ['Reference Spots', None, 'RS'],
                           'value': (3.9027168252815465, -0.014834386535689317, 0.9976046377512487)},
                          {'position': (0, 1), 'info': ['Reference Spots', None, 'RS'],
                           'value': (3.924585455094854, -0.014271288247623928, 0.9966464736583901)}]
        compare_double_just = [(1.15796238349657, -0.001719920301677249, 0.9997724465008571),
                               (0.9686154279191376, 0.00011914453689753496, 0.9998588407500172)]

        normal = self.aa.evaluate('A1')
        just = self.aa.evaluate('C7', just_value=True)
        double = self.aa.evaluate('A1', double_spot=True)
        double_just = self.aa.evaluate('C7', double_spot=True, just_value=True)

        self.assertEqual(normal[0]['position'], compare_normal[0]['position'])
        self.assertAlmostEquals(normal[0]['value'][0], compare_normal[0]['value'][0], delta=1E-13)

        self.assertAlmostEquals(just[0][0], compare_just[0][0], delta=1E-13)

        self.assertEqual(double[0]['position'], compare_double[0]['position'])
        self.assertAlmostEquals(double[0]['value'][0], compare_double[0]['value'][0], delta=1E-13)
        self.assertEqual(double[1]['position'], compare_double[1]['position'])
        self.assertAlmostEquals(double[1]['value'][0], compare_double[1]['value'][0], delta=1E-13)

        self.assertAlmostEquals(double_just[0][0], compare_double_just[0][0], delta=1E-13)
        self.assertAlmostEquals(double_just[1][0], compare_double_just[1][0], delta=1E-13)

    def test_get_spot_empty(self):
        self.assertIsNone(self.aa.get_spot('Ä5'))
        self.assertIsNone(self.aa.get_spot((3, 4, 5)))

    def test_reac(self):
        compare_reac = {'position': (0, 0), 'info': ['Reference Spots', 'N/A', 'RS'],
                        'value': (1.9081492770754633e-10, 2.54922817e-13)}

        data_reac = self.aa.evaluate('A1', norm='reac')
        self.assertEqual(data_reac[0]['position'], compare_reac['position'])
        self.assertAlmostEqual(data_reac[0]['value'][0], compare_reac['value'][0], delta=1E-13)
        self.assertAlmostEqual(data_reac[0]['value'][1], compare_reac['value'][1], delta=1E-15)

    def test_raw_bg(self):

        data_raw_bg = self.aa.evaluate('A1', norm='raw_bg')
        for i in range(len(self.compare_raw_bg['value'])):
            self.assertAlmostEqual(data_raw_bg[0]['value'][i], self.compare_raw_bg['value'][i], delta=1E-6)

    def test_local_bg(self):
        compare_local_bg = {'position': (0, 0), 'info': ['Reference Spots', 'N/A', 'RS'],
                            'value': (2.368006415601608, 1.6560182848362146)}
        data_local_bg = self.aa.evaluate('A1', norm='local_bg')
        self.assertAlmostEqual(data_local_bg[0]['value'][0], compare_local_bg['value'][0], delta=1E-7)
        self.assertAlmostEqual(data_local_bg[0]['value'][1], compare_local_bg['value'][1], delta=1E-7)

    def test_histogram(self):
        compare_hist_fg = {'position': (0, 0), 'info': ['Reference Spots', 'N/A', 'RS'],
                           'value': (2.3353301745304957, -0.009478602375160038, 0.9913012557472861)}
        data_hist_fg = self.aa.evaluate('A1', norm='hist_fg')
        self.assertAlmostEqual(data_hist_fg[0]['value'][0], compare_hist_fg['value'][0], delta=1E-7)
        self.assertAlmostEqual(data_hist_fg[0]['value'][2], compare_hist_fg['value'][2], delta=1E-7)

        compare_hist_raw = {'position': (0, 0), 'info': ['Reference Spots', 'N/A', 'RS'],
                            'value': (3.9027168252815465, -0.014834386535689317, 0.9976046377512487)}
        data_hist_raw = self.aa.evaluate('A1', norm='hist_raw')
        self.assertAlmostEqual(data_hist_raw[0]['value'][0], compare_hist_raw['value'][0], delta=1E-7)
        self.assertAlmostEqual(data_hist_raw[0]['value'][2], compare_hist_raw['value'][2], delta=1E-7)

    def test_raw(self):
        data_raw = self.aa.evaluate('A1', norm='raw')
        compare_raw = {'position': (0, 0), 'info': ['Reference Spots', 'N/A', 'RS'],
                       'value': [0.03877323, 0.04270195, 0.04650554, 0.05025241, 0.05395237,
                                 0.05749112, 0.06095504, 0.06428393, 0.06763055, 0.0708708,
                                 0.07401852, 0.07715833, 0.08024223, 0.08319256, 0.08612504,
                                 0.08909966, 0.09200329, 0.09480581, 0.09751308, 0.10020344,
                                 0.10285004, 0.10553497, 0.1081548, 0.11076575, 0.11330819,
                                 0.11559258, 0.11806939]}

        for i in range(len(self.compare_raw_bg['value'])):
            self.assertAlmostEqual(data_raw[0]['value'][i], compare_raw['value'][i], delta=1E-7)

    def test_figure_contact_sheet(self):
        save_mem = io.BytesIO()
        self.aa.figure_contact_sheet(file=save_mem)
        self.assertEqual(hash_mem(save_mem),
                         'a20deb0027988e5fbd645278a497f965390d5ad2e1555dc15600a24a1fc54a54')

        self.aa.figure_contact_sheet(file=self.out_folder / 'contact_sheet.png', max_size=500)
        self.assertEqual(hash_file(self.out_folder / 'contact_sheet.png'),
                         '9c348336b772c1d68e81ba558d2174c7b52d93ca9d7d10364168693cd9288f7c')

    def test_figure_alignment(self):
        save_mem = io.BytesIO()
        self.aa.figure_alignment(file=save_mem)
        self.assertEqual(hash_mem(save_mem),
                         '87d320e4b4ee5be2bc12f797661def7823217f5bcb6f9454637ccb23d2ec0f6d')

        self.aa.figure_alignment(file=self.out_folder / 'alignment.jpg', max_size=500)
        self.assertEqual(hash_file(self.out_folder / 'alignment.jpg'),
                         'abc22b8192d0e2d5d10e5631ff50e827da29577c6567a3ee309701e90480cdf2')

    def test_figure_contact_sheet_spot(self):
        save_mem = io.BytesIO()
        self.aa.figure_contact_sheet_spot(file=save_mem, position='A1')
        self.assertEqual(hash_mem(save_mem, skip=0),
                         '56c6c69b3de8f0677709e16ac7ac85f0883716713900b5f20ba200aba77ec25e')

        self.aa.figure_contact_sheet_spot(file=self.out_folder / 'contact_sheet_spot.jpg', max_size=150, position='A1')
        self.assertEqual(hash_file(self.out_folder / 'contact_sheet_spot.jpg', skip=0),
                         '10ddfc8494dda2a3a0e29ed0f1fd996347a10d2b806d212358e390ecc26ef450')

    def test_figure_reaction_fit(self):
        self.aa.figure_reaction_fit(file=self.out_folder / 'reaction_fit.png')
        self.assertEqual(hash_file(self.out_folder / 'reaction_fit.png'),
                         '12e95aac956289f7f24b46e5a1ee3e4149aa3a869602bdeff7fa407bab565bab')

    def test_save(self):
        save_mem = io.BytesIO()
        self.aa.save(file=save_mem)
        self.assertEqual(hash_mem(save_mem),
                         'eff29a8d43c76e929d09e90dc7f1cbf2679d5ea73fc5762b3d71ff65c4a29f54')

        self.aa.save(self.out_folder / 'dump.tar')
        self.assertEqual(hash_file(self.out_folder / 'dump.tar'),
                         'eff29a8d43c76e929d09e90dc7f1cbf2679d5ea73fc5762b3d71ff65c4a29f54')


if __name__ == '__main__':
    unittest.main()
