import unittest
from pathlib import Path
import io
import sys
import warnings

from proMAD import ArrayAnalyse


def get_stdout(func, args=[], kwargs={}):
    old_state = sys.stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output
    func(*args, **kwargs)
    sys.stdout = old_state
    return captured_output.getvalue()


class TestARY022B(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.aa = ArrayAnalyse('ARY022B')
        cls.cases = Path(__file__).absolute().resolve().parent / 'cases'

    @classmethod
    def tearDownClass(cls):
        del cls.aa


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
    def test_load_image(self):
        self.aa.debug = True
        self.aa.load_image(self.cases / 'prepared/prepared_00020.tif', rotation=90)
        load_output = get_stdout(self.aa.load_image, args=[self.cases / 'prepared/prepared_00025.tif'],
                                 kwargs=dict(rotation=90))
        self.aa.load_image(self.cases / 'prepared/prepared_00030.tif', rotation=90)
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


class LoadCollection(TestARY022B):
    def test_collection(self):
        self.aa.load_collection(self.cases / 'prepared', rotation=90)

        self.assertEqual(self.aa.is_finalized, True)
        self.assertEqual(self.aa.raw_images.shape[2], 27)
        self.assertEqual(round(self.aa.bg_parameters[0], 4), 0.0144)
        self.assertEqual(round(self.aa.bg_parameters[6], 4), 0.0193)
        self.assertEqual(round(self.aa.bg_parameters[12], 4), 0.0241)
        self.assertEqual(round(self.aa.bg_parameters[18], 4), 0.0289)

        data = self.aa.evaluate()
        compare_11 = {'position': [0, 11], 'info': ['Angiopoietin-2', '285', 'Ang-2, ANGPT2'],
                      'value': (0.9605539141305303, 0.0001096693800271345, 0.9997509967996165)}
        compare_206 = {'position': [9, 0], 'info': ['Reference Spots', 'N/A', 'RS'],
                       'value': (3.830355677396986, -0.018064265580287217, 0.9996281465751826)}

        for i in range(3):
            self.assertAlmostEqual(data[11]['value'][i], compare_11['value'][i], places=7)
            self.assertAlmostEqual(data[206]['value'][i], compare_206['value'][i], places=7)

        compare_reac = {'position': (0, 0), 'info': ['Reference Spots', 'N/A', 'RS'],
                        'value': (1.9247632903443254e-10, 2.48517273e-13)}
        data_reac = self.aa.evaluate('A1', norm='reac')
        self.assertEqual(data_reac[0]['position'], compare_reac['position'])
        self.assertAlmostEquals(data_reac[0]['value'][0], compare_reac['value'][0], delta=1E-13)
        self.assertAlmostEquals(data_reac[0]['value'][1], compare_reac['value'][1], delta=1E-15)

        compare_raw_bg = {'position': (0, 0), 'info': ['Reference Spots', 'N/A', 'RS'],
                          'value': [0.02574705, 0.02901507, 0.03217805, 0.03527297, 0.03834527,
                                    0.04119925, 0.04401591, 0.04664042, 0.04939001, 0.05199787,
                                    0.05443238, 0.05692406, 0.05933491, 0.06165034, 0.06388225,
                                    0.06630212, 0.06840911, 0.07064766, 0.07251287, 0.07481071,
                                    0.07662289, 0.07871264, 0.0808493, 0.08266791, 0.08466276,
                                    0.08624916, 0.08800924]}
        data_raw_bg = self.aa.evaluate('A1', norm='raw_bg')
        for i in range(len(compare_raw_bg['value'])):
            self.assertAlmostEquals(data_raw_bg[0]['value'][i], compare_raw_bg['value'][i], delta=1E-6)

        compare_local_bg = {'position': (0, 0), 'info': ['Reference Spots', 'N/A', 'RS'],
                            'value': (2.4738030480525297, 1.6897023165546854)}
        data_local_bg = self.aa.evaluate('A1', norm='local_bg')
        self.assertAlmostEquals(data_local_bg[0]['value'][0], compare_local_bg['value'][0], delta=1E-7)
        self.assertAlmostEquals(data_local_bg[0]['value'][1], compare_local_bg['value'][1], delta=1E-7)

        compare_hist_fg = {'position': (0, 0), 'info': ['Reference Spots', 'N/A', 'RS'],
                           'value': (2.4721049350532738, -0.009783709382946766, 0.9883517022149169)}
        data_hist_fg = self.aa.evaluate('A1', norm='hist_fg')
        self.assertAlmostEquals(data_hist_fg[0]['value'][0], compare_hist_fg['value'][0], delta=1E-7)
        self.assertAlmostEquals(data_hist_fg[0]['value'][2], compare_hist_fg['value'][2], delta=1E-7)

        compare_hist_raw = {'position': (0, 0), 'info': ['Reference Spots', 'N/A', 'RS'],
                            'value': (4.035250359096405, -0.015125134789062036, 0.9972753160067888)}
        data_hist_raw = self.aa.evaluate('A1', norm='hist_raw')
        self.assertAlmostEquals(data_hist_raw[0]['value'][0], compare_hist_raw['value'][0], delta=1E-7)
        self.assertAlmostEquals(data_hist_raw[0]['value'][2], compare_hist_raw['value'][2], delta=1E-7)

        data_raw = self.aa.evaluate('A1', norm='raw')
        compare_raw = {'position': (0, 0), 'info': ['Reference Spots', 'N/A', 'RS'],
                       'value': [0.04016551, 0.04427588, 0.04823829, 0.05215235, 0.05601633,
                                 0.059707, 0.06331853, 0.06679294, 0.07028141, 0.07365917,
                                 0.07693287, 0.08019991, 0.08341115, 0.08647959, 0.08952598,
                                 0.09261347, 0.09563714, 0.0985555, 0.10136975, 0.10423676,
                                 0.10698961, 0.10977475, 0.11250173, 0.11521494, 0.11785644,
                                 0.12022015, 0.1227878]}
        for i in range(len(compare_raw_bg['value'])):
            self.assertAlmostEquals(data_raw[0]['value'][i], compare_raw['value'][i], delta=1E-7)

        self.aa.reset_collection()
        self.assertEqual(self.aa.source_images, [])


if __name__ == '__main__':
    unittest.main()
