import unittest
from pathlib import Path
import hashlib
import shutil

from proMAD import Cutter
from proMAD.cut import scn_file

from helper import hash_file


class TestCutter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.c = Cutter()
        cls.cases = Path(__file__).absolute().resolve().parent / 'cases'

    @classmethod
    def tearDownClass(cls):
        del cls.c
        del cls.cases

    def test_cut(self):
        self.c.load_collection(self.cases / 'raw')
        self.c.shape = (4, 1)
        self.c.guess_positions()

        guess_cutes = [[0, 343, 687, 1031, 1375], [0, 1100]]
        self.assertListEqual(self.c.cut_positions, guess_cutes)

        self.c.cut_positions = [[210, 450, 695, 930, 1170], [95, 655]]
        self.c.names = [['A'], ['B'], ['C'], ['D']]

        self.c.save_images(self.cases / 'raw_processed')

        image_compare = [
            ['A/A_raw_00000.tif', '4f3e84c1e196c2e2f36afe8f594230fdc9f34bd937cded956a66dc203cee5a87'],
            ['A/A_raw_00001.tif', '2af889a40ba21eceab2031832d5c6a79075c953cc2ee33ec341ddede0c7e3b59'],
            ['A/A_raw_00002.tif', 'f2d2ed2a0dc13b225b580ae01c30595043d42d07b91cdc6a04c9f8e7df03937c'],
            ['A/A_raw_00003.tif', '3ec5dbe41a187f166ff1241ec408250d7e6f83783ff29b51876e57fd94579c0c'],
            ['A/A_raw_00004.tif', '3d21a0687f43bd4400521b89be6d1fb08ceac436d1342f63ca29b0c901140a4a'],
            ['B/B_raw_00000.tif', 'c5fb649418ebfe0289710f5d7ae88ee9dba4957862ffc40cf06197e8a97bc2b6'],
            ['B/B_raw_00001.tif', 'ac29325522323bad737676b8b35552c3a6a11158bf9aeafaf389850e0a844990'],
            ['B/B_raw_00002.tif', '1f129b335d54cec322f7db565215dda7442c583d11ba1d584adbc3ad22ec05fa'],
            ['B/B_raw_00003.tif', '1f6c563b5b96a0088b2950a3adc88bcfdde8964e88f7246c5d80f5c8caef579d'],
            ['B/B_raw_00004.tif', 'e8501cf98bc9b178a835b335684b14ed6923144933ca6fd1e21ea66bda83b461'],
            ['C/C_raw_00000.tif', 'f38f1cbaa6a644a47ca2648c5d0537deaa0b94c8d228309f7cacbef11a9d1651'],
            ['C/C_raw_00001.tif', '79ea81760e191df4999c3f058accf0a5db419e501c80b915dd3138667f93beac'],
            ['C/C_raw_00002.tif', 'ecb1443f8cfb216e7c76d1b307cd4bc2c8d52febe0d64f613a5bc6b03bd214d9'],
            ['C/C_raw_00003.tif', '91eb96fc43acbfe51f550c98f677bf0d5f24a92f666a0c33360235ec24a5d309'],
            ['C/C_raw_00004.tif', '84ff034e5ff9fb4f5e165f5ee786851bb9a00a0cb50090e8aea16492e9fcfb61'],
            ['D/D_raw_00000.tif', 'd5b465c6c3045e81b101504bcbaaa74d5722190d6829dbd38a88311de186463b'],
            ['D/D_raw_00001.tif', 'e8ecc513afc5c775f78bd4501e7953392bab32041d7f88b506c96e9bafa54d2a'],
            ['D/D_raw_00002.tif', '66df0a8c689b60b77bcbc6a13f6c4bce965091a12cabf7ce36f8254e3f3670a2'],
            ['D/D_raw_00003.tif', '3181e7d770c4fc984118aeab37b0a558664c2ba43d71fc7818d05d71fa83c132'],
            ['D/D_raw_00004.tif', '239586301ee883d5acb537553c58d37460df0a5057f2954f47a0f530b82262ee']
        ]

        for image_file, image_hash in image_compare:
            self.assertEqual(hash_file(self.cases / 'raw_processed' / image_file), image_hash)

        shutil.rmtree(self.cases / 'raw_processed')


class TestSCN(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cases = Path(__file__).absolute().resolve().parent / 'cases'

    @classmethod
    def tearDownClass(cls):
        del cls.cases

    def test_load_scn(self):
        test_file_path = self.cases / 'raw' / 'raw_00002.scn'
        data, meta = scn_file(test_file_path)
        meta_compare = {'exposure_time': 1882.524, 'image_date': '2018-12-05T18:45:49',
                        'pixel_mm': (7.638888888888889, 7.638888888888889)}

        self.assertEqual(meta, meta_compare)
        self.assertEqual(hashlib.sha3_256(data.tobytes()).hexdigest(),
                         '3ce238af1c5313d7d28829593ebe914722e4e5a2d0d3dadbc904d202b2efaa4b')

        binary_data, binary_meta = scn_file(test_file_path.read_bytes())
        self.assertEqual(meta, binary_meta)
        self.assertEqual(hashlib.sha3_256(binary_data.tobytes()).hexdigest(),
                         '3ce238af1c5313d7d28829593ebe914722e4e5a2d0d3dadbc904d202b2efaa4b')

        with test_file_path.open('rb') as fo:
            fo_data, fo_meta = scn_file(fo)
        self.assertEqual(meta, fo_meta)
        self.assertEqual(hashlib.sha3_256(fo_data.tobytes()).hexdigest(),
                         '3ce238af1c5313d7d28829593ebe914722e4e5a2d0d3dadbc904d202b2efaa4b')


if __name__ == '__main__':
    unittest.main()
