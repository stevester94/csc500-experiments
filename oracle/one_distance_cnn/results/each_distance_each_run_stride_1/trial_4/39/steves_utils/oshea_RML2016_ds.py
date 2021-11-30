#! /usr/bin/env python3

import pickle
from unittest.case import SkipTest
import numpy as np
import torch
from torch._C import dtype


class OShea_RML2016_DS(torch.utils.data.Dataset):
    """
    snrs: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2]

    modulation_mapping = {
        'AM-DSB': 0,
        'QPSK'  : 1,
        'BPSK'  : 2,
        'QAM64' : 3,
        'CPFSK' : 4,
        '8PSK'  : 5,
        'WBFM'  : 6,
        'GFSK'  : 7,
        'AM-SSB': 8,
        'QAM16' : 9,
        'PAM4'  : 10,
    }
    """
    def __init__(self, path:str="/mnt/wd500GB/CSC500/csc500-super-repo/datasets/RML2016.10a_dict.pkl", 
        snrs_to_get:list=[-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]) -> None:
        """
        args:
            domain_configs: {
                "domain_index":int,
                "min_rotation_degrees":float,
                "max_rotation_degrees":float,
                "num_examples_in_domain":int,
            }
        """
        super().__init__()

        self.modulation_mapping = {
            'AM-DSB': 0,
            'QPSK'  : 1,
            'BPSK'  : 2,
            'QAM64' : 3,
            'CPFSK' : 4,
            '8PSK'  : 5,
            'WBFM'  : 6,
            'GFSK'  : 7,
            'AM-SSB': 8,
            'QAM16' : 9,
            'PAM4'  : 10,
        }

        Xd = pickle.load(open(path,'rb'), encoding="latin1")
        self.Xd = Xd
        snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])       

        data = []  
        lbl = []
        for mod in mods:
            for snr in snrs:
                if snr in snrs_to_get:
                    for x in Xd[(mod,snr)]:
                        data.append(
                            (
                                x.astype(np.single),
                                self.modulation_to_int(mod),
                                np.array([snr], dtype=np.int32)
                            )
                        )

        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def modulation_to_int(self, modulation:str):
        return self.modulation_mapping[modulation]

    # 
    @classmethod
    def get_snrs(cls):
        return [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        


if __name__ == "__main__":
    import unittest
    import random
    import itertools

    LEN_SEQUENCE = 100000
    MAX_CACHE_SIZE = 1000

    class test_OShea_RML2016_DS(unittest.TestCase):
        @classmethod
        def setUpClass(self) -> None:
            # self.ds = OShea_RML2016_DS()
            pass

        @unittest.SkipTest
        def test_snrs(self):
            snrs_to_test = [
                [-18, -12, -6, 0, 6, 12, 18],
                [2, 4, 8, 10, -20, 14, 16, -16, -14, -10, -8, -4, -2],
                [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2]
            ]

            for s in snrs_to_test:
                s.sort()
                unique_snrs = set()
                unique_mods = set()
                ds = OShea_RML2016_DS(snrs_to_get=s)

                for ex in ds:
                    unique_snrs.add(ex[2][0])
                    unique_mods.add(ex[1])

                print(unique_snrs)
                print(unique_mods)

                unique_snrs = list(unique_snrs)
                unique_snrs.sort()


                self.assertEqual(len(unique_mods), 11)
                self.assertEqual(s, unique_snrs)

    class test_normalization(unittest.TestCase):
        @SkipTest
        def test_lazy_normalization(self):
            from steves_utils.utils_v2 import normalize_val, denormalize_val
            from steves_utils.lazy_map import Lazy_Map

            snrs_to_test = [
                [-18, -12, -6, 0, 6, 12, 18],
                [2, 4, 8, 10, -20, 14, 16, -16, -14, -10, -8, -4, -2],
                [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2]
            ]

            for s in snrs_to_test:
                s.sort()
                unique_snrs = set()
                unique_mods = set()
                ds = OShea_RML2016_DS(snrs_to_get=s)

                mapped_ds = Lazy_Map(ds, 
                    lambda ex: (
                        ex[0], ex[1], ex[2], 
                        np.array([normalize_val(min(s), max(s), ex[2][0])], dtype=np.single)
                    )
                )

                for ex in mapped_ds:
                    self.assertAlmostEqual(ex[2][0], denormalize_val(min(s), max(s), ex[3][0]), places=5)


        def test_eyeball_the_snrs(self):
            from steves_utils.utils_v2 import normalize_val, denormalize_val
            from steves_utils.lazy_map import Lazy_Map

            MIN = -20
            MAX = 18

            normalized_original_snrs = list(map(lambda s: normalize_val(MIN, MAX, s), OShea_RML2016_DS.get_snrs()))

            snrs_to_test = [
                [-20, -18, -16],
                [ -14, -12, ],
                [-10, -8, ],
                [ -6, -4, ],
                [-2, 0, ],
                [2, 4, 6, 8, 10, 12, 14, 16, 18],
            ]


            unique_normalized_snrs = set()
            for s in snrs_to_test:
                s.sort()

                ds = OShea_RML2016_DS(snrs_to_get=s)

                mapped_ds = Lazy_Map(ds, 
                    lambda ex: (
                        ex[0], ex[1], ex[2], 
                        np.array([normalize_val(MIN, MAX, ex[2][0])], dtype=np.single)
                    )
                )

                for ex in mapped_ds:
                    unique_normalized_snrs.add(ex[3][0])
            
            unique_normalized_snrs = list(unique_normalized_snrs)
            unique_normalized_snrs.sort()

            print(unique_normalized_snrs)
            print(normalized_original_snrs)

            self.assertEqual(len(unique_normalized_snrs), len(OShea_RML2016_DS.get_snrs()))

            for unique, nrml_original in zip(unique_normalized_snrs, normalized_original_snrs):
                self.assertAlmostEqual(unique, nrml_original, places=5)
            
            for unique, nrml_original in zip(unique_normalized_snrs, OShea_RML2016_DS.get_snrs()):
                self.assertAlmostEqual(
                    denormalize_val(MIN, MAX,unique), nrml_original, places=5
                )

            unique_normalized_snrs



    unittest.main()
    
