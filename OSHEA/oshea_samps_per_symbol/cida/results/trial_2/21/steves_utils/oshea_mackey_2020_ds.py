#! /usr/bin/env python3

import pickle
from unittest.case import SkipTest
import numpy as np
import torch
from torch._C import dtype


class OShea_Mackey_2020_DS(torch.utils.data.Dataset):
    """
    snrs: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2]

    modulation_mapping = {
            'QPSK'  : 0,
            'BPSK'  : 1,
            'QAM64' : 2,
            'CPFSK' : 3,
            '8PSK'  : 4,
            'GFSK'  : 5,
            'QAM16' : 6,
            'PAM4'  : 7,
    }
    """
    def __init__(self, 
        path:str="/mnt/wd500GB/CSC500/csc500-super-repo/datasets/OShea_Mackey_2020_dict.dat",
        snrs_to_get:list=[-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
        samples_per_symbol_to_get=[2,4,6,8,10,12,14,16,18,20]
    ) -> None:

        super().__init__()

        self.modulation_mapping = {
            'QPSK'  : 0,
            'BPSK'  : 1,
            'QAM64' : 2,
            'CPFSK' : 3,
            '8PSK'  : 4,
            'GFSK'  : 5,
            'QAM16' : 6,
            'PAM4'  : 7,
        }

        self.Xd = pickle.load(open(path,'rb'), encoding="latin1")

        all_mods = set([k[0] for k in self.Xd.keys()])
        all_snrs = set([k[1] for k in self.Xd.keys()])
        all_samps_per_symbol = set([k[2] for k in self.Xd.keys()])

        if set(self.modulation_mapping.keys()) != all_mods:
            print(set(self.modulation_mapping.values()))
            print(all_mods)

            raise Exception("Modulations from dataset do not match the expected modulations")


        if not set(snrs_to_get).issubset(all_snrs):
            print("snrs_to_get is not a subset of available snrs")
            print("Requested:", snrs_to_get)
            print("Available:", all_snrs)
            raise Exception("snrs_to_get is not a subset of available snrs")


        if not set(samples_per_symbol_to_get).issubset(all_samps_per_symbol):
            print("samples_per_symbol_to_get is not a subset of available samples_per_symbol")
            print("Requested:", samples_per_symbol_to_get)
            print("Available:", all_samps_per_symbol)
            raise Exception("samples_per_symbol_to_get is not a subset of available samples_per_symbol")

        # Need determinism on this fucker
        all_mods = list(all_mods)
        all_mods.sort()

        data = []  
        for mod in all_mods:
            for snr in snrs_to_get:
                for sps in samples_per_symbol_to_get:
                    for x in self.Xd[(mod,snr,sps)]:
                        data.append(
                            {
                                "IQ": x.astype(np.single),
                                "modulation": self.modulation_to_int(mod),
                                "snr": np.array([snr], dtype=np.int32),
                                "samples_per_symbol": np.array([sps], dtype=np.int32),
                            }
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

    source_ds = OShea_Mackey_2020_DS(samples_per_symbol_to_get=[8], snrs_to_get=[-4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18])

    l = []
    for ex in source_ds:
        l.append(ex["modulation"])

    print(hash(tuple(l)))
    import sys
    sys.exit(1)

    class test_OShea_RML2016_DS(unittest.TestCase):
        @classmethod
        def setUpClass(self) -> None:
            # self.ds = OShea_RML2016_DS()
            pass

        # @unittest.SkipTest
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
                ds = OShea_Mackey_2020_DS(snrs_to_get=s)

                for ex in ds:
                    unique_snrs.add(ex["snr"][0])
                    unique_mods.add(ex["modulation"])

                unique_snrs = list(unique_snrs)
                unique_snrs.sort()


                self.assertEqual(len(unique_mods), 8)
                self.assertEqual(s, unique_snrs)

        # @unittest.SkipTest
        def test_samples_per_symbol(self):
            samps_per_symbol_to_test = [
                [2,4,6,8,10,12,14,16,18,20],
                [2,14,16,18,20],
                [2,20],
                [8,10,16,18,20],
            ]

            for s in samps_per_symbol_to_test:
                s.sort()
                unique_sps = set()
                unique_mods = set()
                ds = OShea_Mackey_2020_DS(samples_per_symbol_to_get=s)

                for ex in ds:
                    unique_sps.add(ex["samples_per_symbol"][0])
                    unique_mods.add(ex["modulation"])

                unique_sps = list(unique_sps)
                unique_sps.sort()


                self.assertEqual(len(unique_mods), 8)
                self.assertEqual(s, unique_sps)

    # class test_normalization(unittest.TestCase):
    #     @SkipTest
    #     def test_lazy_normalization(self):
    #         from steves_utils.utils_v2 import normalize_val, denormalize_val
    #         from steves_utils.lazy_map import Lazy_Map

    #         snrs_to_test = [
    #             [-18, -12, -6, 0, 6, 12, 18],
    #             [2, 4, 8, 10, -20, 14, 16, -16, -14, -10, -8, -4, -2],
    #             [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2]
    #         ]

    #         for s in snrs_to_test:
    #             s.sort()
    #             unique_snrs = set()
    #             unique_mods = set()
    #             ds = OShea_RML2016_DS(snrs_to_get=s)

    #             mapped_ds = Lazy_Map(ds, 
    #                 lambda ex: (
    #                     ex[0], ex[1], ex[2], 
    #                     np.array([normalize_val(min(s), max(s), ex[2][0])], dtype=np.single)
    #                 )
    #             )

    #             for ex in mapped_ds:
    #                 self.assertAlmostEqual(ex[2][0], denormalize_val(min(s), max(s), ex[3][0]), places=5)


    #     def test_eyeball_the_snrs(self):
    #         from steves_utils.utils_v2 import normalize_val, denormalize_val
    #         from steves_utils.lazy_map import Lazy_Map

    #         MIN = -20
    #         MAX = 18

    #         normalized_original_snrs = list(map(lambda s: normalize_val(MIN, MAX, s), OShea_RML2016_DS.get_snrs()))

    #         snrs_to_test = [
    #             [-20, -18, -16],
    #             [ -14, -12, ],
    #             [-10, -8, ],
    #             [ -6, -4, ],
    #             [-2, 0, ],
    #             [2, 4, 6, 8, 10, 12, 14, 16, 18],
    #         ]


    #         unique_normalized_snrs = set()
    #         for s in snrs_to_test:
    #             s.sort()

    #             ds = OShea_RML2016_DS(snrs_to_get=s)

    #             mapped_ds = Lazy_Map(ds, 
    #                 lambda ex: (
    #                     ex[0], ex[1], ex[2], 
    #                     np.array([normalize_val(MIN, MAX, ex[2][0])], dtype=np.single)
    #                 )
    #             )

    #             for ex in mapped_ds:
    #                 unique_normalized_snrs.add(ex[3][0])
            
    #         unique_normalized_snrs = list(unique_normalized_snrs)
    #         unique_normalized_snrs.sort()

    #         print(unique_normalized_snrs)
    #         print(normalized_original_snrs)

    #         self.assertEqual(len(unique_normalized_snrs), len(OShea_RML2016_DS.get_snrs()))

    #         for unique, nrml_original in zip(unique_normalized_snrs, normalized_original_snrs):
    #             self.assertAlmostEqual(unique, nrml_original, places=5)
            
    #         for unique, nrml_original in zip(unique_normalized_snrs, OShea_RML2016_DS.get_snrs()):
    #             self.assertAlmostEqual(
    #                 denormalize_val(MIN, MAX,unique), nrml_original, places=5
    #             )

    #         unique_normalized_snrs



    unittest.main()
    
