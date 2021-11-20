#! /usr/bin/env python3

from steves_utils.oshea_mackey_2020_ds import OShea_Mackey_2020_DS
from steves_utils.oshea_RML2016_ds import OShea_RML2016_DS



snrs_to_get=[-4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
samples_per_symbol_to_get=[8]
print("OShea_Mackey_2020_DS. snrs: {}, sampersym: {}, len: {}".format(
	snrs_to_get,
	samples_per_symbol_to_get,
	len(OShea_Mackey_2020_DS(snrs_to_get=snrs_to_get, samples_per_symbol_to_get=samples_per_symbol_to_get))
))


snrs_to_get=[-4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
print("OShea_RML2016_DS. snrs: {}, len: {}".format(
	snrs_to_get,
	len(OShea_RML2016_DS(snrs_to_get=snrs_to_get))
))

print("OShea_Mackey_2020_DS all len: {}".format(
	len(OShea_Mackey_2020_DS())
))

print("OShea_RML2016_DS all len: {}".format(
	len(OShea_RML2016_DS())
))