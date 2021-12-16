#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import matplotlib.pyplot as plt

# cache
_coos_cache = {}
_checkins_cache = {}


def show_coos_cache():
    print(_coos_cache)


def show_checkins_cache():
    print(_checkins_cache)


def reset_coos_cache():
    global _coos_cache
    _coos_cache = {}


def reset_checkins_cache():
    global _checkins_cache
    _checkins_cache = {}


def _get_poi_coos(dataset):
    global _coos_cache
    if id(dataset) in _coos_cache:
        poi_coos = _coos_cache[id(dataset)]
    else:
        poi_coos = dataset.poi_coos.numpy()
        _coos_cache[id(dataset)] = poi_coos
    return poi_coos


def _get_user_checkins(dataset):
    global _checkins_cache
    if id(dataset) in _checkins_cache:
        all_checkins = _checkins_cache[id(dataset)]
    else:
        all_checkins = {}
        for user, poi in dataset.pos_pairs:
            if user in all_checkins:
                all_checkins[user].append(poi)
            else:
                all_checkins[user] = [poi]
        _checkins_cache[id(dataset)] = all_checkins
    return all_checkins


def show_all_pois(dataset, *args, **kwargs):
    poi_coos = _get_poi_coos(dataset)
    plt.plot(poi_coos[:, 1], poi_coos[:, 0] % 90, *args, **kwargs)


def show_user_checkins(dataset, user, *args, **kwargs):
    poi_coos = _get_poi_coos(dataset)
    checkins = _get_user_checkins(dataset)[user]
    checkins_coos = poi_coos[checkins]
    plt.plot(checkins_coos[:, 1], checkins_coos[:, 0] % 90, *args, **kwargs)


def show_pois(dataset, pois, *args, **kwargs):
    poi_coos = _get_poi_coos(dataset)[pois]
    plt.plot(poi_coos[:, 1], poi_coos[:, 0] % 90, *args, **kwargs)
    
