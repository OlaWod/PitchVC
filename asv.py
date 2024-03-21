# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
This script will download pretrained models from modelscope (https://www.modelscope.cn/models)
based on the given model id, and extract embeddings from input audio. 
Please pre-install "modelscope".
Usage:
    1. extract the embedding from the wav file.
        `python infer_sv.py --model_id $model_id --wavs $wav_path `
    2. extract embeddings from two wav files and compute the similarity score.
        `python infer_sv.py --model_id $model_id --wavs $wav_path1 $wav_path2 `
    3. extract embeddings from the wav list.
        `python infer_sv.py --model_id $model_id --wavs $wav_list `
"""

import os
import sys
import re
import pathlib
import numpy as np
import argparse
import torch
import torchaudio

try:
    from speakerlab.process.processor import FBank
except ImportError:
    sys.path.append('%s/../..'%os.path.dirname(__file__))
    from speakerlab.process.processor import FBank

from speakerlab.utils.builder import dynamic_import

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.pipelines.util import is_official_hub_path


CAMPPLUS_VOX = {
    'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
    },
}

CAMPPLUS_COMMON = {
    'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_VOX = {
    'obj': 'speakerlab.models.eres2net.ResNet.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_COMMON = {
    'obj': 'speakerlab.models.eres2net.ResNet_aug.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_base_COMMON = {
    'obj': 'speakerlab.models.eres2net.ResNet.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
        'm_channels': 32,
    },
}

ERes2Net_Base_3D_Speaker = {
    'obj': 'speakerlab.models.eres2net.ResNet.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
        'm_channels': 32,
    },
}

ERes2Net_Large_3D_Speaker = {
    'obj': 'speakerlab.models.eres2net.ResNet.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
        'm_channels': 64,
    },
}

supports = {
    'damo/speech_campplus_sv_en_voxceleb_16k': {
        'revision': 'v1.0.2', 
        'model': CAMPPLUS_VOX, 
        'model_pt': 'campplus_voxceleb.bin', 
    },
    'damo/speech_campplus_sv_zh-cn_16k-common': {
        'revision': 'v1.0.0', 
        'model': CAMPPLUS_COMMON,
        'model_pt': 'campplus_cn_common.bin',
    },
    'damo/speech_eres2net_sv_en_voxceleb_16k': {
        'revision': 'v1.0.2', 
        'model': ERes2Net_VOX,
        'model_pt': 'pretrained_eres2net.ckpt',
    },
    'damo/speech_eres2net_sv_zh-cn_16k-common': {
        'revision': 'v1.0.4', 
        'model': ERes2Net_COMMON,
        'model_pt': 'pretrained_eres2net_aug.ckpt',
    },
    'damo/speech_eres2net_base_200k_sv_zh-cn_16k-common': {
        'revision': 'v1.0.0', 
        'model': ERes2Net_base_COMMON,
        'model_pt': 'pretrained_eres2net.pt',
    },
    'damo/speech_eres2net_base_sv_zh-cn_3dspeaker_16k': {
        'revision': 'v1.0.1', 
        'model': ERes2Net_Base_3D_Speaker,
        'model_pt': 'eres2net_base_model.ckpt',
    },
    'damo/speech_eres2net_large_sv_zh-cn_3dspeaker_16k': {
        'revision': 'v1.0.0', 
        'model': ERes2Net_Large_3D_Speaker,
        'model_pt': 'eres2net_large_model.ckpt',
    },
}


def get_model(model_id, local_model_dir):
    conf = supports[model_id]

    save_dir = os.path.join(local_model_dir, model_id.split('/')[1])
    save_dir =  pathlib.Path(save_dir)
    pretrained_model = save_dir / conf['model_pt']
    pretrained_state = torch.load(pretrained_model, map_location='cpu')

    # load model
    model = conf['model']
    embedding_model = dynamic_import(model['obj'])(**model['args'])
    embedding_model.load_state_dict(pretrained_state)
    embedding_model.eval()

    return embedding_model


def get_asv_models(model_ids, local_model_dir):
    models = []
    for model_id in model_ids:
        models.append(get_model(model_id, local_model_dir))

    feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)

    return models, feature_extractor


def compute_embedding(wav, embedding_model, feature_extractor):
    # compute feat
    feat = feature_extractor(wav).unsqueeze(0)
    # compute embedding
    with torch.no_grad():
        embedding = embedding_model(feat)
    
    return embedding


def compute_similarity(wav1, wav2, embedding_model, feature_extractor):
    embedding1 = compute_embedding(wav1, embedding_model, feature_extractor)
    embedding2 = compute_embedding(wav2, embedding_model, feature_extractor)

    # compute similarity score
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    scores = similarity(embedding1, embedding2).item()
    
    return scores


def compute_similarity2(wav1, embedding2, embedding_model, feature_extractor):
    embedding1 = compute_embedding(wav1, embedding_model, feature_extractor)

    # compute similarity score
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    scores = similarity(embedding1, embedding2).item()
    
    return scores
