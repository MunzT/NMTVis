# For data loading
import os

from pytorch_lightning.utilities.cloud_io import atomic_save
from pytorch_lightning.utilities.cloud_io import load as pl_load
from shared import MODELS_FOLDER
from tqdm.auto import tqdm

if __name__ == "__main__":
    SRC_LANG = "en"
    TGT_LANG = "de"
    ENSEMBLE = 10
    LAST_CKPT = 63

    trafo_states = []
    print('Loading...')
    for epoch in tqdm(range(LAST_CKPT-ENSEMBLE, LAST_CKPT)):
        ckpt_path = os.path.join(MODELS_FOLDER, 'transformer', f'trafo_{SRC_LANG}_{TGT_LANG}_{epoch}.pt')
        trafo_states.append(pl_load(ckpt_path, map_location=lambda storage, loc: storage)['state_dict'])

    "Average models into model"
    print("Averaging...")
    avg_state = {}
    for key in trafo_states[-1]:
        mean = 0
        for trafo_state in trafo_states:
            mean += trafo_state[key]
        avg_state[key] = mean / len(trafo_states)

    print('saving...')
    ckpt_path = os.path.join(MODELS_FOLDER, 'transformer', f'trafo_{SRC_LANG}_{TGT_LANG}_{LAST_CKPT}.pt')
    avg_ckpt = pl_load(ckpt_path, map_location=lambda storage, loc: storage)
    avg_ckpt['state_dict'] = avg_state
    atomic_save(avg_ckpt, f'.data/models/transformer/trafo_{SRC_LANG}_{TGT_LANG}_ensemble.pt')
