# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

import torch
from models.tts.base.tts_inferece import TTSInference
from models.tts.jets.jets import Jets
from models.tts.jets.jets_dataset import JetsTestCollator, JetsTestDataset
from tqdm import tqdm


class JetsInference(TTSInference):
    def __init__(self, args, cfg):
        TTSInference.__init__(self, args, cfg)
        self.args = args
        self.cfg = cfg
        self.infer_type = args.mode

    def _build_model(self):
        self.model = Jets(self.cfg)
        return self.model

    def _build_test_dataset(self):
        return JetsTestDataset, JetsTestCollator

    def inference_for_batches(self):
        ###### Construct test_batch ######
        n_batch = len(self.test_dataloader)
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        print(
            "Model eval time: {}, batch_size = {}, n_batch = {}".format(
                now, self.test_batch_size, n_batch
            )
        )
        self.model.eval()

        ###### Inference for each batch ######
        pred_res = []
        with torch.no_grad():
            for i, batch_data in enumerate(
                self.test_dataloader if n_batch == 1 else tqdm(self.test_dataloader)
            ):
                outputs = self.model.inference(batch_data)

                audios, d_predictions = outputs
                d_predictions = d_predictions.unsqueeze(-1)

                for idx in range(audios.size(0)):
                    audio = audios[idx, 0, :].data.cpu().float()
                    duration = d_predictions[idx, :, :]
                    audio_length = (
                        duration.sum([0, 1]).long() * self.cfg.preprocess.hop_size
                    )
                    audio_length = audio_length.cpu().numpy()
                    audio = audio[:audio_length]
                    pred_res.append(audio)

        return pred_res
