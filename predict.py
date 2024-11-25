import argparse
import os
from argparse import Namespace
from pathlib import Path

import dill
import torch
import torchaudio
from tqdm import tqdm

from next_frame_classifier import NextFrameClassifier
from utils import detect_peaks, max_min_norm, replicate_first_k_frames

SECOND_THRESHOLD = 300


def generate_aligned_path(
    root_path: str, audio_root_path: str, audio_path: Path, extension: str = ".txt"
):
    return Path(root_path) / audio_path.relative_to(audio_root_path).with_suffix(
        extension
    )


def main(audio_root_path: str, output_path: str, ckpt: str, prominence: float):
    print(f"running inferece using ckpt: {ckpt}")
    print("\n\n", 90 * "-")

    ckpt = torch.load(ckpt, map_location=lambda storage, loc: storage)
    hp = Namespace(**dict(ckpt["hparams"]))

    # load weights and peak detection params
    model = NextFrameClassifier(hp)
    weights = ckpt["state_dict"]
    weights = {k.replace("NFC.", ""): v for k, v in weights.items()}
    model.load_state_dict(weights)
    peak_detection_params = dill.loads(ckpt["peak_detection_params"])["cpc_1"]
    if prominence is not None:
        print(f"overriding prominence with {prominence}")
        peak_detection_params["prominence"] = prominence

    for audio_path in tqdm(list(Path(audio_root_path).glob("**/*.flac"))):
        # load data
        audio, sr = torchaudio.load(audio_path)
        assert (
            sr == 16000
        ), "model was trained with audio sampled at 16khz, please downsample."
        if len(audio) > sr * SECOND_THRESHOLD:
            continue
        audio = audio[0]
        audio = audio.unsqueeze(0)

        # run inference
        try:
            preds = model(audio)  # get scores
            preds = preds[1][0]  # get scores of positive pairs
            preds = replicate_first_k_frames(preds, k=1, dim=1)  # padding
            preds = 1 - max_min_norm(
                preds
            )  # normalize scores (good for visualizations)
            preds = detect_peaks(
                x=preds,
                lengths=[preds.shape[1]],
                prominence=peak_detection_params["prominence"],
                width=peak_detection_params["width"],
                distance=peak_detection_params["distance"],
            )  # run peak detection on scores
            preds = (
                preds[0] * 160 * 1000 / sr
            )  # transform frame indexes to milliseconds
        except:  # if the audio is too short, it seems the prediction fails
            preds = []
        boundary_path = generate_aligned_path(output_path, audio_root_path, audio_path)
        os.makedirs(boundary_path.parent, exist_ok=True)
        with open(boundary_path, "w") as f:
            for b in preds:
                f.write(str(b) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unsupervised segmentation inference script"
    )
    parser.add_argument(
        "--ckpt",
        help="path to checkpoint file",
        default="./pretrained_models/buckeye+_pretrained.ckpt",
    )
    parser.add_argument(
        "--prominence",
        type=float,
        default=0.05,
        help="prominence for peak detection (default: 0.05)",
    )
    parser.add_argument(
        "--audio_path",
        help="path to dataset for evaluation, where .wav, .phn, and .wrd files are stored",
        default="/NAS/Personal/skando/dataset/LibriSpeech/train-clean-100",
    )
    parser.add_argument(
        "--output_boundary_path",
        help="path to output boundaries",
        default="/data/skando/speechLM/experiment/boundaries/librispeech/train-clean-100/phoneme",
    )
    args = parser.parse_args()
    main(args.audio_path, args.output_boundary_path, args.ckpt, args.prominence)
