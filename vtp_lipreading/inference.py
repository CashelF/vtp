import numpy as np
import torch, os, cv2, pickle, sys
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from glob import glob
import mediapipe as mp

from .dataloader import VideoDataset, AugmentationPipeline
from .config import load_args, start_symbol, end_symbol
from .models import builders
from .utils import load
from .search import beam_search

args = load_args()
augmentor = AugmentationPipeline(args)

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

def extract_frames(vidpath, target_frames=20):
    cap = cv2.VideoCapture(vidpath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = np.linspace(0, total_frames - 1, target_frames).astype(int)
    frames = []

    idx_set = set(frame_idxs)
    idx = 0
    success, frame = cap.read()
    while success:
        if idx in idx_set:
            frames.append(frame)
        success, frame = cap.read()
        idx += 1

    cap.release()

    # Padding if less than target_frames
    while len(frames) < target_frames:
        frames.append(frames[-1])

    return frames

def crop_lip(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        lip_landmarks = [landmarks.landmark[i] for i in range(78, 308)]
        h, w, _ = frame.shape
        x_coords = [int(lm.x * w) for lm in lip_landmarks]
        y_coords = [int(lm.y * h) for lm in lip_landmarks]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        lip_crop = frame[y_min:y_max, x_min:x_max]
        lip_crop = cv2.resize(lip_crop, (128, 128))
        return lip_crop
    else:
        return cv2.resize(frame, (128, 128))


def forward_pass(model, src, src_mask):
    encoder_output, src_mask = model.encode(src, src_mask)

    beam_outs, beam_scores = beam_search(
        decoder=model,
        bos_index=start_symbol,
        eos_index=end_symbol,
        max_output_length=args.max_decode_len,
        pad_index=0,
        encoder_output=encoder_output,
        src_mask=src_mask,
        size=args.beam_size,
        alpha=args.beam_len_alpha,
        n_best=args.beam_size,
    )

    return beam_outs, beam_scores

def get_lm_score(lm, lm_tokenizer, texts):
    logloss = nn.CrossEntropyLoss()
    tokens_tensor = lm_tokenizer.batch_encode_plus(texts, return_tensors="pt", padding=True)
    logits = lm(tokens_tensor['input_ids'], attention_mask=tokens_tensor['attention_mask'])[0]
    losses = []
    for logits, m, labels in zip(logits, tokens_tensor['attention_mask'], tokens_tensor['input_ids']):
        loss = logloss(logits[:m.sum() - 1], labels[1:m.sum()])
        losses.append(loss.item())

    losses = 1./np.exp(np.array(losses))  # higher should be treated as better
    return losses

def minmax_normalize(values):
    v = np.array(values)
    v = (v - v.min()) / (v.max() - v.min())
    return v

def run(vidpath, dataloader, model, lm=None, lm_tokenizer=None, display=True):
    raw_frames = extract_frames(vidpath, target_frames=20)
    cropped_frames = [crop_lip(frame) for frame in raw_frames]

    frames_tensor = torch.FloatTensor(np.stack(cropped_frames)).permute(3,0,1,2).unsqueeze(0)
    frames_tensor = augmentor(frames_tensor).detach()

    cur_src = frames_tensor.to(args.device)
    cur_src_mask = torch.ones((1, 1, cur_src.size(2))).to(args.device)

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            beam_outs, beam_scores = forward_pass(model, cur_src, cur_src_mask)
            beam_outs_f, beam_scores_f = forward_pass(model, 
                            augmentor.horizontal_flip(cur_src), cur_src_mask)

            beam_outs = beam_outs[0] + beam_outs_f[0]
            beam_scores = np.array(beam_scores[0] + beam_scores_f[0])

            if lm is not None:
                pred_texts = [dataloader.to_tokens(o.cpu().numpy().tolist()) \
                                for o in beam_outs]

                lm_scores = get_lm_score(lm, lm_tokenizer, pred_texts)
                lm_scores = minmax_normalize(lm_scores)
                beam_scores = minmax_normalize(beam_scores)

                beam_scores = args.lm_alpha * lm_scores + \
                                (1 - args.lm_alpha) * beam_scores

    best_pred_idx = beam_scores.argmax()
    out = beam_outs[best_pred_idx]
    pred = dataloader.to_tokens(out.cpu().numpy().tolist())

    if display: print(pred)
    return pred

if __name__ == '__main__':
    model, video_loader, lm, lm_tokenizer = main(args)
    run(args.fpath, video_loader, model, lm, lm_tokenizer)
