import pickle
import torch
from torch.utils.data.dataloader import DataLoader
from metric_utils.metrics import Metrics
from metric_utils.tracker import Tracker
from data_utils.vocab import Vocab
from model.transformer import make_model
import os
from loss_utils.LabelSmoothingLoss import LabelSmoothing, SimpleLossCompute, NoamOpt
from data_utils.dataloader import Batch, OCRDataset, collate_fn
from tqdm import tqdm

import config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def run_epoch(loaders, train, prefix, epoch, model, loss_compute, metric, tracker):
    if train:
        model.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        model.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}

    for loader in loaders:
        dataset = loader.dataset.dataset
        pbar = tqdm(loader, desc='Epoch {} - {} - Fold {}'.format(epoch+1, prefix, loaders.index(loader)+1), unit='it', ncols=0)
        loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
        cer_tracker = tracker.track('{}_cer'.format(prefix), tracker_class(**tracker_params))
        wer_tracker = tracker.track('{}_wer'.format(prefix), tracker_class(**tracker_params))
        
        for imgs, tokens, shifted_tokens in pbar:
            batch = Batch(imgs, tokens, shifted_tokens, dataset.vocab.padding_idx)
            logprobs = model(batch.imgs, batch.tokens, batch.src_mask, batch.tokens_mask)
            loss = loss_compute(logprobs, batch.shifted_right_tokens, batch.ntokens)
            
            outs = model.get_predictions(batch.imgs, batch.src_mask, dataset.vocab, dataset.max_len)
            scores = metric.get_scores(dataset.vocab.decode_sentence(outs.cpu()), dataset.vocab.decode_sentence(tokens.cpu()))

            loss_tracker.append(loss.item())
            wer_tracker.append(scores["wer"])
            cer_tracker.append(scores["cer"])

            fmt = '{:.4f}'.format
            pbar.set_postfix(loss=fmt(loss_tracker.mean.value), cer=fmt(cer_tracker.mean.value), wer=fmt(wer_tracker.mean.value))
            pbar.update()
            
        if not train:
            return {
                "cer": cer_tracker.mean.value,
                "wer": wer_tracker.mean.value
            }

def train():
    if not os.path.isfile(f"vocab_{config.out_level}.pkl"):
        vocab = Vocab(config.image_dir, config.out_level)
    else:
        vocab = pickle.load(open(f"vocab_{config.out_level}.pkl", "rb"))

    train_dataset = OCRDataset(dir=os.path.join(config.image_dir, "train_data"), image_size=config.image_size, out_level=config.out_level, vocab=vocab)
    test_dataset = OCRDataset(dir=os.path.join(config.image_dir, "test_data"), image_size=config.image_size, out_level=config.out_level, vocab=vocab)
    metric = Metrics(vocab)
    tracker = Tracker()
    
    model = make_model(len(vocab.stoi), N=config.num_layers, image_height=config.image_size[-1], d_model=config.d_model, d_ff=config.dff, 
                            h=config.heads, dropout=config.dropout)

    model.cuda()
    criterion = LabelSmoothing(size=len(vocab.stoi), padding_idx=vocab.padding_idx, smoothing=config.smoothing)
    criterion.cuda()
    model_opt = NoamOpt(model.tgt_embed[0].d_model, 1, 2000,
            torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.98), eps=1e-9))

    if config.start_from is not None:
        saved_info = torch.load(config.start_from)
        model.load_state_dict(saved_info["state_dict"])
        from_stage = saved_info["stage"] + 1
        from_epoch = saved_info["epoch"] + 1
        model.load_state_dict(saved_info["state_dict"])
        model_opt = saved_info["model_opt"]
        folds = saved_info["folds"]
    else:
        from_stage = 0
        from_epoch = 0
        folds = train_dataset.get_folds()

    test_dataloder = DataLoader(test_dataset, 
                                batch_size=config.batch_size, 
                                shuffle=True, 
                                collate_fn=collate_fn)
    for stage in range(from_stage, len(folds)):
        best_scores = {
                "cer": 0,
                "wer": 0
        }

        scores_on_test = {
            "cer": 0,
            "wer": 0
        }

        for epoch in range(from_epoch, config.max_epoch):
            run_epoch(folds[:-1], True, "Training", epoch, model, 
                SimpleLossCompute(model.generator, criterion, model_opt), metric, tracker)
            val_scores = run_epoch([folds[-1]], False, "Validation", epoch, model, 
                SimpleLossCompute(model.generator, criterion, None), metric, tracker)
            test_scores = run_epoch([test_dataloder], False, "Evaluation", epoch, model, 
                SimpleLossCompute(model.generator, criterion, None), metric, tracker)

            if best_scores["cer"] < val_scores["cer"]:
                best_scores = val_scores
                scores_on_test = test_scores
                torch.save({
                    "stage": stage,
                    "epoch": epoch,
                    "folds": folds,
                    "vocab": vocab,
                    "state_dict": model.state_dict(),
                    "model_opt": model_opt,
                    "val_scores": val_scores,
                    "test_scores": test_scores,
                }, os.path.join(config.checkpoint_path, f"best_model_stage_{stage+1}.pth"))

            torch.save({
                "stage": stage,
                "epoch": epoch,
                "folds": folds,
                "vocab": vocab,
                "state_dict": model.state_dict(),
                "model_opt": model_opt,
                "val_scores": val_scores,
                "test_scores": test_scores,
            }, os.path.join(config.checkpoint_path, f"last_model_stage_{stage+1}.pth"))

            print("*"*13)

        print(f"Stage {stage+1} completed. Scores on test set: CER = {scores_on_test['cer']} - WER = {scores_on_test['wer']}.")
        print("="*23)

        # swapping folds
        for idx in range(len(folds)):
            tmp_fold = folds[idx]
            folds[idx] = folds[idx - 1]
            folds[idx-1] = tmp_fold

if __name__=='__main__':

    train()