import fastwer

class Metrics(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def get_error(self, label, true_label, mode):
        error = 0
        if mode == "character":
            error += fastwer.score_sent(label,true_label, char_level=True)
        else:
            error += fastwer.score_sent(label, true_label)

        return error

    def get_scores(self, predicteds, gts):
        cer = 0
        wer = 0
        batch_size = len(gts)
        for predicted, gt in zip(predicteds, gts):
            if len(predicted) == 0:
                cer += len(gt)
                wer += len(gt.split())
                continue
            cer += self.get_error(predicted, gt, mode="character")
            wer += self.get_error(predicted, gt, mode="word")

        return {
            "cer": cer / batch_size,
            "wer": wer / batch_size
        }