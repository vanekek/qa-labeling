import torch

class QuestDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, lengths, labels = None):
        
        self.inputs = inputs
        if labels is not None:
            self.labels = labels
        else:
            self.labels = None
        self.lengths = lengths

    def __getitem__(self, idx):
        
        input_ids       = self.inputs[0][idx]
        input_masks     = self.inputs[1][idx]
        input_segments  = self.inputs[2][idx]
        lengths         = self.lengths[idx]
        if self.labels is not None: # targets
            labels = self.labels[idx]
            return input_ids, input_masks, input_segments, labels, lengths
        return input_ids, input_masks, input_segments, lengths

    def __len__(self):
        return len(self.inputs[0])
