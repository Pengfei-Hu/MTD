import torch
from .dataset import collate_func, valid_collate_func, PickleLoader


def create_train_dataloader(ly_vocab, re_vocab, pickle_path, batch_size, num_workers, all_labels_path):
    dataset = PickleLoader(pickle_path, ly_vocab, re_vocab, 'train', all_labels_path)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        collate_fn=collate_func,
        batch_size=batch_size,
        shuffle=True
    )
    return dataloader

def create_valid_dataloader(ly_vocab, re_vocab, pickle_path, batch_size, num_workers, all_labels_path):
    dataset = PickleLoader(pickle_path, ly_vocab, re_vocab, 'test', all_labels_path)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        collate_fn=valid_collate_func,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    return dataloader



if __name__ == "__main__":
    pass