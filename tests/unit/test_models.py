import torch

from recommender.models import Recommender


def test_recommender():
    n_items = 1000

    recommender = Recommender(vocab_size=1000)

    src_items = torch.randint(low=0, high=n_items, size=(32, 30))
    src_features = torch.rand(32, 30, 1)

    trg_items = torch.randint(low=0, high=n_items, size=(32, 5))
    trg_out = torch.randint(low=0, high=n_items, size=(32, 5, 1))

    out = recommender((src_items, src_features, trg_items))

    assert out.shape == torch.Size([32, 5, 1])

    loss = recommender.training_step(
        (src_items, src_features, trg_items, trg_out), batch_idx=1
    )

    assert isinstance(loss, torch.Tensor)
