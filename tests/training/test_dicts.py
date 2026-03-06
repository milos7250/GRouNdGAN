import numpy as np
import pytest

from .. import pytestmark  # pyright: ignore[reportUnusedImport] # noqa: F401


def test_gan_losses():
    from training.dicts import GANCritLosses, GANGenLosses, GANLosses
    gen_loss = GANGenLosses(**{"Generator Loss": 0.5, "Generator Total Loss": 1.0})
    crit_loss = GANCritLosses(**{"Critic Fake Loss": 0.5, "Critic Real Loss": 0.5, "Critic Gradient Penalty Loss": 0.1, "Critic Total Loss": 1.1})
    gan_loss = GANLosses(**{**gen_loss, **crit_loss, "Total Loss": 2.1}) # pyright: ignore[reportArgumentType]
    assert gan_loss["Generator Loss"] == pytest.approx(0.5)
    assert gan_loss["Generator Total Loss"] == pytest.approx(1.0)
    assert gan_loss["Critic Fake Loss"] == pytest.approx(0.5)
    assert gan_loss["Critic Real Loss"] == pytest.approx(0.5)
    assert gan_loss["Critic Gradient Penalty Loss"] == pytest.approx(0.1)
    assert gan_loss["Critic Total Loss"] == pytest.approx(1.1)
    assert gan_loss["Total Loss"] == pytest.approx(2.1)
    

def test_loss_list():
    from training.dicts import GANLosses, LossList
    losses = LossList([
        GANLosses(**{"Generator Loss": 0.5, "Generator Total Loss": 1.0, "Critic Fake Loss": 0.5, "Critic Real Loss": 0.5, "Critic Gradient Penalty Loss": 0.1, "Critic Total Loss": 1.1, "Total Loss": 2.1}),
        GANLosses(**{"Generator Loss": 1.0, "Generator Total Loss": 2.0, "Critic Fake Loss": 1.0, "Critic Real Loss": 1.0, "Critic Gradient Penalty Loss": 0.2, "Critic Total Loss": 2.2, "Total Loss": 4.2}),
        GANLosses(**{"Generator Loss": 1.5, "Generator Total Loss": 3.0, "Critic Fake Loss": 1.5, "Critic Real Loss": 1.5, "Critic Gradient Penalty Loss": 0.3, "Critic Total Loss": 3.3, "Total Loss": 6.3}),
    ])
    avg_loss = losses.avg()
    assert avg_loss["Generator Loss"] == pytest.approx(1.0)
    assert avg_loss["Generator Total Loss"] == pytest.approx(2.0)
    assert avg_loss["Critic Fake Loss"] == pytest.approx(1.0)
    assert avg_loss["Critic Real Loss"] == pytest.approx(1.0)
    assert avg_loss["Critic Gradient Penalty Loss"] == pytest.approx(0.2)
    assert avg_loss["Critic Total Loss"] == pytest.approx(2.2)
    assert avg_loss["Total Loss"] == pytest.approx(4.2)
    
def test_loss_list_empty():
    from training.dicts import LossList
    losses = LossList([])
    try:
        losses.avg()
        assert False, "Expected ValueError for empty LossList"
    except ValueError as e:
        assert str(e) == "Cannot average an empty list of losses."

def test_loss_list_nan():
    from training.dicts import GANLosses, LossList
    losses = LossList([
        GANLosses(**{"Generator Loss": 0.5, "Generator Total Loss": 1.0, "Critic Fake Loss": 0.5, "Critic Real Loss": 0.5, "Critic Gradient Penalty Loss": 0.1, "Critic Total Loss": 1.1, "Total Loss": 2.1}),
        GANLosses(**{"Generator Loss": 1.0, "Generator Total Loss": 2.0, "Critic Fake Loss": np.nan, "Critic Real Loss": np.nan, "Critic Gradient Penalty Loss": np.nan, "Critic Total Loss": np.nan, "Total Loss": np.nan}),
        GANLosses(**{"Generator Loss": 1.5, "Generator Total Loss": 3.0, "Critic Fake Loss": 1.5, "Critic Real Loss": 1.5, "Critic Gradient Penalty Loss": 0.3, "Critic Total Loss": 3.3, "Total Loss": 6.3}),
    ])
    avg_loss = losses.avg()
    assert avg_loss["Generator Loss"] == pytest.approx(1.0)
    assert avg_loss["Generator Total Loss"] == pytest.approx(2.0)
    assert avg_loss["Critic Fake Loss"] == pytest.approx(1.0)
    assert avg_loss["Critic Real Loss"] == pytest.approx(1.0)
    assert avg_loss["Critic Gradient Penalty Loss"] == pytest.approx(0.2)
    assert avg_loss["Critic Total Loss"] == pytest.approx(2.2)
    assert avg_loss["Total Loss"] == pytest.approx(4.2)