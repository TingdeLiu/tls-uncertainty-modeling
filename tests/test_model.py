import pytest

torch = pytest.importorskip("torch")
RePN = pytest.importorskip("umosls_dl.model").RePN


def test_repn_output_shape():
    model = RePN()
    neighbourhoods = torch.randn(2, 128, 3)
    physical_features = torch.randn(2, 4)

    output = model(neighbourhoods, physical_features)

    assert output.shape == (2, 1)


def test_repn_validates_neighbourhood_size():
    model = RePN()

    with pytest.raises(ValueError, match="neighbourhood size"):
        model(torch.randn(2, 32, 3), torch.randn(2, 4))
