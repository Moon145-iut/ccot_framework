"""CPU smoke tests for paper modules."""
from ccot.paper.config import PaperConfig
from ccot.reasoners.ccot_paper import CCOTPaperReasoner


def test_paper_config_defaults():
    cfg = PaperConfig()
    assert cfg.layer_l > 0
    assert cfg.compression_ratio > 0


def test_reasoner_init_without_weights(monkeypatch):
    cfg = PaperConfig()

    class DummyState:
        hidden_size = 16
        joint_adapter = False

    monkeypatch.setattr(
        "ccot.paper.infer.load_infer_state",
        lambda cfg, device="cpu": DummyState(),
    )
    reasoner = CCOTPaperReasoner(cfg)
    assert reasoner.hidden_size() == 16
