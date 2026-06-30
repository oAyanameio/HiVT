from argparse import ArgumentParser
from argparse import ArgumentTypeError

from utils import make_parser_arg_optional
from utils import merge_checkpoint_hparams
from utils import str2bool


def test_str2bool_parses_common_true_false_literals():
    assert str2bool(True) is True
    assert str2bool(False) is False
    assert str2bool("true") is True
    assert str2bool("false") is False
    assert str2bool("1") is True
    assert str2bool("0") is False
    assert str2bool("yes") is True
    assert str2bool("no") is False


def test_str2bool_rejects_unknown_literals():
    try:
        str2bool("maybe")
    except ArgumentTypeError:
        return
    raise AssertionError("str2bool should reject unknown literals")


def test_merge_checkpoint_hparams_keeps_runtime_args_and_restores_model_args():
    cli_args = {"root": "/tmp/data", "use_reliability": False, "embed_dim": None}
    checkpoint_hparams = {"use_reliability": True, "embed_dim": 64}

    merged = merge_checkpoint_hparams(
        cli_args,
        checkpoint_hparams,
        runtime_arg_names={"root"},
    )

    assert merged["root"] == "/tmp/data"
    assert merged["use_reliability"] is True
    assert merged["embed_dim"] == 64


def test_make_parser_arg_optional_clears_required_flag():
    parser = ArgumentParser()
    parser.add_argument("--embed_dim", type=int, required=True)

    make_parser_arg_optional(parser, "embed_dim", default=None)
    args = parser.parse_args([])

    assert args.embed_dim is None
