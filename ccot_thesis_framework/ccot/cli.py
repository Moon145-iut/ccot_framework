"""CLI entrypoint for the CCOT thesis framework."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

from ccot.config import (
    DEFAULT_COMPRESSION_RATIO,
    DEFAULT_HIDDEN_LAYER,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_ID,
    DEFAULT_NUM_THREADS,
)
from ccot.paper.config import PaperConfig
from ccot.paper.gold import build_gold_targets
from ccot.paper.phi_train import train_phi_layers
from ccot.paper.end_train import train_end_head
from ccot.paper.psi_train import train_psi_decoder
from ccot.paper.export_traces import export_traces
from ccot.paper import infer as paper_infer
from ccot.phase2.truth_vector import build_truth_vector
from ccot.reasoners.ccot_cpu_gru import CCOTCpuGRUReasoner
from ccot.reasoners.ccot_paper import CCOTPaperReasoner
from ccot.report import write_run_report
from ccot.utils.io import read_jsonl


def _positive_float(value: str) -> float:
    val = float(value)
    if val <= 0:
        raise argparse.ArgumentTypeError("Expected positive float")
    return val


def _paper_config_from_args(args: argparse.Namespace) -> PaperConfig:
    artifacts_dir = Path(getattr(args, "artifacts_dir", "artifacts"))
    report_path = artifacts_dir / "reports" / "run_report.md"
    cfg = PaperConfig(
        model_id=args.model_id,
        layer_l=args.layer_l,
        scorer_T=args.scorer_T,
        compression_ratio=args.compression_ratio,
        max_seq_len=args.max_seq_len,
        stop_cap=args.stop_cap,
        subset_method=args.subset_method,
        teacher_jsonl=Path(args.teacher_jsonl) if getattr(args, "teacher_jsonl", None) else None,
        limit_samples=args.limit_samples,
        artifacts_dir=artifacts_dir,
        report_path=report_path,
    )
    return cfg


def _add_paper_args(subparser: argparse.ArgumentParser, require_csv: bool = False) -> None:
    subparser.add_argument("--csv", help="Path to GSM8K CSV", required=require_csv)
    subparser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    subparser.add_argument("--layer-l", type=int, default=16)
    subparser.add_argument("--scorer-T", dest="scorer_T", type=int, default=3)
    subparser.add_argument("--compression-ratio", type=float, default=0.12)
    subparser.add_argument("--max-seq-len", type=int, default=2048)
    subparser.add_argument("--stop-cap", type=int, default=0)
    subparser.add_argument("--subset-method", default="evenly")
    subparser.add_argument("--teacher-jsonl")
    subparser.add_argument("--limit-samples", type=int)
    subparser.add_argument("--device", default="cpu")
    subparser.add_argument("--artifacts-dir", default="artifacts")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CCOT-inspired thesis framework CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare-teacher", help="Build teacher JSONL from GSM8K CSV")
    prep.add_argument("--csv", required=True, help="Path to GSM8K CSV file")
    prep.add_argument("--out", required=True, help="Output JSONL path")
    prep.add_argument("--provider", help="Provider name: gemini or openai_compat")
    prep.add_argument("--teacher-model", help="Remote model id for teacher generation")
    prep.add_argument("--base-url", help="Override OpenAI-compatible base URL")
    prep.add_argument(
        "--use-api-for-rationale",
        action="store_true",
        help="Call the provider to generate rationale + answer",
    )
    prep.add_argument("--limit", type=int, help="Optional cap on processed rows")

    targets = subparsers.add_parser("build-targets", help="Extract hidden-state targets locally")
    targets.add_argument("--teacher-jsonl", required=True)
    targets.add_argument("--out-dir", required=True)
    targets.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    targets.add_argument("--compression-ratio", type=_positive_float, default=DEFAULT_COMPRESSION_RATIO)
    targets.add_argument("--selector", choices=["evenly_spaced", "even", "norm"], default="evenly_spaced")
    targets.add_argument("--hidden-layer-index", type=int, default=DEFAULT_HIDDEN_LAYER)
    targets.add_argument("--num-threads", type=int, default=DEFAULT_NUM_THREADS)
    targets.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    targets.add_argument("--limit", type=int, help="Optional limit for debugging")

    tc = subparsers.add_parser("train-ccot", help="Train latent generator")
    tc.add_argument("--targets-dir", required=True)
    tc.add_argument("--out-dir", required=True)
    tc.add_argument("--epochs", type=int, default=25)
    tc.add_argument("--batch-size", type=int, default=4)
    tc.add_argument("--lr", type=float, default=1e-3)
    tc.add_argument("--num-threads", type=int, default=DEFAULT_NUM_THREADS)
    tc.add_argument("--seed", type=int, default=42)

    td = subparsers.add_parser("train-decoder", help="Train char-level answer decoder")
    td.add_argument("--targets-dir", required=True)
    td.add_argument("--ccot-weights", required=True)
    td.add_argument("--out-dir", required=True)
    td.add_argument("--epochs", type=int, default=30)
    td.add_argument("--batch-size", type=int, default=8)
    td.add_argument("--lr", type=float, default=1e-3)
    td.add_argument("--num-threads", type=int, default=DEFAULT_NUM_THREADS)
    td.add_argument("--seed", type=int, default=42)
    td.add_argument(
        "--use-teacher-latents",
        action="store_true",
        help="Train decoder with teacher latents instead of generated ones",
    )

    inf = subparsers.add_parser("infer", help="Run local inference")
    inf.add_argument("--question", required=True)
    inf.add_argument("--backend", choices=["cpu_gru", "paper"], default="cpu_gru")
    inf.add_argument("--targets-dir", help="Required for cpu_gru backend")
    inf.add_argument("--ccot-weights")
    inf.add_argument("--decoder-weights")
    inf.add_argument("--stop-threshold", type=float, default=0.5)
    inf.add_argument("--max-latents", type=int, default=64)
    inf.add_argument("--num-threads", type=int, default=DEFAULT_NUM_THREADS)
    inf.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    inf.add_argument("--device", default="cpu")

    full = subparsers.add_parser("full-run", help="End-to-end training/eval")
    full.add_argument("--backend", choices=["cpu_gru", "paper"], default="cpu_gru")
    full.add_argument("--csv", required=True)
    full.add_argument("--teacher-jsonl", required=False)
    full.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    full.add_argument("--targets-dir", default="artifacts/targets")
    full.add_argument("--artifacts-dir", default="artifacts")
    full.add_argument("--limit-samples", type=int)
    full.add_argument("--device", default="cpu")

    paper_gold = subparsers.add_parser("paper-build-gold", help="Build gold latents from θ")
    _add_paper_args(paper_gold, require_csv=True)

    paper_phi = subparsers.add_parser("paper-train-phi", help="Train φ adapters")
    _add_paper_args(paper_phi)
    paper_phi.add_argument("--train-path", default="artifacts/gold/paper_train.pt")

    paper_end = subparsers.add_parser("paper-train-end", help="Train ENDψ head")
    _add_paper_args(paper_end)
    paper_end.add_argument("--train-path", default="artifacts/gold/paper_train.pt")
    paper_end.add_argument("--phi-dir", default="artifacts/models/paper_phi")

    paper_psi = subparsers.add_parser("paper-train-psi", help="Train ψ decoder")
    _add_paper_args(paper_psi)
    paper_psi.add_argument("--train-path", default="artifacts/gold/paper_train.pt")
    paper_psi.add_argument("--phi-dir", default="artifacts/models/paper_phi")
    paper_psi.add_argument("--joint-training", action="store_true")

    paper_eval = subparsers.add_parser("paper-eval", help="Evaluate paper backend")
    _add_paper_args(paper_eval, require_csv=True)

    paper_export = subparsers.add_parser("paper-export-traces", help="Export traces to JSONL")
    _add_paper_args(paper_export, require_csv=True)
    paper_export.add_argument("--traces-out", default="artifacts/traces/paper_traces.jsonl")

    paper_full = subparsers.add_parser("paper-full-run", help="Paper backend end-to-end")
    _add_paper_args(paper_full, require_csv=True)

    return parser


def _evaluate_reasoner(reasoner, questions: Iterable[dict], cfg: PaperConfig | None = None) -> dict:
    records = []
    correct = 0
    latencies = []
    ks = []
    for sample in questions:
        start = time.perf_counter()
        trace = reasoner.run_latent(sample["question"])
        answer = reasoner.decode_answer(sample["question"], trace)
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)
        ks.append(trace.k)
        is_correct = answer.strip() == sample["final_answer"].strip()
        if is_correct:
            correct += 1
        records.append(
            {
                "id": sample.get("id"),
                "question": sample["question"],
                "gold_answer": sample["final_answer"],
                "pred_answer": answer,
                "correct": is_correct,
                "trace": trace,
            }
        )
    eval_metrics = {
        "em": correct / max(1, len(records)),
        "avg_latency": sum(latencies) / max(1, len(latencies)),
        "avg_k": sum(ks) / max(1, len(ks)),
        "records": records,
    }
    return eval_metrics


def _execute_full_run(args: argparse.Namespace) -> None:
    if args.backend == "cpu_gru":
        from ccot.pipeline.build_hidden_targets import build_hidden_targets
        from ccot.training.train_ccot import train_ccot
        from ccot.training.train_decoder import train_decoder
        from ccot.pipeline.prepare_teacher_jsonl import prepare_teacher_jsonl

        teacher_jsonl = args.teacher_jsonl or str(Path(args.artifacts_dir) / "teacher.jsonl")
        prepare_teacher_jsonl(args.csv, teacher_jsonl, limit=args.limit_samples)
        build_hidden_targets(
            teacher_jsonl=teacher_jsonl,
            out_dir=args.targets_dir,
            model_id=args.model_id,
            compression_ratio=DEFAULT_COMPRESSION_RATIO,
            selector="evenly_spaced",
            hidden_layer_index=DEFAULT_HIDDEN_LAYER,
            num_threads=DEFAULT_NUM_THREADS,
            max_length=DEFAULT_MAX_LENGTH,
            limit=args.limit_samples,
        )
        ccot_dir = Path(args.artifacts_dir) / "ccot_weights"
        dec_dir = Path(args.artifacts_dir) / "decoder_weights"
        ccot_dir.mkdir(parents=True, exist_ok=True)
        dec_dir.mkdir(parents=True, exist_ok=True)
        train_ccot(args.targets_dir, ccot_dir, epochs=5, batch_size=4)
        train_decoder(
            args.targets_dir,
            ccot_dir / "latent_generator.pt",
            dec_dir,
            epochs=5,
            batch_size=4,
        )
        reasoner = CCOTCpuGRUReasoner(
            targets_dir=args.targets_dir,
            ccot_weights=ccot_dir / "latent_generator.pt",
            decoder_weights=dec_dir / "char_decoder.pt",
        )
        teacher_rows = read_jsonl(teacher_jsonl)[: args.limit_samples or 16]
        eval_metrics = _evaluate_reasoner(reasoner, teacher_rows)
        report_data = {
            "eval": {"em": eval_metrics["em"]},
            "avg_latency": eval_metrics["avg_latency"],
            "avg_k": eval_metrics["avg_k"],
            "phi_losses": {},
            "end_losses": [],
            "psi_losses": [],
            "dataset_sizes": {"train": len(teacher_rows), "val": 0},
            "joint_training": False,
        }
        write_run_report(
            backend="cpu_gru",
            cfg={"model_id": args.model_id},
            report_data=report_data,
            deviations=reasoner.notes(),
            path=Path(args.artifacts_dir) / "reports" / "run_report.md",
        )
    else:
        cfg = _paper_config_from_args(args)
        build_gold_targets(args.csv, cfg, device=args.device, limit=args.limit_samples)
        phi_res = train_phi_layers(
            cfg,
            cfg.gold_dir() / "paper_train.pt",
            cfg.models_dir() / "paper_phi",
            device=args.device,
            limit=args.limit_samples,
        )
        end_res = train_end_head(
            cfg,
            cfg.gold_dir() / "paper_train.pt",
            cfg.models_dir() / "paper_phi",
            cfg.models_dir() / "paper_end.pt",
            device=args.device,
            limit=args.limit_samples,
        )
        psi_res = train_psi_decoder(
            cfg,
            cfg.gold_dir() / "paper_train.pt",
            cfg.models_dir() / "paper_phi",
            cfg.models_dir() / "paper_psi",
            device=args.device,
            limit=args.limit_samples,
            joint_training=True,
        )
        reasoner = CCOTPaperReasoner(cfg, device=args.device)
        teacher_rows = (
            read_jsonl(cfg.teacher_jsonl)[: args.limit_samples or 8]
            if cfg.teacher_jsonl and Path(cfg.teacher_jsonl).exists()
            else []
        )
        if not teacher_rows and args.csv:
            from ccot.data import load_gsm8k_csv

            teacher_rows = [
                {"question": ex.question, "final_answer": ex.final_answer, "id": ex.idx}
                for ex in load_gsm8k_csv(args.csv)[: args.limit_samples or 8]
            ]
        eval_metrics = _evaluate_reasoner(reasoner, teacher_rows, cfg)
        traces_path = cfg.traces_dir() / "paper_traces.jsonl"
        export_traces(eval_metrics["records"], cfg, traces_path)
        build_truth_vector(traces_path, cfg.traces_dir() / "truth_vector.pt")
        report_data = {
            "eval": {"em": eval_metrics["em"]},
            "avg_latency": eval_metrics["avg_latency"],
            "avg_k": eval_metrics["avg_k"],
            "phi_losses": phi_res.layer_losses,
            "end_losses": end_res.loss_history,
            "psi_losses": psi_res.loss_history,
            "dataset_sizes": {"train": len(teacher_rows), "val": 0},
            "joint_training": psi_res.joint_training,
        }
        write_run_report(
            backend="paper",
            cfg=cfg.to_dict(),
            report_data=report_data,
            deviations=reasoner.notes(),
            path=cfg.report_path,
        )


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    cmd = args.command

    if cmd == "prepare-teacher":
        from ccot.pipeline.prepare_teacher_jsonl import prepare_teacher_jsonl

        path = prepare_teacher_jsonl(
            csv_path=args.csv,
            out_path=args.out,
            provider_name=args.provider,
            teacher_model=args.teacher_model,
            base_url=args.base_url,
            use_api_for_rationale=args.use_api_for_rationale,
            limit=args.limit,
        )
        print(f"Saved teacher file to {path}")
    elif cmd == "build-targets":
        from ccot.pipeline.build_hidden_targets import build_hidden_targets

        meta = build_hidden_targets(
            teacher_jsonl=args.teacher_jsonl,
            out_dir=args.out_dir,
            model_id=args.model_id,
            compression_ratio=args.compression_ratio,
            selector=args.selector,
            hidden_layer_index=args.hidden_layer_index,
            num_threads=args.num_threads,
            max_length=args.max_length,
            limit=args.limit,
        )
        print(json.dumps(meta, indent=2))
    elif cmd == "train-ccot":
        from ccot.training.train_ccot import train_ccot

        weights = train_ccot(
            targets_dir=args.targets_dir,
            out_dir=args.out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_threads=args.num_threads,
            seed=args.seed,
        )
        print(f"Saved latent generator weights to {weights}")
    elif cmd == "train-decoder":
        from ccot.training.train_decoder import train_decoder

        weights = train_decoder(
            targets_dir=args.targets_dir,
            ccot_weights=args.ccot_weights,
            out_dir=args.out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_threads=args.num_threads,
            seed=args.seed,
            use_generated_latents=not args.use_teacher_latents,
        )
        print(f"Saved decoder weights to {weights}")
    elif cmd == "infer":
        if args.backend == "cpu_gru":
            from ccot.pipeline.infer_local import infer_local

            result = infer_local(
                question=args.question,
                targets_dir=args.targets_dir,
                ccot_weights=args.ccot_weights,
                decoder_weights=args.decoder_weights,
                stop_threshold=args.stop_threshold,
                max_latents=args.max_latents,
                num_threads=args.num_threads,
            )
            print(json.dumps(result, indent=2))
        else:
            cfg = PaperConfig(model_id=args.model_id, limit_samples=None)
            reasoner = CCOTPaperReasoner(cfg, device=args.device)
            trace = reasoner.run_latent(args.question, max_steps=args.max_latents)
            answer = reasoner.decode_answer(args.question, trace)
            print(json.dumps({"answer": answer, "k": trace.k}, indent=2))
    elif cmd == "full-run":
        _execute_full_run(args)
    elif cmd == "paper-build-gold":
        cfg = _paper_config_from_args(args)
        meta = build_gold_targets(args.csv, cfg, device=args.device, limit=args.limit_samples)
        print(json.dumps(meta, indent=2))
    elif cmd == "paper-train-phi":
        cfg = _paper_config_from_args(args)
        result = train_phi_layers(
            cfg,
            args.train_path,
            Path(cfg.models_dir()) / "paper_phi",
            device=args.device,
        )
        print(json.dumps(result.layer_losses, indent=2))
    elif cmd == "paper-train-end":
        cfg = _paper_config_from_args(args)
        result = train_end_head(
            cfg,
            args.train_path,
            args.phi_dir,
            cfg.models_dir() / "paper_end.pt",
            device=args.device,
        )
        print(json.dumps({"loss": result.loss_history}, indent=2))
    elif cmd == "paper-train-psi":
        cfg = _paper_config_from_args(args)
        result = train_psi_decoder(
            cfg,
            args.train_path,
            args.phi_dir,
            cfg.models_dir() / "paper_psi",
            device=args.device,
            joint_training=args.joint_training,
        )
        print(json.dumps({"loss": result.loss_history, "joint": result.joint_training}, indent=2))
    elif cmd == "paper-eval":
        cfg = _paper_config_from_args(args)
        reasoner = CCOTPaperReasoner(cfg, device=args.device)
        from ccot.data import load_gsm8k_csv

        rows = [
            {"question": ex.question, "final_answer": ex.final_answer, "id": ex.idx}
            for ex in load_gsm8k_csv(args.csv)[: args.limit_samples or 8]
        ]
        metrics = _evaluate_reasoner(reasoner, rows, cfg)
        print(json.dumps({"em": metrics["em"]}, indent=2))
    elif cmd == "paper-export-traces":
        cfg = _paper_config_from_args(args)
        reasoner = CCOTPaperReasoner(cfg, device=args.device)
        from ccot.data import load_gsm8k_csv

        rows = [
            {"question": ex.question, "final_answer": ex.final_answer, "id": ex.idx}
            for ex in load_gsm8k_csv(args.csv)[: args.limit_samples or 8]
        ]
        metrics = _evaluate_reasoner(reasoner, rows, cfg)
        export_traces(metrics["records"], cfg, args.traces_out)
        print(f"Wrote traces to {args.traces_out}")
    elif cmd == "paper-full-run":
        args.backend = "paper"
        _execute_full_run(args)
    else:
        parser.error(f"Unknown command {cmd}")


if __name__ == "__main__":  # pragma: no cover
    main()
