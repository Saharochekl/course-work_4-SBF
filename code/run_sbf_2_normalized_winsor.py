#!/usr/bin/env python3
"""Последовательный runner эксперимента с обрезкой после нормировки.

Production-данные ``sbf-2`` открываются только для чтения.  Результаты и кэш
пишутся исключительно в ``runs/sbf2_normalized_winsor``.
"""

from __future__ import annotations

import argparse
import builtins
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from sbf2_normalized_winsor_core import (
    ExperimentConfig,
    build_evaluation_table,
    discover_galaxies,
    find_project_root,
    inspect_sources,
    load_matching_results,
    prepare_target_status,
    process_target,
    target_status_path,
    update_target_status,
    write_aggregate_tables,
)


_original_print = builtins.print


def _timestamped_print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _original_print(time.strftime("[%Y-%m-%d %H:%M:%S]"), *args, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Сравнить no-winsor, production raw 3.5σ и обрезку после "
            "residual/sqrt(model) на готовых продуктах SBF-2"
        )
    )
    parser.add_argument(
        "--source-batch-root", type=Path,
        help="Каталог успешных NGC_*_result.json production SBF-2",
    )
    parser.add_argument(
        "--output-root", type=Path,
        help="Отдельное дерево результатов эксперимента",
    )
    parser.add_argument(
        "--galaxies", nargs="+",
        help='Подвыборка, например --galaxies "NGC 3379" "NGC 1380"',
    )
    parser.add_argument("--sigma", type=float, default=3.5)
    parser.add_argument("--e-realizations", type=int, default=64)
    parser.add_argument("--random-seed", type=int, default=1489)
    parser.add_argument("--fft-workers", type=int, default=-1)
    parser.add_argument(
        "--save-ring-fft-fits", action="store_true",
        help="Save candidate per-ring PRIMARY/FFTINPUT/WINDOW diagnostics",
    )
    parser.add_argument("--save-all-branch-fits", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rebuild-input-cache", action="store_true")
    parser.add_argument("--rebuild-expectation-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    # Перехват builtins.print добавляет время и к сообщениям из core-модуля.
    builtins.print = _timestamped_print
    args = parse_args()
    project_root = find_project_root()
    source_batch_root = (
        args.source_batch_root.resolve()
        if args.source_batch_root is not None
        else project_root / "runs" / "sbf2_go3055" / "batch"
    )
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else project_root / "runs" / "sbf2_normalized_winsor"
    )
    status_path = target_status_path(output_root)
    galaxies = (
        args.galaxies
        if args.galaxies
        else discover_galaxies(source_batch_root)
    )
    if not galaxies:
        print("Нет успешных production-целей SBF-2", file=sys.stderr)
        return 2

    config = ExperimentConfig(
        normalized_sigma=args.sigma,
        e_realizations=args.e_realizations,
        random_seed=args.random_seed,
        fft_workers=args.fft_workers,
        save_ring_fft_fits=args.save_ring_fft_fits,
        save_all_branch_fits=args.save_all_branch_fits,
    )
    print(f"Source: {source_batch_root}")
    print(f"Output: {output_root}")
    print(f"Targets: {len(galaxies)}")
    print(json.dumps(asdict(config), ensure_ascii=False, indent=2))

    readiness = inspect_sources(galaxies, source_batch_root)
    print(readiness.to_string(index=False))
    if not readiness["ready"].all():
        print("Не все входы готовы; расчёт не начат", file=sys.stderr)
        return 2
    if args.dry_run:
        print("Dry run завершён: FITS не открывались, расчёты не запускались.")
        return 0

    status_table, _ = prepare_target_status(
        galaxies, output_root, config, source_batch_root
    )
    print(f"Состояние очереди: {status_path}")
    print(status_table[["galaxy", "status", "stage"]].to_string(index=False))

    results = []
    failures = []
    for index, galaxy in enumerate(galaxies, start=1):
        print("=" * 88)
        print(f"[{index}/{len(galaxies)}] {galaxy}")
        try:
            result = process_target(
                galaxy=galaxy,
                source_batch_root=source_batch_root,
                output_root=output_root,
                config=config,
                force=args.force,
                rebuild_input_cache=args.rebuild_input_cache,
                rebuild_expectation_cache=args.rebuild_expectation_cache,
            )
            results.append(result)
            failure_path = (
                output_root / "batch" / "failures"
                / f"{galaxy.replace(' ', '_')}.json"
            )
            failure_path.unlink(missing_ok=True)
            cache_state = (
                "result cache" if result.get("result_cache_hit")
                else "E(k) cache" if result.get("expectation_cache_hit")
                else "existing systematics E(k)" if result.get(
                    "expectation_bootstrapped_from"
                )
                else "computed"
            )
            print(
                f"Готово: {result['run_dir']} | {cache_state} | "
                f"closure={result['closure_passed']}"
            )
        except KeyboardInterrupt:
            update_target_status(
                galaxy, "interrupted", output_root, config,
                stage="interrupted", message="KeyboardInterrupt",
            )
            print(
                f"Прервано пользователем. Состояние сохранено в {status_path}",
                file=sys.stderr,
            )
            return 130
        except Exception as error:
            failure = {
                "status": "failed",
                "galaxy": galaxy,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            failure_root = (
                output_root / "batch" / "failures"
            )
            failure_root.mkdir(parents=True, exist_ok=True)
            (failure_root / f"{galaxy.replace(' ', '_')}.json").write_text(
                json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            update_target_status(
                galaxy, "failed", output_root, config,
                stage="failed", message=failure["error"],
            )
            print(f"ОШИБКА: {failure['error']}", file=sys.stderr)
            if args.stop_on_error:
                break

    if results:
        campaign_results = load_matching_results(
            output_root, config, source_batch_root
        )
        aggregate = write_aggregate_tables(campaign_results, output_root)
        evaluation = build_evaluation_table(campaign_results, output_root)
        print("=" * 88)
        print(evaluation.to_string(index=False))
        print("Aggregate tables:")
        for name, path in aggregate.items():
            print(f"  {name}: {path}")

    final_status, _ = prepare_target_status(
        galaxies, output_root, config, source_batch_root
    )
    print(final_status[
        ["galaxy", "status", "stage", "attempt", "message"]
    ].to_string(index=False))
    print(f"Итог: ok={len(results)}, failed={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
