import argparse, os, sys, numpy as np
from pathlib import Path
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Observations
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib3.exceptions import ProtocolError
from urllib.parse import quote
import requests, time #Переход на собственный загрузчик для подгрузки в несколько потоков
import re

def _slug(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s.strip("._-")[:40] or "UNKNOWN"

# на всякий случай увеличим лимит
try:
    Observations.ROW_LIMIT = 100000
except Exception:
    pass

SUFFIX = {"any": ".fits", "i2d": "_i2d.fits", "cal": "_cal.fits", "rate": "_rate.fits"}

def with_retries(fn, *args, retries=5, base=1.7, **kw):
    """Экспоненциальные ретраи для нестабильных вызовов (MAST может рвать соединение)."""
    last = None
    for i in range(retries):
        try:
            return fn(*args, **kw)
        except (requests.RequestException, ProtocolError) as e:
            last = e
            if i == retries - 1:
                raise
            time.sleep(base ** i)
    raise last


def brief_report(obs, prods):
    pids = sorted(set(s(obs["proposal_id"])))
    inst = sorted(set(s(obs["instrument_name"])))
    tgts = sorted(set(s(obs["target_name"])))
    print(f"[Сводка] proposals: {pids}")
    print(f"[Сводка] instruments: {inst}")
    print(f"[Сводка] targets: {tgts[:6]}" + (" ..." if len(tgts) > 6 else ""))
    if "obs_title" in prods.colnames:
        titles = sorted(set(x for x in s(prods["obs_title"]) if x))
        print(f"[Сводка] пример obs_title: {titles[:3]}")


def s(col):
    # безопасно превращаем MaskedColumn в массив строк
    return np.asarray(getattr(col, "filled", lambda x="": col)(""), dtype=str)

def expected_size_bytes(row) -> int | None:
    # MAST иногда отдаёт размер в 'size' (байты) или 'size_kb' (килобайты)
    if "size" in row.colnames:
        try:
            return int(row["size"])
        except Exception:
            return None
    if "size_kb" in row.colnames:
        try:
            return int(float(row["size_kb"]) * 1024)
        except Exception:
            return None
    return None

def make_download_url(data_uri: str) -> str:
    base = "https://mast.stsci.edu/api/v0.1/Download/file?uri="
    return base + quote(data_uri, safe=":")

def fmt_size(n):
    try:
        return f"{n/1024/1024:.1f} MB"
    except Exception:
        return "?"

def head_content_length(url: str, headers: dict | None = None, timeout: int = 60) -> int | None:
    """Вернуть размер файла по заголовку Content-Length или None, если не удалось."""
    try:
        hdrs = {"Accept-Encoding": "identity"}
        if headers:
            hdrs.update(headers)
        r = requests.head(url, headers=hdrs, allow_redirects=True, timeout=timeout)
        if "Content-Length" in r.headers:
            return int(r.headers["Content-Length"])
    except Exception:
        return None
    return None

def download_with_resume(url: str, dest_path: Path, resume: bool = True, exp_size: int | None = None,
                         timeout: int = 120, max_retries: int = 5) -> tuple[bool, str]:
    """
    Скачивание одного файла с поддержкой докачки (HTTP Range).
    Возвращает (ok, message). ok=True, если файл корректно скачан/уже был целым.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    token = os.getenv("MAST_API_TOKEN")
    backoff = 2.0

    for attempt in range(max_retries):
        try:
            # Попробуем определить удалённый размер, если он не задан и файл уже существует
            headers = {"Accept-Encoding": "identity"}
            if token:
                headers["Authorization"] = f"token {token}"

            if dest_path.exists() and exp_size is None:
                remote = head_content_length(url, headers=headers, timeout=timeout)
                if remote:
                    exp_size = remote
                    # если локальный уже полный — пропустим
                    if dest_path.stat().st_size == exp_size:
                        return True, "cached-ok"

            # Если файл уже полный (когда exp_size известен), пропускаем
            if dest_path.exists() and exp_size and dest_path.stat().st_size == exp_size:
                return True, "cached-ok"

            # Вычислим точку возобновления
            start = dest_path.stat().st_size if (resume and dest_path.exists()) else 0
            req_headers = dict(headers)
            if start:
                req_headers["Range"] = f"bytes={start}-"

            print(f"[GET ] {dest_path.name} ({fmt_size(exp_size)}) resume={'yes' if start else 'no'}", flush=True)
            with requests.get(url, stream=True, headers=req_headers, timeout=timeout) as r:
                # Если сервер проигнорировал Range, но файл существует — перекачиваем с нуля
                if start and r.status_code == 200:
                    start = 0
                    dest_path.unlink(missing_ok=True)

                # Если сервер вернул 416 — разрулим
                if r.status_code == 416:
                    remote = head_content_length(url, headers=headers, timeout=timeout)
                    local_sz = dest_path.stat().st_size if dest_path.exists() else 0
                    if remote and local_sz >= remote:
                        return True, "cached-ok-range"
                    # иначе удалим и попробуем заново с нуля
                    dest_path.unlink(missing_ok=True)
                    raise requests.HTTPError("416 Range; retry from zero")

                r.raise_for_status()
                mode = "ab" if start else "wb"
                with open(dest_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=2**20):
                        if chunk:
                            f.write(chunk)

            # Проверка размера
            if exp_size and dest_path.stat().st_size != exp_size:
                raise IOError(f"size mismatch: {dest_path.stat().st_size} != {exp_size}")

            return True, "ok"
        except requests.exceptions.HTTPError as e:
            # 404 — постоянная ошибка: файла нет на CDN (часто candidate-ассоциации)
            if getattr(e, "response", None) is not None and e.response is not None and e.response.status_code == 404:
                return False, "404 Not Found"
            # прочие HTTP — ретраим
            wait = backoff ** attempt
            last_err = f"HTTP {getattr(e.response, 'status_code', '?')}: {e}"
            print(f"[retry {attempt + 1}/{max_retries}] {dest_path.name}: {last_err}; sleep {wait:.1f}s", flush=True)
            time.sleep(wait)
        except Exception as e:
            wait = backoff ** attempt
            last_err = str(e)
            print(f"[retry {attempt + 1}/{max_retries}] {dest_path.name}: {last_err}; sleep {wait:.1f}s", flush=True)
            time.sleep(wait)

    return False, last_err if 'last_err' in locals() else "unknown error"

def main():
    class _HelpFmt(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
        pass

    ap = argparse.ArgumentParser(
        prog="JWST-fetch.py",
        description=(
            "Скачивание продуктов MAST (JWST) по цели или из списка провалов,\n"
            "поддержка докачки (HTTP Range), параллельных потоков и фильтрации."
        ),
        formatter_class=_HelpFmt,
        epilog=(
            "Примеры:\n"
            "  py JWST-fetch.py --name \"NGC 5584\" --radius \"6 arcmin\" --level i2d --only-sci --jobs 6 --resume\n"
            "  py JWST-fetch.py --proposal 1685 --filters F090W F150W --level any --include-candidates\n"
            "  py JWST-fetch.py --from-fails ../data/download_fail.txt --second-pass --jobs 2\n"
        ),
    )
    ap.add_argument("--name", default="NGC 5584", help="Имя цели для резолва координат через SESAME/Simbad.")
    ap.add_argument("--radius", default="4 arcmin", help="Радиус конуса поиска наблюдений вокруг цели (например: '4 arcmin', '0.2 deg').")
    ap.add_argument("--level", default="any", choices=list(SUFFIX), help="Уровень/суффикс продукта по имени файла: any (.fits), i2d (_i2d.fits), cal (_cal.fits), rate (_rate.fits).")
    ap.add_argument("--only-sci", action="store_true", help="Фильтровать только productType=SCIENCE (без калибровочных/интермедиат).")
    ap.add_argument("--filters", nargs="*", default=None, help="Фильтр по имени фильтра инструмента (напр.: F090W F150W). Ищется по колонкам 'filters'/'productSubGroupDescription'.")
    ap.add_argument("--include-candidates", action="store_true", help="Включать stage-3 candidate ассоциации (имена jw<ppppp>-c####_...). По умолчанию исключаем их.")
    ap.add_argument("--proposal", nargs="*", default=None, help="Ограничить по ID программы (напр.: 1685 1638). Можно несколько.")
    ap.add_argument("--all-of-proposal", action="store_true",
                    help="Игнорировать --name/--radius и грузить ВСЕ наблюдения указанных программ (proposal_id)")
    ap.add_argument("--out", default="../data", help="Папка выгрузки (относительно каталога скрипта)")
    ap.add_argument("--dry", action="store_true", help="Сухой прогон: показать список файлов и сводку, но не скачивать.")
    ap.add_argument("--jobs", type=int, default=4, help="Параллельные закачки (потоки)")
    ap.add_argument("--resume", action="store_true", help="Докачивать недокачанные файлы (HTTP Range)")
    ap.add_argument("--timeout", type=int, default=120, help="Таймаут HTTP запроса, сек")

    ap.add_argument("--second-pass", action="store_true", help="Автодокачка провалов вторым проходом с меньшей параллелью")
    ap.add_argument("--from-fails", type=str, default=None, help="Докачать только из файла провалов (tsv: dataURI\\tobs_collection\\tobs_id\\tproductFilename)")
    ap.add_argument("--dir-scheme",
                    choices=["proposal", "object", "proposal_object"],
                    default="proposal_object",
                    help=(
                        "Схема папок под mastDownload: \n"
                        "  'proposal' → JW<id>/<category>/<file>\n"
                        "  'object'   → <TARGET>/<category>/<file>\n"
                        "  'proposal_object' → JW<id>/<TARGET>/<category>/<file>"
                    ))
    args = ap.parse_args()

    # Разрешаем путь относительно каталога скрипта
    script_dir = Path(__file__).resolve().parent
    out_dir = (script_dir / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    auto_fail_file = out_dir / "download_fail.txt"
    if args.second_pass and not args.from_fails and auto_fail_file.exists():
        args.from_fails = str(auto_fail_file)
        print(f"[Автовыбор] --from-fails {args.from_fails}")

    if not args.from_fails:
        if args.proposal and args.all_of_proposal:
            # целиком по программе
            pids = [str(p) for p in args.proposal]
            obs = with_retries(Observations.query_criteria,
                               proposal_id=pids, obs_collection="JWST")
            if len(obs) == 0:
                print("По данным программам нет наблюдений (или они ещё закрыты).", file=sys.stderr)
                sys.exit(2)
        else:
            # старое поведение: по конусу вокруг --name
            coord = with_retries(SkyCoord.from_name, args.name)
            obs = with_retries(Observations.query_region, coord, radius=u.Quantity(args.radius))
            obs = obs[s(obs["obs_collection"]) == "JWST"]
            if args.proposal:
                mask_pid = np.isin(s(obs["proposal_id"]), np.array(args.proposal, dtype=str))
                obs = obs[mask_pid]
            if len(obs) == 0:
                print("JWST наблюдений не найдено (увеличь --radius или используй --all-of-proposal).", file=sys.stderr)
                sys.exit(2)

        # 3) продукты по найденным наблюдениям (важно: передаём ТАБЛИЦУ, а не obsids)
        try:
            prods = with_retries(Observations.get_unique_product_list, obs)
        except Exception:
            prods = with_retries(Observations.get_product_list, obs)
        if len(prods) == 0:
            print("Список продуктов пуст.", file=sys.stderr)
            sys.exit(3)
        # Рекомендуемая фильтрация: минимально рекомендуемый набор и только FITS
        try:
            prods = Observations.filter_products(
                prods,
                mrp_only=True,
                extension="fits",
                productType=["SCIENCE"] if args.only_sci else None
            )
        except Exception:
            pass

        # 4) базовые фильтры
        mask = s(prods["obs_collection"]) == "JWST"
        if args.only_sci and "productType" in prods.colnames:
            mask &= s(prods["productType"]) == "SCIENCE"

        # уровень продукции по суффиксу имени файла
        if "productFilename" not in prods.colnames:
            print("Нет productFilename в выдаче MAST.", file=sys.stderr)
            sys.exit(4)
        fn = s(prods["productFilename"])
        suf = SUFFIX[args.level]
        mask &= np.char.endswith(fn, suf)
        # По умолчанию выкидываем stage-3 candidate (jw<ppppp>-c####_...) — часто нет файлов на CDN → 404
        if not args.include_candidates:
            # правильный паттерн: одна обратная косая в raw-строке
            is_cand = np.array([bool(re.match(r"^jw\d{5}-c\d{4}_", x)) for x in fn])
            mask &= ~is_cand

        # опционально: фильтр по фильтрам инструмента
        if args.filters:
            ok = np.zeros(len(prods), dtype=bool)
            for c in ("filters", "productSubGroupDescription"):
                if c in prods.colnames:
                    col = s(prods[c])
                    ok |= np.array([any(f in x for f in args.filters) for x in col], bool)
            mask &= ok

        take = prods[mask]
        if len(take) == 0:
            print("После фильтров ничего не осталось. Запусти без --only-sci и с --level any для диагностики.", file=sys.stderr)
            sys.exit(5)

        # Убираем дубли: уникально по dataURI (это один и тот же физический ресурс)
        if "dataURI" in take.colnames:
            keys = s(take["dataURI"])
            _, uniq_idx = np.unique(keys, return_index=True)
            take = take[np.sort(uniq_idx)]
        else:
            # запасной вариант — по имени файла
            keys = s(take["productFilename"])
            _, uniq_idx = np.unique(keys, return_index=True)
            take = take[np.sort(uniq_idx)]

        # 5) короткий вывод
        print(f"Файлов к закачке: {len(take)}")
        for i, name in enumerate(s(take['productFilename'])[:20], 1):
            print(f"{i:2d}. {name}")
        brief_report(obs, prods)
        if args.dry:
            return
    else:
        # Режим докачки из файла провалов (tsv: dataURI \t obs_collection \t obs_id \t productFilename)
        fail_file = Path(args.from_fails).expanduser().resolve()
        if not fail_file.exists():
            print(f"Файл не найден: {fail_file}", file=sys.stderr)
            sys.exit(6)
        # Построим минимальную таблицу-like структуру для унифицированного планирования задач
        class RowLike(dict):
            def __getattr__(self, k): return self[k]
        take = []
        with fail_file.open("r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) != 4:
                    continue
                data_uri, obs_coll, obs_id, fname = parts
                take.append(RowLike(dataURI=data_uri, obs_collection=obs_coll, obs_id=obs_id, productFilename=fname))
        if not take:
            print("Список докачки пуст.", file=sys.stderr)
            sys.exit(0)
        print(f"[Режим докачки] элементов: {len(take)}")

    # Формируем задания
    tasks = []
    seen_paths: set[str] = set()
    has_cols = hasattr(take, 'colnames')
    for row in take:
        if hasattr(row, "dataURI"):
            data_uri = str(row.dataURI)
        else:
            data_uri = str(row["dataURI"])
        if hasattr(row, "obs_collection"):
            obs_coll = str(row.obs_collection)
        else:
            obs_coll = str(row["obs_collection"])
        if hasattr(row, "obs_id"):
            obs_id = str(row.obs_id)
        else:
            obs_id = str(row["obs_id"])
        if hasattr(row, "productFilename"):
            fname = str(row.productFilename)
        else:
            fname = str(row["productFilename"])
        url = make_download_url(data_uri)
        # Bucket by detector: nrca1..4 -> nrca, nrcb1..4 -> nrcb, nrcalong -> nrcalong, nrcblong -> nrcblong
        m = re.search(r'_(nrca(?:long|[1-4])|nrcb(?:long|[1-4]))_', fname.lower())
        if m:
            tag = m.group(1)
            if tag.startswith('nrca'):
                category = 'nrcalong' if 'long' in tag else 'nrca'
            else:
                category = 'nrcblong' if 'long' in tag else 'nrcb'
        else:
            category = 'unknown'

        # приоритет: proposal_id -> target_name -> UNKNOWN
        proposal_tag = None
        if has_cols and 'proposal_id' in take.colnames:
            try:
                pid = int(row['proposal_id'])
                proposal_tag = f"JW{pid:05d}"
            except Exception:
                proposal_tag = None

        object_tag = None
        for cname in ("target_name", "obs_target_name", "objName"):
            if has_cols and cname in getattr(take, 'colnames', ()):  # astropy Table case
                val = str(row[cname]).strip()
                if val:
                    object_tag = _slug(val)
                    break

        # схема раскладки папок
        if args.dir_scheme == "proposal":
            folder_root = proposal_tag or object_tag or "UNKNOWN"
        elif args.dir_scheme == "object":
            folder_root = object_tag or proposal_tag or "UNKNOWN"
        else:  # proposal_object
            p = proposal_tag or "UNKNOWN"
            t = object_tag or "UNKNOWN"
            folder_root = Path(p) / t

        local_path = out_dir / "mastDownload" / folder_root / category / fname
        # защита от гонок: не планируем две закачки в один и тот же путь
        lp_str = str(local_path)
        if lp_str in seen_paths:
            continue
        seen_paths.add(lp_str)

        exp_size = None
        try:
            exp_size = int(row["size"])
        except Exception:
            try:
                exp_size = int(float(row["size_kb"]) * 1024)
            except Exception:
                exp_size = None
        tasks.append((data_uri, url, local_path, exp_size))

    todo_file = out_dir / "download_todo.txt"
    with todo_file.open("w") as f:
        for data_uri, url, local_path, exp_size in tasks:
            parts = Path(local_path).parts
            try:
                idx = parts.index("mastDownload")
                folder_tag = "/".join(parts[idx+1:-2])
            except ValueError:
                folder_tag = parts[-3] if len(parts) >= 3 else ""
            category = parts[-2] if len(parts) >= 2 else ""
            fname = parts[-1]
            f.write(f"{data_uri}\t{folder_tag}\t{category}\t{fname}\n")
    print(f"[Файл] TODO → {todo_file}", flush=True)

    # Параллельная загрузка
    ok, skip, fail, scheduled = 0, 0, 0, 0
    ok_list, fail_list = [], []
    futures = {}
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
            for data_uri, url, local_path, exp_size in tasks:
                if local_path.exists() and exp_size and local_path.stat().st_size == exp_size:
                    skip += 1
                    ok_list.append((data_uri, str(local_path)))
                    continue
                fut = ex.submit(download_with_resume, url, local_path, args.resume, exp_size, args.timeout)
                futures[fut] = (data_uri, local_path)
                scheduled += 1

            total_jobs = scheduled
            print(f"[QUEUE] scheduled={scheduled}, skipped={skip}, total={scheduled + skip}", flush=True)

            done = 0
            for fut in as_completed(futures):
                data_uri, path = futures[fut]
                try:
                    ok_flag, msg = fut.result()
                    done += 1
                    if ok_flag:
                        ok += 1
                        ok_list.append((data_uri, str(path)))
                        print(f"[DONE {done}/{total_jobs}] OK   {path.name}", flush=True)
                    else:
                        fail += 1
                        fail_list.append((data_uri, str(path)))
                        print(f"[DONE {done}/{total_jobs}] FAIL {path.name}: {msg}", flush=True)
                except Exception as e:

                    fail += 1
                    fail_list.append((data_uri, str(path)))
                    print(f"[FAIL {done}/{total_jobs}] {path.name}: {e}", flush=True)
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Прервано пользователем, сохраняю хвост как FAIL…", flush=True)
    finally:
        # Добавим незавершённые задачи в FAIL
        for fut, (data_uri, path) in futures.items():
            if not fut.done():
                fail_list.append((data_uri, str(path)))

        total = ok + skip + len(fail_list)
        print(f"\n[Итог] total={total}, ok={ok}, skip={skip}, fail={len(fail_list)}", flush=True)

        # Сохраняем манифесты
        ok_file = out_dir / "download_ok.txt"
        fail_file = out_dir / "download_fail.txt"
        with ok_file.open("w") as f:
            for data_uri, path in ok_list:
                parts = Path(path).parts
                try:
                    idx = parts.index("mastDownload")
                    folder_tag = "/".join(parts[idx+1:-2])
                except ValueError:
                    folder_tag = parts[-3] if len(parts) >= 3 else ""
                category = parts[-2] if len(parts) >= 2 else ""
                fname = parts[-1]
                f.write(f"{data_uri}\t{folder_tag}\t{category}\t{fname}\n")
        with fail_file.open("w") as f:
            for data_uri, path in fail_list:
                parts = Path(path).parts
                try:
                    idx = parts.index("mastDownload")
                    folder_tag = "/".join(parts[idx+1:-2])
                except ValueError:
                    folder_tag = parts[-3] if len(parts) >= 3 else ""
                category = parts[-2] if len(parts) >= 2 else ""
                fname = parts[-1]
                f.write(f"{data_uri}\t{folder_tag}\t{category}\t{fname}\n")
        print(f"[Файлы] OK → {ok_file} | FAIL → {fail_file}", flush=True)

    # Мягкий второй проход
    if args.second_pass and fail_list:
        print(f"[2nd] пытаюсь докачать {len(fail_list)} файлов…", flush=True)
        jobs2 = max(1, args.jobs // 2)
        timeout2 = max(args.timeout, 300)
        ok2 = 0
        with ThreadPoolExecutor(max_workers=jobs2) as ex:
            futures2 = {
                ex.submit(download_with_resume, make_download_url(du), Path(p), True, None, timeout2): Path(p)
                for du, p in fail_list
            }
            for fut in as_completed(futures2):
                p = futures2[fut]
                try:
                    ok_flag, _ = fut.result()
                    if ok_flag:
                        ok2 += 1
                        print(f"[OK 2nd] {p.name}", flush=True)
                    else:
                        print(f"[FAIL 2nd] {p.name}", flush=True)
                except Exception as e:
                    print(f"[FAIL 2nd] {p.name}: {e}", flush=True)
        print(f"[Итог 2nd] докачано {ok2}/{len(fail_list)}", flush=True)

if __name__ == "__main__":
    main()