from pathlib import Path
from pysiril.siril import Siril

SEQ = "nrcblong_"            # базовое имя последовательности (как в Siril, без номера кадра)
INDIR = "/path/to/seq_dir"   # где лежат кадры последовательности
OUTDIR = "/path/to/seq_dir/stretch"  # куда класть результаты

s = Siril()
s.start()  # поднимет siril в headless (-c)
s.exec(f'cd "{INDIR}"')
s.exec(f'load {SEQ}')        # загружаем последовательность (точка . = текущая)
# Получим число кадров из seqstat:
s.exec('seqstat . /tmp/_seqstat.txt')
txt = Path('/tmp/_seqstat.txt').read_text().strip().splitlines()
n = len(txt) - 1  # первая строка — заголовок

Path(OUTDIR).mkdir(parents=True, exist_ok=True)
for i in range(1, n+1):
    s.exec(f'select {i}')                    # выбрать кадр i
    s.exec('autostretch')                    # применить MTF-автотреш к данным
    s.exec(f'save "{OUTDIR}/stretch_{i:05d}.fits"')  # сохранить кадр
s.stop()
print(f"Done: {n} frames → {OUTDIR}")