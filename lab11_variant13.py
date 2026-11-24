"""
Лабораторная работа № 11. Вариант 13.
Кугушева Арина, группа БИ2-1.
Дисциплина: Теория алгоритмов.

Наивный поиск подстроки, алгоритм Кнута — Морриса — Пратта, расстояние
Левенштейна, сопоставление наименований контрагентов.

Условия варианта 13:
  * образец для ручного расчёта префикс-функции: abcaabc;
  * тип текста: случайный; длины n = 10^4; 5·10^4; 10^5;
  * порог сходства 0,80;
  * Л6: блокирование кандидатов (по первой букве и по длине строки),
    измерение сокращения числа сравнений.

Запуск:
    python lab11_variant13.py                       # все эксперименты
    python lab11_variant13.py --plots plots         # + графики (matplotlib)
    python lab11_variant13.py --json results.json   # + результаты в JSON
"""
import argparse
import gc
import json
import random
import re
import statistics
import time

SEED = 42
SIZES = [10 ** 4, 5 * 10 ** 4, 10 ** 5]
PATTERN = "abcaabc"
ALPHABET = "abcd"
THRESHOLD = 0.80
POOL_SIZES = [200, 400, 800]            # число записей в системе А для Л6


# ====================================================================
# Поиск подстроки
# ====================================================================
def naive_search(text, pattern):
    """Наивный поиск. Возвращает (позиции, число сравнений символов)."""
    n, m = len(text), len(pattern)
    positions, comparisons = [], 0
    for i in range(n - m + 1):
        j = 0
        while j < m:
            comparisons += 1
            if text[i + j] != pattern[j]:
                break
            j += 1
        if j == m:
            positions.append(i)
    return positions, comparisons


def prefix_function(pattern):
    """Префикс-функция: π[i] — длина наибольшего собственного префикса
    подстроки P[0..i], совпадающего с её суффиксом."""
    pi = [0] * len(pattern)
    k = 0
    for i in range(1, len(pattern)):
        while k > 0 and pattern[i] != pattern[k]:
            k = pi[k - 1]
        if pattern[i] == pattern[k]:
            k += 1
        pi[i] = k
    return pi


def prefix_function_trace(pattern):
    """Префикс-функция с протоколом шагов (для ручного расчёта)."""
    pi = [0] * len(pattern)
    k = 0
    steps = [{"i": 0, "prefix": pattern[0], "fallbacks": [], "pi": 0}]
    for i in range(1, len(pattern)):
        fallbacks = []
        while k > 0 and pattern[i] != pattern[k]:
            fallbacks.append((k, pi[k - 1]))
            k = pi[k - 1]
        if pattern[i] == pattern[k]:
            k += 1
        pi[i] = k
        steps.append({"i": i, "prefix": pattern[:i + 1], "fallbacks": fallbacks, "pi": k})
    return pi, steps


def kmp_search(text, pattern):
    """Алгоритм Кнута — Морриса — Пратта. Θ(n + m)."""
    n, m = len(text), len(pattern)
    pi = prefix_function(pattern)
    positions, comparisons = [], 0
    k = 0
    for i in range(n):
        while k > 0 and text[i] != pattern[k]:
            comparisons += 1
            k = pi[k - 1]
        comparisons += 1
        if text[i] == pattern[k]:
            k += 1
        if k == m:
            positions.append(i - m + 1)
            k = pi[k - 1]
    return positions, comparisons


def reference_search(text, pattern):
    """Эталон для проверки: все (в том числе перекрывающиеся) вхождения."""
    return [mt.start() for mt in re.finditer("(?=" + re.escape(pattern) + ")", text)]


# ====================================================================
# Расстояние Левенштейна
# ====================================================================
def levenshtein(a, b):
    """Расстояние Левенштейна. Θ(nm) времени, Θ(min(n, m)) памяти."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1,                  # удаление
                               current[j - 1] + 1,               # вставка
                               previous[j - 1] + (ca != cb)))    # замена
        previous = current
    return previous[-1]


def levenshtein_table(a, b):
    """Полная таблица D размера (len(a)+1) × (len(b)+1)."""
    D = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        D[i][0] = i
    for j in range(len(b) + 1):
        D[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1,
                          D[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    return D


def edit_path(D, a, b):
    """Обратный проход по таблице: клетки пути и список операций."""
    i, j = len(a), len(b)
    cells, ops = [(i, j)], []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            ops.append(("совпадение" if a[i - 1] == b[j - 1] else "замена", a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            ops.append(("удаление", a[i - 1], ""))
            i -= 1
        else:
            ops.append(("вставка", "", b[j - 1]))
            j -= 1
        cells.append((i, j))
    return cells[::-1], ops[::-1]


def similarity(a, b):
    """Нормированная мера сходства из [0; 1]."""
    if not a and not b:
        return 1.0
    return 1 - levenshtein(a, b) / max(len(a), len(b))


# ====================================================================
# Сопоставление записей
# ====================================================================
LEGAL_FORMS = {"ооо", "оао", "зао", "пао", "ао", "ип"}


def normalize(name):
    """Нормализация: нижний регистр, ё→е, удаление кавычек и знаков
    препинания, удаление организационно-правовых форм, сжатие пробелов."""
    s = name.lower().replace("ё", "е")
    s = re.sub(r"[\"'«»„“”.,()]", " ", s)
    return " ".join(w for w in s.split() if w not in LEGAL_FORMS)


def length_ok(la, lb, threshold):
    """Без потерь: sim ≥ t возможно только при |la − lb| ≤ (1 − t)·max(la, lb)."""
    return abs(la - lb) <= (1 - threshold) * max(la, lb) + 1e-9


def candidates_ok(na, nb, mode, threshold):
    """Правило блокирования: none / length / first / both."""
    if mode in ("first", "both") and (not na or not nb or na[0] != nb[0]):
        return False
    if mode in ("length", "both") and not length_ok(len(na), len(nb), threshold):
        return False
    return True


def match_records(list_a, list_b, threshold=THRESHOLD, mode="none"):
    """Сопоставление двух списков. Сначала точное сравнение нормализованных
    строк (хеш-таблица), затем нечёткое по расстоянию Левенштейна.

    Возвращает (список результатов, число вызовов similarity).
    Результат: (запись A, индекс записи B или None, сходство, тип).
    """
    norm_b = [normalize(b) for b in list_b]
    index = {}
    for j, nb in enumerate(norm_b):
        index.setdefault(nb, j)
    result = [None] * len(list_a)
    used_b = set()
    rest = []
    for i, a in enumerate(list_a):                      # 1. точное
        na = normalize(a)
        j = index.get(na)
        if j is not None and j not in used_b:
            result[i] = (a, j, 1.0, "точное")
            used_b.add(j)
        else:
            rest.append((i, na))
    free_b = [(j, nb) for j, nb in enumerate(norm_b) if j not in used_b]
    calls = 0
    for i, na in rest:                                   # 2. нечёткое
        best_j, best_s = None, 0.0
        for j, nb in free_b:
            if not candidates_ok(na, nb, mode, threshold):
                continue
            calls += 1
            s = similarity(na, nb)
            if s > best_s:
                best_j, best_s = j, s
        if best_j is not None and best_s >= threshold:
            result[i] = (list_a[i], best_j, best_s, "нечёткое")
        else:
            result[i] = (list_a[i], None, best_s, "не найдено")
    return result, calls


# ====================================================================
# Данные
# ====================================================================
DEMO_A = ['ООО "Ромашка"', "АО Василёк", "ИП Иванов И.И.", 'ООО «Северный ветер»',
          "ПАО Гранит", 'ООО "Альфа-Строй"', "ЗАО Технопарк", 'ООО «Меридиан»',
          'АО "Прогресс-Тех"', "ООО Вектор"]
DEMO_B = ["Ромашка, ООО", "ВАСИЛЕК АО", "Иванов Иван Иванович", "Северный Ветер ООО",
          "Гранит-Строй ПАО", "Альфа Строй ООО", "Технопарк, ЗАО", "Меридан ООО",
          "Прогресс Тех АО", "Вектр ООО"]

ADJ = ["Северный", "Южный", "Восточный", "Западный", "Золотой", "Серебряный", "Новый", "Первый",
       "Большой", "Малый", "Красный", "Синий", "Зелёный", "Белый", "Сибирский", "Уральский",
       "Донской", "Волжский", "Балтийский", "Невский"]
NOUN = ["ветер", "стандарт", "альянс", "мастер", "ресурс", "трейд", "проект", "сервис",
        "континент", "горизонт", "авангард", "прогресс", "импульс", "вектор", "маяк",
        "компас", "фаворит", "лидер", "партнёр", "эксперт", "союз", "гарант", "база", "оптима", "заря",
        "строй", "монтаж", "логистик", "агро", "энерго", "техно", "интер", "метиз", "кристалл", "титан",
        "орбита", "вершина", "родник", "фортуна", "магистраль", "элемент", "пилот", "статус", "аргумент", "дельта"]
FORMS = ["ООО", "АО", "ПАО", "ЗАО"]


def make_pool(n, seed=SEED, typo_share=0.5, first_typo_share=0.1):
    """Две «системы» с n записями; запись i в А соответствует записи i в Б.

    В Б часть наименований искажена опечаткой (1–2 символа); у доли
    first_typo_share искажённых опечатка затрагивает первую букву."""
    rng = random.Random(seed)
    base = [(a + " " + b) for a in ADJ for b in NOUN]
    rng.shuffle(base)
    base = base[:n]
    sysA, sysB = [], []
    for name in base:
        form = rng.choice(FORMS)
        sysA.append(rng.choice([f'{form} "{name}"', f"{form} «{name}»", f"{form} {name}"]))
        nm = name
        if rng.random() < typo_share:
            for _ in range(rng.choice([1, 1, 2])):
                pos = 0 if rng.random() < first_typo_share else rng.randrange(1, len(nm))
                kind = rng.choice(["sub", "del", "ins"])
                letter = rng.choice("абвгдежзиклмнопрстуф")
                if kind == "sub":
                    nm = nm[:pos] + letter + nm[pos + 1:]
                elif kind == "del" and len(nm) > 4:
                    nm = nm[:pos] + nm[pos + 1:]
                else:
                    nm = nm[:pos] + letter + nm[pos:]
        style = rng.choice([f"{nm}, {form}", f"{nm.upper()} {form}", f"{nm} {form}"])
        sysB.append(style)
    order = list(range(n))
    rng.shuffle(order)                                   # перемешиваем Б
    sysB_shuffled = [sysB[k] for k in order]
    truth = {i: order.index(i) for i in range(n)}        # A[i] ↔ B[truth[i]]
    return sysA, sysB_shuffled, truth


# ====================================================================
# Измерения
# ====================================================================
def measure(func, *args, repeats=5):
    """Медиана времени по нескольким запускам, сборщик мусора отключён."""
    times, result = [], None
    for _ in range(repeats):
        gc.collect()
        gc.disable()
        t0 = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter() - t0)
        gc.enable()
    return statistics.median(times), result


def experiment_search(repeats):
    rng = random.Random(SEED)
    rows = []
    for n in SIZES:
        cases = [("случайный", "".join(rng.choice(ALPHABET) for _ in range(n)), PATTERN),
                 ("с повторами", "a" * n, "a" * 50 + "b")]
        for label, text, pat in cases:
            t_naive, (p1, c1) = measure(naive_search, text, pat, repeats=repeats)
            t_kmp, (p2, c2) = measure(kmp_search, text, pat, repeats=repeats)
            ref = reference_search(text, pat) if label == "случайный" else []
            assert p1 == p2 == ref, "алгоритмы дали разные позиции"
            rows.append({"n": n, "text": label, "pattern": pat, "t_naive": t_naive, "t_kmp": t_kmp,
                         "c_naive": c1, "c_kmp": c2, "occurrences": len(p1),
                         "first_positions": p1[:5]})
    return rows


def experiment_prefix():
    pi, steps = prefix_function_trace(PATTERN)
    assert pi == prefix_function(PATTERN)
    return {"pattern": PATTERN, "pi": pi, "steps": steps,
            "example_abcabd": prefix_function("abcabd")}


def experiment_edit():
    pairs = [("кошка", "ложка"), ("кот", "скот"), ("Ромашка", "Ромашки"), ("аналитика", "аналитик")]
    out = {"pairs": []}
    for a, b in pairs:
        D = levenshtein_table(a, b)
        assert D[-1][-1] == levenshtein(a, b)
        cells, ops = edit_path(D, a, b)
        out["pairs"].append({"a": a, "b": b, "d": D[-1][-1], "sim": similarity(a, b),
                             "table": D, "path": cells, "ops": ops})
    return out


def experiment_demo():
    res, calls = match_records(DEMO_A, DEMO_B, THRESHOLD)
    rows = []
    for a, j, s, kind in res:
        rows.append({"a": a, "b": DEMO_B[j] if j is not None else None, "sim": s, "kind": kind,
                     "norm_a": normalize(a), "norm_b": normalize(DEMO_B[j]) if j is not None else None})
    return {"rows": rows, "fuzzy_calls": calls}


def experiment_blocking(repeats=3):
    out = []
    for n in POOL_SIZES:
        A, B, truth = make_pool(n)
        row = {"N": n}
        # число записей, дошедших до нечёткого этапа
        nb = {normalize(b) for b in B}
        rest_a = sum(1 for a in A if normalize(a) not in nb)
        row["rest_a"] = rest_a
        base_pairs = None
        for mode in ["none", "length", "first", "both"]:
            t, (res, calls) = measure(match_records, A, B, THRESHOLD, mode, repeats=repeats)
            found = {i: r[1] for i, r in enumerate(res) if r[1] is not None}
            correct = sum(1 for i, j in found.items() if truth[i] == j)
            if mode == "none":
                base_pairs = found
            row[mode] = {"calls": calls, "time": t, "found": len(found), "correct": correct,
                         "same_as_full": found == base_pairs}
        out.append(row)
    return out


def check_correctness(trials=300):
    """Случайные тесты: КМП и наивный поиск против эталона, Левенштейн
    против полной таблицы, нормированная мера на границах."""
    rng = random.Random(7)
    for _ in range(trials):
        text = "".join(rng.choice("ab") for _ in range(rng.randint(0, 40)))
        pat = "".join(rng.choice("ab") for _ in range(rng.randint(1, 6)))
        assert naive_search(text, pat)[0] == kmp_search(text, pat)[0] == reference_search(text, pat)
        a = "".join(rng.choice("abc") for _ in range(rng.randint(0, 8)))
        b = "".join(rng.choice("abc") for _ in range(rng.randint(0, 8)))
        assert levenshtein(a, b) == levenshtein_table(a, b)[-1][-1] == levenshtein(b, a)
        assert 0 <= similarity(a, b) <= 1
    assert similarity("abc", "abc") == 1.0 and similarity("", "") == 1.0
    assert prefix_function("abcabd") == [0, 0, 0, 1, 2, 0]
    # блокирование по длине не должно терять пары (без потерь)
    A, B, _ = make_pool(60, seed=3)
    r0, _ = match_records(A, B, THRESHOLD, "none")
    r1, _ = match_records(A, B, THRESHOLD, "length")
    assert [x[1] for x in r0] == [x[1] for x in r1]
    return True


# ====================================================================
# Вывод
# ====================================================================
def f(x, d=4):
    return f"{x:.{d}f}".replace(".", ",")


def print_report(res):
    print("=" * 76)
    print("Таблица 1. Поиск подстроки (медиана времени)")
    print(f"{'n':>8}{'текст':>13}{'наивный, с':>12}{'КМП, с':>10}{'сравн. (наив.)':>16}{'сравн. (КМП)':>14}{'вхожд.':>8}")
    for r in res["search"]:
        print(f"{r['n']:>8}{r['text']:>13}{f(r['t_naive']):>12}{f(r['t_kmp']):>10}"
              f"{r['c_naive']:>16}{r['c_kmp']:>14}{r['occurrences']:>8}")
    p = res["prefix"]
    print(f"\nПрефикс-функция для '{p['pattern']}':", p["pi"])
    for s in p["steps"]:
        fb = "; ".join(f"k={a}→{b}" for a, b in s["fallbacks"]) or "—"
        print(f"  i={s['i']} префикс '{s['prefix']}': откаты {fb}; π={s['pi']}")
    print("\nРасстояние Левенштейна:")
    for e in res["edit"]["pairs"]:
        print(f"  d('{e['a']}', '{e['b']}') = {e['d']}, сходство {f(e['sim'], 3)}")
    print("\nСопоставление контрагентов (порог 0,80):")
    for r in res["demo"]["rows"]:
        print(f"  {r['a']:<24} → {r['b'] or '—':<22} {f(r['sim'], 3)} ({r['kind']})")
    print("\nЛ6. Блокирование кандидатов")
    print(f"{'N':>5}{'режим':>8}{'сравнений':>11}{'время, с':>10}{'найдено':>9}{'верных':>8}{'как без блок.':>15}")
    for r in res["blocking"]:
        for mode in ["none", "length", "first", "both"]:
            x = r[mode]
            print(f"{r['N']:>5}{mode:>8}{x['calls']:>11}{f(x['time']):>10}{x['found']:>9}{x['correct']:>8}"
                  f"{'да' if x['same_as_full'] else 'нет':>15}")


def save_plots(res, directory):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(directory, exist_ok=True)
    for label, fname in [("случайный", "search_random.png"), ("с повторами", "search_repeat.png")]:
        rows = [r for r in res["search"] if r["text"] == label]
        ns = [r["n"] for r in rows]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        axes[0].plot(ns, [r["t_naive"] for r in rows], "o-", label="наивный")
        axes[0].plot(ns, [r["t_kmp"] for r in rows], "s-", label="КМП")
        axes[0].set_xscale("log")
        axes[0].set_xlabel("n")
        axes[0].set_ylabel("время, с")
        axes[0].set_title("Время поиска")
        axes[0].legend()
        axes[0].grid(alpha=.3)
        axes[1].plot(ns, [r["c_naive"] for r in rows], "o-", label="наивный")
        axes[1].plot(ns, [r["c_kmp"] for r in rows], "s-", label="КМП")
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("n")
        axes[1].set_ylabel("сравнений символов")
        axes[1].set_title("Число сравнений")
        axes[1].legend()
        axes[1].grid(alpha=.3)
        fig.suptitle(f"Текст: {label}", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(directory, fname), dpi=200)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    Ns = [r["N"] for r in res["blocking"]]
    for mode, lab in [("none", "без блокирования"), ("length", "по длине"), ("first", "по первой букве"),
                      ("both", "по длине и первой букве")]:
        axes[0].plot(Ns, [r[mode]["calls"] for r in res["blocking"]], "o-", label=lab)
        axes[1].plot(Ns, [r[mode]["time"] for r in res["blocking"]], "o-", label=lab)
    axes[0].set_yscale("log")
    axes[0].set_title("Число вызовов similarity")
    axes[1].set_yscale("log")
    axes[1].set_title("Время нечёткого сопоставления, с")
    for ax in axes:
        ax.set_xlabel("число записей N")
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(directory, "blocking.png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plots", metavar="DIR", help="каталог для графиков")
    ap.add_argument("--json", metavar="FILE", help="файл для результатов JSON")
    ap.add_argument("--repeats", type=int, default=5, help="число повторов замера")
    a = ap.parse_args()
    check_correctness()
    print("Проверка корректности на случайных тестах: пройдена")
    res = {"search": experiment_search(a.repeats), "prefix": experiment_prefix(),
           "edit": experiment_edit(), "demo": experiment_demo(),
           "blocking": experiment_blocking()}
    print_report(res)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=1)
    if a.plots:
        save_plots(res, a.plots)


if __name__ == "__main__":
    main()
