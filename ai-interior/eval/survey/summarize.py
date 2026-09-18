"""설문 집계 (DEVLOG §31). python eval/survey/summarize.py

조건별 평균과 95% 신뢰구간(참여자 단위 부트스트랩), 그리고 **참여자 안에서의** 짝 비교를 낸다.
같은 사람이 같은 방·가구를 두 방식으로 본 점수 차이를 보므로 사람마다 다른 채점 습관이 상쇄된다.
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
QS = [("q_position", "위치"), ("q_natural", "자연스러움"), ("q_identity", "가구 보존")]
LABEL = {"local": "로컬", "local_refine": "로컬 + 다듬기", "gemini": "Gemini (마커)"}


def boot_ci(per_rater: dict[str, list[float]], n=2000, seed=0):
    raters = list(per_rater)
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        s = [v for r in (rng.choice(raters) for _ in raters) for v in per_rater[r]]
        means.append(sum(s) / len(s))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main(path: Path = HERE / "responses.csv"):
    trials = {t["id"]: t for t in json.loads((HERE / "trials.json").read_text(encoding="utf-8"))}
    if not path.exists():
        print("아직 응답이 없습니다 (eval/survey/responses.csv)")
        return
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["trial"] in trials]
    raters = sorted({r["rater"] for r in rows})
    done = sum(1 for rt in raters if sum(1 for r in rows if r["rater"] == rt) == len(trials))
    print(f"참여자 {len(raters)}명 (끝까지 {done}명), 응답 {len(rows)}개\n")

    for part, conds in (("A", ["local", "local_refine", "gemini"]), ("B", ["local", "local_refine"])):
        print(f"[{part}] " + ("방식 비교 (파일럿 3쌍)" if part == "A" else "다듬기 효과 (5쌍)"))
        print(f"  {'조건':<14}" + "".join(f"{q[1]:>18}" for q in QS))
        for c in conds:
            line = f"  {LABEL[c]:<14}"
            for key, _ in QS:
                per = defaultdict(list)
                for r in rows:
                    t = trials[r["trial"]]
                    if t["part"] == part and t["condition"] == c:
                        per[r["rater"]].append(int(r[key]))
                allv = [v for vs in per.values() for v in vs]
                if not allv:
                    line += f"{'-':>18}"
                    continue
                lo, hi = boot_ci(per) if len(per) > 1 else (float("nan"),) * 2
                line += f"{sum(allv) / len(allv):>8.2f} [{lo:.2f},{hi:.2f}]"
            print(line)
        # 참여자 안 짝 비교: 기준(로컬) 대비
        for c in conds[1:]:
            for key, qn in QS:
                wins = ties = loses = 0
                for rt in raters:
                    for pair in {t["pair"] for t in trials.values() if t["part"] == part}:
                        def score(cond):
                            v = [int(r[key]) for r in rows if r["rater"] == rt and trials[r["trial"]]["part"] == part
                                 and trials[r["trial"]]["pair"] == pair and trials[r["trial"]]["condition"] == cond]
                            return v[0] if v else None
                        a, b = score("local"), score(c)
                        if a is None or b is None:
                            continue
                        wins += b > a; ties += b == a; loses += b < a
                print(f"    {LABEL[c]} vs 로컬 · {qn}: 높음 {wins} / 같음 {ties} / 낮음 {loses}")
        print()


if __name__ == "__main__":
    import sys
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "responses.csv")
