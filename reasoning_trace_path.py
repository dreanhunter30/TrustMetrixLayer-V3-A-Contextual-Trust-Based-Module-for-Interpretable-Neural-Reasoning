
import numpy as np
from tabulate import tabulate

def trace_reasoning_path(model, step=-1, top_n=5):
    """
    Показывает поэтапно, как признаки трансформировались по слоям,
    какие получили высокий уровень доверия и как это привело к решению.
    """

    trace_map = []

    for i, trust in enumerate(model.trust_layers):
        if not trust.log:
            continue
        record = trust.log[step]
        for j in range(len(record['input'])):
            i1 = np.asarray(record['input'][j]).flatten()[0]
            ctx = record['context'][j] if len(record['context'].shape) > 1 else record['context']
            w = np.asarray(record['weights'][j]).flatten()[0]
            o = np.asarray(record['output'][j]).flatten()[0]
            trace_map.append({
                "layer": f"Trust-{i+1}",
                "feature": f"x{j}",
                "input": float(i1),
                "context": [float(c) for c in ctx],
                "trust": float(w),
                "output": float(o),
                "influence": abs(i1 * w)
            })

    # Сортировка по влиянию
    trace_map = sorted(trace_map, key=lambda x: -x["influence"])

    # Отображение топ-N по влиянию
    print(f"🔍 Наиболее влиятельные признаки в reasoning-пути (top {top_n}):\n")
    rows = []
    for entry in trace_map[:top_n]:
        rows.append([
            entry["layer"],
            entry["feature"],
            f"{entry['input']:.3f}",
            "[" + ", ".join(f"{c:.2f}" for c in entry["context"]) + "]",
            f"{entry['trust']:.3f}",
            f"{entry['output']:.3f}",
            f"{entry['influence']:.3f}"
        ])

    headers = ["Слой", "Признак", "Вход", "Контекст", "Доверие", "Выход", "Влияние"]
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    return trace_map
