# 95% confidence interval

import os
import collections
import argparse
import numpy as np


scenarios = ["mos-u2s", "mos-esd_en", "mos-esd_zh",
             "smos-u2s", "smos-esd_en", "smos-esd_zh"]


evaluators = []

for apath in os.listdir("."):
    if os.path.isdir(apath):
        evaluators.append(apath)


for scenario in scenarios:
    print(f"--- analyzing {scenario}.")
    scores = {}
    scores_sorted = {}

    # 收集该 scenario 下所有的评估者的所有模型的所有分数
    for evaluator in evaluators:
        if not os.path.exists(f"{evaluator}/{scenario}.txt"):
            continue

        # print(f"--- analyzing {evaluator}/{scenario}.")

        scores[evaluator] = {}

        with open(f"{evaluator}/{scenario}.txt", "r") as f:
            results = f.readlines()
            results = [x for x in results if x.strip() != ""]

        for line in results:
            _, path, score = line.strip().split("|")
            scores[evaluator][path] = int(score)

        s = dict(sorted(scores[evaluator].items(), key=lambda x:x[0]))
        scores_sorted[evaluator] = list(s.values())

    # 剔除离谱的评估者
    mean = np.array(list(scores_sorted.values()))
    mean = np.mean(mean, 0)

    valid_evaluators = []
    for evaluator in scores_sorted.keys():
        score = np.array(scores_sorted[evaluator])
        coef = np.corrcoef(score, mean)[0][1]
        if coef > 0.25:
            valid_evaluators.append(evaluator)
        else:
            print(f"- del {evaluator}")
        print(evaluator, coef)
    print(valid_evaluators)

    # 从合理的评估者中统计每个模型的结果
    models = collections.defaultdict(list)
    for evaluator in valid_evaluators:
        for path in scores[evaluator].keys():
            score = scores[evaluator][path]
            model = path.split("/")[-2]
            models[model].append(int(score))

    # 计算最终结果
    with open(f"analyze/delsome/{scenario}.txt", "w") as f:
        for model in models.keys():
            score = np.array(models[model])
            mean = np.mean(score)
            var = np.var(score)
            std = np.sqrt(var)
            n = len(score)
            ci = 1.96 * std / np.sqrt(n)
            f.write(f"{model}: mean={mean}, ci={ci}, variance={var}, std={std}\n")
