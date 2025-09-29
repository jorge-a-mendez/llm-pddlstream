'''
Functions to analyze the results of PDDLStream experiments.

Results are stored in the 'results/' directory, with each experiment in its own subdirectory. The subdirectory names indicate the parameters used in the experiment, given by the following nested loops:

for seed in `seq 0 9`; do
    for algorithm in "sesame" "adaptive"; do
        for llm_args in "" "-integrated_llm -thinking_llm" "-pddl_llm -thinking_llm" "-pose_sampler_llm -thinking_llm" "-pddl_llm -pose_sampler_llm -thinking_llm" "-integrated_llm" "-pddl_llm" "-pose_sampler_llm" "-pddl_llm -pose_sampler_llm"; do
            for domain_command_args in "examples.pybullet.tamp.run -problem packed -n 3" "examples.pybullet.tamp.run -problem packed -n 4" "examples.pybullet.tamp.run -problem packed -n 5" "examples.pybullet.tamp.run -problem blocked -n 1" "examples.pybullet.turtlebot_rovers.run -n 2" "examples.pybullet.turtlebot_rovers.run -n 3" "examples.pybullet.turtlebot_rovers.run -n 4"; do

                # If algorithm is sesame and the domain contains "turtlebot", skip this iteration
                if [[ "$algorithm" == "sesame" && "$domain_command_args" == *"turtlebot"* ]]; then
                    continue
                fi
                
                echo "Running $algorithm$llm_args$domain_command_args with seed $seed"
                export GEMINI_API_KEY=<your_key>
                sudo -E nice -10 python -m $domain_command_args -t 300 -a $algorithm $llm_args -seed $seed --results_dir results/"$algorithm$llm_args$domain_command_args"_$seed > out_"$algorithm$llm_args$domain_command_args"_$seed.log 2>&1 &

                # If we already have MAX_JOBS running, wait for one to finish
                while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
                    sleep 1
                done

                # Wait for 60 seconds to prevent incorrect token limit failures
                sleep 60
            done
        done
    done
done

The following outcomes are possible for each experiment:
    - Solved: The experiment completed successfully and found a solution.
    - Not solved: The experiment completed but did not find a solution.
    - Timeout: The experiment did not complete within the allotted time.
    - APIError: The experiment encountered Gemini APIError due to using up the 1M TPM quota.

In the first three cases, there will be results directory with two files:
    - store.pkl -- dictionary with keys: 'summary', 'cost_over_time', 'best_plan', 'best_cost', 'all_plans'
    - llm_info.pkl -- dictionary with keys: 'integrated_time', 'pddl_time', 'sampling_time', 'integrated_input_tokens', 'integrated_thinking_tokens', 'integrated_output_tokens', 'pddl_input_tokens', 'pddl_thinking_tokens', 'pddl_output_tokens', 'sampling_input_tokens', 'sampling_thinking_tokens', 'sampling_output_tokens', 'pddl_failures', 'num_invalid_actions', 'num_preimages_not_achieved', 'num_axioms_not_achieved', 'local_sampling_failures', 'num_samples_in_collision', 'num_samples_not_stable', 'num_no_plans_returned', 'num_no_samples_returned', 'num_samples_format_failures', 'num_samples_out_of_bounds', 'num_samples_not_visible', 'num_samples_out_of_range', 'num_samples_not_reachable', 'num_samples_not_optimistic', 'num_samples_used', 'num_backtracking_failures'

In the second case, the APIError can be found in the log file.
'''

import os
import pickle
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from glob import glob
from matplotlib import pyplot as plt
import compactletterdisplay as cld

from itertools import combinations
from scipy.stats import binomtest, wilcoxon, ttest_rel
from statsmodels.stats.multitest import multipletests

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 18
color_list = ["#086E96", "#FE4D03", "#5FCCBE", "#9E9E9E"]
color_list_5 = color_list + ["#7030A0"]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter
import seaborn as sns

import numpy as np
from math import pi

algorithm_map = {
    "adaptive-pose_sampler_llm -thinking_llm": ("Adaptive", "Thinking", "Poses"),
    "sesame-integrated_llm -thinking_llm": ("Bilevel", "Thinking", "Integrated"),
    "adaptive-pddl_llm -thinking_llm": ("Adaptive", "Thinking", "PDDL"),
    "adaptive-integrated_llm -thinking_llm": ("Adaptive", "Thinking", "Integrated"),
    "sesame": ("Bilevel", "No LLM"),
    "sesame-integrated_llm": ("Bilevel", "Direct", "Integrated"),
    "sesame-pddl_llm -pose_sampler_llm -thinking_llm": ("Bilevel", "Thinking", "PDDL+Poses"),
    "adaptive-pddl_llm -pose_sampler_llm": ("Adaptive", "Direct", "PDDL+Poses"),
    "adaptive": ("Adaptive", "No LLM"),
    "sesame-pddl_llm -thinking_llm": ("Bilevel", "Thinking", "PDDL"),
    "adaptive-pddl_llm -pose_sampler_llm -thinking_llm": ("Adaptive", "Thinking", "PDDL+Poses"),
    "adaptive-pddl_llm": ("Adaptive", "Direct", "PDDL"),
    "sesame-pose_sampler_llm -thinking_llm": ("Bilevel", "Thinking", "Poses"),
    "sesame-pddl_llm": ("Bilevel", "Direct", "PDDL"),
    "sesame-pose_sampler_llm": ("Bilevel", "Direct", "Poses"),
    "sesame-pddl_llm -pose_sampler_llm": ("Bilevel", "Direct", "PDDL+Poses"),
    "adaptive-pose_sampler_llm": ("Adaptive", "Direct", "Poses"),
    "adaptive-integrated_llm": ("Adaptive", "Direct", "Integrated"),
}

sort_keys = {
    "Adaptive": 0,
    "Bilevel": 1,
    "No LLM": 0,
    "Direct": 1,
    "Thinking": 2,
    np.nan: -1,
    "PDDL": 0,
    "Poses": 1,
    "PDDL+Poses": 2,
    "Integrated": 3,
}

problem_map = {
    ('tamp', '-problem packed -n 4'): "\makecell{Packed\\\\ k=4}",
    ('turtlebot_rovers', '-n 2'): "\makecell{Rovers\\\\ k=2}",
    ('tamp', '-problem packed -n 3'): "\makecell{Packed\\\\ k=3}",
    ('tamp', '-problem blocked -n 1'): "Blocked",
    ('turtlebot_rovers', '-n 3'): "\makecell{Rovers\\\\ k=3}",
    ('tamp', '-problem packed -n 5'): "\makecell{Packed\\\\ k=5}",
    ('turtlebot_rovers', '-n 4'): "\makecell{Rovers\\\\ k=4}",
}

def paired_exact_mcnemar(df, familywise_alpha=0.05, correction='holm', descending=True):
    """
    Perform pairwise exact McNemar (binomial on discordant counts) between columns of df.
    df : Pandas DataFrame (N rows trials, k columns coins, values 0/1)
    familywise_alpha : desired familywise error (e.g., 0.05)
    correction : 'holm' (recommended) or 'bonferroni'
    
    Returns:
      pairwise_df : DataFrame with columns
        ('coin_i','coin_j','n10','n01','k','p_uncorrected','p_adj','significant','winner')
      cld_df : DataFrame with columns ('coin','letters') giving Compact Letter Display
    """
    cols = list(df.columns)
    k = len(cols)
    pairs = list(combinations(range(k), 2))
    m = len(pairs)
    
    pvals = []
    records = []
    
    # compute p-values for each unordered pair
    for (i, j) in pairs:
        xi = df.iloc[:, i].values
        xj = df.iloc[:, j].values
        # discordant counts
        n10 = int(np.sum((xi == 1) & (xj == 0)))  # i wins
        n01 = int(np.sum((xi == 0) & (xj == 1)))  # j wins
        k_discord = n10 + n01
        
        if k_discord == 0:
            p_unc = 1.0  # no info
        else:
            # exact two-sided binomial test under p=0.5 on n10 successes in k trials
            # use scipy.stats.binomtest (two-sided)
            bt = binomtest(n10, n=k_discord, p=0.5, alternative='two-sided')
            p_unc = bt.pvalue
        
        pvals.append(p_unc)
        records.append({
            'i': cols[i], 'j': cols[j],
            'n10': n10, 'n01': n01, 'k': k_discord,
            'p_uncorrected': p_unc
        })
    
    # multiple testing correction
    if correction not in ('holm', 'bonferroni'):
        raise ValueError("correction must be 'holm' or 'bonferroni'")
    method = 'bonferroni' if correction == 'bonferroni' else 'holm'
    reject, pvals_adj, _, _ = multipletests(pvals, alpha=familywise_alpha, method=method)
    
    # annotate records
    for rec, p_adj, rej in zip(records, pvals_adj, reject):
        rec['p_adj'] = p_adj
        rec['significant'] = bool(rej)
        # winner if significant and k>0
        if rec['significant'] and rec['k'] > 0:
            if rec['n10'] > rec['n01']:
                rec['winner'] = rec['i']
            elif rec['n01'] > rec['n10']:
                rec['winner'] = rec['j']
            else:
                rec['winner'] = None
        else:
            rec['winner'] = None
    
    pairwise_df = pd.DataFrame(records)
    cld_df = cld_from_pairwise(pairwise_df, df, descending=descending)
    return pairwise_df, cld_df

def pairwise_ttests(times_df, success_df, test='ttest', familywise_alpha=0.05, correction='holm', descending=False):
    '''
    Paired pairwise t-tests between columns of times_df, considering only trials where both algorithms succeeded.
    '''
    cols = list(times_df.columns)
    k = len(cols)
    pairs = list(combinations(range(k), 2))
    m = len(pairs)

    pvals = []
    records = []
    # compute p-values for each unordered pair
    for (i, j) in pairs:
        ti = times_df.iloc[:, i].values
        tj = times_df.iloc[:, j].values

        ei = success_df.iloc[:, i].values
        ej = success_df.iloc[:, j].values

        # consider only trials where both algorithms succeeded
        mask = np.logical_and(ei, ej)
        ti = ti[mask]
        tj = tj[mask]

        if len(ti) < 2:
            p_unc = 1.0
            mean_diff = np.nan
            std_diff = np.nan
            stat_used = 'N/A'
        elif test == 'ttest':
            # paired t-test
            tt = ttest_rel(ti, tj)
            p_unc = tt.pvalue
            mean_diff = np.mean(ti - tj)
            std_diff = np.std(ti - tj)
            stat_used = 'ttest'
        elif test == 'wilcoxon':
            wt = wilcoxon(ti, tj)
            p_unc = wt.pvalue
            mean_diff = np.mean(ti - tj)
            std_diff = np.std(ti - tj)
            stat_used = 'wilcoxon'
        pvals.append(p_unc)
        records.append({
            'i': cols[i], 'j': cols[j],
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'n': len(ti),
            'p_uncorrected': p_unc,
            'test_used': stat_used
        })
    # multiple testing correction
    if correction not in ('holm', 'bonferroni'):
        raise ValueError("correction must be 'holm' or 'bonferroni'")
    method = 'bonferroni' if correction == 'bonferroni' else 'holm'
    reject, pvals_adj, _, _ = multipletests(pvals, alpha=familywise_alpha, method=method)
    # annotate records
    for rec, p_adj, rej in zip(records, pvals_adj, reject):
        rec['p_adj'] = p_adj
        rec['significant'] = bool(rej)
        # winner if significant and mean_diff != 0
        if rec['significant'] and not np.isnan(rec['mean_diff']) and rec['mean_diff'] != 0:
            if rec['mean_diff'] < 0:
                rec['winner'] = rec['i']
            elif rec['mean_diff'] > 0:
                rec['winner'] = rec['j']
            else:
                rec['winner'] = None
        else:
            rec['winner'] = None
    pairwise_df = pd.DataFrame(records)
    cld_df = cld_from_pairwise(pairwise_df, times_df, descending=descending)
    return pairwise_df, cld_df

def cld_from_pairwise(pairwise_df, df, descending=True):
    cols = list(df.columns)

    # Build adjacency matrix of "significant difference" (undirected)
    k = len(cols)
    sep = np.zeros((k, k), dtype=int)
    name_to_idx = {name: idx for idx, name in enumerate(cols)}
    for _, row in pairwise_df.iterrows():
        if row['significant']:
            a = name_to_idx[row['i']]
            b = name_to_idx[row['j']]
            sep[a, b] = 1
            sep[b, a] = 1
    
    # Utility: convert integer index -> letter name like 'a', 'b', ..., 'z', 'aa', 'ab', ...
    def letter_name(n: int) -> str:
        name = ""
        m = n
        while True:
            char_index = m % 26
            name = chr(ord('a') + char_index) + name
            m //= 26
            if m == 0:
                break
            m -= 1
        return name
    
    ns_edges = set()
    for i in range(k):
        for j in range(i + 1, k):
            if sep[i, j] == 0:
                ns_edges.add((i, j))

    # Greedy clique cover of ns_edges
    cliques = []
    assigned_letters = {i : [] for i in range(k)}

    def uncovered_degree(node):
        return sum(1 for (u, v) in ns_edges if u == node or v == node)
    
    while ns_edges:
        # pick seed node with largest uncovered degree
        degrees = [uncovered_degree(i) for i in range(k)]
        seed = max(range(k), key=lambda i: degrees[i])

        # Build clique starting from seed
        clique = {seed}
        # Candidate nodes: neighbors via uncovered NS edges
        candidates = {v for (u, v) in ns_edges if u == seed} | \
                    {u for (u, v) in ns_edges if v == seed}
        
        # Iteratively expand clique
        expanded = True
        while expanded and candidates:
            expanded = False
            # Sort candidates by uncovered_degree descending
            sorted_cand = sorted(candidates, key=lambda x: (-uncovered_degree(x), x))
            for cand in sorted_cand:
                if all(sep[cand, member] == 0 for member in clique):
                    clique.add(cand)
                    # update candidates: only those still NS with all in clique
                    candidates = {c for c in candidates if all(sep[c, m] == 0 for m in clique) and c not in clique}
                    expanded = True
                    break
        
        cliques.append(clique)
        # Remove edges inside clique from ns_edges
        to_remove = {(u, v) for (u, v) in ns_edges if u in clique and v in clique}
        ns_edges -= to_remove
    
    # Ensure each node has at least one clique
    for i in range(k):
        if not any(i in c for c in cliques):
            cliques.append({i})
        
    # Each clique c has a number of nodes n_c. Each node i in c is associated to a row in the df, with an average 
    # success rate d.iloc[:, i].mean(). We want to sort cliques so that all cliques that contain the node with
    # the highest success rate come before all cliques that do not contain that node. Then, among cliques that contain
    # that node, if there is a singleton clique it should come first, followed by any clique that contains the node
    # with the next highest success rate, and so on recursively. For cliques that do not contain the node with the
    # highest success rate, we repeat the same process with the node with the next highest success rate, and so on.
    success_rates = {i: df.iloc[:, i].mean() for i in range(k)}
    sorted_nodes = sorted(range(k), key=lambda i: success_rates[i], reverse=descending)
    node_rank = {node: rank for rank, node in enumerate(sorted_nodes)}
    def clique_sort_key(clique):
        key = []
        for node in sorted_nodes:
            if node in clique:
                key.append(node_rank[node])
        return key
    cliques.sort(key=clique_sort_key)
    


    # Assign letters to cliques
    for letter_idx, clique in enumerate(cliques):
        for node in clique:
            assigned_letters[node].append(letter_idx)

    # Build cld for each idx
    cld = []
    for i in range(k):
        letter_indices = sorted(assigned_letters[i])
        letter_str = ''.join(letter_name(idx) for idx in letter_indices)
        cld.append(letter_str)
    
    cld_df = pd.DataFrame({'Group': cols, 'CLD': cld})

    return cld_df

def success_df_from_dir(results_dir='results/', n_seeds=10):
    '''
    Create a deaggregated DataFrame of successes, with a multi-index for problem and seed
    '''
    results = defaultdict(lambda: defaultdict(list))
    for exp_dir in glob(os.path.join(results_dir, '*')):
        if not os.path.isdir(exp_dir):
            continue
        exp_name = os.path.basename(exp_dir)
        match = re.match(r'^(.*?)examples\.pybullet\.(.*?)\.run (.*?)_(\d+)$', exp_name)
        if not match:
            print(f"Skipping directory with unexpected name format: {exp_name}")
            continue
        algorithm_llm_args, domain, domain_args, seed = match.groups()
        seed = int(seed)
        # if 'rovers' in domain.lower():
        #     continue
        if seed >= n_seeds:
            continue
        store_path = os.path.join(exp_dir, 'store.pkl')
        if os.path.exists(store_path):
            with open(store_path, 'rb') as f:
                store = pickle.load(f)
            summary = store.get('summary', {})
            solved = int(summary.get('solved', 0))
        else:
            solved = 0
        domain_key = problem_map[domain, domain_args]
        algorithm_key = algorithm_map[algorithm_llm_args]
        results[(domain_key, seed)][algorithm_key] = solved
    # Convert to DataFrame for better visualization
    df = pd.DataFrame.from_dict({k: v for k, v in results.items()}, orient='index').fillna(0).T
    # Sort rows and columns lexicographically
    df = df.sort_index(axis=1)
    df = df.sort_index(axis=0)
    multiindex = pd.MultiIndex.from_tuples(df.columns)
    df.columns = multiindex
    return df

def success_rate_cld_barchart(results_dir='results/', n_seeds=10):
    '''
    Generate a bar chart with the success rates of each algorithm/parameter combination per problem, with the CLD letters above each bar.
    '''
    # Create a deaggregated table of successes, similar to success_rate_table but without averaging across seeds and with a multi-index for problem and seed
    df = success_df_from_dir(results_dir, n_seeds)
    cld_results = []
    for problem in df.columns.levels[0]:
        problem_df = df[problem].T
        # Create a series with the mean success rate for each algorithm/parameter combination
        success_df = problem_df.mean(axis=0)
        # Sort the problem_df by the mean success rate
        problem_df = problem_df[success_df.sort_values(ascending=False).index]
        pairwise_df, problem_cld_df = paired_exact_mcnemar(problem_df, familywise_alpha=0.05, correction='holm')
        # Set the 'Group' column as the index
        problem_cld_df = problem_cld_df.set_index('Group')
        # Add a success rate column
        problem_cld_df['Mean'] = success_df[problem_cld_df.index]

        # Convert tuple index into MultiIndex
        problem_cld_df.index = pd.MultiIndex.from_tuples(problem_cld_df.index)
        problem_cld_df = problem_cld_df.sort_index(axis=0, key=lambda x: [sort_keys[i] for i in x.get_level_values(0)])

        # Create the bar chart for this problem, the same way as the stacked_bar_failures_chart function in results_analyzer.py
        colors = ["black"] + color_list + color_list
        if "Rovers" not in problem:
            colors += colors
        else:
            problem_cld_df = problem_cld_df[problem_cld_df.index.get_level_values(0) != "Bilevel"]
        print(problem_cld_df['Mean'])
        # New df for the "Mean" column level 2 of the index moved to the columns
        new_df = problem_cld_df['Mean'].unstack(level=2)
        # new_df.plot(kind='bar')
        print(new_df)
        
        parent_methods = ['Adaptive', 'Bilevel']
        second_level_methods = ['No LLM', 'Direct', 'Thinking']
        third_level_methods = ['PDDL', 'Poses', 'PDDL+Poses', 'Integrated']

        group_width = 0.8
        bar_width = group_width / len(third_level_methods)

        x = []
        subticks = []
        labels = []
        
        fig, ax = plt.subplots(figsize=(10, 4.5))
        pos = 0

        for parent in parent_methods:
            if "Rovers" in problem and parent == "Bilevel":
                continue
            for second in second_level_methods:
                row = new_df.loc[parent, second]
                if second == "No LLM":
                    val = row[np.nan]
                    ax.bar(pos, val, width=bar_width, color='black')
                    labels.append(f"N/A")
                    x.append(pos)
                    subticks.append(pos)
                    # Advance pos such that spacing is equal to other groups
                    pos += (1 - group_width) + bar_width
                else:
                    for i, third in enumerate(third_level_methods):
                        val = row[third]
                        ax.bar(pos + i * bar_width, val, width=bar_width, color=color_list[i])
                        subticks.append(pos + i * bar_width)
                    labels.append(f"{second}")
                    x.append(pos + group_width / 2 - bar_width / 2)
                    pos += 1
            pos += 0.5  # extra space between parent method groups
            labels[-2] = labels[-2] + f"\n{parent}"
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center")
        ax.set_ylabel('Success Rate')
        ax.set_xlabel('Algorithm / Parameter Combination')
        # Space title out a bit
        ax.set_title(f'Success Rates for {problem}', pad=40)
        plt.ylim(0, 1)
        x_max = (1 * (len(second_level_methods) -1) + bar_width * 2) * len(parent_methods) + 0.5
        plt.xlim(-0.5, x_max)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if "Rovers" in problem:
            # Remove all axis lines, but leave ticks
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            x_max = (1 * (len(second_level_methods) -1) + bar_width * 2) * (len(parent_methods)-1)
            # Add lines to simulate axis from (-0.5, 0) to (x_max, 1)
            plt.plot([-0.5, -0.5], [0, 1], color='black', linewidth=1.5)
            plt.plot([-0.5, x_max], [0, 0], color='black', linewidth=1.5)
        
        # Annotate each bar with the CLD letters, overwriting the ylim if necessary
        for xtick, (idx, row) in zip(subticks, problem_cld_df.iterrows()):
            cld_letters = row['CLD']
            # Split the cld_letters into multiple lines if it has more than 2 letters
            if len(cld_letters) > 2:
                cld_letters = '\n'.join([cld_letters[i:i+2] for i in range(0, len(cld_letters), 2)])
            success_rate = row['Mean']
            ax.text(xtick, success_rate + 0.03, cld_letters, ha='center', va='bottom', fontsize=16)

        plt.tight_layout()
        problem_str = problem.replace(" ", "_").replace("\makecell{", "").replace("}", "").replace("\\", "").replace("k=", "")
        plt.savefig(f'success_rate_{problem_str}.pdf')
        plt.close()

    # Legend
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color='black', label='No LLM')]
    for i, third in enumerate(third_level_methods):
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color_list[i], label=third))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.legend(handles=legend_handles, loc='upper right', ncol=1, title='LLM Usage')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('success_rate_legend.pdf')
        
def time_df_from_dir(results_dir='results/', n_seeds=10):
    '''
    Create a deaggregated DataFrame of times, with a multi-index for problem and seed
    '''
    results = defaultdict(lambda: defaultdict(list))
    for exp_dir in glob(os.path.join('results/', '*')):
        if not os.path.isdir(exp_dir):
            continue
        exp_name = os.path.basename(exp_dir)
        match = re.match(r'^(.*?)examples\.pybullet\.(.*?)\.run (.*?)_(\d+)$', exp_name)
        if not match:
            print(f"Skipping directory with unexpected name format: {exp_name}")
            continue
        algorithm_llm_args, domain, domain_args, seed = match.groups()
        seed = int(seed)
        if seed >= n_seeds:
            continue
        store_path = os.path.join(exp_dir, 'store.pkl')
        if os.path.exists(store_path):
            with open(store_path, 'rb') as f:
                store = pickle.load(f)
            summary = store.get('summary', {})
            if summary.get('solved', False):
                time = summary.get('run_time', None)
            else:
                time = None
        else:
            time = None
        domain_key = problem_map[domain, domain_args]
        algorithm_key = algorithm_map[algorithm_llm_args]
        if time is not None:
            results[(domain_key, seed)][algorithm_key] = time
    # Convert to DataFrame for better visualization
    df = pd.DataFrame.from_dict({k: v for k, v in results.items()}, orient='index').fillna(300).T
    # Sort rows and columns lexicographically
    df = df.sort_index(axis=1)
    df = df.sort_index(axis=0)
    multiindex = pd.MultiIndex.from_tuples(df.columns)
    df.columns = multiindex
    return df

def time_cld_barchart(n_seeds=10, time_thresh=300, success_rate_thresh=0.5, test='ttest'):
    '''
    Generate a bar chart with the average time for each algorithm/parameter combination per problem, with the CLD letters above each bar.
    '''
    time_df = time_df_from_dir(n_seeds=n_seeds)
    success_df = success_df_from_dir(n_seeds=n_seeds)
    for problem in time_df.columns.levels[0]:
        if 'rovers' in problem.lower(): # Only one algorithm works, so timing is irrelevant
            continue
        problem_time_df = time_df[problem].T
        problem_success_df = success_df[problem].T

        

        # Consider only algorithms with > success_rate_thresh success rate for this problem
        success_rate = problem_success_df.mean(axis=0)
        problem_time_df = problem_time_df[success_rate[success_rate > success_rate_thresh].index]
        problem_success_df = problem_success_df[success_rate[success_rate > success_rate_thresh].index]

        # Create a series with the mean time to solve for each algorithm/parameter combination
        # For this, replace all the t>=time_thresh with NaN
        meantime_df = time_df[problem].T.replace(time_thresh, np.nan).mean(axis=0)
        # pairwise_df, problem_cld_df = pairwise_coxph_tests(problem_time_df, problem_success_df, max_time=time_thresh, familywise_alpha=0.05, correction='holm')
        pairwise_df, problem_cld_df = pairwise_ttests(problem_time_df, problem_success_df, test=test, familywise_alpha=0.05, correction='holm')
        print(problem)
        print(pairwise_df.to_markdown(floatfmt=".2f"))
        # Set the 'Group' column as the index
        problem_cld_df = problem_cld_df.set_index('Group')
        # Add a mean time column
        problem_cld_df['Mean'] = meantime_df[problem_cld_df.index]

        # Find out the algorithms with 0 < success rate < success_rate_thresh
        low_success_rate = success_rate[(success_rate > 0) & (success_rate <= success_rate_thresh)].index
        new_rows_df = pd.DataFrame(columns=problem_cld_df.columns, index=low_success_rate)
        new_rows_df['Mean'] = meantime_df[low_success_rate]
        problem_cld_df = pd.concat([problem_cld_df, new_rows_df])

        # For any algorithms with success rate = 0, set the mean time to NaN
        zero_success_rate = success_rate[success_rate == 0].index
        new_rows_df = pd.DataFrame(columns=problem_cld_df.columns, index=zero_success_rate)
        new_rows_df['Mean'] = np.nan
        problem_cld_df = pd.concat([problem_cld_df, new_rows_df])

        # Convert tuple index into MultiIndex
        problem_cld_df.index = pd.MultiIndex.from_tuples(problem_cld_df.index)
        problem_cld_df = problem_cld_df.sort_index(axis=0, key=lambda x: [sort_keys[i] for i in x.get_level_values(0)])

        # New df for the "Mean" column level 2 of the index moved to the columns
        new_df = problem_cld_df['Mean'].unstack(level=2)
        print(new_df)
        
        parent_methods = ['Adaptive', 'Bilevel']
        second_level_methods = ['No LLM', 'Direct', 'Thinking']
        third_level_methods = ['PDDL', 'Poses', 'PDDL+Poses', 'Integrated']

        group_width = 0.8
        bar_width = group_width / len(third_level_methods)

        x = []
        subticks = []
        labels = []

        fig, ax = plt.subplots(figsize=(10, 4.5))
        pos = 0

        for parent in parent_methods:
            for second in second_level_methods:
                row = new_df.loc[parent, second]
                if second == "No LLM":
                    val = row[np.nan]
                    ax.bar(pos, val, width=bar_width, color='black')
                    labels.append(f"N/A")
                    x.append(pos)
                    subticks.append(pos)
                    # Advance pos such that spacing is equal to other groups
                    pos += (1 - group_width) + bar_width
                else:
                    for i, third in enumerate(third_level_methods):
                        val = row[third]
                        ax.bar(pos + i * bar_width, val, width=bar_width, color=color_list[i])
                        subticks.append(pos + i * bar_width)
                    labels.append(f"{second}")
                    x.append(pos + group_width / 2 - bar_width / 2)
                    pos += 1
            pos += 0.5  # extra space between parent method groups
            labels[-2] = labels[-2] + f"\n{parent}"
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center")
        ax.set_ylabel('Time to Solve (s)')
        ax.set_xlabel('Algorithm / Parameter Combination')
        # Space title out a bit
        ax.set_title(f'Time to Solve for {problem}', pad=40)
        plt.ylim(0, time_thresh)
        x_max = (1 * (len(second_level_methods) -1) + bar_width * 2) * len(parent_methods) + 0.5
        plt.xlim(-0.5, x_max)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Annotate each bar with the CLD letters, overwriting the ylim if necessary
        for xtick, (idx, row) in zip(subticks, problem_cld_df.iterrows()):
            cld_letters = row['CLD']
            mean_time = row['Mean']
            if np.isnan(mean_time):
                # There is no bar or CLD
                # Add a cross above the x-axis
                ax.text(xtick, 5, '×', ha='center', va='bottom', fontsize=16, color='red')
            elif not isinstance(cld_letters, str) and np.isnan(cld_letters):
                # There is a mean time but no CLD, so just write \u2014
                ax.text(xtick, mean_time + 5, '\u2014', ha='center', va='bottom', fontsize=16)
            else:
                if len(cld_letters) > 2:
                    cld_letters = '\n'.join([cld_letters[i:i+2] for i in range(0, len(cld_letters), 2)])
                ax.text(xtick, mean_time + 5, cld_letters, ha='center', va='bottom', fontsize=16)

        plt.tight_layout()
        problem_str = problem.replace(" ", "_").replace("\makecell{", "").replace("}", "").replace("\\", "").replace("k=", "")
        plt.savefig(f'time_cld_{problem_str}.pdf')
        plt.close()

    # Legend
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color='black', label='No LLM')]
    for i, third in enumerate(third_level_methods):
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color_list[i], label=third))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.legend(handles=legend_handles, loc='upper right', ncol=1, title='LLM Usage')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('time_cld_legend.pdf')

        # # Creaete the bar chart for this problem, the same way as the stacked_bar_failures_chart function in results_analyzer.py
        # colors = ["black"] + color_list + color_list
        # ax = problem_cld_df['Mean'].plot(kind='bar', figsize=(10, 4.5), width=0.5, color=colors, legend=False, position=0)
        # ax.set_ylabel('Time to Solve (s)')
        # ax.set_xlabel('Algorithm / Parameter Combination')
        # # Space title out a  bit
        # ax.set_title(f'Time to Solve for {problem}', pad=20)
        
        # # Get tick positions
        # xticks = ax.get_xticks()

        # # Build custom labels (sparse style)
        # labels = []
        # prev = (None, None, None)
        # for idx in problem_cld_df.index:
        #     label_parts = []
        #     for i, part in enumerate(idx):
        #         if part != prev[i]:
        #             if part == "Poses":
        #                 label_parts.append("Ps")
        #             elif part == "Integrated":
        #                 label_parts.append("In")
        #             elif part == "PDDL":
        #                 label_parts.append("Pd")
        #             elif part == "PDDL+Poses":
        #                 label_parts.append("PP")
        #             elif part == "No LLM":
        #                 label_parts.append("")
        #             elif not isinstance(part, str) and np.isnan(part):
        #                 label_parts.append("No")
        #             else:
        #                 label_parts.append(str(part))
        #         else:
        #             label_parts.append("")
        #     labels.append(" " + "\n ".join(label_parts[::-1]))  # stacked vertically
        #     prev = idx
        
        # ax.set_xticks(xticks)
        # ax.set_xticklabels(labels, rotation=0, ha="left")

        # plt.ylim(0, time_thresh)
        # plt.xlim(-0.5, 18)
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)

        # # Annotate each bar with the CLD letters, overwriting the ylim if necessary
        # for xtick, (idx, row) in zip(xticks, problem_cld_df.iterrows()):
        #     cld_letters = row['CLD']
        #     mean_time = row['Mean']
        #     if np.isnan(mean_time):
        #         # There is no bar or CLD
        #         # Add a cross above the x-axis
        #         ax.text(xtick + 0.25, 5, '×', ha='center', va='bottom', fontsize=16, color='red')
        #     elif not isinstance(cld_letters, str) and np.isnan(cld_letters):
        #         # There is a mean time but no CLD, so just write \u2014
        #         ax.text(xtick + 0.25, mean_time + 5, '\u2014', ha='center', va='bottom', fontsize=16)
        #     else:
        #         ax.text(xtick + 0.25, mean_time + 5, cld_letters, ha='center', va='bottom', fontsize=16)

        # for idx, tick in enumerate(ax.xaxis.get_major_ticks()):
        #     if idx in [1, 5, 10, 14]:
        #         ms = 50
        #     elif idx == 9:
        #         ms = 80
        #     else:
        #         continue
        #     tick.tick1line.set_markersize(ms)


        # plt.tight_layout()
        # problem_str = problem.replace(" ", "_").replace("\makecell{", "").replace("}", "").replace("\\", "").replace("k=", "")
        # plt.savefig(f'time_cld_{problem_str}.pdf')
        # plt.close()

def failures_table(n_seeds=10):
    '''
    Generate a table that shows the fraction of failures due to timeout, APIError, or the LLM giving up (no plan returned and no timeout).
    '''
    results = defaultdict(lambda: defaultdict(int))
    for exp_dir in glob(os.path.join('results/', '*')):
        if not os.path.isdir(exp_dir):
            continue
        exp_name = os.path.basename(exp_dir)
        match = re.match(r'^(.*?)examples\.pybullet\.(.*?)\.run (.*?)_(\d+)$', exp_name)
        if not match:
            print(f"Skipping directory with unexpected name format: {exp_name}")
            continue
        algorithm_llm_args, domain, domain_args, seed = match.groups()
        seed = int(seed)
        if seed >= n_seeds:
            continue
        store_path = os.path.join(exp_dir, 'store.pkl')
        domain_key = problem_map[domain, domain_args]
        algorithm_key = algorithm_map[algorithm_llm_args]
        if os.path.exists(store_path):
            with open(store_path, 'rb') as f:
                store = pickle.load(f)
            summary = store.get('summary', {})
            if summary.get('solved', False) and summary.get('timeout', False):
                raise ValueError("An experiment cannot be both solved and timed out.")
            if summary.get('solved', False) and summary.get('time', 0) > 300:
                raise ValueError("An experiment cannot be both solved and exceed the time limit.")
            if not summary.get('solved', False):
                if summary.get('timeout', False):
                    results[domain_key,'Timed out'][algorithm_key] += 1
                else:
                    results[domain_key,'Gave up'][algorithm_key] += 1
            else:
                results[domain_key,'S'][algorithm_key] += 1
            
    # Experiments that get APIError do not create an exp_dir, so we need to count them from the log files
    for log_path in glob('logs/out_*.log'):
        with open(log_path, 'r') as f:
            log = f.read()
        if 'RESOURCE_EXHAUSTED' in log:
            match = re.match(r'^out_(.*?)examples\.pybullet\.(.*?)\.run (.*?)_(\d+)\.log$', os.path.basename(log_path))
            if not match:
                print(f"Skipping log file with unexpected name format: {log_path}")
                continue
            algorithm_llm_args, domain, domain_args, seed = match.groups()
            seed = int(seed)
            if seed >= n_seeds:
                continue
            domain_key = problem_map[domain, domain_args]
            algorithm_key = algorithm_map[algorithm_llm_args]
            results[domain_key,'Token limit'][algorithm_key] += 1

    # Convert to DataFrame for better visualization
    df = pd.DataFrame.from_dict({k: {kk: v for kk, v in v2.items()} for k, v2 in results.items()}, orient='index').fillna(0).T

    # Sanity check: each algorithm/problem combination should have n_seeds total runs
    # print(df.sum(axis=1))
    print(df.groupby(axis=1, level=0).sum().to_markdown())

    # Convert counts to fractions
    df = df / n_seeds
    # # Check that grouping by the first level of the column multi-index and summing gives 1.0 for each algorithm/problem combination
    # print(df.groupby(level=0, axis=1).sum())
    # Drop the 'S' column
    df = df.drop(columns='S', level=1)
    total_frac_df = df
    # Re-normalize the groups to sum to 1.0, so that we can interpret the values as fractions of failures -- set to Nan if 'S' = 100%
    df = df.groupby(level=0, axis=1).apply(lambda x: x.div(x.sum(axis=1), axis=0))
    # # Re-check the grouping by the first level of the column multi-index and summing gives 1.0 for each algorithm/problem combination
    # print(df.groupby(level=0, axis=1).sum())

    # This somehow results in two levels of columns with the proble name. Drop the first level
    df.columns = df.columns.droplevel(0)
    failure_frac_df = df

    # Combine each group Gu/Ti/Tk into a single column with the format "Gu/Ti/Tk" with percentages with no decimals
    combined = {}
    for problem in df.columns.levels[0]:
        gu = df.loc[:, (problem, 'Gave up')] if (problem, 'Gave up') in df.columns else pd.Series([0]*len(df), index=df.index)
        ti = df.loc[:, (problem, 'Timed out')] if (problem, 'Timed out') in df.columns else pd.Series([0]*len(df), index=df.index)
        tk = df.loc[:, (problem, 'Token limit')] if (problem, 'Token limit') in df.columns else pd.Series([0]*len(df), index=df.index)
        combined[problem] = [f"{g*100:.0f}\\,/\\,{t*100:.0f}\\,/\\,{k*100:.0f}\\,\\%"  if not any(np.isnan([g,t,k])) else '\\multicolumn{1}{c}{---}' for g, t, k in zip(gu, ti, tk)]
    df = pd.DataFrame(combined, index=df.index)
    # Sort rows and columns lexicographically
    df = df.sort_index(axis=1)
    df = df.sort_index(axis=0, key=lambda x: [sort_keys[i] for i in x.get_level_values(0)])

    ######
    # For markdown only
    df = df.style.format(decimal='.', thousands=',', precision=2)
    print(df.to_latex(sparse_index=True, multicol_align='c', hrules=True))

    return failure_frac_df, total_frac_df

def stacked_bar_failures_chart(n_seeds=10):
    _, df = failures_table(n_seeds=n_seeds)
    # Drop rows that have "No LLM" in level 1 of the index
    df = df[~df.index.get_level_values(1).str.contains('No LLM')]
    df = df.sort_index(axis=1)
    df = df.sort_index(axis=0, key=lambda x: [sort_keys[i] for i in x.get_level_values(0)])

    # For each problem, create a stacked bar chart with a bar for each algorithm/parameter combination and sections for Gu/Ti/Tk
    # columns are (problem, failure_type) and indexes are (algorithm/parameter combination -- 3 levels)
    for problem in df.columns.levels[0]:
        problem_df = df[problem].copy()
        if "Rovers" in problem:
            # Drop the "Bilevel" rows
            problem_df = problem_df[problem_df.index.get_level_values(0) != "Bilevel"]
        
        # New df with the failure modes (current columns) as the inner level of a column multi-index and level 2 of the index as the outer level of the column multi-index
        problem_df = problem_df.unstack(level=2)
        problem_df.columns = problem_df.columns.swaplevel(0, 1)
        problem_df = problem_df.sort_index(axis=1, level=0, key=lambda x: [sort_keys[i] for i in x])
        print(problem)
        print(problem_df)
        # Print only token limit for thinking methods. Idx is (Adaptive, Direct) etc., columns are (PDDL, Gave up) etc.
        # try:
        #     print(problem_df.xs('Thinking', level=1, axis=0).xs('Token limit', level=1, axis=1).to_markdown(floatfmt=".2f"))
        # except:
        #     pass
        
        parent_methods = ['Adaptive', 'Bilevel']
        second_level_methods = ['Direct', 'Thinking']
        third_level_methods = ['PDDL', 'Poses', 'PDDL+Poses', 'Integrated']

        group_width = 0.8
        bar_width = group_width / len(third_level_methods)

        x = []
        subticks = []
        labels = []
        sublabels = []
        sublabels_map = {
            'PDDL': 'Pd',
            'Poses': 'Ps',
            'PDDL+Poses': 'PP',
            'Integrated': 'In'
        }

        fig, ax = plt.subplots(figsize=(10, 4.5))
        pos = 0

        for parent in parent_methods:
            if "Rovers" in problem and parent == "Bilevel":
                continue
            for second in second_level_methods:
                row = problem_df.loc[parent, second]
                for i, third in enumerate(third_level_methods):
                    vals = row[third]
                    # Plot a stacked bar at pos + i * bar_width with the values in vals
                    bottom = 0
                    for j, (failure_type, val) in enumerate(vals.items()):
                        ax.bar(pos + i * bar_width, val, width=bar_width, bottom=bottom, color=color_list[j])
                        bottom += val
                    subticks.append(pos + i * bar_width)
                    sublabels.append(f"{sublabels_map[third]}")
                labels.append(f"{second}\n{parent}")
                x.append(pos + group_width / 2 - bar_width / 2)
                pos += 1
            pos += 0.5  # extra space between parent method groups
        # Add sublabels without tickmarks at the subticks positions
        for xtick, sublabel in zip(subticks, sublabels):
            ax.text(xtick, -0.05, sublabel, ha='center', va='top', transform=ax.get_xaxis_transform())

        ax.set_xticks(x)
        # Space tick marks to make see the sublabels
        ax.set_xticklabels(labels, rotation=0, ha="center", y=-0.15)
        ax.set_ylabel('Fraction of Failures')
        ax.set_xlabel('Algorithm / LLM')
        # Space title out a bit more
        ax.set_title(f'Failure Types for {problem}', pad=40)
        plt.ylim(0, 1)
        x_max = len(second_level_methods) * len(parent_methods) + 0.5
        plt.xlim(-0.5, x_max)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if "Rovers" in problem:
            # Remove all axis lines, but leave ticks
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            x_max = len(second_level_methods) * (len(parent_methods) - 1)
            # Add lines to simulate axis from (-1, 0) to (x_max, 1)
            plt.plot([-0.5, -0.5], [0, 1], color='black', linewidth=1.5)
            plt.plot([-0.5, x_max], [0, 0], color='black', linewidth=1.5)

        plt.tight_layout()
        problem_str = problem.replace(" ", "_").replace("\makecell{", "").replace("}", "").replace("\\", "").replace("k=", "")
        plt.savefig(f'failures_{problem_str}.pdf')
        # plt.show()
        # exit()
        plt.close()

    # Legend
    legend_handles = []
    for i, failure_type in enumerate(['Gave up', 'Timed out', 'Token limit']):
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color_list[i], label=failure_type))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.legend(handles=legend_handles, loc='upper right', ncol=1, title='Failure Type')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('failures_legend.pdf')



        # ax = problem_df.plot(kind='bar', stacked=True, figsize=(10, 4.5), width=0.5, position=0)
        # ax.set_ylabel('Fraction of Failures')
        # ax.set_xlabel('Algorithm / LLM')
        # # Space title out a bit more
        # ax.set_title(f'Failure Types for {problem}', pad=20)

        # # Get tick positions
        # xticks = ax.get_xticks()

        # # Build custom labels (sparse style)
        # labels = []
        # prev = (None, None, None)
        # for idx in problem_df.index:
        #     label_parts = []
        #     for i, part in enumerate(idx):
        #         if part != prev[i]:
        #             if part == "Poses":
        #                 label_parts.append("Ps")
        #             elif part == "Integrated":
        #                 label_parts.append("In")
        #             elif part == "PDDL":
        #                 label_parts.append("Pd")
        #             elif part == "PDDL+Poses":
        #                 label_parts.append("PP")
        #             else:
        #                 label_parts.append(str(part))
        #         else:
        #             label_parts.append("")  # sparse
        #     labels.append(" " + "\n ".join(label_parts[::-1]))  # stacked vertically
        #     prev = idx

        # ax.set_xticks(xticks)
        # ax.set_xticklabels(labels, rotation=0, ha="left")

        # plt.ylim(0, 1)
        # plt.xlim(-0.5, 16)
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # if "Rovers" in problem:
        #     # Remove all axis lines, but leave ticks
        #     ax.spines['left'].set_visible(False)
        #     ax.spines['bottom'].set_visible(False)

        #     # Add lines to simulate axis from (-1, 0) to (8, 1)
        #     plt.plot([-0.5, -0.5], [0, 1], color='black', linewidth=1)
        #     plt.plot([-0.5, 8], [0, 0], color='black', linewidth=1)
        #     # plt.plot([8, 8], [0, 1], color='black', linewidth=1)
        #     # plt.plot([-1, 8], [1, 1], color='black', linewidth=1)

        # for idx, tick in enumerate(ax.xaxis.get_major_ticks()):
        #     if idx in [4, 8, 12]:
        #         ms = 50
        #     else:
        #         continue
        #     tick.tick1line.set_markersize(ms)

        # plt.legend(title='Failure Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        # plt.tight_layout()
        # problem_str = problem.replace(" ", "_").replace("\makecell{", "").replace("}", "").replace("\\", "").replace("k=", "")
        # plt.savefig(f'failures_{problem_str}.pdf')
        # # plt.show()
        # plt.close()

    # Print the sum total of each failure type across algorithms and problems
    print("Total failures across all problems and algorithms:")
    print("index:", df.index)
    print("columns:", df.columns)
    print(df.groupby(axis=1, level=1).sum().sum() * n_seeds)

def llm_table_from_key(key, n_seeds=10, aggregate="mean"):
    '''
    Return a table with a column for each algorithm/parameter combination and a row for each problem. With the value of key in llm_info.pkl averaged across the n_seeds runs.
    '''

    results = defaultdict(lambda: defaultdict(list))
    for exp_dir in glob(os.path.join('results/', '*')):
        if not os.path.isdir(exp_dir):
            continue
        exp_name = os.path.basename(exp_dir)
        match = re.match(r'^(.*?)examples\.pybullet\.(.*?)\.run (.*?)_(\d+)$', exp_name)
        if not match:
            print(f"Skipping directory with unexpected name format: {exp_name}")
            continue
        algorithm_llm_args, domain, domain_args, seed = match.groups()
        seed = int(seed)
        if seed >= n_seeds:
            continue
        llm_info_path = os.path.join(exp_dir, 'llm_info.pkl')
        if os.path.exists(llm_info_path):
            with open(llm_info_path, 'rb') as f:
                llm_info = pickle.load(f)
            value = llm_info.get(key, None)
        else:
            value = None
        domain_key = problem_map[domain, domain_args]
        algorithm_key = algorithm_map[algorithm_llm_args]
        if value is not None:
            results[(domain_key, seed)][algorithm_key] = value
    # Convert to DataFrame for better visualization
    df = pd.DataFrame.from_dict({k: v for k, v in results.items()}, orient='index').fillna(np.nan).T
    # Sort rows and columns lexicographically
    df = df.sort_index(axis=1)
    df = df.sort_index(axis=0, key=lambda x: [sort_keys[i] for i in x.get_level_values(0)])
    multiindex = pd.MultiIndex.from_tuples(df.columns)
    df.columns = multiindex
    if aggregate == "mean":
        df = df.groupby(level=0, axis=1).mean()
    elif aggregate == "sum":
        df = df.groupby(level=0, axis=1).sum()

    return df

def all_llm_tables(n_seeds=10, aggregate="mean"):
    keys = [
        'integrated_time', 'pddl_time', 'sampling_time', 'integrated_input_tokens', 'integrated_thinking_tokens', 'integrated_output_tokens', 
        'pddl_input_tokens', 'pddl_thinking_tokens', 'pddl_output_tokens', 'sampling_input_tokens', 'sampling_thinking_tokens', 
        'sampling_output_tokens', 'pddl_failures', 'num_invalid_actions', 'num_preimages_not_achieved', 'num_axioms_not_achieved', 
        'local_sampling_failures', 'num_samples_in_collision', 'num_samples_not_stable', 'num_no_plans_returned', 'num_no_samples_returned', 
        'num_samples_format_failures', 'num_samples_out_of_bounds', 'num_samples_not_visible', 'num_samples_out_of_range', 
        'num_samples_not_reachable', 'num_samples_not_optimistic', 'num_samples_used', 'num_backtracking_failures'
    ]

    results = {}

    for key in keys:
        df = llm_table_from_key(key, n_seeds, aggregate)
        results[key] = df
    
    return results

def heatmap_plots(n_seeds=10):
    '''
    Radar plots were not very informative. New idea: heatmap plots
    The idea is to create a heatmap for each metric, with a row for each algorithm/parameter combination and a column for
    each problem. The color of each cell represents the value of the metric
    '''
    results = all_llm_tables(n_seeds)
    metrics = {
        'Total Tokens': [
            'integrated_input_tokens', 'integrated_thinking_tokens', 'integrated_output_tokens', 
            'pddl_input_tokens', 'pddl_thinking_tokens', 'pddl_output_tokens', 
            'sampling_input_tokens', 'sampling_thinking_tokens', 'sampling_output_tokens'
        ],
        'PDDL Failures': ['pddl_failures'],
        'PDDL Precondition Failures': ['num_preimages_not_achieved'],
        'PDDL Invalid Action Failures': ['num_invalid_actions'],
        'PDDL Axiom Failures': ['num_axioms_not_achieved'],
        'Sampling failures': ['local_sampling_failures', 'num_backtracking_failures'],
        'Samples Used': ['num_samples_used'],
    }
    aggregated_results = {}
    for metric, keys in metrics.items():
        df = pd.DataFrame(index=results[keys[0]].index, columns=results[keys[0]].columns).fillna(0)
        for key in keys:
            df += results[key]
        aggregated_results[metric] = df
    for metric, df in aggregated_results.items():
        # Create a heatmap for this metric
        fig1 = plt.figure(figsize=(10, 10))
        ax1 = plt.gca()
        if metric == "Samples Used":
            norm = LogNorm()
        else:
            norm = None

        # Add blank rows after No LLM, after Direct, and after Thinking
        new_index = [df.index[0]]
        prev = df.index[0]
        for idx in df.index[1:]:
            if idx[1] != prev[1]:
                new_index.append((None, None, None))
            new_index.append(idx)
            prev = idx
        df = df.reindex(new_index)

        print(df)
        sns.heatmap(df, cmap='Reds', ax=ax1, cbar=False, norm=norm)

        # Add a dashed horizontal line wherever there is a blank row
        # Also drop the tick marks for those rows
        yticks = ax1.get_yticks()
        new_yticks = []
        for i, idx in enumerate(df.index):
            if idx == (np.nan, np.nan, np.nan):
                ax1.hlines(i + 0.5, *ax1.get_xlim(), colors='black', linestyles='--', linewidth=3)
            else:
                new_yticks.append(yticks[i])
        ax1.set_yticks(new_yticks)

        # plt.title(f'{metric} Heatmap', size=20)
        plt.xlabel('Problem')
        plt.ylabel('Algorithm / Parameter Combination')

        # fig2, ax2 = plt.subplots(figsize=(12, 1))
        # mappable = ax1.collections[0]
        # cbar = plt.colorbar(mappable, aspect=40, cax=ax2, orientation='horizontal')
        # cbar.set_label(metric, fontsize=22)
        # cbar.ax.tick_params(labelsize=22)
        # if metric == "Samples Used":
        #     ax2.ticklabel_format(style="sci", axis="x", scilimits=(0,0))

        # Sparsify the y-axis labels
        labels = []
        prev = (None, None, None)
        for idx in df.index:
            if idx == (np.nan, np.nan, np.nan):
                continue
            label_parts = []
            for i, part in enumerate(idx):
                if part != prev[i]:
                    if part == "Poses":
                        label_parts.append("Ps")
                    elif part == "Integrated":
                        label_parts.append("In")
                    elif part == "PDDL":
                        label_parts.append("Pd")
                    elif part == "PDDL+Poses":
                        label_parts.append("PP")
                    elif part == "No LLM":
                        label_parts.append(" " * (len("Thinking") + 5))
                    elif not isinstance(part, str) and np.isnan(part):
                        label_parts.append("No")
                    else:
                        label_parts.append(str(part))
                else:
                    label_parts.append("")
            labels.append("  ".join(label_parts))  # stacked horizontally
            prev = idx
        ax1.set_yticklabels(labels, rotation=0, ha="right", va="center", fontsize=22)
        # For the x-axis, replace problem names with shorter versions using (" ", "_").replace("\makecell{", "").replace("}", "").replace("\\", "")
        problem_labels = [p.replace(" ", "\n").replace("\makecell{", "").replace("}", "").replace("\\", "").replace("k=", "k=") for p in df.columns]
        ax1.set_xticklabels(problem_labels, ha="center", fontsize=22)

        # Add colorbar
        cbar = fig1.colorbar(ax1.collections[0], ax=ax1, location='top', aspect=40)
        # if metric == "Samples Used":
        #     cbar.ax.ticklabel_format(style="sci", axis="x", scilimits=(0,0))
        # # Move cbar tick labels ha="right"
        for label in cbar.ax.get_xticklabels():
            label.set_horizontalalignment('left')

        def sci_format(x,lim):
            # Manual formatting: use scientific notation with 1e6 as the base
            if x == 0:
                return "0"
            else:
                return f"{x/1e6:.1f}e6"

        major_formatter = FuncFormatter(sci_format)
        if metric == "Total Tokens":
            # Set tick marks to [0, 0.3e6 0.6e6, 0.9e6, 1.2e6]
            cbar.set_ticks([0, 300000, 600000, 900000, 1200000])
            cbar.ax.xaxis.set_major_formatter(major_formatter)

        # for idx, tick in enumerate(ax1.yaxis.get_major_ticks()):
        #     tick.label1.set_verticalalignment('top')
        #     if idx in [1, 5, 10, 14]:
        #         ms = 150
        #     elif idx == 9:
        #         ms = 250
        #     else:
        #         continue
        #     tick.tick1line.set_markersize(ms)



        metric_str = metric.replace(" ", "_").lower()
        fig1.tight_layout()
        fig1.savefig(f'heatmap_{metric_str}.pdf')
        # plt.show()
        # exit()
        plt.close()

def resource_usage(n_seeds=10):
    '''
    Count the total number of input, thinking, and output tokens, and time summed across problems and algorithms.
    '''
    results = all_llm_tables(n_seeds, aggregate="sum")
    input_token_keys = [
        'integrated_input_tokens', 'pddl_input_tokens', 'sampling_input_tokens'
    ]
    thinking_token_keys = [
        'integrated_thinking_tokens', 'pddl_thinking_tokens', 'sampling_thinking_tokens'
    ]
    output_token_keys = [
        'integrated_output_tokens', 'pddl_output_tokens', 'sampling_output_tokens'
    ]
    time_keys = [
        'integrated_time', 'pddl_time', 'sampling_time'
    ]
    total_input_tokens = sum(results[key].sum().sum() for key in input_token_keys)
    total_thinking_tokens = sum(results[key].sum().sum() for key in thinking_token_keys)
    total_output_tokens = sum(results[key].sum().sum() for key in output_token_keys)
    total_time = sum(results[key].sum().sum() for key in time_keys)

    # Calculate token cost using $0.3 per 1M input tokens and $2.5 per 1M output (including thinking) tokens
    input_token_cost = total_input_tokens / 1e6 * 0.3
    output_token_cost = (total_thinking_tokens + total_output_tokens) / 1e6 * 2.5
    total_cost = input_token_cost + output_token_cost

    print(f"Total input tokens: {total_input_tokens:.0f}")
    print(f"Total thinking tokens: {total_thinking_tokens:.0f}")
    print(f"Total output tokens: {total_output_tokens:.0f}")
    print(f"Total token cost ($): {total_cost:.2f} (input: ${input_token_cost:.2f}, output+thinking: ${output_token_cost:.2f})")
    # Print total time in hours
    print(f"Total time (s): {total_time:.2f}")

def dollars_table(n_seeds=10):
    '''
    Generate a table that shows the total dollar cost for input, output, and thinking tokens for each algorithm/parameter combination and problem.
    '''
    results = all_llm_tables(n_seeds, aggregate="mean")
    input_token_keys = [
        'integrated_input_tokens', 'pddl_input_tokens', 'sampling_input_tokens'
    ]
    thinking_token_keys = [
        'integrated_thinking_tokens', 'pddl_thinking_tokens', 'sampling_thinking_tokens'
    ]
    output_token_keys = [
        'integrated_output_tokens', 'pddl_output_tokens', 'sampling_output_tokens'
    ]
    cost_results = defaultdict(lambda: defaultdict(float))
    for algorithm in results[input_token_keys[0]].index:
        for problem in results[input_token_keys[0]].columns:
            total_input_tokens = sum(results[key].loc[algorithm, problem] for key in input_token_keys)
            total_thinking_tokens = sum(results[key].loc[algorithm, problem] for key in thinking_token_keys)
            total_output_tokens = sum(results[key].loc[algorithm, problem] for key in output_token_keys)
            input_token_cost = total_input_tokens / 1e6 * 0.3
            thinking_token_cost = total_thinking_tokens / 1e6 * 2.5
            output_token_cost = total_output_tokens / 1e6 * 2.5

            cost_results[problem, 'Input'][algorithm] = input_token_cost
            cost_results[problem, 'Thinking'][algorithm] = thinking_token_cost
            cost_results[problem, 'Output'][algorithm] = output_token_cost
    
    # Convert to DataFrame for better visualization
    df = pd.DataFrame.from_dict({k: {kk: v for kk, v in v2.items()} for k, v2 in cost_results.items()}, orient='index').fillna(0).T
    return df

def stacked_bar_dollars_chart(n_seeds=10):
    df = dollars_table(n_seeds=n_seeds)
    # Drop rows that have "No LLM" in level 1 of the index
    df = df[~df.index.get_level_values(1).str.contains('No LLM')]
    df = df.sort_index(axis=1)
    df = df.sort_index(axis=0, key=lambda x: [sort_keys[i] for i in x.get_level_values(0)])
    # For each problem, create a stacked bar chart with a bar for each algorithm/parameter combination and sections for Input/Thinking/Output costs
    for problem in df.columns.levels[0]:
        problem_df = df[problem].copy()
        print(problem_df.to_markdown())
        if "Rovers" in problem:
            # Drop the "Bilevel" rows
            problem_df = problem_df[problem_df.index.get_level_values(0) != "Bilevel"]

        # New df with the cost types (current columns) as the inner level of a column multi-index and level 2 of the index as the outer level of the column multi-index
        problem_df = problem_df.unstack(level=2)
        problem_df.columns = problem_df.columns.swaplevel(0, 1)
        problem_df = problem_df.sort_index(axis=1, level=0, key=lambda x: [sort_keys[i] for i in x])
        print(problem_df)

        parent_methods = ['Adaptive', 'Bilevel']
        second_level_methods = ['Direct', 'Thinking']
        third_level_methods = ['PDDL', 'Poses', 'PDDL+Poses', 'Integrated']

        group_width = 0.8
        bar_width = group_width / len(third_level_methods)

        x = []
        subticks = []
        labels = []
        sublabels = []
        sublabels_map = {
            'PDDL': 'Pd',
            'Poses': 'Ps',
            'PDDL+Poses': 'PP',
            'Integrated': 'In'
        }

        fig, ax = plt.subplots(figsize=(10, 4.5))
        pos = 0

        for parent in parent_methods:
            if "Rovers" in problem and parent == "Bilevel":
                continue
            for second in second_level_methods:
                row = problem_df.loc[parent, second]
                for i, third in enumerate(third_level_methods):
                    vals = row[third]
                    # Plot a stacked bar at pos + i * bar_width with the values in vals
                    bottom = 0
                    for j, (cost_type, val) in enumerate(vals.items()):
                        ax.bar(pos + i * bar_width, val, width=bar_width, bottom=bottom, color=color_list[j])
                        bottom += val
                    subticks.append(pos + i * bar_width)
                    sublabels.append(f"{sublabels_map[third]}")
                labels.append(f"{second}\n{parent}")
                x.append(pos + group_width / 2 - bar_width / 2)
                pos += 1
            pos += 0.5
        # Add sublabels without tickmarks at the subticks positions
        for xtick, sublabel in zip(subticks, sublabels):
            ax.text(xtick, -0.05, sublabel, ha='center', va='top', transform=ax.get_xaxis_transform())

        ax.set_xticks(x)
        # Space tick marks to make see the sublabels
        ax.set_xticklabels(labels, rotation=0, ha="center", y=-0.15)
        ax.set_ylabel('Cost ($)')
        ax.set_xlabel('Algorithm / LLM')
        # Space title out a bit more
        ax.set_title(f'Token Costs for {problem}', pad=40)
        plt.ylim(0, 0.5)
        x_max = len(second_level_methods) * len(parent_methods) + 0.5
        plt.xlim(-0.5, x_max)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if "Rovers" in problem:
            # Remove all axis lines, but leave ticks
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            x_max = len(second_level_methods) * (len(parent_methods) - 1)
            # Add lines to simulate axis from (-1, 0) to (x_max, 1)
            plt.plot([-0.5, -0.5], [0, ax.get_ylim()[1]], color='black', linewidth=1.5)
            plt.plot([-0.5, x_max], [0, 0], color='black', linewidth=1.5)

        plt.tight_layout()
        problem_str = problem.replace(" ", "_").replace("\makecell{", "").replace("}", "").replace("\\", "").replace("k=", "")
        plt.savefig(f'costs_{problem_str}.pdf')
        # plt.show()
        plt.close()
        
    # Legend
    legend_handles = []
    for i, cost_type in enumerate(['Input', 'Output', 'Thinking']):
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color_list[i], label=cost_type))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.legend(handles=legend_handles, loc='upper right', ncol=1, title='Cost Type')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('costs_legend.pdf')

def chat_object_to_dialog(chat_history, out_path):
    '''
    Convert a Gemini chat history list to a dialog text file that can be viewed in a text editor.
    '''
    with open(out_path, 'w') as f:
        for content in chat_history:
            role = content.role.capitalize()
            if role == 'User':
                assert len(content.parts) == 1
            for part in content.parts:
                role = content.role.capitalize()     # re-set the role
                if part.thought:
                    role = '(Thoughts:'
                else:
                    role = role + ':' + ' ' * (len('Thoughts:') - len(role))
                text = content.parts[0].text
                # Indent the content by the length of the role + 2 (for ": ")
                indent = ' ' * (len('Thoughts:') + 2)
                text = text.replace('\n', '\n' + indent)
                if part.thought:
                    text = text + '\n)'
                f.write(f"{role} {text}\n\n")

def parse_all_chats(n_seeds):
    '''
    Chats are stored in llm_info['chat_histories'] as dictionaries mapping idx to chat history objects.
    '''
    for exp_dir in glob(os.path.join('results/', '*')):
        if not os.path.isdir(exp_dir):
            continue
        exp_name = os.path.basename(exp_dir)
        match = re.match(r'^(.*?)examples\.pybullet\.(.*?)\.run (.*?)_(\d+)$', exp_name)
        if not match:
            print(f"Skipping directory with unexpected name format: {exp_name}")
            continue
        algorithm_llm_args, domain, domain_args, seed = match.groups()
        seed = int(seed)
        if seed >= n_seeds:
            continue
        llm_info_path = os.path.join(exp_dir, 'llm_info.pkl')
        if os.path.exists(llm_info_path):
            with open(llm_info_path, 'rb') as f:
                llm_info = pickle.load(f)
            chat_histories = llm_info.get('chat_histories', {})
            chat_types = llm_info.get('chat_types', {})
            # remove 'results' from exp_dir
            chat_dir = exp_dir.replace('results/', 'chats/')
            os.makedirs(chat_dir, exist_ok=True)
            for idx, chat_object in chat_histories.items():
                chat_type = chat_types.get(idx, 'unknown')
                out_path = os.path.join(chat_dir, f'chat_{idx}_{chat_type}.txt')
                chat_object_to_dialog(chat_object, out_path)
            print(f"Parsed {len(chat_histories)} chats in {exp_dir}")
        else:
            print(f"No llm_info.pkl found in {exp_dir}")

if __name__ == '__main__':
    success_rate_cld_barchart(n_seeds=50)
    # time_cld_barchart(n_seeds=50, success_rate_thresh=0.3, test='wilcoxon')
    # stacked_bar_failures_chart(n_seeds=50)
    # heatmap_plots(n_seeds=50)
    # stacked_bar_dollars_chart(n_seeds=50)
    # resource_usage(n_seeds=50)
    # parse_all_chats(n_seeds=50)
