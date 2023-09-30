import random
import numpy as np


def aggregate_stagewise_values(stagewise):
    means = list()
    medians = list()

    for s in stagewise:
        s_ranks = np.array(stagewise[s])
        means.append(np.mean(s_ranks))
        medians.append(np.median(s_ranks))

    return means, medians


def aggregate_planrank_stats(topic_ref_plan_ranks):
    rank_score_stats = list()
    nneighbor_stats = list()
    rank_score_stagewise = dict()
    n_neighbor_stagewise = dict()

    for clause_plan_neighbors in topic_ref_plan_ranks.values():
        for plan_neighbors in clause_plan_neighbors:
            _, _, plan_ranks, n_neighbors = plan_neighbors
            plan_ranks = np.array(plan_ranks)
            n_neighbors = np.array(n_neighbors)

            rank_score_stats.append((plan_ranks.mean(), np.median(plan_ranks)))
            nneighbor_stats.append((n_neighbors.mean(), np.median(n_neighbors)))

            for i in range(len(plan_ranks)):
                if i not in rank_score_stagewise:
                    rank_score_stagewise[i] = [plan_ranks[i]]
                    n_neighbor_stagewise[i] = [n_neighbors[i]]
                else:
                    rank_score_stagewise[i].append(plan_ranks[i])
                    n_neighbor_stagewise[i].append(n_neighbors[i])

    rank_agg_mean = np.mean([m for m, _ in rank_score_stats])
    rank_agg_median = np.median([m for _, m in rank_score_stats])
    n_neigh_agg_mean = np.mean([m for m, _ in nneighbor_stats])
    n_neigh_agg_median = np.median([m for _, m in nneighbor_stats])
    rank_means, rank_medians = aggregate_stagewise_values(rank_score_stagewise)
    n_neigh_means, n_neigh_medians = aggregate_stagewise_values(n_neighbor_stagewise)

    stats = {
        'aggregated': {
            'rank': {'mean': rank_agg_mean, 'median': rank_agg_median},
            'n_neigh': {'mean': n_neigh_agg_mean, 'median': n_neigh_agg_median}
        },
        'stagewise': {
            'rank': {'mean': rank_means, 'median': rank_medians},
            'n_neigh': {'mean': n_neigh_means, 'median': n_neigh_medians}
        },
        'collated': {
            'mean_median_rank_per_plan': rank_score_stats,
            'mean_median_n_neigh_per_plan': nneighbor_stats,
            'stagewise_ranks': rank_score_stagewise,
            'stagewise_n_neigh': n_neighbor_stagewise
        }
    }

    return stats


def plan_rank_simplesum(topic, ref_plan, graph, vocab, n_steps):
    topic_indx = vocab.lookup_indices([topic])[0]
    i = topic_indx
    ranks = list()
    n_neighbors = list()

    for ref_kwd in ref_plan[:n_steps]:
        neighbors = [n for n in graph.neighbors(i)]
        n_neighbors.append(len(neighbors))
        
        scores = list()
        for j in neighbors:
            score = 0
            topic_edge = graph.get_edge_data(topic_indx, j)
            if topic_edge:
                score += topic_edge['weight']
            score += graph.get_edge_data(i, j)['weight']
            scores.append(score)
        scores = np.array(scores)
        sorted_ids = np.argsort(scores)[::-1]
        candidates = [neighbors[idx] for idx in sorted_ids]

        rank_detected = False
        for r, c in enumerate(candidates):
            cwd = vocab.lookup_token(c)
            if cwd == ref_kwd:
                ranks.append(r+1)
                rank_detected = True
                break
        if not rank_detected:
            ranks.append(len(candidates)+1)
        
        i = vocab.lookup_indices([ref_kwd])[0]

    return ranks, n_neighbors


def generate_plan_simplesum(topic, ref_kwds, graph, vocab, stepwise_topk, n_steps):
    topic_indx = vocab.lookup_indices([topic])[0]
    i = topic_indx
    plan = list()
    n_neighbors = list()
    ref_nodes = set(vocab.lookup_indices(list(ref_kwds)))

    for step_no in range(n_steps):
        neighbors = [n for n in graph.neighbors(i)]
        n_neighbors.append(len(neighbors))
        
        scores = list()
        for j in neighbors:
            score = 0
            topic_edge = graph.get_edge_data(topic_indx, j)
            if topic_edge:
                score += topic_edge['weight']
            score += graph.get_edge_data(i, j)['weight']
            scores.append(score)
        scores = np.array(scores)
        sorted_ids = np.argsort(scores)[::-1]
        threshold = stepwise_topk[step_no]
        candidates = [neighbors[idx] for idx in sorted_ids][:threshold]
        candidates = list(set(candidates).difference(set(plan) | {3}))
        if len(candidates) < 1:
            break
        
        i = random.sample(candidates, 1)[0]
        for c in candidates:
            if c in ref_nodes:
                i = c
                ref_nodes = ref_nodes - {c}
                break
        plan.append(i)
    plan.extend(list(ref_nodes))
    plan = vocab.lookup_tokens(plan)

    return plan, n_neighbors
