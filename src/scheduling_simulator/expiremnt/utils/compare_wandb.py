import wandb


PROJECT = "dev0guy/cluster-scheduling-simulator"

RUN_A = "33xw8fke"  # PPO
RUN_B = "3lm3wkgs"  # Random


METRICS = {
    "eval/reward": "higher",
    "eval/avg_wait_time": "lower",
    "eval/avg_completion_time": "lower",
}

X_AXIS = "evaluation/episode"


def compare_metric(
    run_a,
    run_b,
    metric: str,
    direction: str,
):
    # ---------------------------------------------------------
    # Fetch X + metric
    # ---------------------------------------------------------

    history_a = run_a.history(
        keys=[X_AXIS, metric],
        pandas=True,
    )

    history_b = run_b.history(
        keys=[X_AXIS, metric],
        pandas=True,
    )

    # ---------------------------------------------------------
    # Keep only relevant columns and remove missing values
    # ---------------------------------------------------------

    history_a = history_a[
        [X_AXIS, metric]
    ].dropna()

    history_b = history_b[
        [X_AXIS, metric]
    ].dropna()

    # ---------------------------------------------------------
    # Rename metric columns
    # ---------------------------------------------------------

    history_a = history_a.rename(
        columns={
            metric: "ppo",
        }
    )

    history_b = history_b.rename(
        columns={
            metric: "random",
        }
    )

    # ---------------------------------------------------------
    # Match PPO and Random by evaluation episode
    # ---------------------------------------------------------

    comparison = history_a.merge(
        history_b,
        on=X_AXIS,
        how="inner",
    )

    if comparison.empty:
        print()
        print(metric)
        print("-" * 60)
        print(
            f"No matching {X_AXIS} values found."
        )
        return

    # ---------------------------------------------------------
    # Determine which run is better
    # ---------------------------------------------------------

    if direction == "higher":

        comparison["ppo_better"] = (
            comparison["ppo"]
            > comparison["random"]
        )

        comparison["random_better"] = (
            comparison["random"]
            > comparison["ppo"]
        )

    else:

        comparison["ppo_better"] = (
            comparison["ppo"]
            < comparison["random"]
        )

        comparison["random_better"] = (
            comparison["random"]
            < comparison["ppo"]
        )

    comparison["equal"] = (
        comparison["ppo"]
        == comparison["random"]
    )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    total = len(comparison)

    ppo_better = int(
        comparison["ppo_better"].sum()
    )

    random_better = int(
        comparison["random_better"].sum()
    )

    equal = int(
        comparison["equal"].sum()
    )

    ppo_percentage = (
        ppo_better / total * 100
    )

    random_percentage = (
        random_better / total * 100
    )

    equal_percentage = (
        equal / total * 100
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print()
    print(metric)
    print("=" * 60)

    print(
        f"Comparable evaluation episodes : {total}"
    )

    print(
        f"PPO better                     : "
        f"{ppo_better}/{total} "
        f"({ppo_percentage:.2f}%)"
    )

    print(
        f"Random better                  : "
        f"{random_better}/{total} "
        f"({random_percentage:.2f}%)"
    )

    print(
        f"Equal                          : "
        f"{equal}/{total} "
        f"({equal_percentage:.2f}%)"
    )

    # ---------------------------------------------------------
    # Mean
    # ---------------------------------------------------------

    ppo_mean = comparison["ppo"].mean()
    random_mean = comparison["random"].mean()

    print()
    print("Mean")
    print("-" * 60)

    print(f"PPO    : {ppo_mean:.4f}")
    print(f"Random : {random_mean:.4f}")

    if random_mean != 0:

        difference = (
            (ppo_mean - random_mean)
            / abs(random_mean)
            * 100
        )

        print(
            f"Difference: {difference:+.2f}%"
        )

    # ---------------------------------------------------------
    # Show comparison for every evaluation
    # ---------------------------------------------------------

    print()
    print("Per evaluation")
    print("-" * 60)

    for _, row in comparison.iterrows():

        episode = row[X_AXIS]
        ppo = row["ppo"]
        random = row["random"]

        if ppo > random:
            if direction == "higher":
                winner = "PPO"
            else:
                winner = "Random"

        elif random > ppo:
            if direction == "higher":
                winner = "Random"
            else:
                winner = "PPO"

        else:
            winner = "Equal"

        print(
            f"episode={episode}: "
            f"PPO={ppo:.4f}, "
            f"Random={random:.4f} "
            f"-> {winner}"
        )


def compare_runs(
    project: str,
    run_a_id: str,
    run_b_id: str,
):
    api = wandb.Api()

    run_a = api.run(
        f"{project}/{run_a_id}"
    )

    run_b = api.run(
        f"{project}/{run_b_id}"
    )

    print(
        f"Run A: {run_a.name} ({run_a.id})"
    )

    print(
        f"Run B: {run_b.name} ({run_b.id})"
    )

    print()
    print(
        f"Comparison X-axis: {X_AXIS}"
    )

    print("=" * 60)

    for metric, direction in METRICS.items():

        compare_metric(
            run_a,
            run_b,
            metric,
            direction,
        )


if __name__ == "__main__":

    compare_runs(
        project=PROJECT,
        run_a_id=RUN_A,
        run_b_id=RUN_B,
    )

# reward function:
#  reward calc (job_status) => -1 * (job with status running or pending)
#  ErrorAction (can't allocate) -10 + reward_calac
#  skip_time: reward_calac
#  sucess allocation: 1
#
# NOTICE: Start from here
# TODO: add the distance between poitns (sum)

# eval/reward
# ============================================================
# Comparable evaluation episodes : 100
# PPO better                     : 100/100 (100.00%)
# Random better                  : 0/100 (0.00%)
# Equal                          : 0/100 (0.00%)

# eval/avg_wait_time
# ============================================================
# Comparable evaluation episodes : 100
# PPO better                     : 59/100 (59.00%)
# Random better                  : 37/100 (37.00%)
# Equal                          : 4/100 (4.00%)


# eval/avg_completion_time
# ============================================================
# Comparable evaluation episodes : 100
# PPO better                     : 59/100 (59.00%)
# Random better                  : 37/100 (37.00%)
# Equal                          : 4/100 (4.00%)
#


 #                         ENVIRONMENT
 #                              │
 #                              │
 #                              ▼
 #        ┌────────────────────────────────────────┐
 #        │             OBSERVATION                │
 #        │                                        │
 #        │ Jobs:                                  │
 #        │ status, size, wait_time, TTL           │
 #        │                                        │
 #        │ Machines:                              │
 #        │ capacity, usage                        │
 #        └───────────────────┬────────────────────┘
 #                            │
 #                            ▼
 #                ┌───────────────────────┐
 #                │   FEATURE EXTRACTOR   │
 #                └───────────┬───────────┘
 #                            │
 #             ┌──────────────┴──────────────┐
 #             │                             │
 #             ▼                             ▼
 #       STATUS EMBEDDING              Machine state
 #             │                             │
 #             ▼                             │
 #       Job features                        │
 #             │                             │
 #    ┌────────┼────────┐                    │
 #    │        │        │                    │
 #  size    wait_time   TTL                  │
 #    │        │        │                    │
 #    └────────┼────────┘                    │
 #             ▼                             │
 #          Job MLP                          │
 #             │                             │
 #             ▼                             │
 #       Job embeddings                      │
 #             │                             │
 #             ▼                             │
 #        ATTENTION                          │
 #             │                             │
 #             ▼                             │
 #      Global embedding                     │
 #             │                             │
 #       ┌─────┴───────────────┐             │
 #       │                     │             │
 #       ▼                     ▼             │
 #   ACTOR / POINTER        CRITIC           │
 #       │                     │             │
 #       │                     ▼             │
 #       │                   V(s)            │
 #       │                                   │
 #       ▼                                   │
 # ┌─────────────────────────────┐           │
 # │ For every machine × job:   │◄──────────┘
 # │                             │
 # │ global + machine + job      │
 # │          │                  │
 # │          ▼                  │
 # │       Score Net             │
 # └─────────────┬───────────────┘
 #               │
 #               ▼
 #        ┌──────────────┐
 #        │ Action logits│
 #        └──────┬───────┘
 #               │
 #               ▼
 #        ┌──────────────┐
 #        │ Normal PPO   │
 #        │ Categorical  │
 #        │ distribution │
 #        └──────┬───────┘
 #               │
 #               ▼
 #             ACTION
