# The Scheduling Model, Explained From Scratch

This document explains the whole system assuming no background beyond "I've heard of neural networks." If a term is unfamiliar, it's defined the first time it's used.

---

## 1. What problem are we even solving?

Imagine a single printer in an office, and 10 print jobs that show up throughout the day at different times. Each job takes a different amount of time to print, and once the printer starts a job, it can't be interrupted — it has to finish before the printer can take the next one.

Your job is to decide, every minute, whether the printer should start printing one of the waiting jobs, or just sit idle a bit longer (maybe because nothing has arrived yet, or you're waiting for a strategic reason). The goal: get everything printed as fast as possible, on average — minimize the total time all the jobs together spend waiting plus printing.

That's the whole task. In our simulator it's the same idea, just with "printer" renamed "machine" and "print job" renamed "job," and sometimes there's more than one machine.

**A simple strategy that works surprisingly well**: whenever the printer is free, print whichever waiting job is *shortest*. This is called **SJF (Shortest-Job-First)**, and it turns out to be nearly the best possible strategy when there's one machine — mathematicians proved this decades ago. Beating SJF is genuinely hard, and it's the benchmark this whole project measures itself against.

---

## 2. What is reinforcement learning, in plain terms?

Instead of writing an explicit rule like "always pick the shortest job" (that's what SJF does), we train a neural network to *learn* a good rule by trial and error.

The setup has three pieces, and this vocabulary shows up everywhere in the code:

- **State (a.k.a. observation)**: everything the network is allowed to see right now — which jobs exist, how big they are, whether they've arrived yet, how busy the machine is, etc.
- **Action**: a choice the network makes — "start job #3 on machine #0" or "do nothing this minute."
- **Reward**: a number telling the network how good or bad that choice turned out to be. Here, every minute a job spends waiting or running costs `-1` point — so the network is directly rewarded for finishing everything as fast as possible.

The network plays through thousands of simulated days (called **episodes**), and after each action it gets a reward. Over time, it adjusts itself to prefer actions that lead to higher total reward. The specific training algorithm we use is called **PPO (Proximal Policy Optimization)** — a well-established, stable method for this kind of trial-and-error learning. You don't need to understand PPO's internals to follow the rest of this doc; just think of it as "the trainer" that nudges the network's decisions to be better over time based on the rewards it earned.

---

## 3. What does the network actually see and do?

### What it sees (the state)

For **every job** (whether it's arrived yet or not — more on that below), the network sees:

| Signal | Plain-English meaning |
|---|---|
| `status` | Has this job arrived yet? Is it waiting? Running? Already finished? |
| `size` | How long this job takes to run |
| `wait_time` | How long it's been sitting in the queue so far |
| `ttl` | If it's currently running, how much longer until it's done |
| `arrival` | How many minutes until this job shows up (if it hasn't yet) |

That last one is important and a bit unusual: real life doesn't let you see the future, but **our simulator does** — every job's arrival time is known from the very start of the episode, even for jobs that haven't shown up yet. This means the network can, in principle, plan ahead: "a smaller job is arriving in 2 minutes, is it worth waiting for it instead of starting this bigger one now?" SJF can never do this — it only ever looks at jobs that have already arrived. This foresight is the network's one structural advantage over SJF, and a lot of this project's later refinements (described below) were about helping the network actually use that advantage well, instead of squandering it.

For **every machine**, the network also sees how busy it currently is.

### What it does (the action)

Every minute, the network picks exactly one of:
- **Skip** (do nothing this minute), or
- **Start job J on machine M**, for some specific waiting job and some specific machine.

If there are 10 jobs and 1 machine, that's 11 possible choices each minute (10 "start job J" options plus "skip"). The network produces a numeric score for each option and picks based on those scores (higher score = more likely to be chosen).

---

## 4. The neural network itself

This is the "brain" that turns the state into those action scores. It's built in layers, and each layer solves a specific problem. Here's the walkthrough, from raw numbers to final decision.

### Step 1 — Turn each job into a vector of numbers ("embedding")

A **neural network layer** is just a small mathematical function that takes a list of numbers in and produces a (usually different-length) list of numbers out, with some internal adjustable "dials" (called **weights**) that get tuned during training.

Each job's raw signals (status, size, wait_time, ttl, arrival — described above) get combined into a single list of numbers, and passed through a small stack of these layers (called the **job encoder**). The output is called that job's **embedding** — a richer, learned representation of "what this job is like," rather than just its raw stats. Think of it as the network's internal, more useful way of describing a job, similar to how you might mentally summarize a job as "urgent and quick" rather than listing every stat.

The exact same encoder (the same dials) is applied to every job — the network doesn't have a separate brain per job slot, it has one shared way of understanding "a job," applied 10 times.

### Step 2 — Let jobs compare themselves to each other ("attention")

Here's a subtlety: if each job is processed completely independently in Step 1, the network never gets to directly compare two specific jobs to each other — it can't represent "I am smaller than the job that's currently waiting, but bigger than the one arriving soon."

**Attention** is the mechanism that fixes this. In plain terms: every job "looks around" at every other job and decides how much attention to pay to each one, then blends in a bit of each based on that attention weight. After this step, each job's embedding is no longer just about itself — it's been updated to reflect its *relationship* to the other jobs currently in the picture. This is the same core mechanism used in large language models (it's literally the same word — "attention" — because it's mathematically the same idea, just applied to jobs instead of words in a sentence).

We apply this job-to-job attention with one exception: a job that's already finished (`COMPLETED`) is excluded from being looked at by others, since its old stats are no longer relevant and would just add noise to everyone else's comparison.

### Step 3 — Do the same thing for machines

Machines get their own small encoder too — instead of raw job stats, each machine's numbers describe how busy it currently is. And just like jobs, machines can attend to each other (useful when there's more than one machine, so the network can reason about "which machine is less loaded right now").

### Step 4 — Let jobs and machines "talk" to each other (dual attention)

This is the newest and most advanced piece. So far, jobs only compared themselves to other jobs, and machines only compared themselves to other machines — but a job's embedding never incorporated anything about the machines' actual state, and vice versa. **Dual attention** adds a second attention step where jobs attend directly to machine embeddings (so a job "knows" how busy the machines are) and machines attend directly to job embeddings (so a machine "knows" what's competing for its time). This idea is adapted from a research paper on a related scheduling problem (see the [architecture diagram](dual_attention_network.html) for the full picture and citation).

### Step 5 — Summarize everything into one "big picture" vector

After all that back-and-forth, the network needs one single summary vector describing the *whole* current situation (not any one job or machine specifically) — this is used for two things: helping score every individual action, and letting a separate part of the network (the **critic**, explained below) judge how good the overall situation is. This summary is built by averaging together all the job embeddings and all the machine embeddings (with jobs weighted by a learned importance score, since not all jobs matter equally to the big picture).

### Step 6 — Score every possible action

For every (job, machine) pair, the network takes that specific job's embedding, that specific machine's embedding, and the big-picture summary, and runs them through one more small stack of layers to produce a single number: how good does this action look? The "skip" action gets its own score, computed just from the big-picture summary. All these scores get compared (technically: turned into probabilities via a step called **softmax**), and the network picks — during training, sometimes explores a lower-scored option on purpose; during evaluation, it just takes the highest-scoring one.

### The critic (a second, simpler piece)

Separately from all of the above (which is called the **actor** — it's the part that picks actions), there's a **critic**: a small network that just looks at the big-picture summary and predicts "roughly how much total reward should I expect from here until the episode ends?" This prediction isn't used to pick actions directly — it's used *during training* to help the trainer (PPO) figure out whether a given action turned out better or worse than expected, which is what actually drives the learning signal. You can think of the critic as a coach watching the game and constantly updating its guess of the final score, so it can tell the player "that move was better than I expected" or "worse than I expected" in real time.

---

## 5. Putting it all together: one full decision, step by step

1. The simulator hands the network the current state: every job's stats, every machine's load.
2. Each job and machine gets encoded into its own embedding (Step 1/3 above).
3. Jobs attend to jobs, machines attend to machines (Step 2/3).
4. Jobs and machines cross-attend to each other (Step 4, dual attention).
5. Everything gets pooled into one big-picture summary (Step 5).
6. Every (job, machine) action gets a score, "skip" gets a score, and the network picks the action with the highest score (Step 6).
7. The simulator executes that action, advances time by one minute, and hands back a reward (usually `-1` per minute per still-waiting-or-running job, `+1` for a successful start, big bonuses/penalties at the very end).
8. Repeat until every job is finished or the day (episode) runs out of time.

During training, this whole loop runs for millions of simulated minutes across thousands of episodes, and the trainer (PPO) gradually adjusts every layer's internal dials so that, over time, the actions chosen lead to higher total reward — i.e., jobs get finished faster.

---

## 6. How do we know if it's actually good?

Two ways, both documented in [`docs/ppo_vs_sjf_results.md`](ppo_vs_sjf_results.md) and [`docs/experiment_log.md`](experiment_log.md):

1. **Compare against SJF** across 5000 simulated days: currently the trained network beats SJF outright about 14% of the time, loses about 4% of the time, and ties the rest (mostly because for many job configurations, SJF's simple rule already happens to be the best possible choice).
2. **Compare against the mathematically optimal schedule.** Since the simulator reveals every job's future arrival time upfront, it's technically possible to compute the *exact best possible schedule* for each simulated day (using a technique called **dynamic programming** — essentially, trying all reasonable combinations efficiently instead of brute-force checking literally every ordering). Against that true optimum, the trained network matches the perfect schedule on about 91% of simulated days, versus SJF's 81% — a meaningfully bigger gap than the head-to-head comparison alone would suggest.
