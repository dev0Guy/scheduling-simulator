# E2 job creation

## Equations

$$\rho_j = \frac{j}{N-1}, \qquad j = 0,\dots,N-1$$

$$\tilde\rho_j^{(a)} = \operatorname{clip}\bigl(\rho_j + \epsilon_j^{(a)},\,0,\,1\bigr), \qquad
\tilde\rho_j^{(p)} = \operatorname{clip}\bigl(\rho_j + \epsilon_j^{(p)},\,0,\,1\bigr), \qquad
\epsilon_j^{(a)}, \epsilon_j^{(p)} \sim \mathcal{U}(-0.35,\,0.35)$$

$$a_j = \max\Bigl(0,\ \bigl\lfloor \tilde\rho_j^{(a)} \lfloor T/2 \rfloor \bigr\rfloor + k_j\Bigr), \qquad k_j \sim \mathcal{U}\{-3,\dots,3\}$$

$$p_j = \operatorname{clip}\Bigl(\bigl\lfloor T - \tilde\rho_j^{(p)}(T-1) \bigr\rfloor + k'_j,\ 1,\ T\Bigr), \qquad k'_j \sim \mathcal{U}\{-5,\dots,5\}$$

$$\ell = \max\bigl(1,\ \lfloor C/10 \rfloor\bigr), \qquad
h = \max\bigl(\ell + 1,\ \lfloor C/2 \rfloor\bigr), \qquad
d_j \sim \mathcal{U}\{\ell,\dots,h\}$$

$$u_j[r,\tau] =
\begin{cases}
d_j & \text{if } \tau < p_j \\
0 & \text{otherwise}
\end{cases}
\qquad r = 1,\dots,R, \quad \tau = 0,\dots,T-1$$

## Legend

- **N**: number of jobs (N = 10)
- **R**: number of resources (R = 3)
- **T**: length of the usage window in ticks (T = 10), also the maximum job size
- **C**: machine capacity per resource and tick (C = 255)
- **j**: job index, 0 to N−1
- **ρⱼ**: position of job j in the order, between 0 and 1
- **εⱼ⁽ᵃ⁾, εⱼ⁽ᵖ⁾**: independent random shifts, uniform on [−0.35, 0.35], drawn separately for arrival and for size
- **ρ̃ⱼ⁽ᵃ⁾, ρ̃ⱼ⁽ᵖ⁾**: jittered positions used for arrival and for size, clipped to [0, 1]
- **kⱼ**: random integer jitter on the arrival time, uniform on {−3, …, 3}
- **k′ⱼ**: random integer jitter on the size, uniform on {−5, …, 5}
- **aⱼ**: arrival time of job j (the tick at which it becomes PENDING), aⱼ ≥ 0
- **pⱼ**: size of job j (number of ticks it runs), 1 ≤ pⱼ ≤ T
- **ℓ, h**: lowest and highest demand, ℓ = 25 and h = 127 for C = 255
- **dⱼ**: demand of job j, one integer for every resource and tick, uniform on {ℓ, …, h}
- **uⱼ[r, τ]**: usage of resource r by job j at tick τ after it starts: dⱼ for τ < pⱼ, otherwise 0
- **clip(x, a, b)**: min(max(x, a), b)
- **⌊x⌋**: x rounded down to an integer
- **U(·)**: uniform distribution. U(a, b) is continuous on [a, b]; U{a, …, b} is over the integers from a to b
