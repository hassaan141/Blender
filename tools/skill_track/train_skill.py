#!/usr/bin/env python3
"""PPO for the MuJoCo skill-tracking residual policy (subprocess-parallel envs).

Writes <out>/policy.pt (actor, critic, obs normalizer, config) every --save-every
iterations and at the end, plus train_log.jsonl. Bounded by --iterations.
"""
import argparse, json, math, multiprocessing as mp, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from skill_env import SkillEnv, OBS_DIM


def mlp(i, o):
    return nn.Sequential(nn.Linear(i, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, o))


class Actor(nn.Module):
    def __init__(self):
        super().__init__(); self.net = mlp(OBS_DIM, 12); self.log_std = nn.Parameter(torch.full((12,), -1.0))
    def forward(self, x): return self.net(x)
    def dist(self, x): return Normal(self(x), self.log_std.clamp(-3., 0.).exp())


class Normalizer:
    def __init__(self, n): self.mean = np.zeros(n); self.var = np.ones(n); self.count = 1e-4
    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0), len(x); d = bm - self.mean; tot = self.count + bc
        self.mean = self.mean + d*bc/tot
        self.var = (self.var*self.count + bv*bc + d**2*self.count*bc/tot)/tot; self.count = tot
    def __call__(self, x): return np.clip((x - self.mean)/np.sqrt(self.var + 1e-8), -5, 5).astype(np.float32)


def worker(conn, ref, seeds, stand_prob, focus=None, rscale=None, push=0.):
    envs = [SkillEnv(ref, s, stand_prob, focus, rscale, push) for s in seeds]
    while True:
        cmd, data = conn.recv()
        if cmd == "reset": conn.send(np.stack([e.reset() for e in envs]))
        elif cmd == "step":
            out = []
            for e, a in zip(envs, data):
                o, r, d, i = e.step(a)
                if d: o = e.reset()
                out.append((o, r, d, i["fell"], e.k))
            conn.send(out)
        elif cmd == "close": return


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", default=str(Path(__file__).parent / "out/timid_ref.npz")); p.add_argument("--out", required=True)
    p.add_argument("--iterations", type=int, default=1500); p.add_argument("--workers", type=int, default=20); p.add_argument("--envs-per-worker", type=int, default=4)
    p.add_argument("--horizon", type=int, default=48); p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save-every", type=int, default=100); p.add_argument("--device", default="cuda:0")
    p.add_argument("--stand-prob", type=float, default=.3); p.add_argument("--init", help="resume actor/critic/normalizer from a policy.pt")
    p.add_argument("--residual-scale", type=lambda s: [float(x) for x in s.split(",")], help="1 or 12 values (rad per unit action)")
    p.add_argument("--push", type=float, default=0., help="per-step probability of a random root velocity kick")
    p.add_argument("--focus", type=lambda s: tuple(float(x) for x in s.split(",")), help="lo,hi,prob: extra RSI starts in reference frames [lo,hi]")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(a.seed); np.random.seed(a.seed); dev = torch.device(a.device)
    conns = []
    for w in range(a.workers):
        c1, c2 = mp.Pipe(); seeds = [a.seed*10000 + w*a.envs_per_worker + i for i in range(a.envs_per_worker)]
        mp.Process(target=worker, args=(c2, a.ref, seeds, a.stand_prob, a.focus, a.residual_scale, a.push), daemon=True).start(); conns.append(c1)
    for c in conns: c.send(("reset", None))
    obs = np.concatenate([c.recv() for c in conns]); N = len(obs)
    actor, critic = Actor().to(dev), mlp(OBS_DIM, 1).to(dev)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=a.lr)
    norm = Normalizer(OBS_DIM)
    if a.init:
        c = torch.load(a.init, map_location='cpu', weights_only=False); actor.load_state_dict(c['actor']); critic.load_state_dict(c['critic'])
        norm.mean, norm.var, norm.count = c['obs_mean'], c['obs_var'], 1e6
    t0 = time.time(); log = (out / "train_log.jsonl").open("w")

    def save(tag):
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "obs_mean": norm.mean, "obs_var": norm.var,
                    "config": vars(a), "iteration": tag}, out / "policy.pt")

    ep_len, ep_lens, ep_ends = np.zeros(N), [], []
    for it in range(1, a.iterations + 1):
        B = {k: [] for k in ("o", "a", "lp", "v", "r", "d")}; raw = []
        for _ in range(a.horizon):
            raw.append(obs); x = torch.as_tensor(norm(obs), device=dev)
            with torch.no_grad():
                dist = actor.dist(x); act = dist.sample(); lp = dist.log_prob(act).sum(-1); v = critic(x).squeeze(-1)
            A = act.cpu().numpy()
            for w, c in enumerate(conns): c.send(("step", A[w*a.envs_per_worker:(w+1)*a.envs_per_worker]))
            res = [r for c in conns for r in c.recv()]
            obs = np.stack([r[0] for r in res]); rew = np.array([r[1] for r in res]); done = np.array([r[2] for r in res], np.float32)
            ep_len += 1
            for i, r in enumerate(res):
                if r[2]: ep_lens.append(ep_len[i]); ep_ends.append(0. if r[3] else 1.); ep_len[i] = 0
            for k, val in zip(B, (x, act, lp, v, torch.as_tensor(rew, device=dev, dtype=torch.float32), torch.as_tensor(done, device=dev))): B[k].append(val)
        norm.update(np.concatenate(raw))
        with torch.no_grad(): last = critic(torch.as_tensor(norm(obs), device=dev)).squeeze(-1)
        adv = torch.zeros(a.horizon, N, device=dev); g = torch.zeros(N, device=dev)
        for t in reversed(range(a.horizon)):
            nv = last if t == a.horizon - 1 else B["v"][t+1]; alive = 1 - B["d"][t]
            delta = B["r"][t] + .99*nv*alive - B["v"][t]; g = delta + .99*.95*alive*g; adv[t] = g
        ret = adv + torch.stack(B["v"])
        X, Aa, LP = torch.cat(B["o"]), torch.cat(B["a"]), torch.cat(B["lp"]); ADV, RET = adv.reshape(-1), ret.reshape(-1)
        ADV = (ADV - ADV.mean())/(ADV.std() + 1e-8); idx = np.arange(len(X)); kl_sum = 0.
        for _ in range(5):
            np.random.shuffle(idx)
            for mb in np.array_split(idx, 4):
                j = torch.as_tensor(mb, device=dev); d = actor.dist(X[j]); nlp = d.log_prob(Aa[j]).sum(-1); ratio = (nlp - LP[j]).exp()
                pg = -torch.min(ratio*ADV[j], ratio.clamp(.8, 1.2)*ADV[j]).mean(); vl = (critic(X[j]).squeeze(-1) - RET[j]).pow(2).mean()
                loss = pg + .5*vl - .001*d.entropy().sum(-1).mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 1.); opt.step()
                kl_sum += float((LP[j] - nlp).mean())
        row = {"it": it, "t": round(time.time() - t0, 1), "reward": float(torch.stack(B["r"]).mean()),
               "ep_len": float(np.mean(ep_lens[-200:])) if ep_lens else 0., "complete_frac": float(np.mean(ep_ends[-200:])) if ep_ends else 0.,
               "std": float(actor.log_std.exp().mean()), "kl": kl_sum/20}
        log.write(json.dumps(row) + "\n"); log.flush()
        if it % 25 == 0: print(json.dumps(row), flush=True)
        if it % a.save_every == 0: save(it)
    save(a.iterations)
    for c in conns: c.send(("close", None))
    print(f"CHECKPOINT={out/'policy.pt'}", flush=True)


if __name__ == "__main__":
    mp.set_start_method("fork"); main()
