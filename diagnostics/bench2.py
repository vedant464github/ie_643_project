import torch, time, torch.nn as nn

for n in (14, 8, 4, 2, 1):
    torch.set_num_threads(n)
    m = nn.Sequential(
        nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(),
        nn.Conv1d(32, 32, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(32, 5))
    opt = torch.optim.Adam(m.parameters())
    x, y = torch.randn(64, 1, 30), torch.randn(64, 5)

    def step():
        opt.zero_grad()
        ((m(x) - y) ** 2).mean().backward()
        opt.step()

    for _ in range(20):
        step()
    t = time.time()
    for _ in range(300):
        step()
    print(n, "threads:", round(time.time() - t, 2), "s per 300 steps")