import torch, time
x = torch.randn(1500, 1500)
for n in (14, 8, 4, 1):
    torch.set_num_threads(n)
    x @ x
    t = time.time()
    for _ in range(10):
        x @ x
    print(n, "threads:", round(time.time() - t, 2), "s")