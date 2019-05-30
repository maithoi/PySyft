import syft as sy
import torch as th
from syft.workers import WebsocketClientWorker

hook = sy.TorchHook(th)

charlie = WebsocketClientWorker(hook, id="charlie", host="localhost", port="8776")
alice = WebsocketClientWorker(hook, id="alice", host="localhost", port="8777")
crypto_provider = WebsocketClientWorker(hook, id="crypto_provider", host="localhost", port="8779")

# charlie = sy.VirtualWorker(hook, id="c", verbose=True)
# alice = sy.VirtualWorker(hook, id="a", verbose=True)
# crypto_provider = sy.VirtualWorker(hook, id="cp", verbose=True)

x_data = (
    th.tensor([[1.5], [2.5], [3.5], [15.2], [50.5]])
    .fix_precision()
    .share(charlie, alice, crypto_provider=crypto_provider)
)

# print(x_data)
w = th.tensor([[1.0]]).fix_precision().share(charlie, alice, crypto_provider=crypto_provider)

z = x_data.matmul(w.t())

print("-" * 100)
print()
print()
print(z.get())
print()
print()
print("-" * 100)
