import torch

path = "./final_pt/in-motion/strats_pos_0.pt"
data = torch.load(path, map_location="cpu", weights_only=False)

sample = data["train"][0]

print("label:", sample["label"])
print("times   :", sample["times"])
print("features:", sample["features"])
print("values  :", sample["values"])

# time, beacon 이름과 같이 보기
beacon_cols = data["meta"]["beacon_cols"]  # ['B1','B2','B3','B4','B5']

for t, f, v in zip(sample["times"], sample["features"], sample["values"]):
    print(f"t={int(t):2d}, beacon={beacon_cols[int(f)]}, value={v:.3f}")
