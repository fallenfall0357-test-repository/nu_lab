from datasets import load_dataset

ds = load_dataset("jp1924/AudioCaps", cache_dir=".", split=["train", "validation", "test"])

print(ds)

