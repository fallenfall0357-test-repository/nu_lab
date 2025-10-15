import tarfile

bz2_path = r"nu_lab\diffusion_model\data\QM9\Data for 133885 GDB-9 molecules.bz2"
extract_dir = r"nu_lab\diffusion_model\data\QM9\extracted"

with tarfile.open(bz2_path, "r:bz2") as tar:
    tar.extractall(path=extract_dir)

print("解压完成！")