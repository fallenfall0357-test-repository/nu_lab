from audiocaps_download import Downloader
d = Downloader(root_path='audiocaps/', n_jobs=16)
d.download(format='wav')