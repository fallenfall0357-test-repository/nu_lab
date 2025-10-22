from audiocaps_download import Downloader # type: ignore
d = Downloader(root_path='.', n_jobs=16)
d.download(format='wav')
