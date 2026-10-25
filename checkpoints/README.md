# Checkpoints

The source archive does not contain model weights. The two author-trained
descriptor checkpoints are distributed separately. Public download links have
not been published yet.

Place the two downloaded `.pth` files in this directory, or verify and copy them
from their download folder using:

```bash
python scripts/install_weights.py --source-dir /path/to/downloaded/assets
```

`manifest.json` records filenames, SHA-256 hashes, sizes, tensor dimensions,
and intended datasets. This installer validates hashes before copying. Do not
upload checkpoints through ordinary Git commits; attach them to GitHub Releases
or a suitable model host. Download URLs will be added after a public repository
and release tag exist; no placeholder URL is presented as a working download.

Assets:

- `gsp_gatedreg_3dmatch.pth`: indoor GSP descriptor for 3DMatch and 3DLoMatch.
- `gsp_gatedreg_kitti.pth`: KITTI GSP descriptor.

Official PREDATOR confidence-model weights are separate from these descriptor
checkpoints. Obtain them from the [upstream project](https://github.com/prs-eth/OverlapPredator#datasets-and-pretrained-models),
following [docs/CONFIDENCE.md](../docs/CONFIDENCE.md).
